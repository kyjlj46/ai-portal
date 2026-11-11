"""메일 발송: 설정·검증·확인 절차·SMTP 호출 (가짜 SMTP 서버로)."""

import pytest
from fastapi.testclient import TestClient

import web
from portal import mailer


class FakeSMTP:
    instances: list = []

    def __init__(self, host, port, timeout=None, **kw):
        self.host, self.port, self.calls = host, port, []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, msg, to_addrs=None):
        self.calls.append(("send", msg, to_addrs))


@pytest.fixture
def smtp(monkeypatch):
    for k, v in {"SMTP_HOST": "smtp.example.com", "SMTP_USER": "me@example.com", "SMTP_PASSWORD": "pw-123",
                 "SMTP_FROM": "홍길동 <me@example.com>"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("SMTP_SECURITY", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    FakeSMTP.instances = []
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", FakeSMTP)
    mailer._sent.clear()
    return FakeSMTP


@pytest.fixture
def client():
    return TestClient(web.app)


def test_send_builds_message_with_text_and_html(smtp):
    out = mailer.send("kim@example.com, 이과장 <lee@example.com>", "납품 일정 안내", "안녕하세요.\n• 11/27", cc="park@example.com",
                      html="<p>안녕하세요.</p><ul><li>11/27</li></ul>")
    assert out["to"] == ["kim@example.com", "lee@example.com"] and out["cc"] == ["park@example.com"]
    s = smtp.instances[0]
    assert (s.host, s.port) == ("smtp.example.com", 587) and s.calls[0] == "starttls"
    assert s.calls[1] == ("login", "me@example.com", "pw-123")
    _, msg, rcpts = s.calls[2]
    assert rcpts == ["kim@example.com", "lee@example.com", "park@example.com"]
    assert msg["Subject"] == "납품 일정 안내" and "kim@example.com" in msg["To"] and msg["Cc"] == "park@example.com"
    assert "me@example.com" in msg["From"]
    assert msg.get_body("plain").get_content().startswith("안녕하세요.")
    assert "<li>11/27</li>" in msg.get_body("html").get_content()


def test_ssl_mode_and_port(smtp, monkeypatch):
    monkeypatch.setenv("SMTP_SECURITY", "ssl")
    mailer.send("kim@example.com", "제목", "본문")
    s = smtp.instances[0]
    assert s.port == 465 and "starttls" not in s.calls


@pytest.mark.parametrize("to,subject,text,err", [
    ("", "제목", "본문", "받는 사람"),
    ("not-an-email", "제목", "본문", "올바른 메일 주소"),
    ("kim@example.com\nBcc: evil@x.com", "제목", "본문", "줄바꿈"),
    ("kim@example.com", "제목\nBcc: evil@x.com", "본문", "제목은 한 줄"),
    ("kim@example.com", "", "본문", "제목"),
    ("kim@example.com", "제목", "  ", "본문"),
    (",".join(f"u{i}@example.com" for i in range(21)), "제목", "본문", "20명"),
])
def test_send_validation(smtp, to, subject, text, err):
    with pytest.raises(ValueError, match=err):
        mailer.send(to, subject, text)
    assert not smtp.instances  # 검증에 걸리면 서버에 연결도 하지 않는다


def test_rate_limit(smtp, monkeypatch):
    monkeypatch.setattr(mailer, "RATE_LIMIT", 2)
    mailer.send("a@example.com", "1", "x")
    mailer.send("a@example.com", "2", "x")
    with pytest.raises(RuntimeError, match="1시간에 2통"):
        mailer.send("a@example.com", "3", "x")


def test_not_configured(monkeypatch, client):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    assert mailer.public_status() == {"enabled": False, "sender": ""}
    res = client.post("/api/email/send", json={"to": "a@example.com", "subject": "s", "text": "t", "confirm": True})
    assert res.status_code == 503 and "SMTP_HOST" in res.json()["detail"]


def test_api_requires_confirmation_and_hides_secrets(smtp, client):
    res = client.post("/api/email/send", json={"to": "a@example.com", "subject": "s", "text": "t"})
    assert res.status_code == 400 and not smtp.instances  # 확인 없이 보내지 않는다
    res = client.post("/api/email/send", json={"to": ["a@example.com"], "subject": "회의 결과", "text": "본문", "confirm": True})
    assert res.status_code == 200 and res.json()["to"] == ["a@example.com"]
    assert client.post("/api/email/send", json={"to": "bad", "subject": "s", "text": "t", "confirm": True}).status_code == 400
    status = client.get("/api/status").text
    assert '"enabled":true' in status and "pw-123" not in status and "smtp.example.com" not in status


def test_smtp_failure_is_reported(smtp, client, monkeypatch):
    class Down(FakeSMTP):
        def login(self, user, password):
            raise mailer.smtplib.SMTPAuthenticationError(535, b"bad credentials")

    monkeypatch.setattr(mailer.smtplib, "SMTP", Down)
    res = client.post("/api/email/send", json={"to": "a@example.com", "subject": "s", "text": "t", "confirm": True})
    assert res.status_code == 502 and "SMTPAuthenticationError" in res.json()["detail"]
