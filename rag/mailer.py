"""메일 발송 (SMTP). 사용자가 화면에서 확인하고 '보내기' 를 눌렀을 때만 호출된다 — 에이전트가 스스로 보내지 않는다.

.env 설정:
  SMTP_HOST       예: smtp.gmail.com, smtp.office365.com (비우면 발송 기능 꺼짐 → 화면에서 '메일 앱으로 열기' 만)
  SMTP_PORT       기본 587 (SMTP_SECURITY=ssl 이면 보통 465)
  SMTP_SECURITY   starttls(기본) | ssl | none
  SMTP_USER / SMTP_PASSWORD   로그인 계정 (Gmail 은 앱 비밀번호)
  SMTP_FROM       보내는 사람 (비우면 SMTP_USER). 예: 홍길동 <gildong@example.com>
"""

from __future__ import annotations

import os
import re
import smtplib
import ssl
import time
from email.message import EmailMessage
from email.utils import formatdate, getaddresses, make_msgid, parseaddr

MAX_RECIPIENTS = 20
MAX_BODY = 100_000
RATE_LIMIT = 30  # 1시간에 보낼 수 있는 메일 수 (실수로 반복 발송하는 것을 막는다)
_ADDR = re.compile(r"^[^@\s<>\"',;]+@[^@\s<>\"',;]+\.[^@\s<>\"',;]+$")
_sent: list[float] = []


def config() -> dict | None:
    host = os.getenv("SMTP_HOST", "").strip()
    if not host:
        return None
    security = (os.getenv("SMTP_SECURITY") or "starttls").lower()
    user = os.getenv("SMTP_USER", "").strip()
    return {
        "host": host,
        "port": int(os.getenv("SMTP_PORT") or (465 if security == "ssl" else 587)),
        "security": security if security in ("starttls", "ssl", "none") else "starttls",
        "user": user,
        "password": os.getenv("SMTP_PASSWORD", ""),
        "sender": (os.getenv("SMTP_FROM") or user).strip(),
    }


def public_status() -> dict:
    """화면에 보여줄 정보 (서버 주소·계정·비밀번호는 내보내지 않는다)."""
    cfg = config()
    return {"enabled": bool(cfg and cfg["sender"]), "sender": cfg["sender"] if cfg else ""}


def parse_recipients(value) -> list[str]:
    """'a@x.com, 홍길동 <b@y.com>' 또는 목록 → 주소 목록. 잘못된 주소가 있으면 ValueError."""
    items = value if isinstance(value, list) else [value or ""]
    raw = ",".join(str(v) for v in items).replace(";", ",")
    if "\n" in raw or "\r" in raw:
        raise ValueError("받는 사람에 줄바꿈을 넣을 수 없습니다.")
    out = []
    for name, addr in getaddresses([raw]):
        addr = addr.strip()
        if not addr:
            continue
        if not _ADDR.match(addr):
            raise ValueError(f"올바른 메일 주소가 아닙니다: {addr}")
        if addr.lower() not in (a.lower() for a in out):
            out.append(addr)
    return out


HTML_WRAP = """<!doctype html><html><body style="font-family:'Malgun Gothic',Apple SD Gothic Neo,sans-serif;font-size:14px;line-height:1.6;color:#1d2330">
<style>table{{border-collapse:collapse}}th,td{{border:1px solid #cfd4dc;padding:4px 8px;text-align:left}}th{{background:#f3f5f8}}</style>
{body}
</body></html>"""


def send(to, subject: str, text: str, cc=None, html: str | None = None) -> dict:
    cfg = config()
    if not cfg or not cfg["sender"]:
        raise RuntimeError("메일 발송이 설정되지 않았습니다. .env 에 SMTP_HOST / SMTP_USER / SMTP_PASSWORD / SMTP_FROM 을 넣고 서버를 다시 시작하세요.")
    to_list, cc_list = parse_recipients(to), parse_recipients(cc or [])
    if not to_list:
        raise ValueError("받는 사람을 입력하세요.")
    if len(to_list) + len(cc_list) > MAX_RECIPIENTS:
        raise ValueError(f"받는 사람과 참조는 합쳐서 {MAX_RECIPIENTS}명까지입니다.")
    subject = (subject or "").strip()
    if not subject:
        raise ValueError("제목을 입력하세요.")
    if "\n" in subject or "\r" in subject or len(subject) > 200:
        raise ValueError("제목은 한 줄, 200자 이내여야 합니다.")
    if not (text or "").strip():
        raise ValueError("본문이 비어 있습니다.")
    if len(text) > MAX_BODY or len(html or "") > MAX_BODY * 3:
        raise ValueError("본문이 너무 깁니다.")
    now = time.time()
    _sent[:] = [t for t in _sent if now - t < 3600]
    if len(_sent) >= RATE_LIMIT:
        raise RuntimeError(f"1시간에 {RATE_LIMIT}통까지 보낼 수 있습니다. 잠시 후 다시 시도하세요.")

    msg = EmailMessage()
    msg["From"] = cfg["sender"]
    msg["To"] = ", ".join(to_list)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=(parseaddr(cfg["sender"])[1].split("@")[-1] or None))
    msg.set_content(text)
    if html:
        msg.add_alternative(HTML_WRAP.format(body=html), subtype="html")

    context = ssl.create_default_context()
    smtp_cls = smtplib.SMTP_SSL if cfg["security"] == "ssl" else smtplib.SMTP
    kwargs = {"context": context} if cfg["security"] == "ssl" else {}
    with smtp_cls(cfg["host"], cfg["port"], timeout=20, **kwargs) as s:
        if cfg["security"] == "starttls":
            s.starttls(context=context)
        if cfg["user"]:
            s.login(cfg["user"], cfg["password"])
        s.send_message(msg, to_addrs=to_list + cc_list)
    _sent.append(now)
    return {"ok": True, "message_id": msg["Message-ID"], "to": to_list, "cc": cc_list}
