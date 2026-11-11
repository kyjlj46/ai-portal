"""메인 화면에 마켓 Agent 고정 (📌 홈에 추가)."""

import pytest
from fastapi.testclient import TestClient

import web
from portal import home, store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOAD_DIR", tmp_path / "uploads")
    return TestClient(web.app)


def test_pin_and_unpin_keeps_order_and_saves(client):
    assert client.get("/api/home/pins").json() == {"pins": []}
    assert client.put("/api/home/pins/crawler").json() == {"pins": ["crawler"]}
    assert client.put("/api/home/pins/translator").json() == {"pins": ["crawler", "translator"]}
    assert client.put("/api/home/pins/crawler").json() == {"pins": ["crawler", "translator"]}  # 두 번 눌러도 한 번
    assert (store.DATA_DIR / "home.json").exists()
    assert client.get("/api/home/pins").json() == {"pins": ["crawler", "translator"]}
    assert client.delete("/api/home/pins/crawler").json() == {"pins": ["translator"]}


def test_pin_rejects_unknown_and_items_already_on_home(client):
    assert client.put("/api/home/pins/nope").status_code == 404
    res = client.put("/api/home/pins/summary")  # 기존 '문서 요약' 카드가 이미 홈에 있다
    assert res.status_code == 400 and "이미 메인 화면" in res.json()["detail"]


def test_pins_limit_and_drop_removed_agents(monkeypatch):
    home._write(["translator", "deleted_custom_agent", "summary"])
    assert home.pins() == ["translator"]  # 지운 Agent · 기본 카드 항목은 빠진다
    monkeypatch.setattr(home, "MAX_PINS", 1)
    with pytest.raises(ValueError, match="1개까지"):
        home.pin("crawler")


def test_broken_file_reads_as_empty():
    store.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (store.DATA_DIR / "home.json").write_text("{broken", encoding="utf-8")
    assert home.pins() == []
