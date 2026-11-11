"""웹 크롤링 Agent: crawl_site 도구, URL 입력창 → 요청문, 마켓·앱 표시."""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage

import web
from portal import catalog, extra_tools, market, stream


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "UPLOAD_DIR", tmp_path / "uploads")
    return TestClient(web.app)


def _site(pages: dict[str, tuple[str, str, list[str]]]):
    """가짜 _fetch_page: 주소 → (제목, 본문, 링크). 없는 주소는 404."""
    fetched = []

    def fake(url):
        fetched.append(url)
        if url not in pages:
            return "페이지를 읽지 못했습니다 (HTTP 404)."
        title, text, links = pages[url]
        return {"url": url, "title": title, "text": text, "links": links, "plain": False}

    return fake, fetched


SITE = {
    "https://ex.com/": ("홈", "회사 소개입니다.", ["https://ex.com/notice/", "https://other.com/x", "https://ex.com/logo.png",
                                               "https://ex.com/about"]),
    "https://ex.com/notice/": ("공지", "공지 목록", ["https://ex.com/notice/1", "https://ex.com/notice/2", "https://ex.com/"]),
    "https://ex.com/about": ("소개", "연혁", []),
    "https://ex.com/notice/1": ("공지 1", "11월 점검 안내", []),
    "https://ex.com/notice/2": ("공지 2", "요금 변경", []),
}


def test_crawl_site_follows_same_site_links_only(monkeypatch):
    fake, fetched = _site(SITE)
    monkeypatch.setattr(extra_tools, "_fetch_page", fake)
    monkeypatch.setattr(extra_tools, "_robots", lambda start: lambda url: True)
    out = extra_tools.crawl_site.invoke({"url": "https://ex.com/", "max_pages": 10})
    assert out.startswith("크롤링: https://ex.com/ · 5페이지 읽음")
    assert "### [공지 1] https://ex.com/notice/1\n11월 점검 안내" in out
    assert "other.com" not in "".join(fetched) and "logo.png" not in "".join(fetched)  # 다른 사이트·이미지는 읽지 않음
    assert fetched.count("https://ex.com/") == 1  # 같은 페이지는 한 번만


def test_crawl_site_limits_pages_and_lists_rest(monkeypatch):
    fake, fetched = _site(SITE)
    monkeypatch.setattr(extra_tools, "_fetch_page", fake)
    monkeypatch.setattr(extra_tools, "_robots", lambda start: lambda url: True)
    out = extra_tools.crawl_site.invoke({"url": "https://ex.com/", "max_pages": 2})
    assert "2페이지 읽음" in out and len(fetched) == 2
    assert "## 찾았지만 읽지 않은 링크\n- https://ex.com/about" in out


def test_crawl_site_same_path_robots_and_failures(monkeypatch):
    pages = dict(SITE)
    pages["https://ex.com/notice/"] = ("공지", "공지 목록", ["https://ex.com/about", "https://ex.com/notice/1",
                                                          "https://ex.com/notice/2", "https://ex.com/notice/9"])
    fake, fetched = _site(pages)
    monkeypatch.setattr(extra_tools, "_fetch_page", fake)
    monkeypatch.setattr(extra_tools, "_robots", lambda start: lambda url: not url.endswith("/2"))
    out = extra_tools.crawl_site.invoke({"url": "https://ex.com/notice/", "same_path": True})
    assert "https://ex.com/about" not in fetched  # 경로 밖
    assert "https://ex.com/notice/2" not in fetched and "robots.txt 로 막힌 1개" in out
    assert "## 읽지 못한 페이지\n- https://ex.com/notice/9 — 페이지를 읽지 못했습니다 (HTTP 404)." in out


def test_crawl_site_blocks_internal_start():
    out = extra_tools.crawl_site.invoke({"url": "http://127.0.0.1:8000/"})
    assert out.startswith("시작 페이지를 읽지 못했습니다") and "내부망" in out


def test_text_extractor_collects_links_even_in_nav():
    p = extra_tools._TextExtractor()
    p.feed('<nav><a href="/a">A</a></nav><p>본문 <a href="b.html#top">B</a></p>')
    assert p.links == ["/a", "b.html#top"] and "A" not in p.text()


def test_crawler_in_market_with_url_flag_and_tools():
    item = market.BY_ID["crawler"]
    assert item.needs_url and item.uses_doc is False and item.category == "리서치"
    assert next(m for m in market.public() if m["id"] == "crawler")["needs_url"] is True
    assert next(m for m in market.public() if m["id"] == "web_search")["needs_url"] is False
    assert catalog.get("m:crawler").doc == "none"
    sub = market.subagent("crawler", model=None, doc_tools=["DOC"])
    assert [t.name for t in sub["tools"]] == ["fetch_url", "crawl_site"]


def test_crawl_results_become_sources_and_steps():
    text = "크롤링: https://ex.com/ · 2페이지 읽음\n\n### [홈] https://ex.com/\n본문\n\n### [공지] https://ex.com/notice/\n목록"
    msgs = [ToolMessage(text, tool_call_id="1", name="crawl_site")]
    assert stream.web_sources(msgs) == [("https://ex.com/", "홈"), ("https://ex.com/notice/", "공지")]
    assert stream._describe_result(msgs[0], False)["text"] == "→ 크롤링: https://ex.com/ · 2페이지 읽음"


def test_with_urls_prepends_and_validates():
    q = web.with_urls("공지만 표로", ["https://ex.com/notice", " https://ex.com/notice ", ""])
    assert q == "[크롤링할 URL]\n- https://ex.com/notice\n\n공지만 표로"
    assert web.with_urls("", ["https://ex.com"]).endswith("이 페이지의 내용을 정리해 주세요.")
    assert web.with_urls("그대로", None) == "그대로"
    with pytest.raises(ValueError, match="https://"):
        web.with_urls("q", ["javascript:alert(1)"])
    with pytest.raises(ValueError, match="5개"):
        web.with_urls("q", [f"https://ex.com/{i}" for i in range(6)])


def test_agent_stream_sends_urls_in_request(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost/v1")
    seen = {}

    def fake_stream(path, q, history=None, agent_id="qa"):
        seen.update(q=q, agent=agent_id)
        return iter([{"type": "final", "answer": "정리", "sources": [], "steps": [], "doc_id": None, "agent": agent_id}])

    monkeypatch.setattr(stream, "ask_stream", fake_stream)
    res = client.post("/api/agents/m:crawler/stream", json={"question": "", "urls": ["https://ex.com/"]})
    events = [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]
    assert events[-1]["answer"] == "정리" and seen["agent"] == "m:crawler"
    assert seen["q"].startswith("[크롤링할 URL]\n- https://ex.com/\n\n")
    res = client.post("/api/agents/m:crawler/stream", json={"question": "q", "urls": ["ftp://ex.com"]})
    assert res.status_code == 400 and "https://" in res.json()["detail"]


def test_custom_agents_may_use_crawl_tool():
    from portal import custom_agents

    assert "crawl_site" in custom_agents.ALLOWED_TOOLS
