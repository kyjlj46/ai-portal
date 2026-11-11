"""마켓 Agent 전용 도구: 청킹 점검, OCR 점검·재인식, 웹 검색·페이지 읽기.

TOOLS 의 이름으로 market.MarketAgent.tools 에서 고른다.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import statistics
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from langchain_core.tools import tool

from . import store
from .tools import _parse_pages, chunk_document, ocr_pages, preview_chunks

LOW_QUALITY = 0.85


# ─────────────────────────── 청킹 ───────────────────────────


def chunk_settings(doc_id: str) -> dict:
    path = store.doc_dir(doc_id) / "chunking.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"chunk_size": 800, "chunk_overlap": 120}  # chunk_document 기본값


@tool
def chunk_stats(doc_id: str) -> str:
    """문서의 청킹 결과 통계: 청크 수, 길이 분포, 페이지별 개수, 섹션, 문장 중간에서 잘린 청크 비율, 현재 설정."""
    doc_id = store.resolve(doc_id)  # 표시(D3)도 받는다
    try:
        meta = store.load_meta(doc_id)
    except ValueError:
        return f"알 수 없는 문서: {doc_id}"
    chunks = store.load_chunks(doc_id)
    if not chunks:
        return "아직 청크가 없습니다."
    sizes = [len(c["text"]) for c in chunks]
    per_page: dict[int, int] = {}
    for c in chunks:
        per_page[c["page"]] = per_page.get(c["page"], 0) + 1
    sections = list(dict.fromkeys(c["section"] for c in chunks if c["section"]))
    cut = sum(1 for c in chunks if not re.search(r"[.!?。다요함음됨임\)\]|:]\s*$", c["text"].strip()))
    cfg = chunk_settings(doc_id)
    empty_pages = [p for p in range(1, meta["page_count"] + 1) if p not in per_page]
    lines = [
        f"문서: {meta['path'].split(chr(92))[-1].split('/')[-1]} · {meta['page_count']}{'구간' if meta.get('unit') == 'section' else '페이지'}",
        f"현재 설정: chunk_size={cfg['chunk_size']}, chunk_overlap={cfg['chunk_overlap']}",
        f"청크 수: {len(chunks)}개 (페이지당 평균 {len(chunks) / max(1, len(per_page)):.1f}개)",
        f"길이(자): 최소 {min(sizes)} / 중앙값 {int(statistics.median(sizes))} / 평균 {sum(sizes) // len(sizes)} / 최대 {max(sizes)}",
        f"100자 미만 짧은 청크: {sum(1 for s in sizes if s < 100)}개",
        f"문장 끝이 아닌 곳에서 끝난 청크: {cut}개 ({cut * 100 // len(chunks)}%)",
        "페이지별 청크 수: " + ", ".join(f"p.{p}={n}" for p, n in sorted(per_page.items())),
        f"청크가 없는 페이지: {empty_pages or '없음'}",
        f"인식된 섹션 {len(sections)}개: " + (" / ".join(sections[:12]) + (" …" if len(sections) > 12 else "") if sections else "없음"),
    ]
    return "\n".join(lines)


# ─────────────────────────── OCR ───────────────────────────


@tool
def ocr_report(doc_id: str) -> str:
    """페이지별 OCR 결과 보고: 인식 방식(ocr/text), 신뢰도, 글자 수, 글자 상자 수, 확인이 필요한 페이지."""
    doc_id = store.resolve(doc_id)  # 표시(D3)도 받는다
    try:
        meta = store.load_meta(doc_id)
    except ValueError:
        return f"알 수 없는 문서: {doc_id}"
    rows, low = [], []
    info = {p["page"]: p for p in meta.get("pages", [])}
    for page in store.load_pages(doc_id):
        p = page["page"]
        chars = len(page["text"])
        boxes = info.get(p, {}).get("boxes", "-")
        q = page.get("quality", 0)
        flag = ""
        if page["source"] == "ocr" and (q < LOW_QUALITY or chars < 30):
            flag = "⚠ 확인 필요" + (" (신뢰도 낮음)" if q < LOW_QUALITY else " (글자 거의 없음)")
            low.append(p)
        rows.append(f"p.{p} | {page['source']} | 신뢰도 {q} | {chars}자 | 글자상자 {boxes} | {flag or '정상'}")
    kind = meta.get("kind", "pdf")
    head = [
        f"문서 종류: {kind} · {meta['page_count']}{'구간' if meta.get('unit') == 'section' else '페이지'}"
        + (" · 페이지 이미지 있음 (다시 OCR 가능)" if meta.get("pdf", meta["path"]) else " · 텍스트 문서 (OCR 없음)"),
        f"확인이 필요한 페이지: {low or '없음'} (신뢰도 {LOW_QUALITY} 미만 또는 30자 미만)",
        "페이지 | 방식 | 신뢰도 | 글자 수 | 글자상자 | 상태",
    ]
    return "\n".join(head + rows)


@tool
def reocr_pages(doc_id: str, pages: str, dpi: int = 300) -> str:
    """지정 페이지(예: '2,4-6')를 더 높은 해상도로 다시 OCR 하고, 같은 청킹 설정으로 검색 인덱스를 다시 만든다."""
    doc_id = store.resolve(doc_id)  # 표시(D3)도 받는다
    try:
        meta = store.load_meta(doc_id)
    except ValueError:
        return f"알 수 없는 문서: {doc_id}"
    targets = _parse_pages(pages, meta["page_count"])
    if not targets:
        return f"페이지 범위를 확인하세요 (1~{meta['page_count']})."
    if len(targets) > 20:
        return "한 번에 20페이지까지 다시 OCR 할 수 있습니다."
    before = {p["page"]: p for p in store.load_pages(doc_id)}
    report = ocr_pages.invoke({"doc_id": doc_id, "pages": ",".join(map(str, targets)), "dpi": max(150, min(dpi, 400))})
    after = {p["page"]: p for p in store.load_pages(doc_id)}
    for entry in meta.get("pages", []):  # 문서 정보의 페이지 품질도 갱신
        if entry["page"] in targets and entry["page"] in after:
            entry["quality"] = after[entry["page"]]["quality"]
            entry["chars"] = len(after[entry["page"]]["text"])
    store.save_meta(doc_id, meta)
    cfg = chunk_settings(doc_id)
    rechunk = chunk_document.invoke({"doc_id": doc_id, **cfg})
    diff = "\n".join(
        f"p.{p}: 신뢰도 {before.get(p, {}).get('quality', '-')} → {after[p]['quality']}, "
        f"글자 {len(before.get(p, {}).get('text', ''))} → {len(after[p]['text'])}자"
        for p in targets if p in after
    )
    return f"다시 OCR 한 결과 (dpi={dpi}):\n{diff}\n\n{report}\n\n재인덱싱: {rechunk.strip()}"


# ─────────────────────────── 웹 검색 ───────────────────────────

MAX_FETCH_BYTES = 2_000_000


@tool
def web_search(query: str, max_results: int = 6) -> str:
    """웹에서 검색해 상위 결과의 제목·주소·요약을 돌려준다. 자세한 내용은 fetch_url 로 페이지를 읽는다."""
    from ddgs import DDGS

    backend = os.getenv("WEB_SEARCH_BACKEND", "bing,google,yahoo")
    n = max(1, min(int(max_results), 10))
    # 따옴표 정확 일치·긴 문장 검색어는 결과가 없을 때가 많아, 비면 핵심어만으로 한 번 더 찾는다
    simple = " ".join(re.sub(r"[\"'“”‘’]", " ", query).split()[:8])
    results, error = [], None
    for q in dict.fromkeys([query, simple]):
        try:
            results = DDGS().text(q, max_results=n, backend=backend, region=os.getenv("WEB_SEARCH_REGION", "kr-kr"))
        except Exception as e:  # noqa: BLE001 - 검색 실패는 에이전트가 검색어를 바꾸도록 알려준다
            error = e
            continue
        if results:
            break
    if not results:
        reason = f" ({type(error).__name__})" if error and "No results" not in str(error) else ""
        return f"검색 결과가 없습니다{reason}. 더 짧고 일반적인 검색어로 바꿔 보세요 (따옴표 없이)."
    return "\n\n".join(
        f"[{i}] {r.get('title', '').strip()}\n{r.get('href', '')}\n{(r.get('body') or '').strip()[:300]}"
        for i, r in enumerate(results, 1)
    )


def _public_host(url: str) -> bool:
    """사설·루프백·링크로컬 주소로 가는 요청을 막는다 (서버 내부망을 읽지 못하게)."""
    host = urlparse(url).hostname
    if not host:
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False
    return True


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe", "aside"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in raw.splitlines()]
        return "\n".join(ln for ln in lines if len(ln) > 1)


@tool
def fetch_url(url: str, max_chars: int = 6000) -> str:
    """웹 페이지를 읽어 본문 텍스트를 돌려준다 (http/https 공개 주소만)."""
    import httpx

    max_chars = max(500, min(int(max_chars), 15000))
    for _ in range(4):  # 리디렉션은 직접 따라가며 주소를 매번 다시 검사한다
        if urlparse(url).scheme not in ("http", "https"):
            return "http/https 주소만 읽을 수 있습니다."
        if not _public_host(url):
            return "내부망·사설 주소는 읽을 수 없습니다."
        try:
            with httpx.stream("GET", url, timeout=12, follow_redirects=False,
                              headers={"User-Agent": "Mozilla/5.0 (compatible; work-agent/1.0)", "Accept-Language": "ko,en;q=0.8"}) as res:
                if res.is_redirect and "location" in res.headers:
                    url = urljoin(url, res.headers["location"])
                    continue
                if res.status_code >= 400:
                    return f"페이지를 읽지 못했습니다 (HTTP {res.status_code})."
                ctype = res.headers.get("content-type", "")
                if not any(t in ctype for t in ("text/html", "text/plain", "application/xhtml")):
                    return f"글이 아닌 형식이라 읽지 않았습니다 ({ctype or '알 수 없음'})."
                body = b""
                for chunk in res.iter_bytes():
                    body += chunk
                    if len(body) > MAX_FETCH_BYTES:
                        break
                enc = res.encoding or "utf-8"
        except httpx.HTTPError as e:
            return f"페이지를 읽지 못했습니다 ({type(e).__name__})."
        html = body.decode(enc, errors="replace")
        if "text/plain" in ctype:
            return f"{url}\n{html[:max_chars]}"
        parser = _TextExtractor()
        parser.feed(html)
        text = parser.text()
        return f"[{parser.title.strip()[:120]}] {url}\n{text[:max_chars]}" + ("\n…(이하 생략)" if len(text) > max_chars else "")
    return "리디렉션이 너무 많습니다."


TOOLS = {t.name: t for t in (chunk_stats, preview_chunks, chunk_document, ocr_report, reocr_pages, web_search, fetch_url)}
