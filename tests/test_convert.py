"""Word / PowerPoint / 한글 / 텍스트 문서 처리 테스트. (Word·한글 프로그램 변환은 가짜로 대체해 실패 경로를 검증)"""

import struct
import zipfile

import docx
import pytest

from portal import convert, ingest, store, tools


def test_kind_of():
    assert convert.kind_of("a.PDF") == "pdf"
    assert convert.kind_of("a.docx") == "word"
    assert convert.kind_of("a.PPTX") == "ppt" and convert.kind_of("a.ppt") == "ppt"
    assert convert.kind_of("a.hwpx") == "hwp"
    assert convert.kind_of("a.md") == "text"
    with pytest.raises(ValueError):
        convert.kind_of("a.exe")


def test_read_text_cp949(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("수행팀 PM은 홍길동".encode("cp949"))
    assert convert.read_text(p) == "수행팀 PM은 홍길동"


def test_split_keeps_paragraphs():
    text = "\n\n".join(f"문단{i} " + "가" * 900 for i in range(5))
    segs = convert._split(text, size=2000)
    assert len(segs) == 3 and all(len(s) <= 2000 for s in segs)
    assert segs[0].startswith("문단0") and segs[1].startswith("문단2")


def test_docx_pages_with_heading_table_and_page_break(tmp_path):
    d = docx.Document()
    d.add_heading("사업 계획", 1)
    d.add_paragraph("총 예산은 13.33억 원이다.")
    t = d.add_table(rows=2, cols=2)
    for r, row in enumerate([["이름", "역할"], ["홍길동", "PM"]]):
        for c, v in enumerate(row):
            t.cell(r, c).text = v
    d.add_page_break()
    d.add_paragraph("2쪽 내용")
    path = tmp_path / "a.docx"
    d.save(path)

    segs = convert.extract_segments(path)
    assert len(segs) == 2
    assert segs[0].startswith("# 사업 계획") and "| 홍길동 | PM |" in segs[0]
    assert segs[1] == "2쪽 내용"


def test_hwpx_text_and_table(tmp_path):
    ns = 'xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph" xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section"'
    xml = f"""<hs:sec {ns}>
      <hp:p><hp:run><hp:t>첫 문단</hp:t></hp:run></hp:p>
      <hp:p><hp:run><hp:tbl><hp:tr>
        <hp:tc><hp:subList><hp:p><hp:run><hp:t>이름</hp:t></hp:run></hp:p></hp:subList></hp:tc>
        <hp:tc><hp:subList><hp:p><hp:run><hp:t>역할</hp:t></hp:run></hp:p></hp:subList></hp:tc>
      </hp:tr></hp:tbl></hp:run></hp:p>
      <hp:p pageBreak="1"><hp:run><hp:t>다음 쪽</hp:t></hp:run></hp:p>
    </hs:sec>"""
    path = tmp_path / "a.hwpx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Contents/section0.xml", xml)
    assert convert.extract_segments(path) == ["첫 문단\n\n| 이름 | 역할 |\n|---|---|", "다음 쪽"]


def test_hwp_para_skips_control_chars():
    def w(s):
        return s.encode("utf-16-le")

    # 일반 글자 + 확장 제어(코드 11, 뒤에 7 WCHAR) + 탭(코드 9, 인라인 8 WCHAR) + 문단 끝(13)
    raw = w("수행") + struct.pack("<H", 11) + b"\x00" * 14 + w("팀") + struct.pack("<H", 9) + b"\x00" * 14 + w("PM") + struct.pack("<H", 13)
    assert convert._hwp_para(raw) == "수행팀\tPM\n"


def test_ingest_text_file_without_ocr(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "ocr_page", lambda *a, **k: pytest.fail("텍스트 문서는 OCR 하지 않아야 함"))
    path = tmp_path / "memo.md"
    path.write_text("# 회의록\n\n수행팀 PM은 홍길동이다.", encoding="utf-8")

    doc_id = ingest.ingest(str(path))
    meta = store.load_meta(doc_id)
    assert meta["kind"] == "text" and meta["unit"] == "section" and meta["pdf"] is None
    assert "홍길동" in tools.search_chunks.invoke({"query": "수행팀 PM", "doc_id": doc_id})


def test_ingest_word_falls_back_to_text_when_conversion_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "to_pdf", lambda src, dst: None)
    d = docx.Document()
    d.add_paragraph("연구 인력은 87명이다.")
    path = tmp_path / "보고서.docx"
    d.save(path)

    doc_id = ingest.ingest(str(path))
    meta = store.load_meta(doc_id)
    assert meta["kind"] == "word" and meta["unit"] == "section"
    assert store.load_page(doc_id, 1)["text"] == "연구 인력은 87명이다."


def test_ingest_word_uses_converted_pdf(tmp_path, monkeypatch, sample_pdf):
    """변환에 성공하면 변환된 PDF 로 페이지 OCR 을 한다."""
    import shutil

    def fake_to_pdf(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sample_pdf, dst)
        return str(dst)

    pages = []
    monkeypatch.setattr(ingest, "to_pdf", fake_to_pdf)
    monkeypatch.setattr(
        ingest, "ocr_page",
        lambda pdf, page, dpi=200: pages.append((pdf, page)) or {"text": f"{page}쪽", "raw": "", "quality": 1, "boxes": 1},
    )
    path = tmp_path / "a.docx"
    docx.Document().save(path)

    doc_id = ingest.ingest(str(path))
    meta = store.load_meta(doc_id)
    assert meta["unit"] == "page" and meta["pdf"].endswith("converted.pdf")
    assert sorted(p for _, p in pages) == [1, 2] and all(pdf == meta["pdf"] for pdf, _ in pages)


def _make_pptx(path, slides):
    """최소 PPTX: presentation.xml 순서가 파일 번호와 다를 때도 발표 순서를 따르는지 확인하려고 역순으로 등록."""
    P = (
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    )
    with zipfile.ZipFile(path, "w") as z:
        n = len(slides)
        ids = "".join(f'<p:sldId id="{256 + i}" r:id="rId{n - i}"/>' for i in range(n))  # 슬라이드 i → slide{n-i}.xml
        z.writestr("ppt/presentation.xml", f"<p:presentation {P}><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>")
        rels = "".join(
            f'<Relationship Id="rId{k}" Target="slides/slide{k}.xml" Type="x"/>' for k in range(1, n + 1)
        )
        z.writestr(
            "ppt/_rels/presentation.xml.rels",
            f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>',
        )
        for i, body in enumerate(slides):
            z.writestr(f"ppt/slides/slide{n - i}.xml", f"<p:sld {P}><p:cSld><p:spTree>{body}</p:spTree></p:cSld></p:sld>")


def _sp(text, title=False):
    ph = '<p:ph type="title"/>' if title else ""
    return f"<p:sp><p:nvSpPr><p:nvPr>{ph}</p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp>"


def test_pptx_pages_in_slide_order_with_title_and_table(tmp_path):
    table = (
        "<p:graphicFrame><a:graphic><a:graphicData><a:tbl>"
        "<a:tr><a:tc><a:txBody><a:p><a:r><a:t>역할</a:t></a:r></a:p></a:txBody></a:tc>"
        "<a:tc><a:txBody><a:p><a:r><a:t>인원</a:t></a:r></a:p></a:txBody></a:tc></a:tr>"
        "<a:tr><a:tc><a:txBody><a:p><a:r><a:t>개발</a:t></a:r></a:p></a:txBody></a:tc>"
        "<a:tc><a:txBody><a:p><a:r><a:t>10명</a:t></a:r></a:p></a:txBody></a:tc></a:tr>"
        "</a:tbl></a:graphicData></a:graphic></p:graphicFrame>"
    )
    path = tmp_path / "발표.pptx"
    _make_pptx(path, [_sp("사업 개요", title=True) + _sp("기간은 4년"), _sp("투입 인력") + table])
    pages = convert._pptx_pages(path)
    assert pages[0] == "## 사업 개요\n\n기간은 4년"
    assert pages[1].startswith("투입 인력") and "| 개발 | 10명 |" in pages[1]
    assert convert.extract_segments(path) == pages
