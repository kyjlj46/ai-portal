"""PDF 가 아닌 문서(Word / PowerPoint / 한글 / 텍스트)를 처리 가능한 형태로 바꾼다.

1순위: 설치된 Word / PowerPoint / 한컴오피스로 PDF 변환 → 기존 OCR 파이프라인 사용 (페이지 번호·페이지 이미지 유지)
2순위: 변환 불가(프로그램 없음, 보안 확인 창, 시간 초과)면 파일에서 텍스트를 직접 추출해 '구간' 단위로 나눈다.
텍스트 파일(.txt/.md/.csv 등)은 변환 없이 바로 읽는다.

변환은 별도 프로세스에서 실행한다: COM 자동화가 확인 창 등으로 멈춰도 시간 제한으로 끊을 수 있고,
웹 서버 스레드에서 COM 초기화 문제를 피할 수 있다.
"""

from __future__ import annotations

import re
import struct
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

PDF_EXTS = {".pdf"}
WORD_EXTS = {".docx", ".doc", ".rtf", ".odt"}
PPT_EXTS = {".pptx", ".ppt", ".odp"}
HWP_EXTS = {".hwp", ".hwpx"}
# 소스 코드는 텍스트 문서로 읽는다 (코드 어시스턴트가 리뷰·설명할 수 있게)
CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs",
             ".rb", ".php", ".swift", ".scala", ".sql", ".sh", ".ps1", ".bat", ".yaml", ".yml", ".toml", ".ini", ".cfg",
             ".xml", ".html", ".css", ".vue"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".json"} | CODE_EXTS
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}  # 스캔 이미지 → PDF 로 바꿔 OCR
SUPPORTED_EXTS = PDF_EXTS | WORD_EXTS | PPT_EXTS | HWP_EXTS | TEXT_EXTS | IMAGE_EXTS

CONVERT_TIMEOUT = 180  # 초
SEGMENT_CHARS = 2500  # 텍스트 직접 추출 시 한 '구간'(가짜 페이지) 크기


def kind_of(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in PDF_EXTS:
        return "pdf"
    if ext in WORD_EXTS:
        return "word"
    if ext in PPT_EXTS:
        return "ppt"
    if ext in HWP_EXTS:
        return "hwp"
    if ext in TEXT_EXTS:
        return "text"
    if ext in IMAGE_EXTS:
        return "image"
    raise ValueError(f"지원하지 않는 형식: {ext} (지원: {', '.join(sorted(SUPPORTED_EXTS))})")


def image_to_pdf(src: str | Path, dst: str | Path) -> str:
    """이미지(여러 장짜리 TIFF/GIF 포함)를 장마다 한 페이지인 PDF 로 만든다. 같은 OCR 파이프라인을 타게 하려는 것."""
    from PIL import Image, ImageSequence

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)][:200]
    frames[0].save(dst, "PDF", resolution=200.0, save_all=True, append_images=frames[1:])
    return str(dst)


# ─────────────────────────── 1. PDF 변환 (Word / PowerPoint / 한글 COM) ───────────────────────────


def to_pdf(src: str | Path, dst: str | Path, timeout: int = CONVERT_TIMEOUT) -> str | None:
    """문서를 PDF 로 변환. 성공하면 dst 경로, 실패하면 None (실패 사유는 stderr 로그)."""
    if sys.platform != "win32":
        return None
    src, dst = Path(src).resolve(), Path(dst).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "portal.convert", kind_of(src), str(src), str(dst)],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"[convert] 시간 초과({timeout}s): {src.name} — 텍스트 직접 추출로 진행", file=sys.stderr)
        return None
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        print(f"[convert] 변환 실패: {src.name}\n{proc.stderr[-800:]}", file=sys.stderr)
        return None
    return str(dst)


def _convert_word(src: str, dst: str) -> None:
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    word = win32com.client.DispatchEx("Word.Application")
    try:
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(src, ConfirmConversions=False, ReadOnly=True, AddToRecentFiles=False)
        try:
            doc.ExportAsFixedFormat(dst, 17)  # 17 = wdExportFormatPDF
        finally:
            doc.Close(False)
    finally:
        word.Quit()


def _convert_ppt(src: str, dst: str) -> None:
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    ppt = win32com.client.DispatchEx("PowerPoint.Application")
    try:
        ppt.DisplayAlerts = 1  # ppAlertsNone (PowerPoint 는 Visible=False 를 허용하지 않아 WithWindow 로 숨긴다)
        pres = ppt.Presentations.Open(src, ReadOnly=True, Untitled=False, WithWindow=False)
        try:
            pres.SaveAs(dst, 32)  # 32 = ppSaveAsPDF (슬라이드 1장 = PDF 1페이지)
        finally:
            pres.Close()
    finally:
        ppt.Quit()


def _convert_hwp(src: str, dst: str) -> None:
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    hwp = win32com.client.DispatchEx("HWPFrame.HwpObject")
    try:
        try:  # 보안 모듈이 등록돼 있으면 파일 접근 확인 창이 뜨지 않는다
            hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
        except Exception:
            pass
        try:
            hwp.XHwpWindows.Item(0).Visible = False
        except Exception:
            pass
        if not hwp.Open(src, "", "forceopen:true;versionwarning:false"):
            raise RuntimeError("한글에서 파일을 열지 못했습니다.")
        if not hwp.SaveAs(dst, "PDF", ""):
            raise RuntimeError("한글에서 PDF 저장에 실패했습니다.")
    finally:
        hwp.Quit()


# ─────────────────────────── 2. 텍스트 직접 추출 ───────────────────────────


def extract_segments(path: str | Path) -> list[str]:
    """문서에서 텍스트를 뽑아 '구간' 목록으로 돌려준다 (각 구간이 가짜 페이지 1개)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        pages = [read_text(path)]
    elif ext == ".docx":
        pages = _docx_pages(path)
    elif ext == ".pptx":
        pages = _pptx_pages(path)
    elif ext == ".hwpx":
        pages = _hwpx_pages(path)
    elif ext == ".hwp":
        pages = [_hwp_text(path)]
    else:
        raise ValueError(f"{ext} 는 PDF 변환이 필요합니다 (Word / PowerPoint 가 설치되어 있어야 함).")
    segments = [seg for page in pages for seg in _split(page)]
    return segments or [""]


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp949", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _split(text: str, size: int = SEGMENT_CHARS) -> list[str]:
    """문단 경계를 지키며 size 자 안팎으로 나눈다."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) > size:
            out.append(cur.strip())
            cur = ""
        while len(para) > size * 1.5:  # 문단 하나가 너무 길면 줄 단위로 자른다
            cut = para.rfind("\n", 0, size)
            cut = cut if cut > size // 2 else size
            out.append(para[:cut].strip())
            para = para[cut:]
        cur += para + "\n\n"
    if cur.strip():
        out.append(cur.strip())
    return out


def _md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [[c.replace("\n", " ").replace("|", "/").strip() for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_pages(path: Path) -> list[str]:
    """본문 순서대로 문단(제목은 #)과 표(마크다운 표)를 뽑고, 쪽 나누기로 페이지를 나눈다."""
    import docx

    document = docx.Document(str(path))
    pages, cur = [], []
    for el in document.element.body.iterchildren():
        if el.tag == W + "p":
            text = "".join(t.text or "" for t in el.iter(W + "t")).strip()
            style = el.find(f"{W}pPr/{W}pStyle")
            sval = style.get(W + "val", "") if style is not None else ""
            m = re.search(r"(?:Heading|제목)\s*(\d)", sval)
            if text:
                cur.append(("#" * int(m.group(1)) + " " + text) if m else text)
            if any(br.get(W + "type") == "page" for br in el.iter(W + "br")) and cur:
                pages.append("\n\n".join(cur))
                cur = []
        elif el.tag == W + "tbl":
            rows = [
                ["".join(t.text or "" for t in tc.iter(W + "t")) for tc in tr.iter(W + "tc")]
                for tr in el.iter(W + "tr")
            ]
            cur.append(_md_table(rows))
    if cur:
        pages.append("\n\n".join(cur))
    return pages


_PML = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_DML = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _pptx_slide_names(z: zipfile.ZipFile) -> list[str]:
    """presentation.xml 의 슬라이드 목록 순서(실제 발표 순서)대로 slideN.xml 경로를 돌려준다."""
    names = set(z.namelist())
    try:
        rels = ET.fromstring(z.read("ppt/_rels/presentation.xml.rels"))
        target = {r.get("Id"): r.get("Target", "") for r in rels.iter(_PKG_REL + "Relationship")}
        pres = ET.fromstring(z.read("ppt/presentation.xml"))
        ordered = []
        for sld in pres.iter(_PML + "sldId"):
            t = target.get(sld.get(_REL + "id"), "").lstrip("/")
            t = t if t.startswith("ppt/") else "ppt/" + t
            if t in names:
                ordered.append(t)
        if ordered:
            return ordered
    except (KeyError, ET.ParseError):
        pass
    return sorted(
        (n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)), key=lambda n: int(re.findall(r"\d+", n)[-1])
    )


def _pptx_pages(path: Path) -> list[str]:
    """PPTX(zip+XML): 슬라이드마다 도형 텍스트(제목은 ##)와 표(마크다운 표)를 뽑는다. 슬라이드 1장 = 1페이지."""

    def para_text(p) -> str:
        return "".join(t.text or "" for t in p.iter(_DML + "t")).strip()

    pages: list[str] = []
    with zipfile.ZipFile(path) as z:
        for name in _pptx_slide_names(z):
            root = ET.fromstring(z.read(name))
            parts: list[str] = []
            for el in root.iter():
                if el.tag == _PML + "sp":  # 텍스트 상자/자리 표시자
                    ph = el.find(f"{_PML}nvSpPr/{_PML}nvPr/{_PML}ph")
                    is_title = ph is not None and ph.get("type") in ("title", "ctrTitle")
                    lines = [para_text(p) for p in el.iter(_DML + "p")]
                    text = "\n".join(line for line in lines if line)
                    if text:
                        parts.append("## " + text.replace("\n", " ") if is_title else text)
                elif el.tag == _DML + "tbl":
                    rows = [
                        [" ".join(para_text(p) for p in tc.iter(_DML + "p")) for tc in tr.iter(_DML + "tc")]
                        for tr in el.iter(_DML + "tr")
                    ]
                    parts.append(_md_table(rows))
            pages.append("\n\n".join(parts))
    return pages


def _hwpx_pages(path: Path) -> list[str]:
    """HWPX(zip+XML): 섹션별 문단과 표를 뽑는다. pageBreak 문단에서 페이지를 나눈다."""
    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def para_text(p) -> str:
        return "".join(t.text or "" for t in p.iter() if local(t.tag) == "t").strip()

    pages: list[str] = []
    with zipfile.ZipFile(path) as z:
        sections = sorted(
            (n for n in z.namelist() if re.match(r"Contents/section\d+\.xml$", n)),
            key=lambda n: int(re.findall(r"\d+", n)[-1]),
        )
        for name in sections:
            root = ET.fromstring(z.read(name))
            cur: list[str] = []
            for p in root:  # 섹션 최상위 문단
                if local(p.tag) != "p":
                    continue
                if p.get("pageBreak") == "1" and cur:
                    pages.append("\n\n".join(cur))
                    cur = []
                tables = [t for t in p.iter() if local(t.tag) == "tbl"]
                if tables:
                    for tbl in tables:
                        rows = [
                            [" ".join(para_text(cp) for cp in tc.iter() if local(cp.tag) == "p") for tc in tr if local(tc.tag) == "tc"]
                            for tr in tbl if local(tr.tag) == "tr"
                        ]
                        cur.append(_md_table(rows))
                else:
                    text = para_text(p)
                    if text:
                        cur.append(text)
            if cur:
                pages.append("\n\n".join(cur))
    return pages


def _hwp_text(path: Path) -> str:
    """HWP 5.x(OLE): BodyText/Section* 의 문단 텍스트 레코드(HWPTAG_PARA_TEXT)를 읽는다."""
    import olefile

    with olefile.OleFileIO(str(path)) as ole:
        header = ole.openstream("FileHeader").read()
        compressed = bool(header[36] & 1)
        if header[36] & 2:
            raise ValueError("암호가 걸린 HWP 문서입니다.")
        sections = sorted(
            (e for e in ole.listdir() if e[0] == "BodyText" and e[1].startswith("Section")),
            key=lambda e: int(e[1][7:]),
        )
        paras: list[str] = []
        for entry in sections:
            data = ole.openstream(entry).read()
            if compressed:
                data = zlib.decompress(data, -15)
            pos = 0
            while pos + 4 <= len(data):
                head = struct.unpack_from("<I", data, pos)[0]
                tag, size = head & 0x3FF, (head >> 20) & 0xFFF
                pos += 4
                if size == 0xFFF:
                    size = struct.unpack_from("<I", data, pos)[0]
                    pos += 4
                if tag == 67:  # HWPTAG_PARA_TEXT
                    text = _hwp_para(data[pos : pos + size])
                    if text.strip():
                        paras.append(text.strip())
                pos += size
    return "\n\n".join(paras)


# HWP 문단 텍스트의 제어 문자: 이 코드들은 뒤에 7개의 WCHAR(14바이트)가 더 붙는 확장/인라인 제어
_HWP_LONG_CTRL = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


def _hwp_para(raw: bytes) -> str:
    out, i = [], 0
    while i + 1 < len(raw):
        code = struct.unpack_from("<H", raw, i)[0]
        if code < 32:
            if code == 9:
                out.append("\t")
            elif code in (10, 13):
                out.append("\n")
            i += 16 if code in _HWP_LONG_CTRL else 2
            continue
        out.append(chr(code))
        i += 2
    return "".join(out)


if __name__ == "__main__":  # 변환 하위 프로세스: python -m portal.convert <word|ppt|hwp> <src> <dst>
    kind, src, dst = sys.argv[1:4]
    {"word": _convert_word, "ppt": _convert_ppt, "hwp": _convert_hwp}[kind](src, dst)
