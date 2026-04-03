"""
Ch06-1 Document Loaders — 유틸리티 모듈

로더 로직, 모델 설정, 데이터 변환 함수를 UI와 분리하여 관리합니다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from langchain_core.documents import Document


# ── 상수 ──────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".txt", ".json"}

PDF_LOADERS = {
    "PyMuPDF (권장)": "PyMuPDFLoader",
    "PyPDF": "PyPDFLoader",
    "PDFPlumber": "PDFPlumberLoader",
}

CSV_DELIMITERS = {
    "쉼표 (,)": ",",
    "세미콜론 (;)": ";",
    "탭 (\\t)": "\t",
    "파이프 (|)": "|",
}

# ── 데이터 클래스 ──────────────────────────────────────────────

@dataclass
class LoadResult:
    """Document 로딩 결과를 담는 데이터 클래스."""

    docs: List[Document] = field(default_factory=list)
    loader_name: str = ""
    elapsed_sec: float = 0.0
    error: Optional[str] = None

    # 통계 프로퍼티
    @property
    def count(self) -> int:
        return len(self.docs)

    @property
    def total_chars(self) -> int:
        return sum(len(d.page_content) for d in self.docs)

    @property
    def avg_chars(self) -> int:
        return self.total_chars // self.count if self.count else 0

    @property
    def char_counts(self) -> List[int]:
        return [len(d.page_content) for d in self.docs]

    @property
    def success(self) -> bool:
        return self.error is None and self.count > 0


# ── 로더 함수 ──────────────────────────────────────────────────

def load_pdf(
    file_path: str,
    loader_key: str = "PyMuPDF (권장)",
) -> LoadResult:
    """PDF 파일을 지정된 로더로 로딩합니다."""
    t0 = time.time()
    try:
        if "PyMuPDF" in loader_key:
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(file_path)
            name = "PyMuPDFLoader"
        elif "PDFPlumber" in loader_key:
            from langchain_community.document_loaders import PDFPlumberLoader
            loader = PDFPlumberLoader(file_path)
            name = "PDFPlumberLoader"
        else:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            name = "PyPDFLoader"

        docs = loader.load()
        return LoadResult(docs=docs, loader_name=name, elapsed_sec=time.time() - t0)
    except Exception as exc:
        return LoadResult(error=str(exc), elapsed_sec=time.time() - t0)


def load_csv(
    file_path: str,
    delimiter: str = ",",
    source_column: Optional[str] = None,
) -> LoadResult:
    """CSV 파일을 CSVLoader로 로딩합니다."""
    t0 = time.time()
    try:
        from langchain_community.document_loaders import CSVLoader

        kwargs: dict = {
            "file_path": file_path,
            "csv_args": {"delimiter": delimiter},
        }
        if source_column:
            kwargs["source_column"] = source_column

        loader = CSVLoader(**kwargs)
        docs = loader.load()
        name = f"CSVLoader (delimiter='{delimiter}')"
        return LoadResult(docs=docs, loader_name=name, elapsed_sec=time.time() - t0)
    except Exception as exc:
        return LoadResult(error=str(exc), elapsed_sec=time.time() - t0)


def load_txt(file_path: str) -> LoadResult:
    """텍스트 파일을 TextLoader로 로딩합니다. 인코딩 자동 감지."""
    t0 = time.time()
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            from langchain_community.document_loaders import TextLoader

            loader = TextLoader(file_path, encoding=enc)
            docs = loader.load()
            return LoadResult(
                docs=docs,
                loader_name=f"TextLoader (encoding={enc})",
                elapsed_sec=time.time() - t0,
            )
        except Exception:
            continue
    return LoadResult(
        error="지원되는 인코딩으로 파일을 읽을 수 없습니다 (utf-8, cp949, euc-kr 시도).",
        elapsed_sec=time.time() - t0,
    )


def load_json(file_path: str, source_name: str = "upload.json") -> LoadResult:
    """JSON 파일을 항목별 Document로 변환합니다."""
    t0 = time.time()
    try:
        with open(file_path, encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            docs = [
                Document(
                    page_content=json.dumps(item, ensure_ascii=False, indent=2),
                    metadata={"source": source_name, "index": i},
                )
                for i, item in enumerate(raw)
            ]
        else:
            docs = [
                Document(
                    page_content=json.dumps(raw, ensure_ascii=False, indent=2),
                    metadata={"source": source_name},
                )
            ]
        return LoadResult(
            docs=docs,
            loader_name="JSONLoader (커스텀)",
            elapsed_sec=time.time() - t0,
        )
    except Exception as exc:
        return LoadResult(error=str(exc), elapsed_sec=time.time() - t0)


def load_file(file_path: str, file_ext: str, **kwargs) -> LoadResult:
    """확장자에 따라 적절한 로더를 선택하여 파일을 로딩합니다."""
    ext = file_ext.lower()
    if ext == ".pdf":
        return load_pdf(file_path, loader_key=kwargs.get("pdf_loader_key", "PyMuPDF (권장)"))
    elif ext == ".csv":
        return load_csv(
            file_path,
            delimiter=kwargs.get("delimiter", ","),
            source_column=kwargs.get("source_column"),
        )
    elif ext == ".txt":
        return load_txt(file_path)
    elif ext == ".json":
        return load_json(file_path, source_name=kwargs.get("source_name", "upload.json"))
    else:
        return LoadResult(error=f"지원하지 않는 파일 형식입니다: {file_ext}")


# ── 데이터 변환 함수 ───────────────────────────────────────────

def docs_to_json(docs: List[Document]) -> str:
    """Document 리스트를 JSON 문자열로 변환합니다."""
    data = [
        {"index": i, "page_content": d.page_content, "metadata": d.metadata}
        for i, d in enumerate(docs)
    ]
    return json.dumps(data, ensure_ascii=False, indent=2)


def docs_to_csv_bytes(docs: List[Document]) -> bytes:
    """Document 리스트를 CSV bytes로 변환합니다."""
    rows = []
    for i, doc in enumerate(docs):
        row = {"index": i, "page_content": doc.page_content}
        row.update(doc.metadata)
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def get_metadata_df(docs: List[Document], limit: int = 20) -> pd.DataFrame:
    """Document 목록에서 메타데이터 DataFrame을 생성합니다."""
    rows = []
    for i, doc in enumerate(docs[:limit]):
        row = {"index": i}
        row.update(doc.metadata)
        rows.append(row)
    return pd.DataFrame(rows)


def get_stats_df(char_counts: List[int]) -> pd.DataFrame:
    """글자 수 리스트에서 기술 통계 DataFrame을 생성합니다."""
    s = pd.Series(char_counts)
    return pd.DataFrame(
        {
            "통계": ["최솟값", "최댓값", "평균", "중앙값", "표준편차"],
            "값": [
                f"{s.min():,}자",
                f"{s.max():,}자",
                f"{s.mean():.0f}자",
                f"{s.median():.0f}자",
                f"{s.std():.0f}자",
            ],
        }
    )
