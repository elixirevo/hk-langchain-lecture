"""
Ch06-2 Chunking — 유틸리티 모듈

텍스트 분할기 로직, 모델 설정, 청크 통계 함수를 UI와 분리하여 관리합니다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


# ── 상수 ──────────────────────────────────────────────────────

SPLITTER_KEYS = [
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    "TokenTextSplitter",
    "SemanticChunker (OpenAI)",
]

SPLITTER_DESCRIPTIONS = {
    "CharacterTextSplitter": "단일 구분자(기본: \\n\\n)로 분할. 가장 단순한 방식.",
    "RecursiveCharacterTextSplitter": "여러 구분자를 우선순위대로 재귀 시도. 범용 권장 방식.",
    "TokenTextSplitter": "tiktoken 토큰 수 기준 분할. LLM 토큰 제한 준수에 적합.",
    "SemanticChunker (OpenAI)": "OpenAI 임베딩 유사도로 의미 단위 분할. 높은 RAG 품질.",
}

BREAKPOINT_TYPES = {
    "Percentile (백분위수)": "percentile",
    "Standard Deviation (표준편차)": "standard_deviation",
    "Interquartile (사분위수)": "interquartile",
}

# 청크 색상 팔레트 (배경색, 테두리색)
CHUNK_COLORS = [
    ("#dbeafe", "#3b82f6"),  # 파랑
    ("#dcfce7", "#22c55e"),  # 초록
    ("#fef9c3", "#eab308"),  # 노랑
    ("#fce7f3", "#ec4899"),  # 분홍
    ("#ede9fe", "#8b5cf6"),  # 보라
    ("#ffedd5", "#f97316"),  # 주황
    ("#cffafe", "#06b6d4"),  # 시안
    ("#f1f5f9", "#64748b"),  # 회색
]


# ── 데이터 클래스 ──────────────────────────────────────────────

@dataclass
class ChunkResult:
    """텍스트 분할 결과를 담는 데이터 클래스."""

    chunks: List[str] = field(default_factory=list)
    splitter_name: str = ""
    elapsed_sec: float = 0.0
    error: Optional[str] = None

    @property
    def count(self) -> int:
        return len(self.chunks)

    @property
    def char_counts(self) -> List[int]:
        return [len(c) for c in self.chunks]

    @property
    def total_chars(self) -> int:
        return sum(self.char_counts)

    @property
    def avg_chars(self) -> float:
        return self.total_chars / self.count if self.count else 0.0

    @property
    def min_chars(self) -> int:
        return min(self.char_counts) if self.chunks else 0

    @property
    def max_chars(self) -> int:
        return max(self.char_counts) if self.chunks else 0

    @property
    def success(self) -> bool:
        return self.error is None and self.count > 0

    def to_stats_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "통계": ["청크 수", "최솟값", "최댓값", "평균", "중앙값"],
                "값": [
                    f"{self.count}개",
                    f"{self.min_chars:,}자",
                    f"{self.max_chars:,}자",
                    f"{self.avg_chars:.0f}자",
                    f"{pd.Series(self.char_counts).median():.0f}자",
                ],
            }
        )


# ── 분할 함수 ──────────────────────────────────────────────────

def split_character(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separator: str = "\n\n",
) -> ChunkResult:
    """CharacterTextSplitter로 텍스트를 분할합니다."""
    t0 = time.time()
    try:
        from langchain_text_splitters import CharacterTextSplitter

        splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_text(text)
        return ChunkResult(
            chunks=chunks,
            splitter_name=f"CharacterTextSplitter (separator='{separator}')",
            elapsed_sec=time.time() - t0,
        )
    except Exception as exc:
        return ChunkResult(error=str(exc), elapsed_sec=time.time() - t0)


def split_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[List[str]] = None,
) -> ChunkResult:
    """RecursiveCharacterTextSplitter로 텍스트를 분할합니다."""
    t0 = time.time()
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        kwargs = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "length_function": len,
        }
        if separators:
            kwargs["separators"] = separators

        splitter = RecursiveCharacterTextSplitter(**kwargs)
        chunks = splitter.split_text(text)
        return ChunkResult(
            chunks=chunks,
            splitter_name="RecursiveCharacterTextSplitter",
            elapsed_sec=time.time() - t0,
        )
    except Exception as exc:
        return ChunkResult(error=str(exc), elapsed_sec=time.time() - t0)


def split_token(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str = "cl100k_base",
) -> ChunkResult:
    """TokenTextSplitter로 텍스트를 분할합니다."""
    t0 = time.time()
    try:
        from langchain_text_splitters import TokenTextSplitter

        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
        )
        chunks = splitter.split_text(text)
        return ChunkResult(
            chunks=chunks,
            splitter_name=f"TokenTextSplitter (encoding={encoding_name})",
            elapsed_sec=time.time() - t0,
        )
    except ImportError:
        return ChunkResult(
            error="tiktoken 패키지가 필요합니다. `pip install tiktoken` 을 실행하세요.",
            elapsed_sec=time.time() - t0,
        )
    except Exception as exc:
        return ChunkResult(error=str(exc), elapsed_sec=time.time() - t0)


def split_semantic(
    text: str,
    breakpoint_type: str = "percentile",
    breakpoint_amount: float = 95.0,
) -> ChunkResult:
    """SemanticChunker로 텍스트를 의미 기반 분할합니다. OpenAI API 키 필요."""
    import os

    t0 = time.time()
    if not os.getenv("OPENAI_API_KEY"):
        return ChunkResult(
            error="OpenAI API 키가 설정되지 않았습니다. 사이드바에서 API 키를 입력해 주세요.",
            elapsed_sec=time.time() - t0,
        )
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai.embeddings import OpenAIEmbeddings

        splitter = SemanticChunker(
            OpenAIEmbeddings(),
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_amount,
        )
        chunks = splitter.split_text(text)
        return ChunkResult(
            chunks=chunks,
            splitter_name=f"SemanticChunker (type={breakpoint_type}, amount={breakpoint_amount})",
            elapsed_sec=time.time() - t0,
        )
    except ImportError as exc:
        return ChunkResult(
            error=f"필요한 패키지: {exc}. `pip install langchain-experimental langchain-openai` 실행.",
            elapsed_sec=time.time() - t0,
        )
    except Exception as exc:
        return ChunkResult(error=str(exc), elapsed_sec=time.time() - t0)


def split_text(
    text: str,
    splitter_key: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    **kwargs,
) -> ChunkResult:
    """splitter_key에 따라 적절한 분할기를 선택합니다."""
    if splitter_key == "CharacterTextSplitter":
        return split_character(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=kwargs.get("separator", "\n\n"),
        )
    elif splitter_key == "RecursiveCharacterTextSplitter":
        return split_recursive(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=kwargs.get("separators"),
        )
    elif splitter_key == "TokenTextSplitter":
        return split_token(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=kwargs.get("encoding_name", "cl100k_base"),
        )
    elif "SemanticChunker" in splitter_key:
        return split_semantic(
            text,
            breakpoint_type=kwargs.get("breakpoint_type", "percentile"),
            breakpoint_amount=kwargs.get("breakpoint_amount", 95.0),
        )
    else:
        return ChunkResult(error=f"알 수 없는 분할기: {splitter_key}")


# ── 비교 함수 ─────────────────────────────────────────────────

def compare_splitters(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> pd.DataFrame:
    """Character, Recursive, Token 분할기를 비교합니다 (SemanticChunker 제외)."""
    results = {}
    for key in ["CharacterTextSplitter", "RecursiveCharacterTextSplitter", "TokenTextSplitter"]:
        r = split_text(text, key, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if r.success:
            results[key] = {
                "청크 수": r.count,
                "평균 글자 수": f"{r.avg_chars:.0f}",
                "최솟값 (자)": r.min_chars,
                "최댓값 (자)": r.max_chars,
                "소요 시간 (초)": f"{r.elapsed_sec:.3f}",
            }
        else:
            results[key] = {"청크 수": "오류", "평균 글자 수": "-", "최솟값 (자)": "-", "최댓값 (자)": "-", "소요 시간 (초)": "-"}

    df = pd.DataFrame(results).T
    df.index.name = "분할기"
    return df.reset_index()


# ── 청크 HTML 시각화 ───────────────────────────────────────────

def chunk_to_html(chunk: str, color_idx: int, chunk_num: int) -> str:
    """청크 하나를 색상이 적용된 HTML로 변환합니다."""
    bg, border = CHUNK_COLORS[color_idx % len(CHUNK_COLORS)]
    escaped = chunk.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
<div style="
    background-color: {bg};
    border-left: 4px solid {border};
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    margin: 6px 0;
    font-family: monospace;
    font-size: 0.82rem;
    white-space: pre-wrap;
    word-break: break-word;
">
  <span style="font-size:0.7rem; color:{border}; font-weight:600;">
    청크 {chunk_num}  ({len(chunk):,}자)
  </span><br/>{escaped}
</div>
"""


def build_chunks_html(chunks: List[str], max_display: int = 30) -> str:
    """청크 목록 전체를 색상 구분 HTML로 변환합니다."""
    parts = []
    for i, chunk in enumerate(chunks[:max_display]):
        parts.append(chunk_to_html(chunk, i, i + 1))
    return "".join(parts)


def build_overlap_html(chunks: List[str], max_display: int = 10) -> str:
    """인접 청크 간 오버랩 영역을 강조 표시한 HTML을 생성합니다."""
    if len(chunks) < 2:
        return "<p>청크가 2개 이상이어야 오버랩을 확인할 수 있습니다.</p>"

    parts = []
    for i in range(min(max_display, len(chunks) - 1)):
        c1, c2 = chunks[i], chunks[i + 1]

        # 오버랩 영역 탐색: c1의 끝부분과 c2의 시작부분에서 공통 문자열 추출
        overlap_text = ""
        for length in range(min(len(c1), len(c2), 300), 0, -1):
            if c1.endswith(c2[:length]):
                overlap_text = c2[:length]
                break

        bg1, border1 = CHUNK_COLORS[i % len(CHUNK_COLORS)]
        bg2, border2 = CHUNK_COLORS[(i + 1) % len(CHUNK_COLORS)]

        def esc(s: str) -> str:
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        overlap_html = ""
        if overlap_text:
            overlap_html = f"""
<div style="background-color:#fde68a;border:2px dashed #d97706;border-radius:4px;
    padding:4px 8px;margin:2px 0;font-size:0.78rem;font-family:monospace;">
  <b style="color:#92400e;">오버랩 영역 ({len(overlap_text):,}자):</b><br/>
  {esc(overlap_text)}
</div>"""
        else:
            overlap_html = '<p style="color:#6b7280;font-size:0.78rem;">오버랩 없음</p>'

        parts.append(f"""
<div style="border:1px solid #e5e7eb;border-radius:8px;padding:10px;margin:8px 0;">
  <div style="display:flex;gap:8px;align-items:flex-start;flex-wrap:wrap;">
    <div style="flex:1;min-width:200px;background:{bg1};border-left:4px solid {border1};
        border-radius:0 4px 4px 0;padding:6px 10px;font-family:monospace;font-size:0.78rem;
        white-space:pre-wrap;word-break:break-word;">
      <b style="color:{border1};">청크 {i+1}</b><br/>{esc(c1[-200:] if len(c1) > 200 else c1)}
    </div>
    <div style="flex:1;min-width:200px;background:{bg2};border-left:4px solid {border2};
        border-radius:0 4px 4px 0;padding:6px 10px;font-family:monospace;font-size:0.78rem;
        white-space:pre-wrap;word-break:break-word;">
      <b style="color:{border2};">청크 {i+2}</b><br/>{esc(c2[:200] if len(c2) > 200 else c2)}
    </div>
  </div>
  {overlap_html}
</div>
""")

    return "".join(parts)


# ── 내보내기 함수 ─────────────────────────────────────────────

def chunks_to_json(chunks: List[str], splitter_name: str) -> str:
    data = [
        {"index": i, "content": c, "char_count": len(c)}
        for i, c in enumerate(chunks)
    ]
    return json.dumps(
        {"splitter": splitter_name, "total_chunks": len(chunks), "chunks": data},
        ensure_ascii=False,
        indent=2,
    )


def chunks_to_csv_bytes(chunks: List[str]) -> bytes:
    df = pd.DataFrame(
        [{"index": i, "content": c, "char_count": len(c)} for i, c in enumerate(chunks)]
    )
    return df.to_csv(index=False).encode("utf-8")
