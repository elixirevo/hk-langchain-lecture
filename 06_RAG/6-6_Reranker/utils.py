"""
utils.py — Ch06-6 Reranker 비교 유틸리티
==========================================
Reranker 생성, 검색 실행, Before/After 비교 헬퍼.
Streamlit 의존성 없이 순수 LangChain 로직만 포함.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class RankedDoc:
    """순위와 점수가 부여된 문서."""
    rank: int
    doc: Document
    score: Optional[float] = None   # Reranker 점수 (Before는 None)


@dataclass
class RerankerResult:
    """Reranker 적용 전/후 결과 컨테이너."""
    reranker_name: str
    before_docs: list[RankedDoc]       # 원본 벡터 검색 순서
    after_docs: list[RankedDoc]        # Reranker 재정렬 순서
    retrieval_elapsed: float           # 1단계 검색 소요 시간 (초)
    rerank_elapsed: float              # 2단계 재정렬 소요 시간 (초)
    error: Optional[str] = None


# ─────────────────────────────────────────────
# 벡터스토어 빌드
# ─────────────────────────────────────────────

def build_vectorstore(
    text: str,
    api_key: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    initial_k: int = 10,
):
    """
    문서 텍스트로 FAISS 벡터스토어와 기본 Retriever를 생성합니다.

    Returns
    -------
    (vectorstore, base_retriever, docs)
    """
    import os
    os.environ["OPENAI_API_KEY"] = api_key

    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = splitter.create_documents([text])

    # 문서 ID 부여 (FlashRank 호환)
    for idx, doc in enumerate(docs):
        doc.metadata["id"] = idx

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": initial_k})

    return vectorstore, base_retriever, docs


# ─────────────────────────────────────────────
# Cross-Encoder Reranker
# ─────────────────────────────────────────────

def build_cross_encoder_reranker(model_name: str, top_n: int):
    """HuggingFace Cross-Encoder 기반 ContextualCompressionRetriever를 반환합니다."""
    from langchain_huggingface import HuggingFaceEmbeddings  # noqa: F401 (임베딩 초기화 확인용)
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain.retrievers.document_compressors import CrossEncoderReranker

    model = HuggingFaceCrossEncoder(model_name=model_name)
    reranker = CrossEncoderReranker(model=model, top_n=top_n)
    return reranker


# ─────────────────────────────────────────────
# FlashRank Reranker
# ─────────────────────────────────────────────

def build_flashrank_reranker(model_name: str, top_n: int):
    """FlashRank 기반 Reranker를 반환합니다."""
    from langchain.retrievers.document_compressors import FlashrankRerank

    FlashrankRerank.model_rebuild()
    reranker = FlashrankRerank(model=model_name, top_n=top_n)
    return reranker


# ─────────────────────────────────────────────
# 검색 실행 (Before / After)
# ─────────────────────────────────────────────

def run_reranker_comparison(
    base_retriever,
    reranker,
    query: str,
    top_n: int,
) -> tuple[list[RankedDoc], list[RankedDoc], float, float]:
    """
    Before(벡터 검색 순서)와 After(Reranker 적용) 결과를 반환합니다.

    Returns
    -------
    (before_docs, after_docs, retrieval_elapsed, rerank_elapsed)
    """
    from langchain.retrievers import ContextualCompressionRetriever

    # 1단계: 기본 벡터 검색 (Before)
    t0 = time.perf_counter()
    raw_docs = base_retriever.invoke(query)
    retrieval_elapsed = time.perf_counter() - t0

    before_docs = [RankedDoc(rank=i + 1, doc=d) for i, d in enumerate(raw_docs)]

    # 2단계: Reranker 적용 (After)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

    t1 = time.perf_counter()
    reranked_raw = compression_retriever.invoke(query)
    rerank_elapsed = time.perf_counter() - t1 - retrieval_elapsed  # 재정렬만의 시간

    after_docs = []
    for i, doc in enumerate(reranked_raw):
        score = doc.metadata.get("relevance_score", None)
        after_docs.append(RankedDoc(rank=i + 1, doc=doc, score=score))

    return before_docs, after_docs, retrieval_elapsed, rerank_elapsed


# ─────────────────────────────────────────────
# 순위 변화 계산
# ─────────────────────────────────────────────

def compute_rank_changes(
    before: list[RankedDoc],
    after: list[RankedDoc],
) -> dict[str, int]:
    """
    문서 내용을 키로, Before→After 순위 변화(양수=상승)를 값으로 반환합니다.
    """
    before_ranks = {d.doc.page_content: d.rank for d in before}
    changes: dict[str, int] = {}

    for d in after:
        content = d.doc.page_content
        if content in before_ranks:
            changes[content] = before_ranks[content] - d.rank  # 양수 = 순위 상승
        else:
            changes[content] = 0

    return changes


# ─────────────────────────────────────────────
# 점수 정규화
# ─────────────────────────────────────────────

def normalize_scores(docs: list[RankedDoc]) -> list[RankedDoc]:
    """
    relevance_score를 0~1 범위로 min-max 정규화합니다.
    점수가 없는 경우 원본 반환.
    """
    scores = [d.score for d in docs if d.score is not None]
    if not scores:
        return docs

    min_s, max_s = min(scores), max(scores)
    rng = max_s - min_s if max_s != min_s else 1.0

    normalized = []
    for d in docs:
        if d.score is not None:
            norm = (d.score - min_s) / rng
            normalized.append(RankedDoc(rank=d.rank, doc=d.doc, score=norm))
        else:
            normalized.append(d)

    return normalized
