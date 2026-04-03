"""
Ch06-4 VectorStore — 유틸리티 함수

FAISS 벡터스토어 구축, 검색, 결과 시각화 헬퍼 함수 모음
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import streamlit as st


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """단일 검색 결과."""
    rank: int
    content: str
    source: str
    score: float          # 유사도 검색: L2 거리 / MMR: 코사인 유사도
    score_type: str       # "l2_distance" | "cosine_similarity"
    chunk_index: int = 0


@dataclass
class SearchSession:
    """검색 세션 정보."""
    query: str
    search_type: str
    k: int
    latency_ms: float
    results: list[SearchResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 임베딩 모델
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="OpenAI 임베딩 모델을 로드하는 중...")
def load_openai_embeddings(model_name: str, api_key: str):
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model_name, api_key=api_key)


# ---------------------------------------------------------------------------
# 텍스트 분할
# ---------------------------------------------------------------------------

def split_text(
    text: str,
    source_name: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
):
    """텍스트를 RecursiveCharacterTextSplitter로 분할하여 Document 리스트 반환."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    doc = Document(page_content=text, metadata={"source": source_name})
    return splitter.split_documents([doc])


# ---------------------------------------------------------------------------
# 벡터스토어 구축
# ---------------------------------------------------------------------------

def build_faiss_vectorstore(documents, embeddings):
    """FAISS VectorStore를 생성하여 반환."""
    from langchain_community.vectorstores import FAISS
    return FAISS.from_documents(documents, embeddings)


# ---------------------------------------------------------------------------
# 검색
# ---------------------------------------------------------------------------

def search_similarity(
    vectorstore,
    embeddings,
    query: str,
    k: int,
) -> SearchSession:
    """유사도 검색 수행. FAISS L2 거리 기반."""
    t0 = time.perf_counter()
    raw = vectorstore.similarity_search_with_score(query, k=k)
    latency = (time.perf_counter() - t0) * 1000

    results = [
        SearchResult(
            rank=i + 1,
            content=doc.page_content,
            source=doc.metadata.get("source", "알 수 없음"),
            score=float(score),
            score_type="l2_distance",
        )
        for i, (doc, score) in enumerate(raw)
    ]
    return SearchSession(query=query, search_type="similarity", k=k, latency_ms=latency, results=results)


def search_mmr(
    vectorstore,
    embeddings,
    query: str,
    k: int,
    fetch_k: int,
    lambda_mult: float,
) -> SearchSession:
    """MMR 검색 수행. 코사인 유사도 계산 포함."""
    t0 = time.perf_counter()
    docs = vectorstore.max_marginal_relevance_search(
        query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
    )
    # MMR은 점수를 반환하지 않으므로 코사인 유사도 직접 계산
    query_vec = np.array(embeddings.embed_query(query), dtype=float)
    doc_texts = [d.page_content for d in docs]
    doc_vecs = np.array(embeddings.embed_documents(doc_texts), dtype=float)
    norms_d = np.linalg.norm(doc_vecs, axis=1)
    norms_q = np.linalg.norm(query_vec)
    scores = doc_vecs @ query_vec / (norms_d * norms_q + 1e-10)
    latency = (time.perf_counter() - t0) * 1000

    results = [
        SearchResult(
            rank=i + 1,
            content=doc.page_content,
            source=doc.metadata.get("source", "알 수 없음"),
            score=float(scores[i]),
            score_type="cosine_similarity",
        )
        for i, doc in enumerate(docs)
    ]
    return SearchSession(query=query, search_type="mmr", k=k, latency_ms=latency, results=results)


# ---------------------------------------------------------------------------
# 다양성 점수 계산 (MMR 시각화용)
# ---------------------------------------------------------------------------

def compute_pairwise_diversity(results: list[SearchResult], embeddings) -> np.ndarray:
    """결과 문서 간 코사인 유사도 매트릭스 반환 (다양성 시각화용)."""
    if len(results) < 2:
        return np.array([[1.0]])
    texts = [r.content for r in results]
    vecs = np.array(embeddings.embed_documents(texts), dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / (norms + 1e-10)
    return np.clip(vecs_n @ vecs_n.T, 0, 1)


# ---------------------------------------------------------------------------
# Plotly 시각화
# ---------------------------------------------------------------------------

def build_score_bar(session: SearchSession):
    """검색 결과 점수 막대그래프 Figure 반환."""
    import plotly.graph_objects as go

    results = session.results
    labels = [f"#{r.rank}: {r.content[:22]}..." for r in results]

    if session.search_type == "similarity":
        # L2 거리: 낮을수록 더 유사 → 역수로 높이를 표현
        raw_scores = [r.score for r in results]
        max_s = max(raw_scores) if max(raw_scores) > 0 else 1.0
        bar_vals = [1 - s / max_s for s in raw_scores]
        score_labels = [f"L2={s:.4f}" for s in raw_scores]
        y_title = "정규화된 유사도 (1 - L2/max)"
        bar_title = "검색 결과 유사도 점수 (Similarity Search)"
    else:
        bar_vals = [r.score for r in results]
        score_labels = [f"{v:.4f}" for v in bar_vals]
        y_title = "코사인 유사도"
        bar_title = "검색 결과 유사도 점수 (MMR Search)"

    colors = ["#1565C0" if i == 0 else "#64B5F6" for i in range(len(results))]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=bar_vals,
            marker_color=colors,
            text=score_labels,
            textposition="outside",
            hovertext=[r.content for r in results],
            hoverinfo="text+y",
        )
    )
    fig.update_layout(
        title=bar_title,
        xaxis_title="검색 결과",
        yaxis_title=y_title,
        height=360,
        margin={"t": 50, "b": 80, "l": 20, "r": 20},
        yaxis={"range": [0, max(bar_vals) * 1.25 if bar_vals else 1]},
    )
    return fig


def build_mmr_diversity_heatmap(sim_matrix: np.ndarray, labels: list[str]):
    """MMR 결과 문서 간 유사도 히트맵 Figure 반환 (다양성 확인용)."""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Heatmap(
            z=sim_matrix,
            x=labels,
            y=labels,
            colorscale="Blues_r",   # 값이 낮을수록 (다양할수록) 파란색
            zmin=0,
            zmax=1,
            text=np.round(sim_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="문서 A: %{y}<br>문서 B: %{x}<br>유사도: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="결과 문서 간 유사도 (낮을수록 다양성 높음)",
        height=360,
        margin={"t": 50, "b": 10, "l": 10, "r": 10},
    )
    return fig
