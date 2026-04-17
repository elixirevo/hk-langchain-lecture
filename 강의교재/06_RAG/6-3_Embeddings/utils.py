"""
Ch06-3 Embeddings — 유틸리티 함수

임베딩 생성, 유사도 계산, 차원 축소 시각화 등의 헬퍼 함수 모음
"""

from __future__ import annotations

import numpy as np
import streamlit as st


# ---------------------------------------------------------------------------
# 임베딩 모델 로드
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="OpenAI 임베딩 모델을 로드하는 중...")
def load_openai_embeddings(model_name: str, api_key: str):
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model_name, api_key=api_key)


@st.cache_resource(show_spinner="HuggingFace 모델을 다운로드하는 중... (최초 실행 시 시간이 걸립니다)")
def load_hf_embeddings(model_name: str):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# 유사도 계산
# ---------------------------------------------------------------------------

def cosine_similarity_pair(a: list[float], b: list[float]) -> float:
    """두 벡터 간 코사인 유사도 반환 (0~1)."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-10))


def cosine_similarity_matrix(vectors: list[list[float]]) -> np.ndarray:
    """N개 벡터에 대한 (N, N) 코사인 유사도 매트릭스 반환."""
    mat = np.array(vectors, dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_norm = mat / (norms + 1e-10)
    sim = mat_norm @ mat_norm.T
    return np.clip(sim, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 차원 축소
# ---------------------------------------------------------------------------

def reduce_pca(vectors: list[list[float]], n_components: int = 2) -> np.ndarray:
    """PCA로 차원 축소. (N, n_components) 배열 반환."""
    from sklearn.decomposition import PCA
    mat = np.array(vectors, dtype=float)
    if mat.shape[0] <= n_components:
        # 샘플 수가 너무 적으면 패딩
        pad = np.zeros((n_components + 1 - mat.shape[0], mat.shape[1]))
        mat = np.vstack([mat, pad])
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(mat)
    return reduced[: len(vectors)]


def reduce_tsne(vectors: list[list[float]], n_components: int = 2, perplexity: int = 5) -> np.ndarray:
    """t-SNE로 차원 축소. 샘플이 4개 미만이면 PCA로 대체."""
    mat = np.array(vectors, dtype=float)
    if mat.shape[0] < 4:
        return reduce_pca(vectors, n_components)
    from sklearn.manifold import TSNE
    perp = min(perplexity, mat.shape[0] - 1)
    tsne = TSNE(n_components=n_components, perplexity=perp, random_state=42, max_iter=500)
    return tsne.fit_transform(mat)


# ---------------------------------------------------------------------------
# Plotly 시각화
# ---------------------------------------------------------------------------

def build_similarity_heatmap(sim_matrix: np.ndarray, labels: list[str]):
    """코사인 유사도 매트릭스 히트맵 Figure 반환."""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Heatmap(
            z=sim_matrix,
            x=labels,
            y=labels,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=np.round(sim_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate=(
                "문장 A: %{y}<br>문장 B: %{x}<br>유사도: %{z:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="코사인 유사도 매트릭스",
        xaxis={"tickangle": -30},
        height=520,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def build_scatter_2d(reduced: np.ndarray, labels: list[str], title: str = "임베딩 공간 (2D)"):
    """2D 산점도 Figure 반환."""
    import plotly.graph_objects as go

    colors = [f"hsl({int(i * 360 / len(labels))}, 70%, 50%)" for i in range(len(labels))]

    fig = go.Figure(
        go.Scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            mode="markers+text",
            marker=dict(size=14, color=colors, line=dict(width=1, color="white")),
            text=[lb[:20] + "..." if len(lb) > 20 else lb for lb in labels],
            textposition="top center",
            textfont=dict(size=10),
            hovertext=labels,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        height=480,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def build_scatter_3d(reduced: np.ndarray, labels: list[str], title: str = "임베딩 공간 (3D)"):
    """3D 산점도 Figure 반환."""
    import plotly.graph_objects as go

    colors = [f"hsl({int(i * 360 / len(labels))}, 70%, 50%)" for i in range(len(labels))]

    fig = go.Figure(
        go.Scatter3d(
            x=reduced[:, 0],
            y=reduced[:, 1],
            z=reduced[:, 2],
            mode="markers+text",
            marker=dict(size=8, color=colors, opacity=0.85),
            text=[lb[:18] + "..." if len(lb) > 18 else lb for lb in labels],
            hovertext=labels,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="PC 1", yaxis_title="PC 2", zaxis_title="PC 3"),
        height=520,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return fig


def similarity_level(score: float) -> tuple[str, str]:
    """유사도 점수에 따른 수준 레이블과 색상 반환."""
    if score >= 0.90:
        return "매우 높음 — 거의 동일한 의미", "green"
    elif score >= 0.75:
        return "높음 — 유사한 주제", "blue"
    elif score >= 0.55:
        return "보통 — 부분적으로 관련", "orange"
    else:
        return "낮음 — 의미적으로 다름", "red"
