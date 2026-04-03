"""
Ch06-5 Retriever — 검색 전략 비교 데모
========================================
실행: streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ch06-5 검색 전략 비교",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────────
st.title("🔍 Ch06-5 Retriever — 검색 전략 비교")
st.markdown(
    "문서를 업로드하고 검색어를 입력해 "
    "**Similarity / MMR / BM25 / Ensemble / MultiQuery** 결과를 나란히 비교하세요."
)

# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ 설정")

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
    help="환경변수 OPENAI_API_KEY가 설정되어 있으면 자동 입력됩니다.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📄 문서 설정")

chunk_size = st.sidebar.slider("청크 크기", 200, 1000, 500, 100)
chunk_overlap = st.sidebar.slider("청크 오버랩", 0, 200, 50, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("🔢 검색 파라미터")

top_k = st.sidebar.slider("반환 문서 수 (k)", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 검색 방식 선택")

search_methods = st.sidebar.multiselect(
    "비교할 검색 방식",
    options=["Similarity", "MMR", "BM25", "Ensemble", "MultiQuery"],
    default=["Similarity", "MMR", "BM25", "Ensemble"],
    help="MultiQuery는 LLM을 추가 호출하므로 시간과 비용이 증가합니다.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Ensemble 가중치")

bm25_weight = st.sidebar.slider(
    "BM25 가중치",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
)
vector_weight = round(1.0 - bm25_weight, 1)
st.sidebar.info(f"BM25 **{bm25_weight}** : Vector **{vector_weight}**")

st.sidebar.markdown("---")
st.sidebar.subheader("🔀 MMR 설정")

lambda_mult = st.sidebar.slider(
    "Lambda (관련성 ↔ 다양성)",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.1,
    help="1.0 = 관련성만 / 0.0 = 다양성만",
)

# ─────────────────────────────────────────────
# 문서 업로드
# ─────────────────────────────────────────────
st.header("📄 문서 준비")

col_up, col_def = st.columns([2, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "텍스트 파일 업로드 (.txt)",
        type=["txt"],
    )

with col_def:
    use_default = st.checkbox("기본 문서 사용 (ai-story.txt)", value=True)

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ai-story.txt")


def load_doc_text() -> str | None:
    if uploaded_file is not None:
        return uploaded_file.read().decode("utf-8")
    if use_default and os.path.exists(DEFAULT_DATA_PATH):
        with open(DEFAULT_DATA_PATH, encoding="utf-8") as f:
            return f.read()
    return None


# ─────────────────────────────────────────────
# RetrieverBundle 캐시
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="벡터스토어 구축 중... (최초 1회)")
def cached_bundle(text: str, api_key: str, _chunk_size: int, _chunk_overlap: int):
    from utils import build_retriever_bundle
    return build_retriever_bundle(
        text=text,
        api_key=api_key,
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
    )


# ─────────────────────────────────────────────
# 검색 실행 섹션
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🔎 검색 실행")

query = st.text_input(
    "검색어를 입력하세요",
    placeholder="예: Transformer 모델의 핵심 메커니즘은 무엇인가요?",
)

run_btn = st.button("🚀 검색 실행", type="primary", use_container_width=True)

if run_btn:
    # ── 유효성 검사 ──────────────────────────────
    if not openai_api_key:
        st.error("사이드바에서 OpenAI API Key를 입력해주세요.")
        st.stop()
    if not query.strip():
        st.error("검색어를 입력해주세요.")
        st.stop()
    if not search_methods:
        st.error("비교할 검색 방식을 하나 이상 선택해주세요.")
        st.stop()

    doc_text = load_doc_text()
    if doc_text is None:
        st.error("문서를 찾을 수 없습니다. 파일을 업로드하거나 '기본 문서 사용'을 체크해주세요.")
        st.stop()

    # ── 벡터스토어 빌드 ──────────────────────────
    bundle = cached_bundle(doc_text, openai_api_key, chunk_size, chunk_overlap)

    # ── 검색 실행 ─────────────────────────────────
    from utils import run_all_searches, compute_overlap_matrix

    with st.spinner("검색 중..."):
        search_start = time.perf_counter()
        results = run_all_searches(
            bundle=bundle,
            query=query,
            methods=search_methods,
            top_k=top_k,
            lambda_mult=lambda_mult,
            bm25_weight=bm25_weight,
            openai_api_key=openai_api_key,
        )
        total_elapsed = time.perf_counter() - search_start

    # ── 오류 표시 ─────────────────────────────────
    for r in results:
        if r.error:
            st.error(f"[{r.method}] 오류: {r.error}")

    ok_results = [r for r in results if not r.error]
    if not ok_results:
        st.stop()

    # ─────────────────────────────────────────────
    # 상단 요약 지표
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 검색 결과 요약")

    metric_cols = st.columns(len(ok_results) + 1)
    metric_cols[0].metric("총 소요 시간", f"{total_elapsed:.2f}s")

    method_colors = {
        "Similarity": "#4A90D9",
        "MMR": "#27AE60",
        "BM25": "#E67E22",
        "Ensemble": "#8E44AD",
        "MultiQuery": "#C0392B",
    }

    for col, r in zip(metric_cols[1:], ok_results):
        delta_color = "normal"
        col.metric(
            label=r.method,
            value=f"{len(r.docs)}개 문서",
            delta=f"{r.elapsed:.2f}s",
            delta_color=delta_color,
        )

    # ─────────────────────────────────────────────
    # 나란히 비교 (최대 3열)
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 검색 결과 나란히 비교")
    st.caption(f"검색어: **{query}**")

    COLS_PER_ROW = 3
    chunks = [
        ok_results[i : i + COLS_PER_ROW]
        for i in range(0, len(ok_results), COLS_PER_ROW)
    ]

    METHOD_DESC = {
        "Similarity": lambda: f"코사인 유사도 · k={top_k}",
        "MMR": lambda: f"다양성 고려 · λ={lambda_mult} · k={top_k}",
        "BM25": lambda: f"키워드 빈도 · k={top_k}",
        "Ensemble": lambda: f"BM25({bm25_weight}) + Vector({vector_weight}) · k={top_k}",
        "MultiQuery": lambda: f"LLM 쿼리 다각화 · k={top_k}",
    }
    METHOD_ICON = {
        "Similarity": "🔵",
        "MMR": "🟢",
        "BM25": "🟠",
        "Ensemble": "🟣",
        "MultiQuery": "🔴",
    }

    for row_results in chunks:
        cols = st.columns(len(row_results))
        for col, r in zip(cols, row_results):
            color = method_colors.get(r.method, "#888")
            icon = METHOD_ICON.get(r.method, "⚪")
            desc = METHOD_DESC.get(r.method, lambda: "")()

            with col:
                st.markdown(
                    f"<div style='background:{color}22;border-left:4px solid {color};"
                    f"padding:8px 12px;border-radius:4px;margin-bottom:8px'>"
                    f"<b>{icon} {r.method}</b><br>"
                    f"<small style='color:#555'>{desc}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # MultiQuery: 생성된 쿼리 표시
                if r.method == "MultiQuery" and r.generated_queries:
                    with st.expander("LLM 생성 쿼리 보기", expanded=False):
                        for qi, q_text in enumerate(r.generated_queries, 1):
                            st.markdown(f"{qi}. *{q_text}*")

                if not r.docs:
                    st.info("검색 결과 없음")
                    continue

                for i, doc in enumerate(r.docs, 1):
                    preview = doc.page_content[:50].replace("\n", " ")
                    with st.expander(
                        f"문서 {i}  —  {preview}...",
                        expanded=(i == 1),
                    ):
                        # 관련성 색상 바 (순위 기반)
                        relevance_ratio = 1.0 - (i - 1) / max(len(r.docs), 1)
                        bar_color = color
                        st.markdown(
                            f"<div style='width:{relevance_ratio*100:.0f}%;"
                            f"height:6px;background:{bar_color};"
                            f"border-radius:3px;margin-bottom:6px'></div>",
                            unsafe_allow_html=True,
                        )
                        st.write(doc.page_content)
                        if doc.metadata:
                            st.caption(f"메타데이터: {doc.metadata}")

    # ─────────────────────────────────────────────
    # 공통 문서 분석
    # ─────────────────────────────────────────────
    if len(ok_results) >= 2:
        st.markdown("---")
        st.subheader("🔁 방식 간 공통 문서 분석")

        overlap_matrix = compute_overlap_matrix(ok_results)
        if overlap_matrix:
            ov_cols = st.columns(len(overlap_matrix))
            for col, ((m1, m2), cnt) in zip(ov_cols, overlap_matrix.items()):
                total = max(
                    len([r for r in ok_results if r.method == m1][0].docs),
                    len([r for r in ok_results if r.method == m2][0].docs),
                    1,
                )
                col.metric(
                    label=f"{m1} ∩ {m2}",
                    value=f"{cnt}개",
                    delta=f"전체 최대 {total}개 중",
                )

    # ─────────────────────────────────────────────
    # 검색 시간 비교 차트
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⏱️ 검색 방식별 소요 시간")

    import pandas as pd

    time_data = pd.DataFrame(
        {
            "방식": [r.method for r in ok_results],
            "소요 시간 (초)": [round(r.elapsed, 3) for r in ok_results],
        }
    ).set_index("방식")

    st.bar_chart(time_data, height=200)

# ─────────────────────────────────────────────
# 하단 설명
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 검색 방식 설명", expanded=False):
    st.markdown(
        """
| 방식 | 원리 | 장점 | 단점 |
|------|------|------|------|
| **Similarity** | 코사인 유사도 | 빠르고 간단 | 중복 결과 발생 가능 |
| **MMR** | 유사도 + 다양성 | 다양한 관점 반환 | 약간 느림 |
| **BM25** | 키워드 빈도(TF-IDF 계열) | 정확한 용어 매칭 | 동의어·의미 검색 미흡 |
| **Ensemble** | BM25 + Vector RRF 결합 | 두 방식의 장점 결합 | 가중치 튜닝 필요 |
| **MultiQuery** | LLM이 쿼리를 다각화 | recall(재현율) 향상 | LLM 추가 비용 발생 |

**Ensemble 가중치 가이드**
- 전문 용어·고유명사 중심 → BM25 높임 (0.6 ~ 0.7)
- 자연어 질문·동의어 많음 → Vector 높임 (0.6 ~ 0.7)
- 일반 사용 → 균등 (0.5)
"""
    )
