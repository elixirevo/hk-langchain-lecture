"""
Ch06-6 Reranker — Before vs After 비교 데모
=============================================
실행: streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Ch06-6 Reranker 비교",
    page_icon="🎯",
    layout="wide",
)

# ─────────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────────
st.title("🎯 Ch06-6 Reranker — Before vs After 비교")
st.markdown(
    "벡터 검색 결과(Before)와 Reranker 적용 결과(After)를 **나란히 비교**하고, "
    "순위 변화와 관련성 점수를 시각화합니다."
)

# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ 설정")

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
    help="벡터 임베딩에 사용됩니다. 환경변수 OPENAI_API_KEY가 있으면 자동 입력.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("📄 문서 설정")

chunk_size = st.sidebar.slider("청크 크기", 200, 1000, 500, 100)
chunk_overlap = st.sidebar.slider("청크 오버랩", 0, 200, 100, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("🔢 검색 파라미터")

initial_k = st.sidebar.slider(
    "초기 검색 수 (k)",
    min_value=5,
    max_value=20,
    value=10,
    help="Reranker가 재정렬할 후보 문서 수입니다.",
)

top_n = st.sidebar.slider(
    "최종 반환 수 (top_n)",
    min_value=1,
    max_value=10,
    value=3,
    help="Reranker가 최종 선택하는 문서 수입니다.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Reranker 선택")

reranker_choice = st.sidebar.radio(
    "사용할 Reranker",
    options=["Cross-Encoder (BAAI/bge-reranker-v2-m3)", "FlashRank (ms-marco-MultiBERT-L-12)"],
    index=0,
)

# 선택된 Reranker 설명
if "Cross-Encoder" in reranker_choice:
    st.sidebar.info(
        "**Cross-Encoder**: 쿼리-문서 쌍을 함께 분석해 정확한 관련성 평가.\n\n"
        "다국어 지원 · 높은 정확도 · 로컬 실행 · GPU 권장"
    )
else:
    st.sidebar.info(
        "**FlashRank**: 초경량·초고속 로컬 Reranker.\n\n"
        "다국어 지원 · API 키 불필요 · CPU에서도 빠름"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("📊 점수 표시 옵션")

show_normalized = st.sidebar.checkbox(
    "점수 정규화 (0~1)",
    value=True,
    help="체크 시 점수를 min-max 정규화하여 표시합니다.",
)

# ─────────────────────────────────────────────
# 문서 업로드
# ─────────────────────────────────────────────
st.header("📄 문서 준비")

col_up, col_def = st.columns([2, 1])

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "data", "appendix-keywords.txt"
)

with col_up:
    uploaded_file = st.file_uploader("텍스트 파일 업로드 (.txt)", type=["txt"])

with col_def:
    use_default = st.checkbox("기본 문서 사용 (appendix-keywords.txt)", value=True)


def load_doc_text() -> str | None:
    if uploaded_file is not None:
        return uploaded_file.read().decode("utf-8")
    if use_default and os.path.exists(DEFAULT_DATA_PATH):
        with open(DEFAULT_DATA_PATH, encoding="utf-8") as f:
            return f.read()
    return None


# ─────────────────────────────────────────────
# 벡터스토어 캐시
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="벡터스토어 구축 중... (최초 1회)")
def cached_vectorstore(
    text: str,
    api_key: str,
    _chunk_size: int,
    _chunk_overlap: int,
    _initial_k: int,
):
    from utils import build_vectorstore
    return build_vectorstore(
        text=text,
        api_key=api_key,
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        initial_k=_initial_k,
    )


@st.cache_resource(show_spinner="Reranker 모델 로드 중... (최초 1회, 시간이 걸릴 수 있습니다)")
def cached_reranker(reranker_type: str, _top_n: int):
    from utils import build_cross_encoder_reranker, build_flashrank_reranker

    if reranker_type == "cross_encoder":
        return build_cross_encoder_reranker(
            model_name="BAAI/bge-reranker-v2-m3",
            top_n=_top_n,
        )
    else:
        return build_flashrank_reranker(
            model_name="ms-marco-MultiBERT-L-12",
            top_n=_top_n,
        )


# ─────────────────────────────────────────────
# 검색 섹션
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🔎 검색 실행")

query = st.text_input(
    "검색어를 입력하세요",
    placeholder="예: 자연어 처리에서 단어를 벡터로 변환하는 기술은?",
)

run_btn = st.button("🚀 검색 + Reranking 실행", type="primary", use_container_width=True)

if run_btn:
    # ── 유효성 검사 ──────────────────────────────
    if not openai_api_key:
        st.error("사이드바에서 OpenAI API Key를 입력해주세요.")
        st.stop()
    if not query.strip():
        st.error("검색어를 입력해주세요.")
        st.stop()

    doc_text = load_doc_text()
    if doc_text is None:
        st.error("문서를 찾을 수 없습니다. 파일을 업로드하거나 '기본 문서 사용'을 체크해주세요.")
        st.stop()

    # ── 벡터스토어 빌드 ──────────────────────────
    vectorstore, base_retriever, docs = cached_vectorstore(
        doc_text, openai_api_key, chunk_size, chunk_overlap, initial_k
    )

    # ── Reranker 빌드 ─────────────────────────────
    reranker_type = "cross_encoder" if "Cross-Encoder" in reranker_choice else "flashrank"
    reranker = cached_reranker(reranker_type, top_n)

    # ── 검색 실행 ─────────────────────────────────
    from utils import run_reranker_comparison, compute_rank_changes, normalize_scores

    with st.spinner("검색 및 Reranking 중..."):
        try:
            before_docs, after_docs, retrieval_elapsed, rerank_elapsed = run_reranker_comparison(
                base_retriever=base_retriever,
                reranker=reranker,
                query=query,
                top_n=top_n,
            )
        except Exception as exc:
            st.error(f"검색 중 오류가 발생했습니다: {exc}")
            st.stop()

    if show_normalized:
        after_docs = normalize_scores(after_docs)

    # ─────────────────────────────────────────────
    # 상단 요약 지표
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 검색 결과 요약")

    m_cols = st.columns(4)
    m_cols[0].metric("사용 Reranker", reranker_type.replace("_", " ").title())
    m_cols[1].metric("초기 검색 문서", f"{len(before_docs)}개")
    m_cols[2].metric("최종 선택 문서", f"{len(after_docs)}개")
    m_cols[3].metric(
        "검색 소요 시간",
        f"{retrieval_elapsed:.2f}s",
        delta=f"재정렬: {rerank_elapsed:.2f}s",
    )

    # 순위 변화 계산
    rank_changes = compute_rank_changes(before_docs, after_docs)

    # ─────────────────────────────────────────────
    # Before / After 나란히 비교
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Before vs After 순위 비교")
    st.caption(f"검색어: **{query}**")

    col_before, col_after = st.columns(2)

    # ── Before (벡터 검색 원본 순서) ─────────────
    with col_before:
        st.markdown(
            "<div style='background:#4A90D922;border-left:4px solid #4A90D9;"
            "padding:8px 12px;border-radius:4px;margin-bottom:12px'>"
            "<b>🔵 Before — 벡터 유사도 검색 (원본 순서)</b><br>"
            f"<small style='color:#555'>상위 {len(before_docs)}개 후보 문서</small>"
            "</div>",
            unsafe_allow_html=True,
        )

        for rd in before_docs[:max(initial_k, len(before_docs))]:
            preview = rd.doc.page_content[:60].replace("\n", " ")
            is_selected = any(
                rd.doc.page_content == ad.doc.page_content for ad in after_docs
            )

            border = "#4A90D9" if is_selected else "#ccc"
            bg = "#EBF5FB" if is_selected else "#fafafa"
            badge = "✅ 선택됨" if is_selected else "❌ 제외됨"
            badge_color = "#27AE60" if is_selected else "#999"

            st.markdown(
                f"<div style='border:1px solid {border};background:{bg};"
                f"border-radius:6px;padding:8px 12px;margin-bottom:6px'>"
                f"<b>#{rd.rank}</b> &nbsp;"
                f"<span style='color:{badge_color};font-size:0.8em'>{badge}</span><br>"
                f"<small>{preview}...</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── After (Reranker 적용 순서) ────────────────
    with col_after:
        reranker_label = "Cross-Encoder" if reranker_type == "cross_encoder" else "FlashRank"
        st.markdown(
            "<div style='background:#27AE6022;border-left:4px solid #27AE60;"
            "padding:8px 12px;border-radius:4px;margin-bottom:12px'>"
            f"<b>🟢 After — {reranker_label} 재정렬 결과</b><br>"
            f"<small style='color:#555'>최종 상위 {len(after_docs)}개 선택</small>"
            "</div>",
            unsafe_allow_html=True,
        )

        for rd in after_docs:
            content = rd.doc.page_content
            preview = content[:60].replace("\n", " ")
            change = rank_changes.get(content, 0)

            # 순위 변화 표시
            if change > 0:
                arrow = f"⬆️ +{change}"
                arrow_color = "#27AE60"
            elif change < 0:
                arrow = f"⬇️ {change}"
                arrow_color = "#E74C3C"
            else:
                arrow = "➡️ 신규"
                arrow_color = "#8E44AD"

            score_html = ""
            if rd.score is not None:
                bar_width = int(rd.score * 100) if show_normalized else min(int(abs(rd.score) * 10), 100)
                score_val = f"{rd.score:.4f}"
                score_html = (
                    f"<div style='margin-top:4px'>"
                    f"<small style='color:#555'>점수: <b>{score_val}</b></small>"
                    f"<div style='width:100%;background:#eee;border-radius:3px;height:6px;margin-top:2px'>"
                    f"<div style='width:{bar_width}%;background:#27AE60;height:6px;border-radius:3px'></div>"
                    f"</div></div>"
                )

            st.markdown(
                f"<div style='border:1px solid #27AE60;background:#F0FFF4;"
                f"border-radius:6px;padding:8px 12px;margin-bottom:6px'>"
                f"<b>#{rd.rank}</b> &nbsp;"
                f"<span style='color:{arrow_color};font-size:0.85em'>{arrow}</span><br>"
                f"<small>{preview}...</small>"
                f"{score_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ─────────────────────────────────────────────
    # 점수 바 차트
    # ─────────────────────────────────────────────
    if any(rd.score is not None for rd in after_docs):
        st.markdown("---")
        st.subheader("📉 관련성 점수 시각화")

        score_data = []
        for rd in after_docs:
            label = f"#{rd.rank} {rd.doc.page_content[:30].replace(chr(10), ' ')}..."
            score_data.append({"문서": label, "점수": rd.score if rd.score is not None else 0.0})

        df_scores = pd.DataFrame(score_data).set_index("문서")
        st.bar_chart(df_scores, height=300)

        # 수치 테이블
        with st.expander("점수 테이블 보기", expanded=False):
            st.dataframe(df_scores.reset_index(), use_container_width=True)

    # ─────────────────────────────────────────────
    # 순위 변화 요약
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔄 순위 변화 요약")

    change_rows = []
    for rd in after_docs:
        content = rd.doc.page_content
        change = rank_changes.get(content, 0)
        before_rank = rd.rank + change if change != 0 else "신규"
        change_rows.append(
            {
                "최종 순위": f"#{rd.rank}",
                "이전 순위": f"#{before_rank}" if isinstance(before_rank, int) else before_rank,
                "변화": (
                    f"⬆️ +{change}" if isinstance(change, int) and change > 0
                    else f"⬇️ {change}" if isinstance(change, int) and change < 0
                    else "➡️ 신규"
                ),
                "점수": f"{rd.score:.4f}" if rd.score is not None else "N/A",
                "문서 미리보기": rd.doc.page_content[:80].replace("\n", " ") + "...",
            }
        )

    st.dataframe(pd.DataFrame(change_rows), use_container_width=True)

    # ─────────────────────────────────────────────
    # 전체 문서 내용 (펼치기)
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 최종 선택 문서 전체 내용")

    for rd in after_docs:
        with st.expander(
            f"#{rd.rank} — {rd.doc.page_content[:60].replace(chr(10), ' ')}...",
            expanded=False,
        ):
            if rd.score is not None:
                st.caption(
                    f"관련성 점수: **{rd.score:.4f}** "
                    f"{'(정규화됨)' if show_normalized else ''}"
                )
            st.write(rd.doc.page_content)
            if rd.doc.metadata:
                st.caption(f"메타데이터: {rd.doc.metadata}")

# ─────────────────────────────────────────────
# 하단 설명
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 Reranker 설명", expanded=False):
    st.markdown(
        """
## Two-Stage Retrieval 개념

| 단계 | 방식 | 역할 | 속도 |
|------|------|------|------|
| **1단계** | Bi-Encoder (벡터 검색) | 후보 문서 빠르게 추출 (k=10~20) | 빠름 |
| **2단계** | Cross-Encoder (Reranker) | 쿼리-문서 상호작용 분석, 정밀 재정렬 | 느리지만 정확 |

## Reranker 비교

| 특징 | Cross-Encoder (BAAI) | FlashRank |
|------|:---:|:---:|
| 속도 | 보통 | 초고속 |
| 정확도 | 높음 | 보통 |
| 비용 | 무료 (로컬) | 무료 (로컬) |
| GPU | 권장 | 불필요 |
| 다국어 | ✅ | ✅ |

## 파라미터 가이드

- **초기 검색 수 (k)**: 충분한 후보 확보 (보통 10~20개)
- **최종 반환 수 (top_n)**: LLM 컨텍스트 크기 고려 (보통 3~5개)
- k가 클수록 Reranker가 더 다양한 후보를 검토해 품질이 향상됩니다.
"""
    )
