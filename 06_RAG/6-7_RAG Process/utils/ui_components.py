"""
ui_components.py — 재사용 가능한 Streamlit UI 위젯 모음

포함 내용:
  - render_sidebar()           : 사이드바 전체 렌더링
  - render_doc_stats_dashboard(): 문서 통계 대시보드
  - render_retrieved_docs()    : 참조 문서 카드 (유사도 점수 포함)
  - render_pipeline_diagram()  : 파이프라인 시각화
  - render_conversation_stats(): 대화형 RAG 통계 행
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document


# ──────────────────────────────────────────────
# CSS 주입 (최초 1회)
# ──────────────────────────────────────────────
_CSS_INJECTED = False


def inject_css() -> None:
    global _CSS_INJECTED
    if _CSS_INJECTED:
        return
    st.markdown(
        """
        <style>
        .metric-card {
            background: #f0f2f6;
            border-radius: 8px;
            padding: 12px 16px;
            text-align: center;
        }
        .metric-card h3 { margin: 0; font-size: 1.6rem; color: #1f77b4; }
        .metric-card p  { margin: 0; font-size: 0.8rem; color: #555; }
        .step-badge {
            display: inline-block;
            background: #1f77b4;
            color: white;
            border-radius: 50%;
            width: 24px; height: 24px;
            line-height: 24px;
            text-align: center;
            font-weight: bold;
            margin-right: 6px;
        }
        .doc-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            background: #fafafa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _CSS_INJECTED = True


# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────

def render_sidebar() -> dict:
    """사이드바를 렌더링하고 설정값 딕셔너리를 반환합니다.

    Returns:
        dict with keys:
            active_key, llm_model, temperature, uploaded_file,
            chunk_size, chunk_overlap, top_k, search_type,
            lambda_mult, bm25_weight
    """
    inject_css()
    cfg: dict = {}

    with st.sidebar:
        st.header("⚙️ 설정")

        # ── API 키 ──────────────────────────────
        st.subheader("🔑 API 키")
        import os
        env_key = os.getenv("OPENAI_API_KEY", "")
        if env_key:
            st.success("환경 변수에서 API 키 로드됨", icon="✅")
            cfg["active_key"] = env_key
        else:
            input_key = st.text_input(
                "OpenAI API 키",
                type="password",
                placeholder="sk-...",
                help=".env 파일에 OPENAI_API_KEY를 설정하거나 여기에 입력하세요.",
            )
            if input_key:
                os.environ["OPENAI_API_KEY"] = input_key
                cfg["active_key"] = input_key
            else:
                cfg["active_key"] = ""

        st.divider()

        # ── 모델 설정 ────────────────────────────
        st.subheader("🤖 모델 설정")
        cfg["llm_model"] = st.selectbox(
            "LLM 모델",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
        )
        cfg["temperature"] = st.slider(
            "Temperature",
            min_value=0.0, max_value=1.0, value=0.0, step=0.1,
            help="0=결정적, 1=창의적",
        )

        st.divider()

        # ── PDF 업로드 ───────────────────────────
        st.subheader("📄 문서 업로드")
        cfg["uploaded_file"] = st.file_uploader(
            "PDF 파일",
            type=["pdf"],
            help="업로드 즉시 자동으로 Load → Split → Embed → Store 진행",
        )

        st.divider()

        # ── 검색 설정 ────────────────────────────
        st.subheader("🔍 검색 설정")
        cfg["chunk_size"] = st.slider("Chunk Size", 200, 2000, 1000, 100,
                                      help="각 청크의 최대 문자 수")
        cfg["chunk_overlap"] = st.slider("Chunk Overlap", 0, 400, 100, 50,
                                         help="인접 청크 간 겹치는 문자 수")
        cfg["top_k"] = st.slider("검색 문서 수 (k)", 1, 10, 4,
                                 help="질문당 가져올 청크 수")
        cfg["search_type"] = st.selectbox(
            "검색 방식",
            ["similarity", "mmr", "ensemble"],
            help=(
                "similarity: 코사인 유사도\n"
                "mmr: 다양성 고려\n"
                "ensemble: BM25+FAISS 혼합"
            ),
        )

        cfg["lambda_mult"] = 0.5
        cfg["bm25_weight"] = 0.4

        if cfg["search_type"] == "mmr":
            cfg["lambda_mult"] = st.slider(
                "MMR lambda", 0.0, 1.0, 0.5, 0.1,
                help="1.0=관련성 우선, 0.0=다양성 우선",
            )
        if cfg["search_type"] == "ensemble":
            cfg["bm25_weight"] = st.slider(
                "BM25 가중치", 0.0, 1.0, 0.4, 0.1,
                help="나머지는 FAISS에 자동 배분",
            )

        st.divider()

        # ── 문서 통계 ────────────────────────────
        if st.session_state.get("doc_stats"):
            stats = st.session_state["doc_stats"]
            st.subheader("📊 문서 정보")
            st.markdown(
                f"""
                | 항목 | 값 |
                |------|-----|
                | 파일명 | {stats.get('filename', '-')} |
                | 페이지 수 | {stats.get('pages', 0)} |
                | 청크 수 | {stats.get('chunks', 0)} |
                | 평균 청크 크기 | {stats.get('avg_chunk', 0):.0f} 자 |
                | 임베딩 차원 | {stats.get('embed_dim', 1536)} |
                """
            )

        st.divider()

        # ── 대화 관리 ────────────────────────────
        st.subheader("💬 대화 관리")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ 초기화", use_container_width=True):
                for key in ["basic_messages", "adv_messages", "conv_messages", "conv_history"]:
                    st.session_state[key] = []
                st.toast("모든 대화가 초기화되었습니다.")
                st.rerun()
        with col_b:
            all_msgs = (
                st.session_state.get("basic_messages", [])
                + st.session_state.get("adv_messages", [])
                + st.session_state.get("conv_messages", [])
            )
            if all_msgs:
                export_data = json.dumps(
                    [{"role": m["role"], "content": m["content"]} for m in all_msgs],
                    ensure_ascii=False,
                    indent=2,
                )
                st.download_button(
                    "💾 내보내기",
                    data=export_data,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

    return cfg


# ──────────────────────────────────────────────
# 문서 통계 대시보드
# ──────────────────────────────────────────────

def render_doc_stats_dashboard(stats: dict) -> None:
    """문서 처리 통계를 4컬럼 카드로 표시합니다."""
    inject_css()
    c1, c2, c3, c4 = st.columns(4)
    items = [
        (c1, "총 페이지", stats.get("pages", 0)),
        (c2, "총 청크 수", stats.get("chunks", 0)),
        (c3, "평균 청크 크기", f"{stats.get('avg_chunk', 0):.0f}자"),
        (c4, "임베딩 차원", stats.get("embed_dim", 1536)),
    ]
    for col, label, value in items:
        col.markdown(
            f'<div class="metric-card"><h3>{value}</h3><p>{label}</p></div>',
            unsafe_allow_html=True,
        )
    st.write("")


# ──────────────────────────────────────────────
# 참조 문서 카드
# ──────────────────────────────────────────────

def render_retrieved_docs(
    docs: list[Document],
    question: str,
    vectorstore=None,
    expanded: bool = False,
) -> None:
    """검색된 문서를 유사도 점수와 함께 expander 카드로 표시합니다."""
    if not docs:
        return

    # 유사도 점수 계산
    if vectorstore is not None:
        from utils.rag_pipeline import get_relevance_scores
        scores = get_relevance_scores(question, docs, vectorstore)
    else:
        scores = [0.0] * len(docs)

    with st.expander(
        f"🔍 참조 문서 ({len(docs)}개) — 클릭하여 펼치기",
        expanded=expanded,
    ):
        for i, (doc, score) in enumerate(zip(docs, scores), 1):
            page = doc.metadata.get("page", "?")
            source = Path(doc.metadata.get("source", "unknown")).name
            pct = int(score * 100)
            color = "#28a745" if pct >= 70 else "#ffc107" if pct >= 40 else "#dc3545"

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**[{i}] {source} — 페이지 {page}**")
            with col2:
                st.markdown(
                    f'<span style="color:{color};font-weight:bold">유사도 {pct}%</span>',
                    unsafe_allow_html=True,
                )
            st.caption(
                doc.page_content[:500]
                + ("..." if len(doc.page_content) > 500 else "")
            )
            if i < len(docs):
                st.divider()


# ──────────────────────────────────────────────
# 파이프라인 다이어그램
# ──────────────────────────────────────────────

def render_pipeline_diagram(chunk_size: int, top_k: int, llm_model: str) -> None:
    """RAG 8단계 파이프라인을 5컬럼 배지로 시각화합니다."""
    inject_css()
    with st.expander("📊 RAG 파이프라인 구성 보기", expanded=False):
        cols = st.columns(5)
        steps = [
            ("1", "문서 로드", "PyMuPDFLoader"),
            ("2", "텍스트 분할", f"chunk={chunk_size}"),
            ("3", "임베딩", "text-embedding-3-small"),
            ("4", "벡터 DB", "FAISS"),
            ("5", "검색 + LLM", f"k={top_k}, {llm_model}"),
        ]
        for col, (num, title, desc) in zip(cols, steps):
            col.markdown(
                f'<div class="metric-card">'
                f'<span class="step-badge">{num}</span>'
                f'<strong>{title}</strong><br><small>{desc}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────
# 대화형 RAG 통계
# ──────────────────────────────────────────────

def render_conversation_stats(conv_history: list) -> None:
    """대화형 RAG의 현재 세션 통계를 3컬럼으로 표시합니다."""
    n_turns = len(conv_history) // 2
    n_messages = len(conv_history)
    approx_tokens = sum(len(m.content) // 4 for m in conv_history)

    c1, c2, c3 = st.columns(3)
    c1.metric("대화 턴 수", n_turns)
    c2.metric("누적 메시지", n_messages)
    c3.metric("토큰 예상 (근사)", approx_tokens)
