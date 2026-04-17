"""
Ch06-7 RAG Process — 플래그십 RAG 챗봇 데모
=============================================

탭 구성:
탭 1 — 기본 RAG       : 8단계 파이프라인 (similarity 검색)
탭 2 — 고급 RAG       : Ensemble(BM25+FAISS) + MMR 검색
탭 3 — 대화형 RAG     : 대화 이력 유지, 후속 질문 지원

실행:
    streamlit run streamlit_app.py

필요 패키지:
    pip install streamlit langchain langchain-openai langchain-community
    pip install faiss-cpu pymupdf rank-bm25 python-dotenv
"""

from __future__ import annotations

import os
import sys

import streamlit as st
from dotenv import load_dotenv

# utils 패키지를 현재 파일 기준으로 임포트
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()

from utils.rag_pipeline import build_vectorstore, get_retriever, stream_response
from utils.ui_components import (
    inject_css,
    render_conversation_stats,
    render_doc_stats_dashboard,
    render_pipeline_diagram,
    render_retrieved_docs,
    render_sidebar,
)

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG 챗봇",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

st.title("📚 RAG 문서 기반 챗봇")
st.caption(
    "PDF를 업로드하면 3가지 RAG 방식(기본 / 고급 / 대화형)으로 문서 Q&A를 체험할 수 있습니다."
)

# ──────────────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────────────
_DEFAULTS: dict = {
    "vectorstore": None,
    "split_docs": [],
    "doc_stats": {},
    "uploaded_file_key": "",
    "basic_messages": [],
    "adv_messages": [],
    "conv_messages": [],
    "conv_history": [],      # list[HumanMessage | AIMessage]
    "_last_docs": [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ──────────────────────────────────────────────
# 사이드바 렌더링 → 설정값 수집
# ──────────────────────────────────────────────
cfg = render_sidebar()

# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def require_api_key() -> bool:
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("사이드바에서 OpenAI API 키를 먼저 입력해 주세요.", icon="🔑")
        return False
    return True


def require_vectorstore() -> bool:
    if st.session_state.vectorstore is None:
        st.info("사이드바에서 PDF 파일을 업로드해 주세요.", icon="👈")
        return False
    return True


def _make_retriever():
    return get_retriever(
        vectorstore=st.session_state.vectorstore,
        split_docs=st.session_state.split_docs,
        search_type=cfg["search_type"],
        k=cfg["top_k"],
        lambda_mult=cfg["lambda_mult"],
        bm25_weight=cfg["bm25_weight"],
    )


def _make_adv_retriever():
    """고급 탭은 항상 ensemble + mmr 조합."""
    return get_retriever(
        vectorstore=st.session_state.vectorstore,
        split_docs=st.session_state.split_docs,
        search_type="ensemble",
        k=cfg["top_k"],
        lambda_mult=cfg["lambda_mult"],
        bm25_weight=cfg["bm25_weight"],
    )


# ──────────────────────────────────────────────
# 문서 업로드 처리
# ──────────────────────────────────────────────
uploaded = cfg["uploaded_file"]
if uploaded is not None:
    file_key = f"{uploaded.name}_{cfg['chunk_size']}_{cfg['chunk_overlap']}"
    if st.session_state.uploaded_file_key != file_key:
        if not cfg["active_key"]:
            st.error("API 키를 먼저 입력해 주세요.")
        else:
            try:
                pdf_bytes = uploaded.read()
                vectorstore, split_docs, stats = build_vectorstore(
                    pdf_bytes=pdf_bytes,
                    filename=uploaded.name,
                    chunk_size=cfg["chunk_size"],
                    chunk_overlap=cfg["chunk_overlap"],
                )
                st.session_state.vectorstore = vectorstore
                st.session_state.split_docs = split_docs
                st.session_state.doc_stats = stats
                st.session_state.uploaded_file_key = file_key
                # 대화 초기화
                st.session_state.basic_messages = []
                st.session_state.adv_messages = []
                st.session_state.conv_messages = []
                st.session_state.conv_history = []

                st.toast(f"'{uploaded.name}' 처리 완료!", icon="✅")
            except Exception as exc:
                st.error(f"문서 처리 오류: {exc}")

# ──────────────────────────────────────────────
# 문서 통계 대시보드
# ──────────────────────────────────────────────
if st.session_state.doc_stats:
    render_doc_stats_dashboard(st.session_state.doc_stats)

# ──────────────────────────────────────────────
# 메인 탭
# ──────────────────────────────────────────────
tab_basic, tab_adv, tab_conv = st.tabs(
    ["🔰 기본 RAG", "🚀 고급 RAG (Ensemble + MMR)", "💬 대화형 RAG"]
)


# ══════════════════════════════════════════════
# 탭 1: 기본 RAG
# ══════════════════════════════════════════════
with tab_basic:
    st.subheader("🔰 기본 RAG — 8단계 파이프라인")
    st.markdown(
        "**사전 작업**: 문서 로드 → 분할 → 임베딩 → 벡터 DB 저장  \n"
        "**실행 단계**: 질문 → 검색 → 프롬프트 → LLM → 답변"
    )

    if not require_vectorstore():
        pass
    else:
        render_pipeline_diagram(
            chunk_size=cfg["chunk_size"],
            top_k=cfg["top_k"],
            llm_model=cfg["llm_model"],
        )
        st.divider()

        # 기존 메시지 표시
        for msg in st.session_state.basic_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    render_retrieved_docs(
                        docs=msg["sources"],
                        question=msg.get("question", ""),
                        vectorstore=st.session_state.vectorstore,
                    )

        # 채팅 입력
        if prompt := st.chat_input("문서에 대해 질문하세요...", key="basic_input"):
            if not require_api_key():
                pass
            else:
                st.chat_message("user").markdown(prompt)
                st.session_state.basic_messages.append({"role": "user", "content": prompt})

                try:
                    retriever = _make_retriever()
                    with st.chat_message("assistant"):
                        response = st.write_stream(
                            stream_response(
                                question=prompt,
                                retriever=retriever,
                                chat_history=None,
                                llm_model=cfg["llm_model"],
                                temperature=cfg["temperature"],
                            )
                        )
                        docs = st.session_state.get("_last_docs", [])
                        render_retrieved_docs(
                            docs=docs,
                            question=prompt,
                            vectorstore=st.session_state.vectorstore,
                        )

                    st.session_state.basic_messages.append(
                        {"role": "assistant", "content": response,
                         "sources": docs, "question": prompt}
                    )
                except Exception as e:
                    st.error(f"오류: {e}")


# ══════════════════════════════════════════════
# 탭 2: 고급 RAG
# ══════════════════════════════════════════════
with tab_adv:
    st.subheader("🚀 고급 RAG — Ensemble + MMR")
    st.markdown(
        "**BM25 (키워드 검색)** + **FAISS with MMR (의미 + 다양성)** 결합.  \n"
        "기본 RAG보다 더 풍부하고 다양한 문서를 검색합니다."
    )

    if not require_vectorstore():
        pass
    else:
        with st.expander("⚙️ 고급 검색 설정 미리보기", expanded=False):
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**BM25 Retriever** (키워드 기반)")
                st.markdown(f"- 가중치: `{cfg['bm25_weight']}`")
                st.markdown(f"- 반환 문서: `{cfg['top_k']}개`")
            with col_r:
                st.markdown("**FAISS + MMR** (의미 + 다양성)")
                st.markdown(f"- 가중치: `{round(1.0 - cfg['bm25_weight'], 2)}`")
                st.markdown(f"- lambda_mult: `{cfg['lambda_mult']}` (1=관련성, 0=다양성)")

        st.divider()

        # 기존 메시지 표시
        for msg in st.session_state.adv_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    render_retrieved_docs(
                        docs=msg["sources"],
                        question=msg.get("question", ""),
                        vectorstore=st.session_state.vectorstore,
                    )

        # 채팅 입력
        if prompt := st.chat_input("문서에 대해 질문하세요...", key="adv_input"):
            if not require_api_key():
                pass
            else:
                st.chat_message("user").markdown(prompt)
                st.session_state.adv_messages.append({"role": "user", "content": prompt})

                try:
                    retriever = _make_adv_retriever()
                    with st.chat_message("assistant"):
                        response = st.write_stream(
                            stream_response(
                                question=prompt,
                                retriever=retriever,
                                chat_history=None,
                                llm_model=cfg["llm_model"],
                                temperature=cfg["temperature"],
                            )
                        )
                        docs = st.session_state.get("_last_docs", [])
                        render_retrieved_docs(
                            docs=docs,
                            question=prompt,
                            vectorstore=st.session_state.vectorstore,
                        )

                    st.session_state.adv_messages.append(
                        {"role": "assistant", "content": response,
                        "sources": docs, "question": prompt}
                    )
                except Exception as e:
                    st.error(f"오류: {e}")


# ══════════════════════════════════════════════
# 탭 3: 대화형 RAG
# ══════════════════════════════════════════════
with tab_conv:
    st.subheader("💬 대화형 RAG — 문맥을 기억하는 챗봇")
    st.markdown(
        "**대화 이력(Chat History)** 을 유지하여 후속 질문의 문맥을 이해합니다.  \n"
        "예시: _\"그것의 역사는?\"_ → 이전 대화에서 지시 대상을 자동으로 파악합니다."
    )

    if not require_vectorstore():
        pass
    else:
        render_conversation_stats(st.session_state.conv_history)
        st.divider()

        # 기존 메시지 표시
        for msg in st.session_state.conv_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    render_retrieved_docs(
                        docs=msg["sources"],
                        question=msg.get("question", ""),
                        vectorstore=st.session_state.vectorstore,
                    )

        # 사용 팁 (첫 대화 전)
        if not st.session_state.conv_messages:
            with st.expander("💡 대화형 RAG 사용 팁", expanded=True):
                st.markdown(
                    """
                    1. **첫 질문** — 주제를 명확히 질문합니다.
                       예) *"디지털 전환이란 무엇인가요?"*
                    2. **후속 질문** — 대명사를 사용해도 문맥을 이해합니다.
                       예) *"그것의 주요 목표는 무엇인가요?"*
                    3. **심화 질문** — 이전 답변을 기반으로 확장합니다.
                       예) *"방금 언급한 추진 과제 중 첫 번째를 자세히 설명해주세요."*
                    """
                )

        # 채팅 입력
        if prompt := st.chat_input("이전 대화를 이어서 질문하세요...", key="conv_input"):
            if not require_api_key():
                pass
            else:
                st.chat_message("user").markdown(prompt)
                st.session_state.conv_messages.append({"role": "user", "content": prompt})

                try:
                    retriever = _make_retriever()
                    with st.chat_message("assistant"):
                        response = st.write_stream(
                            stream_response(
                                question=prompt,
                                retriever=retriever,
                                chat_history=st.session_state.conv_history,
                                llm_model=cfg["llm_model"],
                                temperature=cfg["temperature"],
                            )
                        )
                        docs = st.session_state.get("_last_docs", [])
                        render_retrieved_docs(
                            docs=docs,
                            question=prompt,
                            vectorstore=st.session_state.vectorstore,
                        )

                    # 대화 이력 업데이트
                    from langchain_core.messages import AIMessage, HumanMessage

                    st.session_state.conv_history.append(HumanMessage(content=prompt))
                    st.session_state.conv_history.append(AIMessage(content=response))

                    st.session_state.conv_messages.append(
                        {"role": "assistant", "content": response,
                         "sources": docs, "question": prompt}
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"오류: {e}")
