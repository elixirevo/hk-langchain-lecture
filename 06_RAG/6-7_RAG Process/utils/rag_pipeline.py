"""
rag_pipeline.py — RAG 파이프라인 핵심 로직

담당:
  - PDF 로드 → 텍스트 분할 → 임베딩 → FAISS 저장 (build_vectorstore)
  - 검색기 생성: similarity / mmr / ensemble  (get_retriever)
  - LLM 스트리밍 응답 생성 (stream_response)
  - 문서 포매팅 유틸 (format_docs)
"""

from __future__ import annotations

import os
import tempfile
from typing import Generator, Iterator

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from utils.llm_config import create_embeddings, create_llm, get_embedding_dim


# ──────────────────────────────────────────────
# 1~4단계: 문서 처리 → 벡터 DB
# ──────────────────────────────────────────────

def build_vectorstore(
    pdf_bytes: bytes,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    embedding_model: str = "text-embedding-3-small",
) -> tuple[FAISS, list[Document], dict]:
    """PDF 바이트를 받아 FAISS 벡터스토어를 구성합니다.

    Args:
        pdf_bytes: PDF 파일 바이트
        filename: 원본 파일명 (표시용)
        chunk_size: 청크 최대 문자 수
        chunk_overlap: 청크 간 겹치는 문자 수
        embedding_model: OpenAI 임베딩 모델 ID

    Returns:
        (vectorstore, split_docs, stats_dict) 튜플
        stats_dict 키: filename, pages, chunks, avg_chunk, embed_dim
    """
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with st.status("📦 문서를 처리하고 있습니다...", expanded=True) as status:
        # 1/4 문서 로드
        status.update(label="1/4 문서 로딩 중...")
        st.write(f"**{filename}** 읽는 중...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            loader = PyMuPDFLoader(tmp_path)
            raw_docs = loader.load()
        finally:
            os.unlink(tmp_path)
        st.write(f"✅ {len(raw_docs)} 페이지 로드 완료")

        # 2/4 텍스트 분할
        status.update(label="2/4 텍스트 분할 중...")
        st.write(f"chunk_size={chunk_size}, overlap={chunk_overlap} 으로 분할 중...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        split_docs = splitter.split_documents(raw_docs)
        avg_len = int(
            sum(len(d.page_content) for d in split_docs) / max(len(split_docs), 1)
        )
        st.write(f"✅ {len(split_docs)} 청크 생성 (평균 {avg_len}자)")

        # 3/4 임베딩 생성
        status.update(label="3/4 임베딩 생성 중...")
        st.write(f"`{embedding_model}` 으로 임베딩 중...")
        embeddings = create_embeddings(embedding_model)

        # 4/4 벡터 DB 저장
        status.update(label="4/4 벡터 DB 저장 중...")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        embed_dim = get_embedding_dim(embedding_model)
        st.write(f"✅ FAISS 벡터 DB 저장 완료 (차원: {embed_dim})")

        status.update(label="✅ 문서 처리 완료!", state="complete", expanded=False)

    stats = {
        "filename": filename,
        "pages": len(raw_docs),
        "chunks": len(split_docs),
        "avg_chunk": avg_len,
        "embed_dim": embed_dim,
    }
    return vectorstore, split_docs, stats


# ──────────────────────────────────────────────
# 5단계: 검색기 생성
# ──────────────────────────────────────────────

def get_retriever(
    vectorstore: FAISS,
    split_docs: list[Document],
    search_type: str = "similarity",
    k: int = 4,
    lambda_mult: float = 0.5,
    bm25_weight: float = 0.4,
):
    """검색 방식에 따라 적절한 Retriever를 반환합니다.

    Args:
        vectorstore: FAISS 벡터스토어
        split_docs: 청크 문서 목록 (ensemble 시 필요)
        search_type: 'similarity' | 'mmr' | 'ensemble'
        k: 반환 문서 수
        lambda_mult: MMR 다양성 조절 (0~1)
        bm25_weight: Ensemble에서 BM25 가중치

    Returns:
        LangChain Retriever 인스턴스
    """
    if search_type == "similarity":
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    elif search_type == "mmr":
        fetch_k = max(k * 5, 20)
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )

    elif search_type == "ensemble":
        try:
            from langchain.retrievers import EnsembleRetriever
            from langchain_community.retrievers import BM25Retriever

            bm25 = BM25Retriever.from_documents(split_docs)
            bm25.k = k
            faiss_r = vectorstore.as_retriever(search_kwargs={"k": k})
            faiss_w = round(1.0 - bm25_weight, 2)
            return EnsembleRetriever(
                retrievers=[bm25, faiss_r],
                weights=[bm25_weight, faiss_w],
            )
        except ImportError:
            st.warning("`rank-bm25` 패키지가 없어 similarity 검색으로 대체합니다.")
            return vectorstore.as_retriever(search_kwargs={"k": k})

    # fallback
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ──────────────────────────────────────────────
# 6~8단계: 프롬프트 + LLM + 스트리밍
# ──────────────────────────────────────────────

def format_docs(docs: list[Document]) -> str:
    """Document 목록을 하나의 문자열로 합칩니다."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def stream_response(
    question: str,
    retriever,
    chat_history: list[BaseMessage] | None = None,
    llm_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Iterator[str]:
    """RAG 체인을 실행하고 스트리밍 응답을 생성합니다.

    Args:
        question: 사용자 질문
        retriever: LangChain Retriever 인스턴스
        chat_history: 이전 대화 이력 (None이면 단일 QA 모드)
        llm_model: OpenAI 모델 ID
        temperature: LLM temperature

    Yields:
        응답 텍스트 청크

    Side-effect:
        st.session_state._last_docs 에 검색된 Document 목록 저장
    """
    retrieved = retriever.invoke(question)
    st.session_state._last_docs = retrieved
    context = format_docs(retrieved)

    if chat_history is not None:
        # 대화형 RAG: 이전 대화 이력 포함
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "당신은 문서 기반 질의응답 AI입니다.\n"
                "주어진 문맥(Context)과 대화 이력을 참고하여 한국어로 답변하세요.\n"
                "문맥에 없는 정보는 '주어진 문서에서 해당 정보를 찾을 수 없습니다.'라고 답하세요.\n\n"
                "#Context:\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        input_data = {
            "context": context,
            "question": question,
            "chat_history": chat_history,
        }
    else:
        # 기본 RAG: 단일 QA
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "당신은 문서 기반 질의응답 AI입니다.\n"
                "주어진 문맥(Context)을 바탕으로 한국어로 답변하세요.\n"
                "문맥에 없는 정보는 '주어진 문서에서 해당 정보를 찾을 수 없습니다.'라고 답하세요.\n\n"
                "#Context:\n{context}",
            ),
            ("human", "{question}"),
        ])
        input_data = {
            "context": context,
            "question": question,
        }

    llm = create_llm(model=llm_model, temperature=temperature, streaming=True)
    chain = prompt | llm | StrOutputParser()

    for chunk in chain.stream(input_data):
        yield chunk


# ──────────────────────────────────────────────
# 유사도 점수 조회
# ──────────────────────────────────────────────

def get_relevance_scores(
    question: str,
    docs: list[Document],
    vectorstore: FAISS,
) -> list[float]:
    """각 문서의 유사도 점수(0~1)를 반환합니다.

    FAISS L2 거리를 근사 코사인 유사도로 변환합니다.
    """
    try:
        results_with_scores = vectorstore.similarity_search_with_score(question, k=len(docs))
        score_map = {r[0].page_content[:80]: r[1] for r in results_with_scores}
        scores: list[float] = []
        for doc in docs:
            raw = score_map.get(doc.page_content[:80], 1.0)
            sim = max(0.0, 1.0 - raw / 2.0)
            scores.append(round(sim, 3))
        return scores
    except Exception:
        return [0.0] * len(docs)
