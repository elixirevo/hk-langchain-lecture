"""
utils.py — Ch06-5 Retriever 검색 전략 비교 유틸리티
=======================================================
Retriever 생성, 검색 실행, 결과 비교 헬퍼 함수 모음.
Streamlit 의존성 없이 순수 LangChain 로직만 포함.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class SearchResult:
    """단일 검색 방식의 결과를 담는 컨테이너."""
    method: str
    docs: list[Document]
    elapsed: float          # 검색 소요 시간 (초)
    generated_queries: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class RetrieverBundle:
    """빌드된 Retriever와 공유 컴포넌트 묶음."""
    vectorstore: object
    docs: list[Document]
    similarity_retriever: object
    bm25_base_docs: list[Document]   # BM25 재생성용 원본 docs


# ─────────────────────────────────────────────
# Retriever 빌드
# ─────────────────────────────────────────────

def build_retriever_bundle(
    text: str,
    api_key: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    top_k: int = 3,
) -> RetrieverBundle:
    """
    문서 텍스트로 FAISS 벡터스토어와 기본 Retriever를 생성합니다.

    Parameters
    ----------
    text : str
        원시 텍스트
    api_key : str
        OpenAI API 키
    chunk_size : int
        청크 최대 크기
    chunk_overlap : int
        인접 청크 중복 문자 수
    top_k : int
        기본 반환 문서 수

    Returns
    -------
    RetrieverBundle
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

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)

    similarity_ret = vectorstore.as_retriever(search_kwargs={"k": top_k})

    return RetrieverBundle(
        vectorstore=vectorstore,
        docs=docs,
        similarity_retriever=similarity_ret,
        bm25_base_docs=docs,
    )


# ─────────────────────────────────────────────
# 개별 Retriever 생성 헬퍼
# ─────────────────────────────────────────────

def make_similarity_retriever(bundle: RetrieverBundle, k: int):
    return bundle.vectorstore.as_retriever(search_kwargs={"k": k})


def make_mmr_retriever(bundle: RetrieverBundle, k: int, lambda_mult: float):
    return bundle.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(k * 3, 10), "lambda_mult": lambda_mult},
    )


def make_bm25_retriever(bundle: RetrieverBundle, k: int):
    from langchain_community.retrievers import BM25Retriever

    ret = BM25Retriever.from_documents(bundle.bm25_base_docs)
    ret.k = k
    return ret


def make_ensemble_retriever(
    bundle: RetrieverBundle,
    k: int,
    bm25_weight: float,
    vector_weight: float,
):
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever

    bm25 = BM25Retriever.from_documents(bundle.bm25_base_docs)
    bm25.k = k
    faiss_ret = bundle.vectorstore.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25, faiss_ret],
        weights=[bm25_weight, vector_weight],
    )


# ─────────────────────────────────────────────
# 검색 실행
# ─────────────────────────────────────────────

def run_search(method: str, retriever, query: str) -> tuple[list[Document], float]:
    """주어진 retriever로 검색하고 (docs, elapsed) 반환."""
    start = time.perf_counter()
    docs = retriever.invoke(query)
    elapsed = time.perf_counter() - start
    return docs, elapsed


def run_multi_query_search(
    bundle: RetrieverBundle,
    llm,
    query: str,
    k: int,
) -> tuple[list[Document], float, list[str]]:
    """
    MultiQueryRetriever를 실행하고 LLM이 생성한 쿼리 목록도 반환합니다.

    Returns
    -------
    (docs, elapsed, generated_queries)
    """
    import ast
    from langchain.retrievers.multi_query import MultiQueryRetriever

    generated_queries: list[str] = []

    class _QueryCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if "Generated queries:" in msg:
                try:
                    raw = msg.split("Generated queries:")[1].strip()
                    generated_queries.extend(ast.literal_eval(raw))
                except Exception:
                    pass

    logger = logging.getLogger("langchain.retrievers.multi_query")
    logger.setLevel(logging.INFO)
    handler = _QueryCapture()
    logger.addHandler(handler)

    try:
        base_ret = bundle.vectorstore.as_retriever(search_kwargs={"k": k})
        mq_ret = MultiQueryRetriever.from_llm(retriever=base_ret, llm=llm)

        start = time.perf_counter()
        docs = mq_ret.invoke(query)
        elapsed = time.perf_counter() - start
    finally:
        logger.removeHandler(handler)

    return docs, elapsed, generated_queries


# ─────────────────────────────────────────────
# 일괄 검색 실행
# ─────────────────────────────────────────────

def run_all_searches(
    bundle: RetrieverBundle,
    query: str,
    methods: list[str],
    top_k: int,
    lambda_mult: float = 0.6,
    bm25_weight: float = 0.5,
    openai_api_key: str = "",
) -> list[SearchResult]:
    """
    선택된 검색 방식을 모두 실행하고 SearchResult 목록을 반환합니다.
    """
    import os
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    results: list[SearchResult] = []

    for method in methods:
        try:
            if method == "Similarity":
                ret = make_similarity_retriever(bundle, top_k)
                docs, elapsed = run_search(method, ret, query)
                results.append(SearchResult(method=method, docs=docs, elapsed=elapsed))

            elif method == "MMR":
                ret = make_mmr_retriever(bundle, top_k, lambda_mult)
                docs, elapsed = run_search(method, ret, query)
                results.append(SearchResult(method=method, docs=docs, elapsed=elapsed))

            elif method == "BM25":
                ret = make_bm25_retriever(bundle, top_k)
                docs, elapsed = run_search(method, ret, query)
                results.append(SearchResult(method=method, docs=docs, elapsed=elapsed))

            elif method == "Ensemble":
                vector_weight = round(1.0 - bm25_weight, 2)
                ret = make_ensemble_retriever(bundle, top_k, bm25_weight, vector_weight)
                docs, elapsed = run_search(method, ret, query)
                results.append(SearchResult(method=method, docs=docs, elapsed=elapsed))

            elif method == "MultiQuery":
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    temperature=0,
                    model="gpt-4o-mini",
                    api_key=openai_api_key,
                )
                docs, elapsed, gen_q = run_multi_query_search(bundle, llm, query, top_k)
                results.append(
                    SearchResult(
                        method=method,
                        docs=docs,
                        elapsed=elapsed,
                        generated_queries=gen_q,
                    )
                )

        except Exception as exc:
            results.append(
                SearchResult(method=method, docs=[], elapsed=0.0, error=str(exc))
            )

    return results


# ─────────────────────────────────────────────
# 결과 비교 분석
# ─────────────────────────────────────────────

def compute_overlap_matrix(results: list[SearchResult]) -> dict[tuple[str, str], int]:
    """
    방식 쌍별 공통 문서 수를 계산합니다.

    Returns
    -------
    {(method_a, method_b): overlap_count}
    """
    content_sets = {r.method: {d.page_content for d in r.docs} for r in results}
    methods = list(content_sets.keys())
    matrix: dict[tuple[str, str], int] = {}

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            overlap = len(content_sets[m1] & content_sets[m2])
            matrix[(m1, m2)] = overlap

    return matrix
