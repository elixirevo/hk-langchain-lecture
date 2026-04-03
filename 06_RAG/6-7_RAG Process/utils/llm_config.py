"""
llm_config.py — LLM 및 임베딩 모델 설정 팩토리

지원 모델:
  LLM     : gpt-4o-mini, gpt-4o, gpt-3.5-turbo
  Embedding: text-embedding-3-small (고정)
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def create_llm(model: str = "gpt-4o-mini", temperature: float = 0.0, streaming: bool = True) -> ChatOpenAI:
    """ChatOpenAI 인스턴스를 반환합니다.

    Args:
        model: OpenAI 모델 ID
        temperature: 응답 무작위성 (0=결정적, 1=창의적)
        streaming: 스트리밍 여부

    Returns:
        ChatOpenAI 인스턴스
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=streaming,
    )


def create_embeddings(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """OpenAIEmbeddings 인스턴스를 반환합니다.

    Args:
        model: OpenAI 임베딩 모델 ID

    Returns:
        OpenAIEmbeddings 인스턴스
    """
    return OpenAIEmbeddings(model=model)


# 임베딩 차원 매핑
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def get_embedding_dim(model: str = "text-embedding-3-small") -> int:
    """모델에 해당하는 임베딩 차원을 반환합니다."""
    return EMBEDDING_DIMENSIONS.get(model, 1536)
