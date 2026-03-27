"""
Ch04 Memory — 유틸리티 모듈
메모리 클래스, 체인 로직, 모델 설정을 담당합니다.
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationTokenBufferMemory,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# ─────────────────────────────────────────
# 상수
# ─────────────────────────────────────────

SESSION_IDS = ["세션-A", "세션-B", "세션-C"]

MEMORY_OPTIONS: dict[str, dict] = {
    "Buffer (전체 저장)": {
        "icon": "📦",
        "desc": "모든 대화를 그대로 저장합니다. 대화가 길어질수록 토큰 비용이 증가합니다.",
        "legacy": "ConversationBufferMemory",
        "modern": "RunnableWithMessageHistory + ChatMessageHistory",
    },
    "Window (최근 K턴)": {
        "icon": "🪟",
        "desc": "최근 K개 대화 턴만 유지합니다. 슬라이딩 윈도우로 오래된 대화를 자동 제거합니다.",
        "legacy": "ConversationBufferWindowMemory",
        "modern": "last_k_messages() 트리밍",
    },
    "Token (토큰 제한)": {
        "icon": "🔢",
        "desc": "토큰 수 기준으로 오래된 메시지를 제거합니다. API 비용을 일정하게 유지합니다.",
        "legacy": "ConversationTokenBufferMemory",
        "modern": "trim_messages() 토큰 트리밍",
    },
    "Summary (요약)": {
        "icon": "📝",
        "desc": "LLM이 이전 대화를 요약하여 보관합니다. 긴 대화에서도 핵심 맥락을 유지합니다.",
        "legacy": "ConversationSummaryMemory",
        "modern": "요약 체인 미들웨어",
    },
}

TOKEN_LIMIT_APPROX = 500  # 토큰 모드의 근사 제한값


# ─────────────────────────────────────────
# 모델 초기화 (Streamlit 캐시)
# ─────────────────────────────────────────


@st.cache_resource
def get_llm(api_key: str, model_name: str, temperature: float = 0.7) -> ChatOpenAI:
    """채팅용 LLM 인스턴스를 반환합니다 (캐시됨)."""
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key or None)


@st.cache_resource
def get_summary_llm(api_key: str, model_name: str) -> ChatOpenAI:
    """요약용 LLM 인스턴스를 반환합니다 (temperature=0, 캐시됨)."""
    return ChatOpenAI(model=model_name, temperature=0, api_key=api_key or None)


# ─────────────────────────────────────────
# API 키 해결
# ─────────────────────────────────────────


def resolve_api_key() -> str:
    """사이드바 입력 → 환경 변수 순으로 API 키를 반환합니다."""
    sidebar_key = st.session_state.get("openai_api_key", "")
    if sidebar_key:
        return sidebar_key
    return os.getenv("OPENAI_API_KEY", "")


# ─────────────────────────────────────────
# 세션 히스토리 관리
# ─────────────────────────────────────────


def get_session_history(session_id: str) -> ChatMessageHistory:
    """session_id에 해당하는 ChatMessageHistory를 반환합니다 (없으면 생성)."""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


def get_messages(session_id: str) -> List[BaseMessage]:
    """해당 세션의 메시지 리스트를 반환합니다."""
    return get_session_history(session_id).messages


def count_chars(messages: List[BaseMessage]) -> int:
    """메시지 리스트의 전체 문자 수를 반환합니다."""
    return sum(len(m.content) for m in messages)


def clear_session(session_id: str) -> None:
    """특정 세션의 대화 기록과 Legacy 메모리 객체를 모두 제거합니다."""
    if session_id in st.session_state.store:
        del st.session_state.store[session_id]
    legacy_keys = [k for k in st.session_state if k.startswith(f"legacy_{session_id}_")]
    for k in legacy_keys:
        del st.session_state[k]


def clear_all_sessions() -> None:
    """모든 세션 데이터를 초기화합니다."""
    st.session_state.store = {}
    legacy_keys = [k for k in st.session_state if k.startswith("legacy_")]
    for k in legacy_keys:
        del st.session_state[k]


# ─────────────────────────────────────────
# 메시지 트리밍 헬퍼
# ─────────────────────────────────────────


def last_k_messages(messages: List[BaseMessage], k: int) -> List[BaseMessage]:
    """메시지 리스트에서 마지막 k*2개를 반환합니다 (human+AI 쌍 기준)."""
    return messages[-(k * 2):] if messages else []


# ─────────────────────────────────────────
# 대화 요약
# ─────────────────────────────────────────


def build_summary(session_id: str, llm: ChatOpenAI) -> str:
    """LLM을 사용하여 세션 대화를 3~5문장으로 요약합니다."""
    messages = get_messages(session_id)
    if not messages:
        return "(아직 대화 없음)"
    history_text = "\n".join(
        f"{'사용자' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in messages
    )
    prompt_text = (
        "다음 대화를 3~5문장으로 핵심만 요약해 주세요. 한국어로 작성하세요.\n\n"
        f"대화 내용:\n{history_text}\n\n요약:"
    )
    try:
        result = llm.invoke(prompt_text)
        return result.content
    except Exception as e:
        return f"요약 생성 실패: {e}"


# ─────────────────────────────────────────
# 대화 내보내기
# ─────────────────────────────────────────


def export_conversation(session_id: str) -> str:
    """세션 대화를 JSON 문자열로 직렬화하여 반환합니다."""
    messages = get_messages(session_id)
    data = {
        "session_id": session_id,
        "memory_type": st.session_state.get("memory_type", ""),
        "pattern_mode": st.session_state.get("pattern_mode", ""),
        "messages": [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in messages
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────
# LCEL 체인 빌더
# ─────────────────────────────────────────


def build_base_chain(llm: ChatOpenAI):
    """기본 LCEL 체인을 생성합니다 (메모리 없음)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 한국어로 답변해 주세요."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()


# ─────────────────────────────────────────
# 메인: 메모리 유형별 LLM 호출
# ─────────────────────────────────────────


def invoke_with_memory(
    question: str,
    session_id: str,
    memory_type: str,
    pattern_mode: str,
    window_k: int,
    llm: ChatOpenAI,
) -> Tuple[str, str]:
    """
    메모리 유형과 패턴에 따라 LLM을 호출하고 응답과 내부 동작 설명을 반환합니다.

    Returns:
        (answer, operation_detail)
    """
    history = get_session_history(session_id)
    all_messages = history.messages

    if pattern_mode == "Modern (RunnableWithMessageHistory)":
        return _invoke_modern(
            question, session_id, memory_type, window_k, llm, history, all_messages
        )
    else:
        return _invoke_legacy(question, session_id, memory_type, window_k, llm, history)


# ── Modern 패턴 구현 ──────────────────────────────────────


def _invoke_modern(
    question: str,
    session_id: str,
    memory_type: str,
    window_k: int,
    llm: ChatOpenAI,
    history: ChatMessageHistory,
    all_messages: List[BaseMessage],
) -> Tuple[str, str]:
    """Modern 패턴 (RunnableWithMessageHistory)으로 LLM을 호출합니다."""

    if memory_type == "Buffer (전체 저장)":
        op_detail = f"전체 {len(all_messages)}개 메시지를 컨텍스트에 포함"
        chain = build_base_chain(llm)
        wrapped = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        answer = wrapped.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )
        return answer, op_detail

    elif memory_type == "Window (최근 K턴)":
        window = last_k_messages(all_messages, window_k)
        dropped = len(all_messages) - len(window)
        op_detail = (
            f"전체 {len(all_messages)}개 중 최근 {len(window)}개 사용 "
            f"(k={window_k}, {dropped}개 제외)"
        )
        prompt_tpl = ChatPromptTemplate.from_messages([
            ("system", "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 한국어로 답변해 주세요."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        msg = prompt_tpl.invoke({"chat_history": window, "question": question})
        answer = (llm | StrOutputParser()).invoke(msg)
        history.add_user_message(question)
        history.add_ai_message(answer)
        return answer, op_detail

    elif memory_type == "Token (토큰 제한)":
        token_window = all_messages[-10:] if all_messages else []
        approx_tokens = count_chars(token_window) // 2
        op_detail = f"최근 {len(token_window)}개 메시지 사용 (예상 토큰: ~{approx_tokens})"
        prompt_tpl = ChatPromptTemplate.from_messages([
            ("system", "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 한국어로 답변해 주세요."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        msg = prompt_tpl.invoke({"chat_history": token_window, "question": question})
        answer = (llm | StrOutputParser()).invoke(msg)
        history.add_user_message(question)
        history.add_ai_message(answer)
        return answer, op_detail

    elif memory_type == "Summary (요약)":
        api_key = resolve_api_key()
        model_name = st.session_state.get("model_name", "gpt-4o-mini")
        s_llm = get_summary_llm(api_key, model_name)
        if len(all_messages) >= 2:
            current_summary = build_summary(session_id, s_llm)
            op_detail = f"LLM이 {len(all_messages)}개 메시지를 요약하여 컨텍스트 주입"
        else:
            current_summary = ""
            op_detail = "첫 대화 — 요약 없이 직접 응답"
        summary_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 한국어로 답변해 주세요.\n\n"
                + (f"[이전 대화 요약]\n{current_summary}" if current_summary else ""),
            ),
            ("human", "{question}"),
        ])
        answer = (summary_prompt | llm | StrOutputParser()).invoke({"question": question})
        history.add_user_message(question)
        history.add_ai_message(answer)
        return answer, op_detail

    return "알 수 없는 메모리 유형입니다.", ""


# ── Legacy 패턴 구현 ──────────────────────────────────────


def _get_or_create_legacy_memory(
    session_id: str,
    memory_type: str,
    window_k: int,
    llm: ChatOpenAI,
):
    """Legacy 메모리 객체를 세션 상태에서 가져오거나 생성합니다."""
    legacy_key = f"legacy_{session_id}_{memory_type}"
    if legacy_key not in st.session_state:
        api_key = resolve_api_key()
        model_name = st.session_state.get("model_name", "gpt-4o-mini")
        if memory_type == "Buffer (전체 저장)":
            st.session_state[legacy_key] = ConversationBufferMemory(return_messages=True)
        elif memory_type == "Window (최근 K턴)":
            st.session_state[legacy_key] = ConversationBufferWindowMemory(
                k=window_k, return_messages=True
            )
        elif memory_type == "Token (토큰 제한)":
            st.session_state[legacy_key] = ConversationTokenBufferMemory(
                llm=llm, max_token_limit=TOKEN_LIMIT_APPROX, return_messages=True
            )
        elif memory_type == "Summary (요약)":
            st.session_state[legacy_key] = ConversationSummaryMemory(
                llm=get_summary_llm(api_key, model_name), return_messages=True
            )
    return st.session_state[legacy_key]


def _invoke_legacy(
    question: str,
    session_id: str,
    memory_type: str,
    window_k: int,
    llm: ChatOpenAI,
    history: ChatMessageHistory,
) -> Tuple[str, str]:
    """Legacy 패턴 (langchain.memory)으로 LLM을 호출합니다."""
    memory = _get_or_create_legacy_memory(session_id, memory_type, window_k, llm)

    mem_vars = memory.load_memory_variables({})
    legacy_hist = mem_vars.get("history", [])
    n_stored = len(legacy_hist) if isinstance(legacy_hist, list) else 0
    op_detail = f"Legacy {type(memory).__name__} — {n_stored}개 메시지 보관 중"

    legacy_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 한국어로 답변해 주세요."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    conv_chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=legacy_prompt,
        verbose=False,
    )
    answer = conv_chain.predict(input=question)

    # Modern 뷰 동기화 (메모리 상태 패널에서 공통으로 표시)
    history.add_user_message(question)
    history.add_ai_message(answer)
    return answer, op_detail
