"""
Ch04 Memory - 대화 메모리 데모
Streamlit 앱: LangChain 메모리 유형 비교 데모

실행 방법:
    streamlit run notebooks/04_Memory/streamlit_app.py
"""

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from utils import (
    MEMORY_OPTIONS,
    SESSION_IDS,
    TOKEN_LIMIT_APPROX,
    build_summary,
    clear_all_sessions,
    clear_session,
    count_chars,
    export_conversation,
    get_llm,
    get_messages,
    get_summary_llm,
    invoke_with_memory,
    last_k_messages,
    resolve_api_key,
)

load_dotenv()

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="메모리 데모",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────────────
DEFAULTS = {
    "store": {},
    "active_session": "세션-A",
    "memory_type": "Buffer (전체 저장)",
    "pattern_mode": "Modern (RunnableWithMessageHistory)",
    "window_k": 3,
    "openai_api_key": "",
    "model_name": "gpt-4o-mini",
}
for _key, _default in DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ─────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ 설정")
    st.markdown("---")

    # ── API 키 ──
    st.subheader("🔑 OpenAI API 키")
    import os
    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key:
        st.success("환경 변수에서 API 키를 감지했습니다.")
        st.caption("직접 입력하면 환경 변수 키보다 우선 적용됩니다.")
    st.text_input(
        "API Key (선택)",
        type="password",
        placeholder="sk-...",
        key="openai_api_key",
        help=".env 파일에 OPENAI_API_KEY가 있으면 생략 가능합니다.",
    )

    # ── 모델 선택 ──
    st.subheader("🤖 모델")
    st.selectbox(
        "OpenAI 모델",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        key="model_name",
    )

    st.markdown("---")

    # ── 패턴 모드 ──
    st.subheader("🔄 패턴 모드")
    st.radio(
        "Legacy vs Modern",
        options=[
            "Modern (RunnableWithMessageHistory)",
            "Legacy (langchain.memory)",
        ],
        key="pattern_mode",
        help=(
            "Modern: LangChain 1.0.x 공식 권장 방식\n"
            "Legacy: 구식 ConversationBufferMemory 등"
        ),
    )

    st.markdown("---")

    # ── 메모리 유형 ──
    st.subheader("🧠 메모리 유형")
    st.selectbox(
        "메모리 유형 선택",
        options=list(MEMORY_OPTIONS.keys()),
        key="memory_type",
    )
    selected_mem = MEMORY_OPTIONS[st.session_state.memory_type]
    st.info(f"{selected_mem['icon']} {selected_mem['desc']}")
    with st.expander("클래스 매핑 보기"):
        st.markdown(
            f"**Legacy**: `{selected_mem['legacy']}`\n\n"
            f"**Modern**: `{selected_mem['modern']}`"
        )

    # ── Window k 슬라이더 ──
    if st.session_state.memory_type == "Window (최근 K턴)":
        st.markdown("---")
        st.subheader("📐 윈도우 크기")
        st.slider(
            "기억할 대화 턴 수 (k)",
            min_value=1,
            max_value=10,
            value=st.session_state.window_k,
            step=1,
            key="window_k",
            help="k=3이면 최근 3턴(사용자+AI 쌍 3개)만 LLM에 전달됩니다.",
        )
        st.caption(
            f"현재 설정: 최근 **{st.session_state.window_k}턴** "
            f"= {st.session_state.window_k * 2}개 메시지"
        )

    st.markdown("---")

    # ── 세션 관리 ──
    st.subheader("🗂️ 세션 관리")
    st.selectbox(
        "활성 세션",
        options=SESSION_IDS,
        key="active_session",
        help="서로 다른 세션은 완전히 독립된 대화 공간입니다.",
    )
    for sid in SESSION_IDS:
        n_msgs = len(get_messages(sid))
        turns = n_msgs // 2
        marker = " ◀ 현재" if sid == st.session_state.active_session else ""
        dot = "▶" if sid == st.session_state.active_session else "·"
        st.caption(f"{dot} **{sid}**{marker}: {turns}턴")

    col_clr, col_all = st.columns(2)
    with col_clr:
        if st.button("현재 초기화", use_container_width=True):
            clear_session(st.session_state.active_session)
            st.success(f"{st.session_state.active_session} 초기화!")
            st.rerun()
    with col_all:
        if st.button("전체 초기화", use_container_width=True, type="secondary"):
            clear_all_sessions()
            st.success("전체 초기화!")
            st.rerun()

    st.markdown("---")

    # ── 대화 내보내기 ──
    st.subheader("💾 대화 내보내기")
    msgs_now = get_messages(st.session_state.active_session)
    if msgs_now:
        export_data = export_conversation(st.session_state.active_session)
        st.download_button(
            label=f"{st.session_state.active_session} 내보내기 (JSON)",
            data=export_data,
            file_name=f"conversation_{st.session_state.active_session}.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.caption("대화가 없습니다.")

    st.markdown("---")
    st.caption(
        "**빠른 가이드**\n"
        "1. API 키 입력 (또는 .env 파일)\n"
        "2. 패턴·메모리·세션 선택\n"
        "3. 채팅 후 오른쪽 패널에서 메모리 상태 확인"
    )


# ─────────────────────────────────────────
# 메인 영역 헤더
# ─────────────────────────────────────────

st.title("🧠 Ch04 대화 메모리 데모")

hc1, hc2, hc3, hc4 = st.columns(4)
with hc1:
    st.markdown(f"**패턴**: `{st.session_state.pattern_mode.split('(')[0].strip()}`")
with hc2:
    icon = MEMORY_OPTIONS[st.session_state.memory_type]["icon"]
    st.markdown(f"**메모리**: {icon} `{st.session_state.memory_type}`")
with hc3:
    st.markdown(f"**세션**: `{st.session_state.active_session}`")
with hc4:
    active_turns = len(get_messages(st.session_state.active_session)) // 2
    st.markdown(f"**대화**: `{active_turns}턴`")

st.markdown("---")

# ─────────────────────────────────────────
# 레이아웃: 채팅(3) | 메모리 상태(2)
# ─────────────────────────────────────────

col_chat, col_memory = st.columns([3, 2])

# ─────────────────────────────────────────
# 왼쪽: 채팅 인터페이스
# ─────────────────────────────────────────

with col_chat:
    st.subheader(f"💬 채팅 — {st.session_state.active_session}")

    messages = get_messages(st.session_state.active_session)

    chat_area = st.container(height=480)
    with chat_area:
        if not messages:
            st.info(
                "아직 대화가 없습니다. 아래 입력창에 메시지를 입력하세요.\n\n"
                "**예시 질문**: '안녕하세요, 제 이름은 김철수입니다.' → '제 이름이 뭔가요?'"
            )
        else:
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    with st.chat_message("user"):
                        st.write(msg.content)
                elif isinstance(msg, AIMessage):
                    with st.chat_message("assistant"):
                        st.write(msg.content)

    api_key = resolve_api_key()
    if not api_key:
        st.warning(
            "OpenAI API 키를 사이드바에 입력하거나 .env 파일에 OPENAI_API_KEY를 설정해 주세요."
        )
    else:
        user_input = st.chat_input("메시지를 입력하세요...", key="chat_input")

        if user_input:
            llm = get_llm(api_key, st.session_state.model_name)
            with st.status("메모리 처리 중...", expanded=True) as status:
                st.write(f"세션 `{st.session_state.active_session}` 히스토리 로드 중...")
                cur_msgs = get_messages(st.session_state.active_session)
                st.write(f"현재 저장된 메시지: {len(cur_msgs)}개")
                st.write(f"메모리 유형 `{st.session_state.memory_type}` 적용 중...")
                if st.session_state.memory_type == "Window (최근 K턴)":
                    st.write(
                        f"윈도우 크기 k={st.session_state.window_k} 적용 "
                        f"(최근 {st.session_state.window_k * 2}개 메시지 사용)"
                    )
                elif st.session_state.memory_type == "Summary (요약)":
                    st.write("LLM으로 이전 대화 요약 생성 중...")
                elif st.session_state.memory_type == "Token (토큰 제한)":
                    st.write("토큰 수 기준으로 메시지 트리밍 중...")
                st.write(f"LLM(`{st.session_state.model_name}`) 호출 중...")

                try:
                    answer, op_detail = invoke_with_memory(
                        question=user_input,
                        session_id=st.session_state.active_session,
                        memory_type=st.session_state.memory_type,
                        pattern_mode=st.session_state.pattern_mode,
                        window_k=st.session_state.window_k,
                        llm=llm,
                    )
                    st.write(f"완료: {op_detail}")
                    status.update(label="완료!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label=f"오류: {e}", state="error", expanded=True)

            st.rerun()


# ─────────────────────────────────────────
# 오른쪽: 메모리 상태 시각화
# ─────────────────────────────────────────

with col_memory:
    st.subheader("📊 메모리 상태")

    messages = get_messages(st.session_state.active_session)
    total_msgs = len(messages)
    total_turns = total_msgs // 2
    total_chars = count_chars(messages)
    approx_tokens = total_chars // 2

    # 메트릭
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("대화 턴", f"{total_turns}턴")
    with m2:
        st.metric("총 메시지", f"{total_msgs}개")
    with m3:
        st.metric("예상 토큰", f"~{approx_tokens:,}")

    st.markdown("---")

    mem_type = st.session_state.memory_type
    window_k = st.session_state.window_k

    # ── Buffer ──
    if mem_type == "Buffer (전체 저장)":
        with st.expander("📦 버퍼 전체 내용", expanded=True):
            if messages:
                for i, msg in enumerate(messages):
                    role_icon = "👤" if isinstance(msg, HumanMessage) else "🤖"
                    role_name = "사용자" if isinstance(msg, HumanMessage) else "AI"
                    st.markdown(f"**{role_icon} {role_name}** `#{i + 1}`")
                    preview = msg.content[:200]
                    st.text(preview + ("..." if len(msg.content) > 200 else ""))
            else:
                st.write("저장된 내용 없음")

        if total_msgs >= 2:
            with st.expander("📈 토큰 누적 추이", expanded=False):
                token_history = []
                running_chars = 0
                for msg in messages:
                    running_chars += len(msg.content)
                    if isinstance(msg, AIMessage):
                        token_history.append(running_chars // 2)
                if token_history:
                    df_tok = pd.DataFrame(
                        {
                            "턴": list(range(1, len(token_history) + 1)),
                            "누적 토큰": token_history,
                        }
                    ).set_index("턴")
                    st.line_chart(df_tok)
                    st.caption("대화가 길어질수록 토큰 비용이 선형적으로 증가합니다.")

    # ── Window ──
    elif mem_type == "Window (최근 K턴)":
        windowed = last_k_messages(messages, window_k)
        dropped_count = total_msgs - len(windowed)

        st.markdown("**윈도우 범위 시각화**")
        if total_msgs > 0:
            ratio = len(windowed) / total_msgs
            st.progress(
                ratio,
                text=f"전체 중 {len(windowed)}/{total_msgs}개 사용 ({int(ratio * 100)}%)",
            )

        with st.expander(f"🪟 윈도우 내 메시지 ({len(windowed)}개)", expanded=True):
            if dropped_count > 0:
                st.warning(f"⚠️ {dropped_count}개 메시지가 LLM에 전달되지 않음")
            if windowed:
                st.success(f"✅ LLM 컨텍스트: {len(windowed)}개 메시지")
                for msg in windowed:
                    role_icon = "👤" if isinstance(msg, HumanMessage) else "🤖"
                    role_name = "사용자" if isinstance(msg, HumanMessage) else "AI"
                    st.markdown(f"**{role_icon} {role_name}**")
                    st.text(msg.content[:200] + ("..." if len(msg.content) > 200 else ""))
            else:
                st.write("저장된 내용 없음")

        if dropped_count > 0:
            with st.expander(f"🗑️ 제외된 메시지 ({dropped_count}개)", expanded=False):
                excluded = messages[:dropped_count]
                for msg in excluded:
                    role = "사용자" if isinstance(msg, HumanMessage) else "AI"
                    st.markdown(f"~~**{role}**: {msg.content[:80]}...~~")

    # ── Token ──
    elif mem_type == "Token (토큰 제한)":
        token_window_msgs = messages[-10:] if messages else []
        window_tokens = count_chars(token_window_msgs) // 2
        usage_ratio = min(window_tokens / TOKEN_LIMIT_APPROX, 1.0)

        with st.expander("🔢 토큰 사용 현황", expanded=True):
            st.progress(
                usage_ratio,
                text=(
                    f"토큰 사용량: ~{window_tokens} / {TOKEN_LIMIT_APPROX} "
                    f"({int(usage_ratio * 100)}%)"
                ),
            )
            if usage_ratio >= 0.9:
                st.error("토큰 한도에 근접! 오래된 메시지가 제거됩니다.")
            elif usage_ratio >= 0.7:
                st.warning("토큰 사용량이 높습니다.")
            else:
                st.success("토큰 사용량이 적정 수준입니다.")

            st.markdown(f"**현재 컨텍스트**: 최근 {len(token_window_msgs)}개 메시지")
            for msg in token_window_msgs:
                role_icon = "👤" if isinstance(msg, HumanMessage) else "🤖"
                role_name = "사용자" if isinstance(msg, HumanMessage) else "AI"
                n_tok = len(msg.content) // 2
                st.markdown(
                    f"**{role_icon} {role_name}** (~{n_tok} tok): "
                    f"{msg.content[:100]}..."
                )

    # ── Summary ──
    elif mem_type == "Summary (요약)":
        with st.expander("📝 대화 요약", expanded=True):
            if messages:
                api_key_now = resolve_api_key()
                if api_key_now:
                    s_llm = get_summary_llm(api_key_now, st.session_state.model_name)
                    with st.spinner("요약 생성 중..."):
                        summary = build_summary(st.session_state.active_session, s_llm)
                    st.markdown("**현재 대화 요약:**")
                    st.info(summary)
                    summary_chars = len(summary)
                    compression = max(
                        0, 100 - int(summary_chars / max(total_chars, 1) * 100)
                    )
                    st.caption(
                        f"원본 ~{approx_tokens:,} tok → "
                        f"요약 ~{summary_chars // 2:,} tok "
                        f"(약 {compression}% 압축)"
                    )
                else:
                    st.warning("요약 생성에는 API 키가 필요합니다.")
            else:
                st.write("저장된 내용 없음")

    st.markdown("---")

    # ── 세션 비교 ──
    with st.expander("🔀 세션별 대화 현황", expanded=False):
        rows = []
        for sid in SESSION_IDS:
            msgs = get_messages(sid)
            rows.append(
                {
                    "세션": f"{'▶ ' if sid == st.session_state.active_session else ''}{sid}",
                    "대화 턴": len(msgs) // 2,
                    "메시지 수": len(msgs),
                    "예상 토큰": count_chars(msgs) // 2,
                }
            )
        st.dataframe(
            pd.DataFrame(rows).set_index("세션"), use_container_width=True
        )
        st.caption("session_id가 다르면 메모리를 공유하지 않습니다.")

    # ── Legacy vs Modern 비교 ──
    with st.expander("📚 Legacy vs Modern 비교표", expanded=False):
        comp_data = {
            "기능": ["메모리 저장", "세션 분리", "백엔드 교체", "LangChain 버전"],
            "Legacy": [
                "자동 (체인 내부)",
                "별도 구현 필요",
                "코드 전체 수정",
                "0.x (deprecated)",
            ],
            "Modern": [
                "invoke() 시 자동",
                "session_id만 변경",
                "함수 한 곳만 수정",
                "1.0.x (공식 권장)",
            ],
        }
        st.dataframe(
            pd.DataFrame(comp_data).set_index("기능"), use_container_width=True
        )
        st.markdown("""
**클래스 매핑:**

| Legacy | Modern |
|--------|--------|
| `ConversationBufferMemory` | `ChatMessageHistory` + `RunnableWithMessageHistory` |
| `ConversationBufferWindowMemory` | `last_k_messages()` 트리밍 |
| `ConversationTokenBufferMemory` | `trim_messages()` 토큰 트리밍 |
| `ConversationSummaryMemory` | 요약 체인 미들웨어 |
        """)
