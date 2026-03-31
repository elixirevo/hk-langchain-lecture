import os

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PERSONAS = {
    "해적": "당신은 말끝마다 '야호!'를 외치는 유쾌한 해적입니다. 항상 보물과 바다 이야기를 합니다.",
    "과학자": "당신은 논리적이고 분석적인 과학자입니다. 모든 현상을 과학적으로 설명하며 데이터를 중시합니다.",
    "시인": "당신은 감성적인 시인입니다. 답변을 시적인 언어와 은유로 표현합니다.",
    "요리사": "당신은 열정적인 요리사입니다. 모든 대화를 요리와 음식에 비유합니다.",
    "탐정": "당신은 냉철한 탐정입니다. 모든 것을 논리적으로 분석하고 단서를 찾습니다.",
}


def build_persona_chain(model: ChatOpenAI, persona_name: str):
    """페르소나 기반 대화 체인 생성"""
    system_msg = PERSONAS[persona_name]
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{input}"),
    ])
    return prompt | model | StrOutputParser()


def build_email_chain(model: ChatOpenAI):
    """이메일 작성 체인 생성"""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "당신은 전문 이메일 작성 도우미입니다. "
            "{tone} 톤으로 {language}로 이메일을 작성합니다. "
            "이메일 목적은 '{purpose}'입니다.",
        ),
        (
            "human",
            "발신자: {sender}\n수신자: {recipient}\n제목: {subject}\n"
            "핵심 내용:\n{content}\n\n"
            "위 정보를 바탕으로 완성된 이메일을 작성해주세요. "
            "적절한 인사말, 본문, 마무리 인사를 포함해주세요.",
        ),
    ])
    return prompt | model | StrOutputParser()


def build_translate_style_chains(model: ChatOpenAI):
    """번역 + 스타일 변환 체인 생성 (2단계 파이프라인)"""
    translate_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 전문 번역가입니다. 자연스럽고 정확한 번역을 제공합니다."),
        ("human", "다음 텍스트를 {target_lang}로 번역해주세요:\n{text}"),
    ])
    style_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 글쓰기 전문가입니다."),
        ("human", "다음 텍스트를 {style} 스타일로 다시 작성해주세요:\n{translated}"),
    ])
    translate_chain = translate_prompt | model | StrOutputParser()
    style_chain = style_prompt | model | StrOutputParser()
    return translate_chain, style_chain


def build_quiz_chain(model: ChatOpenAI):
    """Few-shot 기반 퀴즈 생성 체인"""
    examples = [
        {
            "input": "파이썬 프로그래밍 - 3개 단답형 (초급)",
            "output": (
                "**Q1. 파이썬에서 리스트와 튜플의 차이점은?**\n"
                "A: 리스트는 수정 가능(mutable), 튜플은 수정 불가(immutable)\n\n"
                "**Q2. 파이썬의 들여쓰기(indentation)의 역할은?**\n"
                "A: 코드 블록을 구분하는 문법적 요소 (다른 언어의 중괄호 역할)\n\n"
                "**Q3. `len()` 함수의 역할은?**\n"
                "A: 시퀀스(리스트, 문자열 등)의 길이를 반환"
            ),
        },
        {
            "input": "머신러닝 기초 - 2개 O/X 문제 (중급)",
            "output": (
                "**Q1. 과적합(Overfitting)은 훈련 데이터에 너무 잘 맞춰진 모델을 의미한다. (O/X)**\n"
                "A: O - 훈련 데이터에 과하게 최적화되어 새로운 데이터에 성능이 저하됨\n\n"
                "**Q2. 학습률(Learning Rate)이 클수록 항상 빠르게 최적해에 수렴한다. (O/X)**\n"
                "A: X - 학습률이 너무 크면 발산할 수 있음"
            ),
        },
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples, example_prompt=example_prompt
    )
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 교육 전문가입니다. 예시 형식과 동일하게 퀴즈를 만들어주세요."),
        few_shot_prompt,
        ("human", "{topic}"),
    ])
    return final_prompt | model | StrOutputParser()


def init_session_state():
    defaults = {
        "openai_api_key": "",
        "model_name": "gpt-4o-mini",
        "temperature": 0.3,
        "selected_persona": list(PERSONAS.keys())[0],
        "persona_histories": {name: [] for name in PERSONAS},
        "email_history": [],
        "translate_history": [],
        "quiz_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def resolve_api_key():
    if st.session_state.get("openai_api_key"):
        return st.session_state["openai_api_key"]

    env_key = os.getenv("OPENAI_API_KEY", "")
    if env_key:
        return env_key

    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return ""


@st.cache_resource(show_spinner=False)
def get_model(model_name: str, temperature: float, api_key: str):
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )


def render_message_history(history):
    if not history:
        st.info("아직 대화가 없습니다.")
        return

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def build_persona_input(history, user_input: str):
    if not history:
        return user_input

    transcript = []
    for message in history:
        speaker = "사용자" if message["role"] == "user" else "AI"
        transcript.append(f"{speaker}: {message['content']}")

    history_text = "\n".join(transcript)
    return (
        "다음은 지금까지의 대화입니다.\n"
        f"{history_text}\n"
        f"사용자: {user_input}\n\n"
        "위 대화 흐름을 반영해서 마지막 사용자 메시지에 자연스럽게 답변해주세요."
    )


def render_persona_tab(model: ChatOpenAI):
    st.subheader("페르소나 챗봇")
    col_persona, col_action = st.columns([3, 1])

    with col_persona:
        persona_name = st.selectbox(
            "페르소나 선택",
            options=list(PERSONAS.keys()),
            key="selected_persona",
        )
        st.caption(PERSONAS[persona_name])

    with col_action:
        st.write("")
        st.write("")
        if st.button("현재 대화 초기화", use_container_width=True):
            st.session_state["persona_histories"][persona_name] = []
            st.rerun()

    history = st.session_state["persona_histories"][persona_name]
    chat_area = st.container(height=420, border=True)
    with chat_area:
        render_message_history(history)

    user_input = st.chat_input("메시지를 입력하세요", key="persona_chat_input")
    if not user_input:
        return

    chain = build_persona_chain(model, persona_name)
    history.append({"role": "user", "content": user_input})

    with chat_area:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                answer = chain.invoke({"input": build_persona_input(history[:-1], user_input)})
            st.markdown(answer)

    history.append({"role": "assistant", "content": answer})


def render_email_tab(model: ChatOpenAI):
    st.subheader("이메일 작성 도우미")

    with st.form("email_form"):
        tone = st.selectbox("톤", ["정중한", "친근한", "격식 있는", "간결한"])
        language = st.selectbox("언어", ["한국어", "영어", "일본어", "중국어"])
        purpose = st.text_input("이메일 목적", value="미팅 일정 조율")
        sender = st.text_input("발신자", value="김철수")
        recipient = st.text_input("수신자", value="홍길동")
        subject = st.text_input("제목", value="다음 주 미팅 일정 조율")
        content = st.text_area(
            "핵심 내용",
            value="다음 주 중 가능한 미팅 시간을 몇 가지 제안해 주세요.",
            height=160,
        )
        submitted = st.form_submit_button("이메일 생성")

    render_message_history(st.session_state["email_history"])

    if not submitted:
        return

    chain = build_email_chain(model)
    email_text = chain.invoke(
        {
            "tone": tone,
            "language": language,
            "purpose": purpose,
            "sender": sender,
            "recipient": recipient,
            "subject": subject,
            "content": content,
        }
    )

    st.session_state["email_history"].extend(
        [
            {
                "role": "user",
                "content": (
                    "이메일 작성 요청\n\n"
                    f"- 톤: {tone}\n"
                    f"- 언어: {language}\n"
                    f"- 목적: {purpose}\n"
                    f"- 발신자: {sender}\n"
                    f"- 수신자: {recipient}\n"
                    f"- 제목: {subject}\n"
                    f"- 핵심 내용: {content}"
                ),
            },
            {"role": "assistant", "content": email_text},
        ]
    )
    st.rerun()


def render_translate_tab(model: ChatOpenAI):
    st.subheader("번역 + 스타일 변환")

    with st.form("translate_form"):
        text = st.text_area(
            "원문",
            value="LangChain helps developers build applications powered by language models.",
            height=160,
        )
        target_lang = st.selectbox("번역 언어", ["한국어", "영어", "일본어", "중국어"])
        style = st.selectbox("스타일", ["뉴스 기사", "시적 표현", "마케팅 문구", "쉬운 설명"])
        submitted = st.form_submit_button("변환 실행")

    render_message_history(st.session_state["translate_history"])

    if not submitted:
        return

    translate_chain, style_chain = build_translate_style_chains(model)
    translated = translate_chain.invoke({"target_lang": target_lang, "text": text})
    styled = style_chain.invoke({"style": style, "translated": translated})

    st.session_state["translate_history"].extend(
        [
            {
                "role": "user",
                "content": (
                    "텍스트 변환 요청\n\n"
                    f"- 번역 언어: {target_lang}\n"
                    f"- 스타일: {style}\n"
                    f"- 원문: {text}"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"**번역 결과**\n\n{translated}\n\n"
                    f"**스타일 변환 결과**\n\n{styled}"
                ),
            },
        ]
    )
    st.rerun()


def render_quiz_tab(model: ChatOpenAI):
    st.subheader("Few-shot 퀴즈 생성기")

    with st.form("quiz_form"):
        topic = st.text_input(
            "퀴즈 요청",
            value="LangChain 기초 - 3개 단답형 (초급)",
        )
        submitted = st.form_submit_button("퀴즈 생성")

    render_message_history(st.session_state["quiz_history"])

    if not submitted:
        return

    chain = build_quiz_chain(model)
    quiz_text = chain.invoke({"topic": topic})
    st.session_state["quiz_history"].extend(
        [
            {"role": "user", "content": f"퀴즈 생성 요청\n\n{topic}"},
            {"role": "assistant", "content": quiz_text},
        ]
    )
    st.rerun()


def render_app():
    st.set_page_config(
        page_title="LangChain Chatbot Demo",
        page_icon="🤖",
        layout="wide",
    )
    init_session_state()

    with st.sidebar:
        st.title("설정")
        if os.getenv("OPENAI_API_KEY"):
            st.caption("환경 변수의 OPENAI_API_KEY를 사용할 수 있습니다.")

        st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            key="openai_api_key",
        )
        st.selectbox(
            "모델",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            key="model_name",
        )
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            key="temperature",
        )
        st.divider()
        if st.button("모든 기록 초기화", use_container_width=True):
            st.session_state["persona_histories"] = {name: [] for name in PERSONAS}
            st.session_state["email_history"] = []
            st.session_state["translate_history"] = []
            st.session_state["quiz_history"] = []
            st.rerun()

    st.title("LangChain Streamlit 챗봇")
    st.caption("이 파일에 정의된 LangChain 체인을 바로 사용하는 UI입니다.")

    api_key = resolve_api_key()
    if not api_key:
        st.warning("OpenAI API 키를 입력하거나 환경 변수에 OPENAI_API_KEY를 설정하세요.")
        return

    model = get_model(
        st.session_state["model_name"],
        st.session_state["temperature"],
        api_key,
    )

    persona_tab, email_tab, translate_tab, quiz_tab = st.tabs(
        ["페르소나 채팅", "이메일 작성", "번역 + 스타일", "퀴즈 생성"]
    )

    with persona_tab:
        render_persona_tab(model)
    with email_tab:
        render_email_tab(model)
    with translate_tab:
        render_translate_tab(model)
    with quiz_tab:
        render_quiz_tab(model)


if __name__ == "__main__":
    render_app()
