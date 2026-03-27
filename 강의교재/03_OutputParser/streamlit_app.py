"""
Ch03 OutputParser 데모 앱 — UI 레이아웃

LangChain 로직은 utils.py 에서 가져옵니다.
실행 방법: streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import streamlit as st

# ── 페이지 설정 (반드시 최상단) ───────────────────────────────────────────────
st.set_page_config(
    page_title="Ch03 OutputParser 데모",
    page_icon="🔧",
    layout="wide",
)

# ── utils 임포트 ──────────────────────────────────────────────────────────────
from utils import (
    CSV_CODE,
    DATETIME_CODE,
    ENUM_CODE,
    FIXING_CHAIN_CODE,
    FIXING_CODE,
    JSON_CODE,
    PYDANTIC_CODE,
    STR_CODE,
    Sentiment,
    build_csv_chain,
    create_model,
    run_csv_parser,
    run_datetime_parser,
    run_enum_parser,
    run_fixing_chain,
    run_fixing_parser_compare,
    run_json_parser,
    run_pydantic_parser,
    run_str_parser,
)


# ── 세션 스테이트 초기화 ───────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "api_key": "",
        "parse_attempts": 0,
        "parse_successes": 0,
        "total_elapsed": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ── 캐시된 모델 초기화 ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_cached_model(model_name: str, temperature: float, api_key: str):
    return create_model(model_name, temperature, api_key or None)


def get_model():
    return get_cached_model(
        st.session_state["model_name"],
        st.session_state["temperature"],
        st.session_state["api_key"],
    )


# ── 통계 업데이트 헬퍼 ─────────────────────────────────────────────────────────
def _record_success(elapsed: float):
    st.session_state["parse_attempts"] += 1
    st.session_state["parse_successes"] += 1
    st.session_state["total_elapsed"] += elapsed


def _record_failure():
    st.session_state["parse_attempts"] += 1


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정")

    # API 키
    env_key = os.environ.get("OPENAI_API_KEY", "")
    try:
        secrets_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secrets_key = ""
    default_key = env_key or secrets_key or ""

    api_key_input = st.text_input(
        "OpenAI API Key",
        value=default_key,
        type="password",
        help=".env 파일 또는 st.secrets 에서 자동으로 불러옵니다.",
        key="sidebar_api_key",
    )
    st.session_state["api_key"] = api_key_input

    st.divider()

    # 모델 선택
    model_choice = st.selectbox(
        "모델 선택",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        key="sidebar_model",
    )
    st.session_state["model_name"] = model_choice

    # Temperature
    temp_choice = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["temperature"],
        step=0.1,
        key="sidebar_temp",
    )
    st.session_state["temperature"] = temp_choice

    st.divider()

    # 파싱 통계
    st.subheader("파싱 통계")
    attempts = st.session_state["parse_attempts"]
    successes = st.session_state["parse_successes"]
    total_elapsed = st.session_state["total_elapsed"]

    rate = (successes / attempts * 100) if attempts > 0 else 0.0
    avg_elapsed = (total_elapsed / successes) if successes > 0 else 0.0

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("시도", attempts)
    col_m2.metric("성공", successes)
    st.metric("성공률", f"{rate:.0f}%")
    st.metric("평균 응답시간", f"{avg_elapsed:.1f}s")

    if st.button("통계 초기화", key="reset_stats"):
        st.session_state["parse_attempts"] = 0
        st.session_state["parse_successes"] = 0
        st.session_state["total_elapsed"] = 0.0
        st.rerun()

    st.divider()
    st.markdown(
        """
**파서별 출력 타입**

| 파서 | 출력 |
|------|------|
| StrOutputParser | `str` |
| PydanticOutputParser | Pydantic 객체 |
| JsonOutputParser | `dict` |
| CommaSeparatedList | `list` |
| Datetime + Enum | `datetime` / Enum |
| OutputFixingParser | 오류 자동 수정 |
"""
    )


# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.title("Ch03 OutputParser 데모")
st.markdown(
    "LangChain의 다양한 **OutputParser**를 탭별로 직접 체험해보세요.  \n"
    "왼쪽 사이드바에서 모델과 API 키를 설정한 뒤 실행 버튼을 누르세요."
)

# ── 탭 ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. StrOutputParser",
    "2. PydanticOutputParser",
    "3. JsonOutputParser",
    "4. CommaSeparatedList",
    "5. Datetime / Enum",
    "6. OutputFixingParser",
])


# ────────────────────────────────────────────────────────────────────────────
# 탭 1: StrOutputParser
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("StrOutputParser")
    st.markdown(
        "`AIMessage` 객체에서 텍스트만 꺼내 `str`로 반환합니다.  \n"
        "파서가 없으면 어떤 차이가 있는지 직접 비교해보세요."
    )

    col_in, col_opt = st.columns([2, 1])
    with col_in:
        topic = st.text_input(
            "주제를 입력하세요",
            value="양자 컴퓨팅",
            key="str_topic",
        )
    with col_opt:
        show_compare = st.checkbox("파서 유무 비교", value=True, key="str_compare")

    st.code(STR_CODE, language="python")

    if st.button("실행", key="run_str"):
        if not topic.strip():
            st.warning("주제를 입력해주세요.")
        else:
            with st.status("StrOutputParser 체인 실행 중...", expanded=True) as status:
                try:
                    st.write("모델 초기화...")
                    llm = get_model()
                    st.write("체인 실행...")
                    raw_type, parsed, elapsed = run_str_parser(topic, llm)
                    status.update(label="완료!", state="complete")
                    _record_success(elapsed)
                    st.toast(f"파싱 성공! ({elapsed:.1f}s)", icon="✅")
                except Exception as e:
                    status.update(label="오류 발생", state="error")
                    _record_failure()
                    st.error(f"오류: {e}")
                    st.stop()

            if show_compare:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("파서 없이 (AIMessage)")
                    st.caption(f"타입: `{raw_type}`")
                    st.info(
                        "AIMessage 객체에는 content 외에도\n"
                        "response_metadata, id 등이 포함됩니다."
                    )
                with c2:
                    st.subheader("StrOutputParser 사용 (str)")
                    st.caption("타입: `str`")
                    st.success(parsed)
            else:
                st.subheader("결과")
                st.success(parsed)

            st.metric("응답 시간", f"{elapsed:.1f}s")


# ────────────────────────────────────────────────────────────────────────────
# 탭 2: PydanticOutputParser
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("PydanticOutputParser")
    st.markdown(
        "Pydantic 모델을 정의하면 LLM 출력을 **타입 검증이 보장된 객체**로 변환합니다.  \n"
        "아래 도서 리뷰 텍스트를 분석해보세요."
    )

    review_text = st.text_area(
        "도서 리뷰 텍스트",
        value=(
            "'파이썬 마스터하기'는 김파이썬 저자의 역작입니다.\n"
            "초보자부터 고급 개발자까지 모두에게 유용한 내용을 담고 있으며,\n"
            "실용적인 예제가 풍부합니다. 특히 데이터 분석과 웹 개발 파트가 인상적이었습니다.\n"
            "5점 만점에 5점을 주고 싶습니다!"
        ),
        height=110,
        key="pydantic_review",
    )

    st.code(PYDANTIC_CODE, language="python")

    if st.button("분석 실행", key="run_pydantic"):
        if not review_text.strip():
            st.warning("리뷰 텍스트를 입력해주세요.")
        else:
            with st.status("PydanticOutputParser 체인 실행 중...", expanded=True) as status:
                try:
                    st.write("모델 초기화...")
                    llm = get_model()
                    st.write("체인 실행 및 Pydantic 파싱...")
                    result, format_instr, elapsed = run_pydantic_parser(review_text, llm)
                    status.update(label="완료!", state="complete")
                    _record_success(elapsed)
                    st.toast(f"BookReview 객체 생성 성공! ({elapsed:.1f}s)", icon="📚")
                except Exception as e:
                    status.update(label="오류 발생", state="error")
                    _record_failure()
                    st.error(f"오류: {e}")
                    st.stop()

            st.success(f"파싱 성공 — 타입: `BookReview` (Pydantic 객체)  |  {elapsed:.1f}s")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("추출된 도서 정보")
                st.write(f"**제목:** {result.title}")
                st.write(f"**저자:** {result.author}")
                stars = "⭐" * result.rating
                st.write(f"**평점:** {stars} ({result.rating}/5)")
                st.write(f"**장르:** {', '.join(result.genres)}")
                st.write(f"**요약:** {result.summary}")
                st.metric("평점", f"{result.rating} / 5")
            with col2:
                st.subheader("형식 지침 (LLM에 전달)")
                st.code(format_instr, language="text")


# ────────────────────────────────────────────────────────────────────────────
# 탭 3: JsonOutputParser
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("JsonOutputParser")
    st.markdown(
        "LLM 출력을 Python `dict`로 변환합니다.  \n"
        "Pydantic 스키마 없이 프롬프트에서 구조를 직접 명시해도 됩니다."
    )

    recipe_request = st.text_input(
        "요청할 레시피",
        value="간단한 파스타 레시피를 알려주세요.",
        key="json_request",
    )

    st.code(JSON_CODE, language="python")

    if st.button("생성 실행", key="run_json"):
        if not recipe_request.strip():
            st.warning("레시피 요청을 입력해주세요.")
        else:
            with st.status("JsonOutputParser 체인 실행 중...", expanded=True) as status:
                try:
                    st.write("모델 초기화...")
                    llm = get_model()
                    st.write("체인 실행 및 JSON 파싱...")
                    result, elapsed = run_json_parser(recipe_request, llm)
                    status.update(label="완료!", state="complete")
                    _record_success(elapsed)
                    st.toast(f"dict 생성 성공! ({elapsed:.1f}s)", icon="🍝")
                except Exception as e:
                    status.update(label="오류 발생", state="error")
                    _record_failure()
                    st.error(f"오류: {e}")
                    st.stop()

            st.success(f"파싱 성공 — 타입: `{type(result).__name__}` (dict)  |  {elapsed:.1f}s")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("레시피 정보")
                st.write(f"**레시피명:** {result.get('recipe_name', '-')}")
                st.write(f"**재료:** {', '.join(result.get('ingredients', []))}")
                st.write(f"**조리 시간:** {result.get('cooking_time', '-')}분")
                st.write(f"**난이도:** {result.get('difficulty', '-')}")
                st.metric("재료 수", len(result.get("ingredients", [])))
            with col2:
                st.subheader("원본 dict")
                st.json(result)


# ────────────────────────────────────────────────────────────────────────────
# 탭 4: CommaSeparatedListOutputParser
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("CommaSeparatedListOutputParser")
    st.markdown(
        "쉼표로 구분된 LLM 출력을 Python `list`로 변환합니다.  \n"
        "키워드 추출, 추천 목록 생성 등 단순 배열 데이터에 최적입니다."
    )

    scenario = st.selectbox(
        "시나리오 선택",
        ["키워드 추출", "영화 추천", "학습 로드맵"],
        key="csv_scenario",
    )

    invoke_input: dict = {}
    label = ""

    if scenario == "키워드 추출":
        csv_input_text = st.text_area(
            "키워드를 추출할 텍스트",
            value=(
                "인공지능 기술이 빠르게 발전하면서 챗봇 개발이 더욱 쉬워졌습니다. "
                "특히 LangChain 프레임워크는 대화형 AI 애플리케이션 구축을 간소화합니다. "
                "Python을 사용하여 자연어 처리 파이프라인을 만들 수 있으며, "
                "GPT 모델과의 통합도 매우 간단합니다."
            ),
            height=90,
            key="csv_keyword_text",
        )
        invoke_input = {"input": csv_input_text}
        label = "추출된 키워드"

    elif scenario == "영화 추천":
        c1, c2 = st.columns(2)
        genre = c1.text_input("좋아하는 장르", value="SF, 스릴러", key="csv_genre")
        recent = c2.text_input("최근에 본 영화", value="인터스텔라", key="csv_recent")
        invoke_input = {"genre": genre, "recent": recent}
        label = "추천 영화 목록"

    else:
        c1, c2 = st.columns(2)
        subject = c1.text_input("학습 분야", value="데이터 과학", key="csv_subject")
        level = c2.selectbox("수준", ["초급", "중급", "고급"], key="csv_level")
        invoke_input = {"level": level, "subject": subject}
        label = "학습 로드맵"

    st.code(CSV_CODE, language="python")

    if st.button("실행", key="run_csv"):
        with st.status("CommaSeparatedListOutputParser 체인 실행 중...", expanded=True) as status:
            try:
                st.write("모델 초기화...")
                llm = get_model()
                st.write("체인 실행 및 리스트 파싱...")
                items, elapsed = run_csv_parser(scenario, invoke_input, llm)
                status.update(label="완료!", state="complete")
                _record_success(elapsed)
                st.toast(f"list 생성 성공! 항목 {len(items)}개  ({elapsed:.1f}s)", icon="📋")
            except Exception as e:
                status.update(label="오류 발생", state="error")
                _record_failure()
                st.error(f"오류: {e}")
                st.stop()

        st.success(
            f"파싱 성공 — 타입: `{type(items).__name__}` (list)  |  "
            f"항목 수: {len(items)}  |  {elapsed:.1f}s"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(label)
            for i, item in enumerate(items, 1):
                st.write(f"{i}. {item}")
            st.metric("항목 수", len(items))
        with col2:
            st.subheader("원본 list 값")
            st.code(str(items), language="python")


# ────────────────────────────────────────────────────────────────────────────
# 탭 5: Datetime / Enum
# ────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("Datetime / Enum 파싱 (PydanticOutputParser)")
    st.markdown(
        "Pydantic 모델에 `datetime` 또는 `Enum` 필드를 선언하면  \n"
        "LLM 출력을 날짜 객체 또는 열거형 값으로 자동 변환합니다."
    )

    demo_type = st.radio(
        "데모 타입 선택",
        ["Datetime — 역사적 사건 날짜", "Enum — 감정 분류"],
        horizontal=True,
        key="special_demo",
    )

    if demo_type == "Datetime — 역사적 사건 날짜":
        event_q = st.text_input(
            "날짜를 알고 싶은 사건",
            value="ChatGPT가 처음 출시된 날짜",
            key="dt_event",
        )
        st.code(DATETIME_CODE, language="python")

        if st.button("실행", key="run_dt"):
            if not event_q.strip():
                st.warning("사건명을 입력해주세요.")
            else:
                with st.status("datetime 파싱 중...", expanded=True) as status:
                    try:
                        llm = get_model()
                        result, elapsed = run_datetime_parser(event_q, llm)
                        status.update(label="완료!", state="complete")
                        _record_success(elapsed)
                        st.toast(f"datetime 변환 성공! ({elapsed:.1f}s)", icon="📅")
                    except Exception as e:
                        status.update(label="오류 발생", state="error")
                        _record_failure()
                        st.error(f"오류: {e}")
                        st.stop()

                st.success(
                    f"파싱 성공 — event_date 타입: `{type(result.event_date).__name__}`  |  {elapsed:.1f}s"
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("추출 결과")
                    st.write(f"**사건명:** {result.event_name}")
                    st.write(f"**날짜 (포맷):** {result.event_date.strftime('%Y년 %m월 %d일')}")
                    st.write(f"**요일:** {result.event_date.strftime('%A')}")
                    st.write(f"**설명:** {result.description}")
                    st.metric("연도", result.event_date.year)
                with col2:
                    st.subheader("datetime 객체 정보")
                    st.code(
                        f"타입:  {type(result.event_date)}\n"
                        f"값:    {result.event_date}\n"
                        f"year:  {result.event_date.year}\n"
                        f"month: {result.event_date.month}\n"
                        f"day:   {result.event_date.day}",
                        language="text",
                    )

    else:  # Enum
        feedback_text = st.text_area(
            "감정을 분류할 피드백 텍스트",
            value="제품이 정말 훌륭해요! 배송도 빠르고 품질도 최고입니다.",
            height=80,
            key="enum_feedback",
        )
        st.code(ENUM_CODE, language="python")

        if st.button("실행", key="run_enum"):
            if not feedback_text.strip():
                st.warning("피드백 텍스트를 입력해주세요.")
            else:
                with st.status("Enum 파싱 중...", expanded=True) as status:
                    try:
                        llm = get_model()
                        result, elapsed = run_enum_parser(feedback_text, llm)
                        status.update(label="완료!", state="complete")
                        _record_success(elapsed)
                        st.toast(f"Enum 변환 성공! ({elapsed:.1f}s)", icon="😊")
                    except Exception as e:
                        status.update(label="오류 발생", state="error")
                        _record_failure()
                        st.error(f"오류: {e}")
                        st.stop()

                sentiment_val = result.sentiment.value
                emoji_map = {"긍정": "😊", "중립": "😐", "부정": "😞"}
                emoji = emoji_map.get(sentiment_val, "?")

                st.success(
                    f"파싱 성공 — sentiment 타입: `{type(result.sentiment).__name__}` (Enum)  |  {elapsed:.1f}s"
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("감정 분석 결과")
                    st.metric(label="감정", value=f"{emoji} {sentiment_val}")
                with col2:
                    st.subheader("Enum 객체 정보")
                    st.code(
                        f"타입:   {type(result.sentiment)}\n"
                        f"Enum:   {result.sentiment}\n"
                        f".value: {result.sentiment.value}\n"
                        f".name:  {result.sentiment.name}",
                        language="text",
                    )


# ────────────────────────────────────────────────────────────────────────────
# 탭 6: OutputFixingParser
# ────────────────────────────────────────────────────────────────────────────
with tab6:
    st.header("OutputFixingParser")
    st.markdown(
        "파싱이 실패할 경우 LLM이 자동으로 출력을 수정하여 재파싱합니다.  \n"
        "아래에서 **일부러 잘못된 형식의 JSON**을 입력해 자동 수정 과정을 확인해보세요."
    )
    st.info(
        "OutputFixingParser는 파싱 실패 시 LLM 호출을 **1회 추가**합니다.  \n"
        "오류 발생 자체를 줄이고 안전망으로만 활용하는 것이 좋습니다."
    )

    st.subheader("파트 1: 잘못된 JSON 직접 파싱")

    preset_map = {
        "직접 입력": "",
        "작은따옴표 (JSON 규격 위반)": (
            "{'artist': 'BTS', 'album_title': 'MAP OF THE SOUL: 7', "
            "'release_year': 2020, 'genres': ['K-Pop', 'Hip Hop']}"
        ),
        "필드 누락 (stock 없음)": '{"name": "노트북", "price": 1200000}',
        "타입 불일치 (price가 문자열)": '{"name": "마우스", "price": "삼만원", "stock": 50}',
        "따옴표 없는 키": "{name: '키보드', price: 80000, stock: 30}",
    }

    col_preset, col_model_info = st.columns([2, 1])
    with col_preset:
        error_scenario = st.selectbox(
            "오류 시나리오 선택",
            list(preset_map.keys()),
            key="fix_scenario",
        )
    with col_model_info:
        st.subheader("대상 Pydantic 모델")
        st.code(
            "class MusicAlbum(BaseModel):\n"
            "    artist: str\n"
            "    album_title: str\n"
            "    release_year: int\n"
            "    genres: List[str]\n\n"
            "class Product(BaseModel):\n"
            "    name: str\n"
            "    price: int\n"
            "    stock: int",
            language="python",
        )

    if error_scenario == "직접 입력":
        bad_json = st.text_area(
            "잘못된 JSON 직접 입력",
            value="{'artist': 'BTS', 'album_title': 'MAP OF THE SOUL: 7', 'release_year': 2020, 'genres': ['K-Pop', 'Hip Hop']}",
            height=70,
            key="fix_json_direct",
        )
    else:
        bad_json = st.text_area(
            "잘못된 JSON (프리셋)",
            value=preset_map[error_scenario],
            height=70,
            key="fix_json_preset",
        )

    st.code(FIXING_CODE, language="python")

    if st.button("기본 파서 / OutputFixingParser 비교 실행", key="run_fix"):
        if not bad_json.strip():
            st.warning("JSON을 입력해주세요.")
        else:
            with st.status("기본 파서 및 OutputFixingParser 실행 중...", expanded=True) as status:
                try:
                    st.write("모델 초기화...")
                    llm = get_model()
                    st.write("기본 파서 시도...")
                    st.write("OutputFixingParser 시도 (LLM 수정 요청 포함)...")
                    compare = run_fixing_parser_compare(bad_json, llm)
                    status.update(label="완료!", state="complete")
                    if compare["fixed_result"] is not None:
                        _record_success(compare["elapsed"])
                    else:
                        _record_failure()
                    st.toast("비교 완료!", icon="🔧")
                except Exception as e:
                    status.update(label="오류 발생", state="error")
                    _record_failure()
                    st.error(f"오류: {e}")
                    st.stop()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("기본 파서 결과")
                if compare["base_ok"]:
                    st.success("파싱 성공 (오류 없음)")
                else:
                    st.error("파싱 실패")
                    st.code(compare["base_error"] or "", language="text")

            with col2:
                st.subheader("OutputFixingParser 결과")
                fixed = compare["fixed_result"]
                if fixed is not None:
                    st.success(
                        f"자동 수정 후 파싱 성공! (타입: `{compare['model_label']}`)  |  {compare['elapsed']:.1f}s"
                    )
                    if compare["model_label"] == "MusicAlbum":
                        st.write(f"**아티스트:** {fixed.artist}")
                        st.write(f"**앨범:** {fixed.album_title}")
                        st.write(f"**발매 연도:** {fixed.release_year}")
                        st.write(f"**장르:** {', '.join(fixed.genres)}")
                    else:
                        st.write(f"**제품명:** {fixed.name}")
                        st.write(f"**가격:** {fixed.price:,}원")
                        st.write(f"**재고:** {fixed.stock}개")
                else:
                    st.error(f"OutputFixingParser도 실패: {compare['fix_error']}")

    st.divider()
    st.subheader("파트 2: LCEL 체인에 OutputFixingParser 통합하기")
    st.markdown("체인의 파서 자리에 그대로 꽂아서 안정적인 파이프라인을 구성합니다.")

    chain_dish = st.text_input("요리 이름", value="김치찌개", key="fix_dish")
    st.code(FIXING_CHAIN_CODE, language="python")

    if st.button("체인 실행", key="run_fix_chain"):
        if not chain_dish.strip():
            st.warning("요리 이름을 입력해주세요.")
        else:
            with st.status("OutputFixingParser 체인 실행 중...", expanded=True) as status:
                try:
                    st.write("모델 초기화...")
                    llm = get_model()
                    st.write("체인 실행 (파싱 오류 시 자동 수정)...")
                    result, elapsed = run_fixing_chain(chain_dish, llm)
                    status.update(label="완료!", state="complete")
                    _record_success(elapsed)
                    st.toast(f"RecipeInfo 객체 생성 성공! ({elapsed:.1f}s)", icon="🍲")
                except Exception as e:
                    status.update(label="오류 발생", state="error")
                    _record_failure()
                    st.error(f"오류: {e}")
                    st.stop()

            st.success(f"파싱 성공 — `RecipeInfo` 객체 반환  |  {elapsed:.1f}s")

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**요리명:** {result.name}")
                st.write(f"**재료:** {', '.join(result.ingredients)}")
                st.write(f"**조리 시간:** {result.cooking_time_minutes}분")
                st.write(f"**난이도:** {result.difficulty}")
                st.metric("재료 수", len(result.ingredients))
            with col2:
                st.code(
                    f"name:                 {result.name}\n"
                    f"ingredients:          {result.ingredients}\n"
                    f"cooking_time_minutes: {result.cooking_time_minutes}\n"
                    f"difficulty:           {result.difficulty}",
                    language="text",
                )
