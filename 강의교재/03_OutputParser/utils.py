"""
Ch03 OutputParser 데모 — LangChain 로직 모듈

모든 파서 정의, 모델 초기화, 체인 실행 로직을 담당합니다.
streamlit_app.py 는 UI 레이아웃만 담당합니다.
"""

from __future__ import annotations

import time
from datetime import datetime as dt_type
from enum import Enum as PyEnum
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


# ── Pydantic 모델 정의 ────────────────────────────────────────────────────────

class BookReview(BaseModel):
    """도서 리뷰에서 추출한 정보"""
    title: str = Field(description="도서 제목")
    author: str = Field(description="저자명")
    rating: int = Field(description="평점 (1-5점)", ge=1, le=5)
    genres: List[str] = Field(description="장르 목록 (최대 3개)")
    summary: str = Field(description="한 문장 요약")


class HistoricalEvent(BaseModel):
    """역사적 사건 정보"""
    event_name: str = Field(description="사건 이름")
    event_date: dt_type = Field(description="날짜 (YYYY-MM-DD 형식)")
    description: str = Field(description="한 문장 설명")


class Sentiment(PyEnum):
    POSITIVE = "긍정"
    NEUTRAL = "중립"
    NEGATIVE = "부정"


class SentimentAnalysis(BaseModel):
    """감정 분석 결과"""
    sentiment: Sentiment = Field(
        description="감정 분류 (긍정, 중립, 부정 중 하나)"
    )


class MusicAlbum(BaseModel):
    """음악 앨범 정보"""
    artist: str = Field(description="아티스트 이름")
    album_title: str = Field(description="앨범 제목")
    release_year: int = Field(description="발매 연도")
    genres: List[str] = Field(description="장르 목록")


class Product(BaseModel):
    """제품 정보"""
    name: str = Field(description="제품명")
    price: int = Field(description="가격 (원)")
    stock: int = Field(description="재고 수량")


class RecipeInfo(BaseModel):
    """요리 레시피 정보"""
    name: str = Field(description="요리 이름")
    ingredients: List[str] = Field(description="주요 재료 (3-5개)")
    cooking_time_minutes: int = Field(description="조리 시간 (분)")
    difficulty: str = Field(description="난이도 (쉬움/보통/어려움)")


# ── 모델 초기화 (Streamlit cache 사용 시 app에서 래핑) ───────────────────────

def create_model(model_name: str, temperature: float, api_key: str | None = None) -> ChatOpenAI:
    """ChatOpenAI 인스턴스를 생성합니다."""
    kwargs: Dict[str, Any] = {"model": model_name, "temperature": temperature}
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)


# ── 탭 1: StrOutputParser ─────────────────────────────────────────────────────

def run_str_parser(
    topic: str,
    llm: ChatOpenAI,
) -> Tuple[str, str, float]:
    """
    StrOutputParser 체인을 실행합니다.

    Returns:
        (raw_type, parsed_result, elapsed_seconds)
    """
    prompt = PromptTemplate.from_template("{topic}에 대해 한 문장으로 설명해주세요.")

    chain_raw = prompt | llm
    chain_parsed = prompt | llm | StrOutputParser()

    t0 = time.perf_counter()
    raw = chain_raw.invoke({"topic": topic})
    parsed = chain_parsed.invoke({"topic": topic})
    elapsed = time.perf_counter() - t0

    return type(raw).__name__, parsed, elapsed


STR_CODE = '''\
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = PromptTemplate.from_template(
    "{topic}에 대해 한 문장으로 설명해주세요."
)
output_parser = StrOutputParser()

chain = prompt | model | output_parser
result = chain.invoke({"topic": topic})
# result 타입: str  (파서 없으면 AIMessage)
'''


# ── 탭 2: PydanticOutputParser ────────────────────────────────────────────────

def run_pydantic_parser(
    review_text: str,
    llm: ChatOpenAI,
) -> Tuple[BookReview, str, float]:
    """
    PydanticOutputParser 체인을 실행합니다.

    Returns:
        (result_object, format_instructions, elapsed_seconds)
    """
    parser = PydanticOutputParser(pydantic_object=BookReview)
    prompt = PromptTemplate.from_template(
        "다음 텍스트에서 도서 정보를 추출하세요.\n\n{text}\n\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    t0 = time.perf_counter()
    result = chain.invoke({"text": review_text})
    elapsed = time.perf_counter() - t0

    return result, parser.get_format_instructions(), elapsed


PYDANTIC_CODE = '''\
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class BookReview(BaseModel):
    title: str = Field(description="도서 제목")
    author: str = Field(description="저자명")
    rating: int = Field(description="평점 (1-5점)", ge=1, le=5)
    genres: List[str] = Field(description="장르 목록 (최대 3개)")
    summary: str = Field(description="한 문장 요약")

parser = PydanticOutputParser(pydantic_object=BookReview)
prompt = PromptTemplate.from_template(
    "다음 텍스트에서 도서 정보를 추출하세요.\\n\\n{text}\\n\\n{format_instructions}"
)
prompt = prompt.partial(
    format_instructions=parser.get_format_instructions()
)

chain = prompt | model | parser
result = chain.invoke({"text": review_text})
# result 타입: BookReview (Pydantic 객체)
# 접근: result.title, result.author, result.rating ...
'''


# ── 탭 3: JsonOutputParser ────────────────────────────────────────────────────

def run_json_parser(
    request: str,
    llm: ChatOpenAI,
) -> Tuple[Dict[str, Any], float]:
    """
    JsonOutputParser 체인을 실행합니다.

    Returns:
        (result_dict, elapsed_seconds)
    """
    json_parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 레시피 전문가입니다."),
        (
            "user",
            "{request}\n\n응답은 다음 JSON 형식으로 작성하세요:\n"
            "- recipe_name: 레시피 이름\n"
            "- ingredients: 재료 목록 (배열)\n"
            "- cooking_time: 조리 시간 (분)\n"
            "- difficulty: 난이도 (쉬움/보통/어려움)\n\n"
            "{format_instructions}",
        ),
    ]).partial(format_instructions=json_parser.get_format_instructions())

    chain = prompt | llm | json_parser

    t0 = time.perf_counter()
    result = chain.invoke({"request": request})
    elapsed = time.perf_counter() - t0

    return result, elapsed


JSON_CODE = '''\
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

json_parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 레시피 전문가입니다."),
    ("user", """{request}

응답은 다음 JSON 형식으로 작성하세요:
- recipe_name: 레시피 이름
- ingredients: 재료 목록 (배열)
- cooking_time: 조리 시간 (분)
- difficulty: 난이도 (쉬움/보통/어려움)

{format_instructions}
"""),
])
prompt = prompt.partial(
    format_instructions=json_parser.get_format_instructions()
)

chain = prompt | model | json_parser
result = chain.invoke({"request": request})
# result 타입: dict
# 접근: result["recipe_name"], result["ingredients"] ...
'''


# ── 탭 4: CommaSeparatedListOutputParser ─────────────────────────────────────

def build_csv_chain(scenario: str, llm: ChatOpenAI):
    """시나리오에 맞는 CommaSeparatedListOutputParser 체인을 반환합니다."""
    output_parser = CommaSeparatedListOutputParser()
    fi = output_parser.get_format_instructions()

    if scenario == "키워드 추출":
        prompt = PromptTemplate(
            template=(
                "다음 텍스트에서 SEO에 효과적인 키워드 6개를 추출하세요.\n\n"
                "텍스트:\n{input}\n\n{format_instructions}"
            ),
            input_variables=["input"],
            partial_variables={"format_instructions": fi},
        )
    elif scenario == "영화 추천":
        prompt = PromptTemplate(
            template=(
                "사용자의 취향에 맞는 영화 제목 5개를 추천하세요.\n\n"
                "장르: {genre}\n최근 본 영화: {recent}\n\n{format_instructions}"
            ),
            input_variables=["genre", "recent"],
            partial_variables={"format_instructions": fi},
        )
    else:  # 학습 로드맵
        prompt = PromptTemplate(
            template=(
                "{level} 수준에서 {subject}를 배우기 위한 핵심 학습 주제 6개를 순서대로 나열하세요.\n\n"
                "{format_instructions}"
            ),
            input_variables=["level", "subject"],
            partial_variables={"format_instructions": fi},
        )

    return prompt | llm | output_parser


def run_csv_parser(
    scenario: str,
    invoke_input: Dict[str, str],
    llm: ChatOpenAI,
) -> Tuple[List[str], float]:
    """
    CommaSeparatedListOutputParser 체인을 실행합니다.

    Returns:
        (result_list, elapsed_seconds)
    """
    chain = build_csv_chain(scenario, llm)

    t0 = time.perf_counter()
    result = chain.invoke(invoke_input)
    elapsed = time.perf_counter() - t0

    return result, elapsed


CSV_CODE = '''\
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
# → "Your response should be a list of comma separated values, eg: `foo, bar, baz`"

prompt = PromptTemplate(
    template="주제 목록을 나열하세요.\\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | model | output_parser
result = chain.invoke({"subject": "한국의 유명한 산"})
# result 타입: list
# 예: ["한라산", "지리산", "설악산", "북한산", "덕유산"]
'''


# ── 탭 5: Datetime / Enum ─────────────────────────────────────────────────────

def run_datetime_parser(
    event_q: str,
    llm: ChatOpenAI,
) -> Tuple[HistoricalEvent, float]:
    parser = PydanticOutputParser(pydantic_object=HistoricalEvent)
    prompt = PromptTemplate.from_template(
        "다음 사건에 대한 정보를 제공하세요.\n\n"
        "사건: {question}\n\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm | parser

    t0 = time.perf_counter()
    result = chain.invoke({"question": event_q})
    elapsed = time.perf_counter() - t0

    return result, elapsed


DATETIME_CODE = '''\
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class HistoricalEvent(BaseModel):
    event_name: str = Field(description="사건 이름")
    event_date: datetime = Field(description="날짜 (YYYY-MM-DD 형식)")
    description: str = Field(description="한 문장 설명")

parser = PydanticOutputParser(pydantic_object=HistoricalEvent)
chain = prompt | model | parser

result = chain.invoke({"question": event_q})
# result.event_date 는 datetime 객체!
formatted = result.event_date.strftime("%Y년 %m월 %d일")
'''


def run_enum_parser(
    feedback_text: str,
    llm: ChatOpenAI,
) -> Tuple[SentimentAnalysis, float]:
    parser = PydanticOutputParser(pydantic_object=SentimentAnalysis)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 고객 피드백 분석 전문가입니다."),
        (
            "user",
            "다음 고객 피드백의 감정을 분석하세요.\n\n"
            "피드백: {feedback}\n\n{format_instructions}",
        ),
    ]).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm | parser

    t0 = time.perf_counter()
    result = chain.invoke({"feedback": feedback_text})
    elapsed = time.perf_counter() - t0

    return result, elapsed


ENUM_CODE = '''\
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class Sentiment(Enum):
    POSITIVE = "긍정"
    NEUTRAL  = "중립"
    NEGATIVE = "부정"

class SentimentAnalysis(BaseModel):
    sentiment: Sentiment = Field(
        description="감정 분류 (긍정, 중립, 부정 중 하나)"
    )

parser = PydanticOutputParser(pydantic_object=SentimentAnalysis)
chain = prompt | model | parser

result = chain.invoke({"feedback": feedback_text})
# result.sentiment 는 Sentiment Enum 값!
print(result.sentiment.value)   # "긍정"
print(result.sentiment.name)    # "POSITIVE"
'''


# ── 탭 6: OutputFixingParser ──────────────────────────────────────────────────

def run_fixing_parser_compare(
    bad_json: str,
    llm: ChatOpenAI,
) -> Dict[str, Any]:
    """
    기본 파서와 OutputFixingParser를 동시에 실행하고 결과를 비교합니다.

    Returns:
        {
            "model_label": str,
            "base_ok": bool,
            "base_error": str | None,
            "fixed_result": MusicAlbum | Product | None,
            "fix_error": str | None,
            "elapsed": float,
        }
    """
    has_album = any(k in bad_json for k in ["album_title", "BTS", "K-Pop"])
    has_product = any(k in bad_json for k in ["price", "stock", "노트북", "마우스", "키보드"])

    if has_album or not has_product:
        pydantic_model = MusicAlbum
        model_label = "MusicAlbum"
    else:
        pydantic_model = Product
        model_label = "Product"

    base_parser = PydanticOutputParser(pydantic_object=pydantic_model)
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

    base_ok = False
    base_error = None
    try:
        base_parser.parse(bad_json)
        base_ok = True
    except Exception as e:
        base_error = f"{type(e).__name__}: {str(e)[:200]}"

    fixed_result = None
    fix_error = None
    t0 = time.perf_counter()
    try:
        fixed_result = fixing_parser.parse(bad_json)
    except Exception as e:
        fix_error = str(e)
    elapsed = time.perf_counter() - t0

    return {
        "model_label": model_label,
        "base_ok": base_ok,
        "base_error": base_error,
        "fixed_result": fixed_result,
        "fix_error": fix_error,
        "elapsed": elapsed,
    }


def run_fixing_chain(dish_name: str, llm: ChatOpenAI) -> Tuple[RecipeInfo, float]:
    base_parser = PydanticOutputParser(pydantic_object=RecipeInfo)
    fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 요리 전문가입니다. 레시피 정보를 정확하게 제공하세요."),
        (
            "user",
            "다음 요리의 레시피 정보를 알려주세요.\n\n"
            "요리: {dish_name}\n\n{format_instructions}",
        ),
    ]).partial(format_instructions=base_parser.get_format_instructions())

    chain = prompt | llm | fixing_parser

    t0 = time.perf_counter()
    result = chain.invoke({"dish_name": dish_name})
    elapsed = time.perf_counter() - t0

    return result, elapsed


FIXING_CODE = '''\
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser

base_parser = PydanticOutputParser(pydantic_object=MusicAlbum)

# 기본 파서를 LLM으로 감싸서 오류 복구 기능 추가
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser, llm=model
)

# 잘못된 JSON도 자동 수정 후 파싱
result = fixing_parser.parse(bad_json)
print(result.artist, result.album_title)
'''

FIXING_CHAIN_CODE = '''\
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import OutputFixingParser

base_parser = PydanticOutputParser(pydantic_object=RecipeInfo)
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser, llm=model
)

prompt = ChatPromptTemplate.from_messages([...])
# 체인 파서 자리에 fixing_parser 사용
chain = prompt | model | fixing_parser

result = chain.invoke({"dish_name": "김치찌개"})
'''
