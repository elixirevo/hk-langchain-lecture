from __future__ import annotations

import json
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

DEFAULT_MODEL_NAME = "gpt-4o"
DEFAULT_NUM_FOLLOW_UP_QUESTIONS = 4
MODEL_OPTIONS = [
    "gpt-5",
    "gpt-5-nano",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o-mini",
]

FIRST_STAGE_QUESTIONS = [
    {
        "id": "P1",
        "text": "최근 새로운 사람들을 여러 명 만났던 상황을 떠올려 주세요. 그 자리에서 처음 20~30분 동안 당신은 어떻게 행동했고, 속으로는 어떤 생각을 했나요?",
    },
    {
        "id": "P2",
        "text": "최근 어떤 문제를 해결해야 했던 경험을 하나 골라 주세요. 그때 무엇부터 확인했고, 어떤 순서로 접근했는지 구체적으로 설명해 주세요.",
    },
    {
        "id": "P3",
        "text": "중요한 결정을 내려야 했던 최근 경험을 말해 주세요. 그때 무엇을 가장 중요하게 봤고, 왜 그렇게 판단했나요?",
    },
    {
        "id": "P4",
        "text": "해야 할 일이 많았던 한 주를 떠올려 주세요. 당신은 일정을 어떻게 관리했고, 계획이 틀어졌을 때는 어떻게 반응했나요?",
    },
    {
        "id": "P5",
        "text": "에너지가 많이 차오르거나 반대로 크게 소모됐던 최근 상황을 각각 하나씩 설명해 주세요. 어떤 환경과 사람이 영향을 줬는지도 포함해 주세요.",
    },
    {
        "id": "P6",
        "text": "누군가와 의견이 부딪혔던 최근 경험을 말해 주세요. 그때 당신은 어떤 방식으로 반응했고, 무엇을 해결하려고 했나요?",
    },
]

PurposeTag = Literal[
    "energy_direction",
    "information_focus",
    "decision_basis",
    "structure_preference",
]

AXIS_ORDER: list[PurposeTag] = [
    "energy_direction",
    "information_focus",
    "decision_basis",
    "structure_preference",
]

AXIS_METADATA = {
    "energy_direction": {
        "label": "에너지 방향",
        "negative_letter": "I",
        "positive_letter": "E",
        "description": "혼자 정리하며 충전하는 편인지, 외부 상호작용에서 에너지를 얻는 편인지",
    },
    "information_focus": {
        "label": "정보 처리",
        "negative_letter": "S",
        "positive_letter": "N",
        "description": "구체적 사실 중심인지, 패턴과 가능성 중심인지",
    },
    "decision_basis": {
        "label": "의사결정 기준",
        "negative_letter": "T",
        "positive_letter": "F",
        "description": "논리와 일관성 중심인지, 사람과 가치 중심인지",
    },
    "structure_preference": {
        "label": "구조 선호",
        "negative_letter": "J",
        "positive_letter": "P",
        "description": "미리 정리하고 닫아두는 편인지, 열어두고 유연하게 가는 편인지",
    },
}


class UserProfile(BaseModel):
    summary: str = Field(description="이 사용자를 3~4문장으로 요약")
    social_style: str = Field(description="낯선 관계/익숙한 관계에서의 행동 패턴")
    energy_pattern: str = Field(description="어떤 상황에서 에너지가 차고 소모되는지")
    decision_style: str = Field(description="결정 시 주로 보는 기준")
    planning_style: str = Field(description="계획/즉흥/변수 대응 방식")
    conflict_style: str = Field(description="갈등 상황에서의 반응 방식")
    notable_patterns: list[str] = Field(description="반복적으로 드러난 행동 패턴")
    ambiguity_points: list[str] = Field(description="아직 더 물어봐야 하는 애매한 지점")


class GeneratedQuestion(BaseModel):
    id: str
    question_text: str
    purpose_tag: PurposeTag
    why_this_question: str


class GeneratedQuestionSet(BaseModel):
    questions: list[GeneratedQuestion] = Field(
        description="후속 질문 4~6개",
        min_length=4,
        max_length=6,
    )


class AnswerAnalysis(BaseModel):
    question_id: str
    purpose_tag: PurposeTag
    evidence: list[str] = Field(description="답변에서 직접 드러난 근거")
    score_hint: float = Field(description="해당 purpose_tag 기준 -1.0 ~ 1.0")
    confidence: float = Field(description="0.0 ~ 1.0")
    contradictions: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    mbti_type: str
    summary: str
    strengths: list[str]
    cautions: list[str]
    growth_tips: list[str]


class InterviewPreparation(BaseModel):
    user_profile: UserProfile
    follow_up_questions: list[GeneratedQuestion]


class MBTIInterviewResult(BaseModel):
    user_profile: UserProfile
    follow_up_questions: list[GeneratedQuestion]
    answer_analyses: list[AnswerAnalysis]
    axis_scores: dict[str, float]
    final_type: str
    final_report: FinalReport


PROFILE_INTERPRETER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 서술형 인터뷰를 읽고 사용자의 행동 프로필을 요약하는 분석가입니다.

목표:
- 결과물은 한국어로 생성해야 합니다.
- MBTI를 직접 판단하지 않습니다.
- 오직 사용자의 행동 패턴과 사고 방식, 에너지 패턴을 정리합니다.
- 답변에 없는 내용은 추론하지 않습니다.

반드시 아래 항목으로 구조화하세요:
- summary
- social_style
- energy_pattern
- decision_style
- planning_style
- conflict_style
- notable_patterns
- ambiguity_points
""".strip(),
        ),
        (
            "human",
            """
다음은 사용자의 1차 인터뷰 답변입니다.

{first_stage_answers}
""".strip(),
        ),
    ]
)

QUESTION_GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 성격 이해용 2차 인터뷰 질문 생성기입니다.

목표:
- 사용자의 1차 프로필을 바탕으로, 그 사람에게 맞는 후속 질문 4~6개를 생성합니다.
- 질문은 모두 행동과 실제 사례 중심이어야 합니다.
- 사용자가 스스로 '나는 외향적이다/계획적이다'처럼 라벨링하게 만드는 질문은 금지합니다.
- 질문은 자연스럽고 개인화되어야 합니다.

내부 규칙:
- 질문은 한국어로 생성해야 합니다.
- 각 질문에는 아래 내부 purpose_tag 중 하나를 붙입니다.
- energy_direction
- information_focus
- decision_basis
- structure_preference
- 질문 텍스트에는 purpose_tag를 드러내지 않습니다.
- ambiguity_points를 우선 해소하도록 질문을 만듭니다.
- 질문마다 why_this_question을 짧게 적습니다.
""".strip(),
        ),
        (
            "human",
            """
다음은 사용자 프로필입니다.

{user_profile}

질문 개수: {num_questions}
""".strip(),
        ),
    ]
)

ANSWER_ANALYZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 성격 인터뷰 답변 분석기입니다.

목표:
- 각 답변을 읽고 내부 점수 힌트를 구조화합니다.
- 최종 MBTI를 직접 말하지 않습니다.
- 근거는 반드시 답변 내용에 기반해야 합니다.

점수 기준:
- energy_direction: -1.0(I 성향) ~ 1.0(E 성향)
- information_focus: -1.0(S 성향) ~ 1.0(N 성향)
- decision_basis: -1.0(T 성향) ~ 1.0(F 성향)
- structure_preference: -1.0(J 성향) ~ 1.0(P 성향)

지침:
- question_id는 입력으로 받은 값을 그대로 사용합니다.
- purpose_tag는 입력으로 받은 값을 그대로 사용합니다.
- 점수는 question의 purpose_tag에 맞는 항목만 강하게 주고, 다른 축 해석은 끌고 오지 않습니다.
- evidence에는 답변에서 직접 확인 가능한 문장 또는 행동 단서를 짧게 정리합니다.
- contradictions에는 답변 내부의 상충 요소가 있을 때만 적습니다.
""".strip(),
        ),
        (
            "human",
            """
question_id:
{question_id}

question:
{question}

purpose_tag:
{purpose_tag}

answer:
{answer}
""".strip(),
        ),
    ]
)

REPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 MBTI형 성향 결과를 설명하는 리포트 작성기입니다.

원칙:
- 이미 판정은 끝났습니다. 판정을 바꾸지 마세요.
- 과장하지 말고 자기이해용 피드백처럼 작성하세요.
- '절대적 성격 진단'처럼 말하지 마세요.

반드시 아래 필드를 채워 구조화하세요:
- mbti_type
- summary
- strengths
- cautions
- growth_tips
""".strip(),
        ),
        (
            "human",
            """
final_type: {final_type}
axis_scores: {axis_scores}
evidence_summary: {evidence_summary}
user_profile: {user_profile}
""".strip(),
        ),
    ]
)


def get_llm(
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = 0,
    api_key: str | None = None,
) -> ChatOpenAI:
    llm_kwargs = {"model": model_name}

    # GPT-5 family models can reject temperature depending on model/reasoning mode.
    if not model_name.startswith("gpt-5"):
        llm_kwargs["temperature"] = temperature

    if api_key:
        llm_kwargs["api_key"] = api_key
    return ChatOpenAI(**llm_kwargs)


def dump_model(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def validate_model(model_cls: type[BaseModel], data: dict) -> BaseModel:
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def format_first_stage_answers(first_stage_answers: dict[str, str]) -> str:
    sections = []
    for question in FIRST_STAGE_QUESTIONS:
        answer = first_stage_answers.get(question["id"], "").strip()
        sections.append(
            f"[{question['id']}] {question['text']}\n답변: {answer or '(미입력)'}"
        )
    return "\n\n".join(sections)


def normalize_user_profile(user_profile: UserProfile | dict) -> UserProfile:
    if isinstance(user_profile, UserProfile):
        return user_profile
    return validate_model(UserProfile, user_profile)


def normalize_questions(
    questions: list[GeneratedQuestion] | list[dict],
) -> list[GeneratedQuestion]:
    normalized: list[GeneratedQuestion] = []
    for item in questions:
        if isinstance(item, GeneratedQuestion):
            normalized.append(item)
        else:
            normalized.append(validate_model(GeneratedQuestion, item))
    return normalized


def validate_answers(
    question_ids: list[str],
    answers: dict[str, str],
    label: str,
) -> None:
    missing = [question_id for question_id in question_ids if not answers.get(question_id, "").strip()]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{label} 답변이 비어 있습니다: {joined}")


def create_user_profile(
    first_stage_answers: dict[str, str],
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str | None = None,
) -> UserProfile:
    validate_answers(
        [question["id"] for question in FIRST_STAGE_QUESTIONS],
        first_stage_answers,
        "1차 질문",
    )

    llm = get_llm(model_name=model_name, api_key=api_key)
    chain = PROFILE_INTERPRETER_PROMPT | llm.with_structured_output(UserProfile)
    return chain.invoke(
        {"first_stage_answers": format_first_stage_answers(first_stage_answers)}
    )


def generate_follow_up_questions(
    user_profile: UserProfile,
    num_questions: int = DEFAULT_NUM_FOLLOW_UP_QUESTIONS,
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str | None = None,
) -> list[GeneratedQuestion]:
    if num_questions < 4 or num_questions > 6:
        raise ValueError("2차 질문 개수는 4~6개여야 합니다.")

    llm = get_llm(model_name=model_name, api_key=api_key)
    chain = QUESTION_GENERATOR_PROMPT | llm.with_structured_output(GeneratedQuestionSet)
    response = chain.invoke(
        {
            "user_profile": json.dumps(
                dump_model(user_profile),
                ensure_ascii=False,
                indent=2,
            ),
            "num_questions": num_questions,
        }
    )

    questions: list[GeneratedQuestion] = []
    for index, question in enumerate(response.questions[:num_questions], start=1):
        if hasattr(question, "model_copy"):
            normalized = question.model_copy(update={"id": f"F{index}"})
        else:
            normalized = question.copy(update={"id": f"F{index}"})
        questions.append(normalized)

    return questions


def analyze_follow_up_answers(
    follow_up_questions: list[GeneratedQuestion] | list[dict],
    follow_up_answers: dict[str, str],
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str | None = None,
) -> list[AnswerAnalysis]:
    normalized_questions = normalize_questions(follow_up_questions)
    validate_answers(
        [question.id for question in normalized_questions],
        follow_up_answers,
        "2차 질문",
    )

    llm = get_llm(model_name=model_name, api_key=api_key)
    chain = ANSWER_ANALYZER_PROMPT | llm.with_structured_output(AnswerAnalysis)

    analyses: list[AnswerAnalysis] = []
    for question in normalized_questions:
        analysis = chain.invoke(
            {
                "question_id": question.id,
                "question": question.question_text,
                "purpose_tag": question.purpose_tag,
                "answer": follow_up_answers[question.id].strip(),
            }
        )

        if hasattr(analysis, "model_copy"):
            normalized = analysis.model_copy(
                update={
                    "question_id": question.id,
                    "purpose_tag": question.purpose_tag,
                }
            )
        else:
            normalized = analysis.copy(
                update={
                    "question_id": question.id,
                    "purpose_tag": question.purpose_tag,
                }
            )

        analyses.append(normalized)

    return analyses


def calculate_axis_scores(answer_analyses: list[AnswerAnalysis]) -> dict[str, float]:
    axis_scores: dict[str, float] = {}

    for tag in AXIS_ORDER:
        related = [analysis for analysis in answer_analyses if analysis.purpose_tag == tag]
        if not related:
            axis_scores[tag] = 0.0
            continue

        total_weight = sum(max(analysis.confidence, 0.05) for analysis in related)
        weighted_sum = sum(
            analysis.score_hint * max(analysis.confidence, 0.05)
            for analysis in related
        )
        score = weighted_sum / total_weight
        axis_scores[tag] = round(max(-1.0, min(1.0, score)), 3)

    return axis_scores


def determine_mbti_type(axis_scores: dict[str, float]) -> str:
    return "".join(
        [
            "E" if axis_scores.get("energy_direction", 0) >= 0 else "I",
            "N" if axis_scores.get("information_focus", 0) >= 0 else "S",
            "F" if axis_scores.get("decision_basis", 0) >= 0 else "T",
            "P" if axis_scores.get("structure_preference", 0) >= 0 else "J",
        ]
    )


def build_evidence_summary(answer_analyses: list[AnswerAnalysis]) -> str:
    sections: list[str] = []

    for tag in AXIS_ORDER:
        related = [analysis for analysis in answer_analyses if analysis.purpose_tag == tag]
        if not related:
            continue

        evidence_lines: list[str] = []
        for analysis in related:
            evidence_lines.extend(analysis.evidence[:2])

        if evidence_lines:
            sections.append(
                f"{tag}: " + " / ".join(evidence_lines[:3])
            )

    return "\n".join(sections)


def create_final_report(
    user_profile: UserProfile | dict,
    axis_scores: dict[str, float],
    answer_analyses: list[AnswerAnalysis],
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str | None = None,
) -> FinalReport:
    normalized_profile = normalize_user_profile(user_profile)
    final_type = determine_mbti_type(axis_scores)

    llm = get_llm(model_name=model_name, api_key=api_key)
    chain = REPORT_PROMPT | llm.with_structured_output(FinalReport)
    report = chain.invoke(
        {
            "final_type": final_type,
            "axis_scores": json.dumps(axis_scores, ensure_ascii=False),
            "evidence_summary": build_evidence_summary(answer_analyses),
            "user_profile": json.dumps(
                dump_model(normalized_profile),
                ensure_ascii=False,
                indent=2,
            ),
        }
    )

    if hasattr(report, "model_copy"):
        return report.model_copy(update={"mbti_type": final_type})
    return report.copy(update={"mbti_type": final_type})


def prepare_interview(
    first_stage_answers: dict[str, str],
    num_questions: int = DEFAULT_NUM_FOLLOW_UP_QUESTIONS,
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str | None = None,
) -> InterviewPreparation:
    user_profile = create_user_profile(
        first_stage_answers=first_stage_answers,
        model_name=model_name,
        api_key=api_key,
    )
    follow_up_questions = generate_follow_up_questions(
        user_profile=user_profile,
        num_questions=num_questions,
        model_name=model_name,
        api_key=api_key,
    )
    return InterviewPreparation(
        user_profile=user_profile,
        follow_up_questions=follow_up_questions,
    )


def complete_interview(
    user_profile: UserProfile | dict,
    follow_up_questions: list[GeneratedQuestion] | list[dict],
    follow_up_answers: dict[str, str],
    model_name: str = DEFAULT_MODEL_NAME,
    api_key: str | None = None,
) -> MBTIInterviewResult:
    normalized_profile = normalize_user_profile(user_profile)
    normalized_questions = normalize_questions(follow_up_questions)
    answer_analyses = analyze_follow_up_answers(
        follow_up_questions=normalized_questions,
        follow_up_answers=follow_up_answers,
        model_name=model_name,
        api_key=api_key,
    )
    axis_scores = calculate_axis_scores(answer_analyses)
    final_type = determine_mbti_type(axis_scores)
    final_report = create_final_report(
        user_profile=normalized_profile,
        axis_scores=axis_scores,
        answer_analyses=answer_analyses,
        model_name=model_name,
        api_key=api_key,
    )

    return MBTIInterviewResult(
        user_profile=normalized_profile,
        follow_up_questions=normalized_questions,
        answer_analyses=answer_analyses,
        axis_scores=axis_scores,
        final_type=final_type,
        final_report=final_report,
    )
