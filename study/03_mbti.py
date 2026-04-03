import time

from dotenv import load_dotenv

from mbti_core import (
    AXIS_METADATA,
    AXIS_ORDER,
    FIRST_STAGE_QUESTIONS,
    complete_interview,
    prepare_interview,
)


def collect_answers(questions: list[dict[str, str]]) -> dict[str, str]:
    answers: dict[str, str] = {}

    for index, question in enumerate(questions, start=1):
        print(f"\n[{index}] {question['text']}")
        answers[question["id"]] = input("> ").strip()

    return answers


def print_axis_scores(axis_scores: dict[str, float]) -> None:
    print("\n[축별 점수]")
    for tag in AXIS_ORDER:
        metadata = AXIS_METADATA[tag]
        score = axis_scores[tag]
        print(
            f"- {metadata['label']}: {score:+.2f} "
            f"({metadata['negative_letter']} <-> {metadata['positive_letter']})"
        )


def main() -> None:
    start_time = time.time()
    load_dotenv()

    print("MBTI 서술형 인터뷰 CLI")
    print("더 편한 UI가 필요하면 `streamlit run mbti_streamlit_app.py`를 실행하세요.")

    first_stage_answers = collect_answers(FIRST_STAGE_QUESTIONS)
    preparation = prepare_interview(first_stage_answers)

    print("\n[1차 분석 요약]")
    print(preparation.user_profile.summary)

    print("\n[2차 질문]")
    follow_up_answers: dict[str, str] = {}
    for question in preparation.follow_up_questions:
        print(f"\n[{question.id}] {question.question_text}")
        print(f"질문 의도: {question.why_this_question}")
        follow_up_answers[question.id] = input("> ").strip()

    result = complete_interview(
        user_profile=preparation.user_profile,
        follow_up_questions=preparation.follow_up_questions,
        follow_up_answers=follow_up_answers,
    )

    print(f"\n[최종 결과] {result.final_type}")
    print(result.final_report.summary)
    print_axis_scores(result.axis_scores)

    print("\n[강점]")
    for item in result.final_report.strengths:
        print(f"- {item}")

    print("\n[주의할 점]")
    for item in result.final_report.cautions:
        print(f"- {item}")

    print("\n[성장 팁]")
    for item in result.final_report.growth_tips:
        print(f"- {item}")

    elapsed = time.time() - start_time
    if elapsed < 60:
        print(f"\n소요 시간: {elapsed:.1f}초")
    else:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"\n소요 시간: {minutes}분 {seconds:.1f}초")


if __name__ == "__main__":
    main()
