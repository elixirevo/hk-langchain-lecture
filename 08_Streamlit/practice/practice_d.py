import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    chain,
)

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

classify_prompt = ChatPromptTemplate.from_template("""
다음 텍스트를 확인하고 '뉴스', '리뷰', '기타' 중 어울리는 카테고리를 선택해서 반환하시오

{text}
""")

classify_chain = (
    classify_prompt
    | model
    | StrOutputParser()
)

news_prompt = ChatPromptTemplate.from_template("""
    다음 텍스트는 뉴스입니다.
    뉴스에 맞는 분석을 하고 분석한 내용을 반환하시오

    텍스트:
    {text}
""")

review_prompt = ChatPromptTemplate.from_template("""
    다음 텍스트는 리뷰입니다.
    리뷰에 맞는 분석을 하고 분석한 내용을 반환하시오

    텍스트:
    {text}
""")

etc_prompt = ChatPromptTemplate.from_template("""
    다음 텍스트는 기준이 정해지지 않은 텍스트 입니다.
    자유롭게 분석하고 분석한 내용을 반환하시오

    텍스트:
    {text}
""")

news_chain = (
    news_prompt
    | model
    | StrOutputParser()
)

review_chain = (
    review_prompt
    | model
    | StrOutputParser()
)

etc_chain = (
    etc_prompt
    | model
    | StrOutputParser()
)

@chain
def smart_transform_pipeline(input_text: str):
    """
    5단계 파이프라인 구성
    """
    info = {"text": input_text}
    category = classify_chain.invoke(info)

    if "뉴스" in category:
        res = news_chain.invoke(info)
    elif "리뷰" in category:
        res = review_chain.invoke(info)
    else:
        res = etc_chain.invoke(info)

    final_report = f"""
    [ 텍스트 분석 통합 리포트 ]
    ------------------------------------------------------------
    - 텍스트 유형: {category}
    - 분류 근거: 원본 텍스트 내용을 바탕으로 LLM이 판별함
    ------------------------------------------------------------
    [ 분석 상세 내용 ]
    {res}
    ------------------------------------------------------------
    """

    return final_report


# ============================================================
# Streamlit UI 구성
# ============================================================

st.set_page_config(page_title="스마트 텍스트 분석기", page_icon="📝")

st.title("📝 스마트 텍스트 분석기")
st.markdown("텍스트를 입력하시면 제가 자동으로 카테고리를 분류하고 맞춤형 분석 리포트를 써드릴게요!")

user_input = st.text_area("분석할 텍스트를 여기에 입력해주세요:", height=150, placeholder="예: 어제 먹은 사과는 정말 맛있었지만 가격이 너무 비쌌다.")

if st.button("분석 시작! 🚀"):
    if not user_input.strip():
        st.warning("앗! 텍스트가 비어있어요! 빈 칸을 주시면 저도 어쩔 수 없다구요!")
    else:
        with st.spinner("열심히 텍스트를 읽고 분석하는 중이에요... 잠시만요! 💦"):
            try:
                # 파이프라인 실행
                report = smart_transform_pipeline.invoke(user_input)
                
                # 결과 출력
                st.success("분석이 무사히 완료되었어요!")
                st.markdown(report)
            except Exception as e:
                st.error(f"뭔가 문제가 생겼어요! 에러 내용: {str(e)}")