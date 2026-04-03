from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    chain,
)
import time
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

"""
맛점을 하기 위해 맛있는 식당을 찾아주는 에이전트
위치, 사람, 인원수를 고려해서 오늘 뭘 먹을지에 대한 음식 카테고리와 관련 맛집들을 5순위까지 추천해주는 에이전트

1. 위치, 인원수, 사람 특성 정보를 입력받는다.
2. 웹 서치로 원하는 위치 주변의 걸어서 10분 거리 이내에 있는 식당들을 검색한다.
3. 평점이 낮은 식당을 제외한다.
4. 리뷰를 분석해서 괜찮
4. 검색된 식당을 카테고리와 식당으로 분류한다.

"""

SYSTEM_PROMPT = """
당신은 맛집 추천 전문 에이전트입니다.
1. 사용자의 위치, 인원수, 특성을 파악한다.
2. 도보 10분 이내 식당을 검색한다.
3. 평점 4.0 미만 식당은 제외한다.
4. 리뷰를 분석하여 실제 방문자 경험을 파악한다.
5. 음식 카테고리별로 분류하고 상위 5개를 추천한다.
최종 응답은 카테고리 → 추천 식당 5순위 형태로 제공한다.
"""

llm = ChatOpenAI(model="gpt-4o", temperature=0)
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[search_restaurants, analyze_reviews],
    checkpointer=memory,
    state_modifier=SYSTEM_PROMPT
)

# 실행
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "위치: 충정로역, 인원: 4명, 특성: 회사 점심"
    }]
})

result
print(f' ==> [Line 52]: \033[38;2;56;204;135m[result]\033[0m({type(result).__name__}) = \033[38;2;21;53;31m{result}\033[0m')
