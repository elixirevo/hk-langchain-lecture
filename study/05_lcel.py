from datetime import datetime
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{topic}에 대해 유저가 지정한 형식으로 간략하게 설명하시오"),
        ("human", "{format}"),
    ]
)

chain = prompt | llm | output_parser

runnable_chain = (
    {"topic": RunnablePassthrough(), "format": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)


def get_today(a):
    return datetime.today().strftime("%m-%d")


prompt = PromptTemplate.from_template(
    "{today} 가 생일인 유명인 {n} 명을 나열하세요. 생년월일을 표기해 주세요."
)

chain = (
    {"today": RunnableLambda(get_today), "n": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

chain = (
    {"today": RunnableLambda(get_today), "n": itemgetter("n")}
    | prompt
    | llm
    | output_parser
)

res = chain.invoke({"n": "2"})
print(res)


# res = runnable_chain.stream(
#     {"topic": "스토아 철학", "format": "1. 정의 2. 핵심 사상 3. 현대적 의의"}
# )
# for chunk in res:
#     print(chunk, end="", flush=True)
