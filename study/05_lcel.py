from datetime import datetime

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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


def get_today():
    return datetime.today().strftime("%Y-%m-%d")


res = runnable_chain.stream(
    {"topic": "스토아 철학", "format": "1. 정의 2. 핵심 사상 3. 현대적 의의"}
)
for chunk in res:
    print(chunk, end="", flush=True)
