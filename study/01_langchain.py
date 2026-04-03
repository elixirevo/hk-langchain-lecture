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

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트입니다."),
    ("user", "{question}"),
])

output_parser = StrOutputParser()

chain = (prompt | model | output_parser)

start_time = time.time()

print(chain.invoke({"question": "안녕하세요?"}))

end_time = time.time()
elapsed = end_time - start_time

if elapsed < 60:
    print(f"소요 시간: {elapsed:.1f}초")
else:
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"소요 시간: {minutes}분 {seconds:.1f}초")
