from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    chain,
)
import time
from pydantic import BaseModel, Field

load_dotenv()

class EmailSummary(BaseModel):
    person: str = Field(description="이메일을 보낸 사람")
    company: str = Field(description="이메일을 보낸 회사의 이름")
    email: str = Field(description="이메일 주소")
    subject: str = Field(description="이메일 제목")
    summary: str = Field(description="이메일 내용 요약")
    date: str = Field(description="이메일 수신 날짜")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_parser = PydanticOutputParser(pydantic_object=EmailSummary)

email_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 이메일 요약 전문가입니다."),
    ("human", """
    이메일을 아래 형식에 맞춰 요약해주세요.

    이메일: {email}
    형식: {format}

    요약:
    """),
])

email_prompt = email_prompt.partial(format=output_parser.get_format_instructions())

email_chain = (
    email_prompt
    | llm
    | output_parser
)

email_conversation = """
From: 테디 (teddy@teddynote.com)
To: 이은채 대리님 (eunchae@teddyinternational.me)
Subject: RAG 솔루션 시연 관련 미팅 제안
안녕하세요, 이은채 대리님,
저는 테디노트의 테디입니다. 최근 귀사에서 AI를 활용한 혁신적인 솔루션을 모색 중이라는 소
식을 들었습니다. 테디노트는 AI 및 RAG 솔루션 분야에서 다양한 경험과 노하우를 가진 기업으
로, 귀사의 요구에 맞는 최적의 솔루션을 제공할 수 있다고 자부합니다.
저희 테디노트의 RAG 솔루션은 귀사의 데이터 활용을 극대화하고, 실시간으로 정확한 정보 제
공을 통해 비즈니스 의사결정을 지원하는 데 탁월한 성능을 보입니다. 이 솔루션은 특히 다양
한 산업에서의 성공적인 적용 사례를 통해 그 효과를 입증하였습니다.
귀사와의 협력 가능성을 논의하고, 저희 RAG 솔루션의 구체적인 기능과 적용 방안을 시연하기 위해 미팅을 제안드립니다. 다음 주 목요일(7월 18일) 오전 10시에 귀사 사무실에서 만나 뵐 
수 있을까요?
미팅 시간을 조율하기 어려우시다면, 편하신 다른 일정을 알려주시면 감사하겠습니다. 이은채 
대리님과의 소중한 만남을 통해 상호 발전적인 논의가 이루어지길 기대합니다.
감사합니다.
테디
테디노트 AI 솔루션팀
"""

res = email_chain.invoke({"email": email_conversation})
print(f' ==> [Line 38]: \033[38;2;30;122;206m[res]\033[0m({type(res).__name__}) = \033[38;2;189;177;222m{res}\033[0m')












