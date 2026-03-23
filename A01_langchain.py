from dotenv import load_dotenv

# .env 파일에서 OPENAI_API_KEY 등 환경 변수를 로드
load_dotenv()

from langchain_openai import ChatOpenAI
import os

# ---------------------------------------------------
# ChatOpenAI 모델 초기화
# ---------------------------------------------------
# ChatOpenAI: OpenAI의 채팅 완성 API를 감싸는 LangChain 클래스
# model 파라미터로 사용할 모델 이름을 지정해요
# gpt-4o-mini는 속도·비용 면에서 입문 실습에 적합해요
model = ChatOpenAI(model="gpt-4o-mini")
model
print(f' ==> [Line 16]: \033[38;2;77;178;99m[model]\033[0m({type(model).__name__}) = \033[38;2;108;148;244m{model}\033[0m')
