from ast import parse
from dotenv import load_dotenv

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 친절한 챗봇입니다."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

store = {}

conversation_log = [
    {
        "human": "안녕하세요, 제 이름은 elixir입니다. 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?",
        "ai": "안녕하세요! 계좌 개설을 원하신다니 기쁩니다. 먼저, 본인 인증을 위해 신분증을 준비해 주시겠어요?",
    },
    {
        "human": "신분증은 준비했습니다. 다음 단계는 무엇인가요?",
        "ai": "신분증을 준비하셨군요! 좋습니다. 이제 앱에서 '계좌 개설'을 선택하시고, 안내에 따라 신분증 촬영과 본인 인증 절차를 진행해 주세요.",
    },
    {
        "human": "앱에서 '계좌 개설'을 선택했는데, '본인 인증' 단계에서 오류가 발생합니다.",
        "ai": "오류가 발생했다니 죄송합니다. 혹시 신분증 촬영 시 빛 반사가 심하거나, 신분증이 너무 멀리서 촬영되지 않았는지 확인해 주시겠어요?",
    },
]

history = InMemoryChatMessageHistory()

messages = []
for log in conversation_log:
    messages.append(HumanMessage(content=log["human"]))
    messages.append(AIMessage(content=log["ai"]))

history.add_messages(messages)

chain = (
    prompt
    | llm
    | parser
)

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

store["elixir"] = history

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",       # 질문이 들어가는 키
    history_messages_key="history",   # 프롬프트의 MessagesPlaceholder 이름
)



# res = chain.invoke({
#     "history": history.messages,
#     "input": "제 이름이 무엇이었죠?"
# })

res = chain_with_history.invoke(
    {"input": "제 이름이 무엇이었죠?"},
    config={"configurable": {"session_id": "elixir"}}
)


print(res)




