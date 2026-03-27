# 05-Memory: 대화 메모리 및 상태 관리

## 📋 Overview
대화형 AI 애플리케이션에서 대화 컨텍스트를 유지하고 관리하는 다양한 메모리 시스템을 학습합니다. 단순 버퍼부터 벡터 기반 장기 메모리까지 포괄적으로 다룹니다.

## 🎯 Prerequisites
- `01-Basic` ~ `04-Model` 완료
- 대화형 AI 개념 이해
- VectorStore 기본 지식 (선택사항)

## 📚 Contents

| Notebook | Description | Use Case |
|----------|-------------|----------|
| `01-ConversationBufferMemory.ipynb` | 전체 대화 이력 저장 | 짧은 대화, 컨텍스트 중요 |
| `02-ConversationBufferWindowMemory.ipynb` | 최근 N개 대화만 유지 | 긴 대화, 토큰 제한 |
| `03-ConversationTokenBufferMemory.ipynb` | 토큰 수 기반 메모리 제한 | 정확한 토큰 관리 |
| `04-ConversationEntityMemory.ipynb` | 엔티티 정보 추출 및 저장 | 인물/장소 정보 기억 |
| `05-ConversationKnowledgeGraph.ipynb` | 지식 그래프 기반 메모리 | 관계성 파악, 복잡한 정보 |
| `06-ConversationSummary.ipynb` | 대화 요약 저장 | 매우 긴 대화, 핵심만 보존 |
| `07-VectorStoreRetrieverMemory.ipynb` | 벡터 검색 기반 메모리 | 장기 메모리, 관련성 검색 |
| `08-LCEL-add-memory.ipynb` | LCEL 체인에 메모리 통합 | 최신 패턴, 권장 방식 |
| `09-Memory-using-SQLite.ipynb` | SQLite 영구 저장 | 데이터베이스 기반 보존 |
| `10-Conversation-With-History.ipynb` | 대화 이력 관리 패턴 | 실전 구현 패턴 |

## 🔑 Key Concepts

### 1. Memory Types

#### Short-term Memory
- **Buffer**: 전체 대화 기록
- **Window**: 최근 N턴만 보존
- **Token Buffer**: 토큰 수 제한

#### Long-term Memory
- **Summary**: 주기적 요약
- **VectorStore**: 의미 기반 검색
- **Knowledge Graph**: 구조화된 지식

### 2. Memory Management
- **Context Length**: 토큰 제한 관리
- **Relevance**: 관련 정보만 로드
- **Persistence**: 세션 간 보존

### 3. LCEL Integration
- **RunnableWithMessageHistory**: 최신 권장 방식
- **Session Management**: 세션 ID 기반 분리
- **Flexible Backends**: 다양한 저장소 지원

## 🛤️ Learning Path

```
1. 기본 메모리 (01, 02, 03)
   ├→ Buffer (01)
   ├→ Window (02)
   └→ Token Buffer (03)
   ↓
2. 고급 메모리 (04, 05, 06, 07)
   ├→ Entity (04)
   ├→ Knowledge Graph (05)
   ├→ Summary (06)
   └→ VectorStore (07)
   ↓
3. 통합 및 영구화 (08, 09, 10)
   ├→ LCEL Integration (08) ← 중요
   ├→ SQLite (09)
   └→ Production Pattern (10)
```

## 💡 Quick Start Examples

### ConversationBufferMemory
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory
)

conversation.predict(input="Hi, I'm Alice")
conversation.predict(input="What's my name?")
# Output: "Your name is Alice"
```

### ConversationBufferWindowMemory
```python
from langchain.memory import ConversationBufferWindowMemory

# 최근 2턴만 기억
memory = ConversationBufferWindowMemory(k=2)
```

### LCEL with RunnableWithMessageHistory (권장)
```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 세션별로 대화 관리
chain_with_history.invoke(
    {"input": "Hi!"},
    config={"configurable": {"session_id": "user1"}}
)
```

### VectorStore Memory (장기 메모리)
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma

vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

memory = VectorStoreRetrieverMemory(retriever=retriever)
```

## 🎯 Memory Selection Guide

| Scenario | Recommended Memory | Reason |
|----------|-------------------|--------|
| 짧은 대화 (<10 턴) | ConversationBufferMemory | 전체 컨텍스트 보존 |
| 긴 대화, 토큰 제한 | ConversationBufferWindowMemory | 최근 대화만 유지 |
| 정확한 토큰 관리 | ConversationTokenBufferMemory | 토큰 예산 준수 |
| 매우 긴 대화 | ConversationSummaryMemory | 요약으로 압축 |
| 장기 메모리 필요 | VectorStoreRetrieverMemory | 관련성 기반 검색 |
| 복잡한 정보 관계 | ConversationKnowledgeGraph | 구조화된 지식 |
| **Production** | **RunnableWithMessageHistory** | **최신 권장 패턴** |

## 🔗 Related Sections
- **Previous**: `04-Model` - LLM 모델
- **Next**: `06-DocumentLoader` - 문서 로딩
- **Related**: `12-RAG` - RAG with conversation history

## 📖 Resources
- [LangChain Memory](https://python.langchain.com/docs/modules/memory/)
- [RunnableWithMessageHistory](https://python.langchain.com/docs/expression_language/how_to/message_history)
- [Chat Message History](https://python.langchain.com/docs/modules/memory/chat_messages/)
