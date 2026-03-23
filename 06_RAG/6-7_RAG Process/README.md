# 6-7_RAG Process

RAG(Retrieval-Augmented Generation) 시스템의 전체 프로세스를 학습합니다.
기본 RAG부터 고급 기법까지 단계적으로 구현합니다.

## 📚 노트북 목록

| 번호 | 노트북 | 주제 | 난이도 | 핵심 기법 |
|------|--------|------|--------|----------|
| 00 | RAG-Basic.ipynb | 기본 RAG 파이프라인 | ⭐ | 8단계 구축 (PDF) |
| 01 | RAG-Web-Based.ipynb | 웹 문서 RAG | ⭐ | WebBaseLoader |
| 02 | RAG-Advanced.ipynb | 고급 검색 기법 | ⭐⭐⭐ | Ensemble, MMR |
| 03 | Conversation-RAG.ipynb | 대화 이력 RAG | ⭐⭐ | RunnableWithMessageHistory |
| 04 | RAPTOR.ipynb | 계층적 RAG | ⭐⭐⭐⭐ | 계층적 요약 |
| 05 | Web-Summarization.ipynb | 웹 요약 RAG | ⭐⭐⭐ | Map-Reduce |

## 🎯 학습 경로

```
1. 기본 RAG 이해 (00, 01)
   ├→ PDF 기반 RAG (00)
   └→ 웹 기반 RAG (01)
   ↓
2. 검색 품질 향상 (02)
   ├→ Ensemble Retriever
   └→ MMR 검색
   ↓
3. 대화형 RAG (03)
   └→ 세션 관리
   ↓
4. 긴 문서 처리 (04, 05)
   ├→ RAPTOR (04)
   └→ Map-Reduce 요약 (05)
```

## 🔑 핵심 개념

### RAG 기본 파이프라인 (8단계)

```
사전작업 (Pre-processing):
1. 문서 로드 → 2. 분할 → 3. 임베딩 → 4. 벡터 저장

실행 (Runtime):
5. 검색기 → 6. 프롬프트 → 7. LLM → 8. 체인
```

### 고급 기법

**Ensemble Retriever**:
- BM25 (키워드) + FAISS (의미) 결합
- 정확한 용어와 유사 개념 모두 검색

**MMR (Maximum Marginal Relevance)**:
- 관련성 + 다양성 균형
- 중복 제거, 다양한 관점 확보

**RAPTOR**:
- 계층적 문서 요약
- 다양한 추상화 레벨에서 검색

## 💡 실습 데이터

- **PDF**: `sample-rag-brief.pdf` (디지털 정책 문서)
- **웹**: Wikipedia 한국어 페이지
- 위치: `../6-1_DocumentLoaders/data/`

## ⚙️ 환경 설정

```bash
pip install langchain langchain-community langchain-openai
pip install faiss-cpu  # 또는 faiss-gpu
pip install beautifulsoup4 pymupdf
```

## 🚀 빠른 시작

### 기본 RAG (00번)

```python
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1-4단계: 문서 준비
loader = PyMuPDFLoader("document.pdf")
docs = loader.load()
# ... (분할, 임베딩, 저장)

# 5-8단계: RAG 체인
retriever = vectorstore.as_retriever()
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# 질의
answer = rag_chain.invoke("질문")
```

### 고급 RAG (02번)

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Ensemble Retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)

# MMR Retriever
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "lambda_mult": 0.5}
)
```

## 📊 성능 비교

| 기법 | 검색 정확도 | 다양성 | 처리 시간 | 적용 시나리오 |
|------|-----------|--------|----------|-------------|
| 기본 RAG | ⭐⭐⭐ | ⭐⭐ | 빠름 | 간단한 QA |
| Ensemble | ⭐⭐⭐⭐ | ⭐⭐ | 보통 | 전문 용어 많은 문서 |
| MMR | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 약간 느림 | 다양한 관점 필요 |
| RAPTOR | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 느림 | 긴 문서 (>50페이지) |

## 🎓 학습 팁

1. **순서대로 학습**: 00 → 01 → 02 → 03 → 04 → 05
2. **실습 중심**: 각 노트북의 코드를 직접 실행
3. **파라미터 실험**: chunk_size, k, weights 등 조정 테스트
4. **자신의 데이터 적용**: 실습 데이터를 자신의 문서로 교체

## 📖 참고 자료

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [RAG Paper (2020)](https://arxiv.org/abs/2005.11401)
- [RAPTOR Paper (2024)](https://arxiv.org/abs/2401.18059)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/index/contextual-retrieval)

## 🔗 관련 섹션

- **이전**: `6-6_Reranker` - 검색 결과 재정렬
- **다음**: `7_Evaluation` - RAG 성능 평가
- **관련**: `5_Advanced-LCEL` - LCEL 고급 기법

---

이 교재는 LangChain 1.0.x 기준으로 작성되었습니다.
