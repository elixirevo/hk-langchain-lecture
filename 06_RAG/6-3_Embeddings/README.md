# 6-3. Embeddings: 텍스트 임베딩 및 벡터화

RAG 시스템의 핵심 구성요소인 텍스트 임베딩을 학습하는 섹션입니다.

## 📋 Overview

텍스트를 벡터로 변환하는 다양한 임베딩 모델을 학습합니다. 의미 기반 검색을 가능하게 하는 RAG의 핵심 기술입니다.

## 🎯 Prerequisites

- `6-2_Chunking` 완료
- 벡터 및 임베딩 기본 개념 이해
- 각 제공자의 API 키 (해당하는 경우)

## 📚 Contents

| 노트북 | 주제 | 난이도 | 주요 내용 |
|--------|------|--------|-----------|
| `01-OpenAI-and-Cache-Embeddings.ipynb` | OpenAI + 캐싱 | ⭐ | Production 필수, 비용 절감 |
| `02-HuggingFace-Embeddings.ipynb` | 오픈소스 모델 | ⭐⭐ | 무료, 다양한 선택지 |
| `03-Korean-Embeddings.ipynb` | 한국어 특화 | ⭐ | Upstage Solar |
| `04-Local-Embeddings.ipynb` | 로컬 실행 | ⭐⭐ | 프라이버시, 완전 무료 |

## 🔑 Key Concepts

### 1. 임베딩의 기본 원리

- **벡터 표현**: 텍스트 → 고차원 벡터 (예: 1536차원)
- **의미 유사도**: 비슷한 의미 = 가까운 벡터
- **코사인 유사도**: 벡터 간 유사도 측정 방법

### 2. 임베딩 모델의 특성

- **차원(Dimensionality)**: 512, 768, 1024, 1536 등
- **컨텍스트 길이**: 입력 가능한 최대 토큰 수
- **언어 지원**: 영어 전용 vs 다국어

### 3. 최적화 전략

- **캐싱**: 동일 텍스트 재임베딩 방지
- **배치 처리**: 여러 텍스트 일괄 처리
- **로컬 vs 클라우드**: 비용/성능 트레이드오프

## 🛤️ Learning Path

```
1. OpenAI + Cache (01) ← 시작점: Production 기본
   ↓
2. HuggingFace (02) ← 무료 옵션 탐색
   ↓
3. Korean (03) ← 한국어 프로젝트
   ↓
4. Local (04) ← 프라이버시/비용 최적화
```

## 💡 Quick Start Examples

### OpenAI Embeddings + 캐싱 (권장)

```python
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

# 기본 임베딩
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 캐싱 추가 (비용 절감!)
store = LocalFileStore("./cache/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings,
    document_embedding_cache=store,
    namespace="openai-embeddings"
)

# 사용
vectors = cached_embeddings.embed_documents(["문서1", "문서2"])
```

### HuggingFace Local (무료)

```python
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},  # or "cuda"
    encode_kwargs={"normalize_embeddings": True},
)
```

### Upstage Korean (한국어)

```python
from langchain_upstage import UpstageEmbeddings

# Query와 Passage 분리
query_embeddings = UpstageEmbeddings(
    model="solar-embedding-1-large-query"
)
passage_embeddings = UpstageEmbeddings(
    model="solar-embedding-1-large-passage"
)
```

### Ollama (로컬, 쉬움)

```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

## 🎯 임베딩 모델 선택 가이드

### 용도별 추천

| 용도 | 추천 모델 | 이유 |
|------|----------|------|
| **Production (영어)** | OpenAI text-embedding-3-small | 안정적, 비용 효율적 |
| **Production (한국어)** | Upstage Solar | 한국어 최적화 |
| **최고 품질** | OpenAI text-embedding-3-large | 최고 성능 |
| **무료 (GPU)** | HuggingFace BGE-M3 | 고품질, 오픈소스 |
| **무료 (CPU)** | Ollama + nomic-embed-text | 쉬운 설치 |
| **프라이버시** | 로컬 옵션 (02, 04) | 데이터 외부 전송 없음 |

### 비용 비교

| 모델 | 초기 비용 | 1M 토큰 비용 | 특징 |
|------|----------|-------------|------|
| OpenAI small | $0 | $0.02 | 저렴, 고품질 |
| OpenAI large | $0 | $0.13 | 최고 품질 |
| Upstage | $0 | 가격 문의 | 한국어 특화 |
| **로컬 (HF, Ollama)** | **시간** | **$0** | **완전 무료** |

### 성능 vs 비용

```
고비용, 고성능
    ↑
    │ OpenAI text-embedding-3-large
    │ Upstage Solar
    │ OpenAI text-embedding-3-small
    │ HuggingFace (GPU)
    │ Ollama
    │ HuggingFace (CPU)
    ↓
저비용, 저성능
```

## ⚙️ Performance Optimization

### 1. 캐싱 (필수!)

```python
# Production 환경에서는 반드시 캐싱 사용
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

store = LocalFileStore("./cache/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=OpenAIEmbeddings(),
    document_embedding_cache=store,
    namespace="model-name"
)
```

**효과**:
- 속도: 10-100배 빠름
- 비용: 재실행 시 $0

### 2. 배치 처리

```python
# Bad: 하나씩
for text in texts:
    vector = embeddings.embed_query(text)  # N번 API 호출

# Good: 배치
vectors = embeddings.embed_documents(texts)  # 1번 API 호출
```

### 3. 차원 축소

```python
# 저장 공간 절약 (약간의 정확도 감소)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024  # 1536 → 1024
)
```

## 🔬 실습 예제: RAG에서의 임베딩 활용

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

# 1. 문서 로드
loader = TextLoader("document.txt")
documents = loader.load()

# 2. 문서 분할
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 3. 캐싱 임베딩 설정
store = LocalFileStore("./cache/")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    OpenAIEmbeddings(model="text-embedding-3-small"),
    store,
    namespace="my-embeddings"
)

# 4. 벡터 DB 생성
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. 검색
results = vectorstore.similarity_search("검색 질문", k=3)
```

## 🎓 학습 순서 권장

### 초급 (필수)

1. **01-OpenAI-and-Cache-Embeddings.ipynb**
   - OpenAI 임베딩 기본
   - 캐싱으로 비용 절감
   - Production 필수 지식

### 중급 (선택)

2. **02-HuggingFace-Embeddings.ipynb**
   - 무료 오픈소스 옵션
   - BGE-M3 고급 기능

3. **03-Korean-Embeddings.ipynb**
   - 한국어 프로젝트 필수
   - Query/Passage 분리

### 고급 (선택)

4. **04-Local-Embeddings.ipynb**
   - 완전 로컬 실행
   - 프라이버시 중시 프로젝트

## 📊 한국어 모델 성능 비교

한국어 검색 성능 (참고):

| 모델 | NDCG@10 | Recall@10 | 특징 |
|------|---------|-----------|------|
| Upstage Solar | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 한국어 최고 |
| BGE-M3 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 다국어 우수 |
| multilingual-e5 | ⭐⭐⭐ | ⭐⭐⭐ | 무료 옵션 |
| OpenAI (다국어) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 영어 최적화 |

## 🚨 Common Issues

### 1. API 키 오류

```python
# .env 파일 확인
OPENAI_API_KEY=sk-...
UPSTAGE_API_KEY=up-...
HUGGINGFACEHUB_API_TOKEN=hf_...
```

### 2. 캐시 문제

```python
# 캐시 디렉토리 확인
import os
os.makedirs("./cache/", exist_ok=True)

# 캐시 초기화 (필요시)
import shutil
shutil.rmtree("./cache/", ignore_errors=True)
```

### 3. 로컬 모델 실행 오류

```python
# GPU 메모리 부족 시 CPU 사용
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}  # GPU → CPU
)
```

## 🔗 Related Sections

- **Previous**: `6-2_Chunking` - 텍스트 분할
- **Next**: `6-4_VectorStore` - 벡터 저장소
- **Related**: `6-7_RAG Process` - 전체 RAG 시스템

## 📖 Additional Resources

### 공식 문서
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- [Upstage Solar](https://console.upstage.ai/)
- [Ollama](https://ollama.com/)

### 벤치마크
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Kor-IR (한국어 임베딩)](https://github.com/teddylee777/Kor-IR)

### 추가 학습
- [임베딩 이론](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)
- [벡터 검색 최적화](https://www.pinecone.io/learn/vector-search/)

## 💪 실전 프로젝트 아이디어

1. **Q&A 챗봇**: 문서 기반 질문 답변 시스템
2. **의미 검색 엔진**: 키워드가 아닌 의미로 검색
3. **문서 클러스터링**: 유사 문서 자동 그룹화
4. **중복 탐지**: 유사 텍스트 찾기
5. **추천 시스템**: 콘텐츠 유사도 기반 추천

---

## 📝 작성 및 검수 정보

- **작성 일자**: 2025-11-21
- **LangChain 버전**: 1.0.x
- **검수 태그**: 모든 노트북에 `@TAG-REVIEW-POINT` 포함
- **예제 데이터**: 레퍼런스와 차별화된 독창적 예제 사용
- **규칙 준수**: `langchain_teddynote` 제거, 최신 API 사용


