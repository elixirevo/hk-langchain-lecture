# 6-4_VectorStore: 벡터 데이터베이스

임베딩 벡터를 효율적으로 저장하고 검색하는 벡터 데이터베이스를 학습합니다. RAG 시스템의 핵심 인프라로, 대규모 문서 검색을 가능하게 합니다.

## 📚 노트북 목록

| 번호 | 노트북 | VectorStore | Type | Best For |
|------|--------|-------------|------|----------|
| 01 | `01-Chroma.ipynb` | ChromaDB | Local/Persistent | 개발, 소규모 |
| 02 | `02-FAISS.ipynb` | FAISS (Meta) | In-memory | 고속 검색, 중규모 |
| 03 | `03-Pinecone.ipynb` | Pinecone | Cloud | Production, 관리형 |
| 04 | `04-Milvus.ipynb` | **Milvus** | Self-hosted | 대규모, 확장성 |

## 🎯 학습 경로

```
1. 로컬 개발 (01)
   └→ Chroma ← 시작점
   ↓
2. 성능 테스트 (02)
   └→ FAISS ← 고속 검색
   ↓
3. 프로덕션 선택
   ├→ Pinecone (03) ← 관리형 클라우드
   └→ Milvus (04) ← 자체 호스팅
```

## 🔑 VectorStore 비교

### 상세 비교표

| 특징 | Chroma | FAISS | Milvus | Pinecone |
|------|--------|-------|--------|----------|
| **설치** | pip install | pip install | Docker/pip | API 가입 |
| **저장** | Persistent | In-memory + save | Persistent | Cloud |
| **확장성** | 소~중규모 | 중규모 | **대규모** | 무제한 |
| **최대 벡터** | ~1M | ~10M | **수십억** | 무제한 |
| **분산 처리** | ❌ | ❌ | **✅** | ✅ |
| **GPU 지원** | ❌ | 제한적 | **✅** | ✅ |
| **비용** | 무료 | 무료 | 무료 | 유료 |
| **관리 부담** | 낮음 | 낮음 | 중간 | 없음 |
| **메타데이터 필터링** | 풍부 | 제한적 | **풍부** | 풍부 |

### 사용 시나리오별 추천

| 시나리오 | 추천 VectorDB | 이유 |
|----------|--------------|------|
| **로컬 개발/프로토타입** | Chroma | 설치 간편, 영구 저장 |
| **성능 벤치마크** | FAISS | 최고 속도 |
| **소규모 프로덕션** (<100만) | Chroma + Docker | 간단한 배포 |
| **중규모 프로덕션** (100만~1천만) | FAISS 또는 Milvus | 성능 vs 확장성 |
| **대규모 프로덕션** (1천만+) | **Milvus** | 확장성, 분산 처리 |
| **초대규모** (수억~수십억) | **Milvus 클러스터** | 분산, GPU 가속 |
| **관리 부담 최소화** | Pinecone | 완전 관리형 |
| **비용 민감 + 대규모** | **Milvus** | 오픈소스, 자체 호스팅 |

## 💡 주요 기능 비교

### 1. 검색 알고리즘

**Chroma**:
- HNSW (기본)
- 코사인 유사도

**FAISS**:
- Flat, IVF, HNSW, PQ 등
- 다양한 인덱스 선택 가능

**Milvus**:
- FLAT, IVF_FLAT, IVF_SQ8, HNSW, DiskANN
- GPU 가속 인덱스 (IVF_PQ_GPU 등)

**Pinecone**:
- 자동 최적화
- 인덱스 선택 불필요

### 2. 검색 방식

**공통**:
- Similarity Search (유사도 검색)
- MMR (Maximum Marginal Relevance)

**Milvus 추가**:
- Range Search (거리 범위 검색)
- Hybrid Search (Dense + Sparse)
- Time Travel (과거 시점 검색)

## 🚀 빠른 시작

### Chroma (로컬 개발)

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

results = vectorstore.similarity_search("query", k=4)
```

### FAISS (고속 검색)

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# 디스크 저장/로드
vectorstore.save_local("faiss_index")
loaded_vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings())
```

### Milvus (대규모 확장)

```python
from langchain_community.vectorstores import Milvus

# Milvus Lite (개발)
vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    collection_name="my_collection",
    connection_args={"uri": "./milvus_demo.db"}
)

# Milvus Server (프로덕션)
vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    collection_name="my_collection",
    connection_args={"host": "localhost", "port": "19530"}
)

# 메타데이터 필터링
results = vectorstore.similarity_search(
    "query",
    k=4,
    expr="year >= 2020"  # Milvus 필터 표현식
)
```

### Pinecone (관리형)

```python
from langchain_community.vectorstores import Pinecone
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENV")

vectorstore = Pinecone.from_documents(
    docs,
    OpenAIEmbeddings(),
    index_name="langchain-index"
)
```

## ⚙️ 설치 및 설정

### Chroma

```bash
pip install chromadb
```

### FAISS

```bash
pip install faiss-cpu  # CPU 버전
# 또는
pip install faiss-gpu  # GPU 버전
```

### Milvus

**Milvus Lite (간편 버전)**:
```bash
pip install milvus
```

**Milvus Standalone (Docker)**:
```bash
# docker-compose.yml 다운로드
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 실행
docker-compose up -d
```

**Milvus Client**:
```bash
pip install pymilvus
```

### Pinecone

```bash
pip install pinecone-client
```

API 키 발급: https://app.pinecone.io/

## 📊 성능 벤치마크 (참고)

| VectorDB | 100K 벡터 검색 | 1M 벡터 검색 | 10M 벡터 검색 |
|----------|---------------|-------------|--------------|
| FAISS (Flat) | ~1ms | ~10ms | ~100ms |
| FAISS (IVF) | ~1ms | ~5ms | ~20ms |
| Milvus (HNSW) | ~2ms | ~5ms | ~10ms |
| Chroma | ~5ms | ~30ms | N/A |

*실제 성능은 하드웨어, 벡터 차원, 인덱스 설정에 따라 다릅니다.*

## 🎓 학습 팁

1. **순서대로 학습**: 01 → 02 → 03 → 04
2. **실습 중심**: 각 VectorDB를 직접 실행
3. **비교 분석**: 동일 데이터로 성능 비교
4. **메타데이터 활용**: 필터링 기능 테스트
5. **RAG 통합**: Retriever로 변환하여 RAG 구축

## 💰 비용 비교 (대략적)

### 무료 옵션
- **Chroma**: 무료 (자체 호스팅)
- **FAISS**: 무료 (자체 호스팅)
- **Milvus**: 무료 (자체 호스팅, 인프라 비용만)

### 유료 옵션
- **Pinecone**:
  - Free Tier: 월 100만 쿼리 무료
  - Serverless: 사용량 기반
  - Dedicated: $70+/월
- **Zilliz Cloud** (Milvus 관리형):
  - Free Tier 있음
  - 사용량 기반 과금

## 🔗 관련 섹션

- **이전**: `6-3_Embeddings` - 임베딩
- **다음**: `6-5_Retriever` - 검색기
- **관련**: `6-7_RAG Process` - RAG 시스템

## 📖 추가 자료

- [ChromaDB 공식 문서](https://docs.trychroma.com/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Milvus 공식 문서](https://milvus.io/docs)
- [Pinecone 공식 문서](https://docs.pinecone.io/)
- [Vector Database Comparison](https://lakefs.io/blog/12-vector-databases-2023/)

---

**추천 선택 플로우차트**:

```
시작
 ↓
로컬 개발? ─YES→ Chroma (01)
 ↓ NO
 ↓
고속 검색 필요? ─YES→ FAISS (02)
 ↓ NO
 ↓
자체 호스팅 가능? ─YES→ Milvus (04)
 ↓ NO
 ↓
관리 부담 싫음? ─YES→ Pinecone (03)
```

