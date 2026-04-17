import json

def create_milvus_notebook():
    """04-Milvus.ipynb 생성"""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Milvus VectorStore\n",
                "\n",
                "Milvus는 대규모 벡터 데이터를 위한 오픈소스 벡터 데이터베이스입니다. 수십억 개의 벡터를 밀리초 단위로 검색할 수 있는 고성능 시스템입니다.\n",
                "\n",
                "**주요 특징:**\n",
                "- 대규모 확장성 (수십억 개 벡터 지원)\n",
                "- 다양한 인덱스 알고리즘 (HNSW, IVF, DiskANN 등)\n",
                "- 분산 아키텍처 (클러스터 구성 가능)\n",
                "- 풍부한 메타데이터 필터링\n",
                "- GPU 가속 지원\n",
                "\n",
                "**언제 Milvus를 사용할까?**\n",
                "- ✅ **중대규모 프로덕션**: 수백만~수십억 벡터\n",
                "- ✅ **온프레미스 배포**: 자체 인프라에서 운영\n",
                "- ✅ **고성능 요구**: 밀리초 단위 응답 필요\n",
                "- ✅ **커스터마이징**: 세밀한 성능 튜닝 가능\n",
                "- ✅ **비용 효율**: 클라우드 비용 절감 (오픈소스)\n",
                "\n",
                "**Chroma/FAISS와의 차이:**\n",
                "- Chroma: 소규모, 개발용\n",
                "- FAISS: 중규모, 단일 서버\n",
                "- **Milvus**: 대규모, 분산 시스템, 프로덕션급\n",
                "\n",
                "**참고 문서:**\n",
                "- [Milvus 공식 문서](https://milvus.io/docs)\n",
                "- [LangChain Milvus 통합](https://python.langchain.com/docs/integrations/vectorstores/milvus/)\n",
                "- [Milvus vs 다른 VectorDB 비교](https://milvus.io/docs/comparison.md)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 설치 및 실행\n",
                "\n",
                "### 1. Milvus Standalone (Docker)\n",
                "\n",
                "가장 간단한 방법은 Docker Compose로 실행하는 것입니다.\n",
                "\n",
                "```bash\n",
                "# Milvus standalone 다운로드\n",
                "wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml\n",
                "\n",
                "# 실행\n",
                "docker-compose up -d\n",
                "\n",
                "# 상태 확인\n",
                "docker-compose ps\n",
                "```\n",
                "\n",
                "### 2. Python 클라이언트 설치\n",
                "\n",
                "```bash\n",
                "pip install pymilvus\n",
                "```\n",
                "\n",
                "**기본 포트:** `http://localhost:19530`\n",
                "\n",
                "### 3. Milvus Lite (간편 버전)\n",
                "\n",
                "Docker 없이 Python으로만 실행 가능한 경량 버전:\n",
                "\n",
                "```bash\n",
                "pip install milvus\n",
                "```\n",
                "\n",
                "이 노트북에서는 **Milvus Lite**를 사용합니다 (설치 간편)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 환경 설정"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "'''\n",
                "@TAG-SETUP\n",
                "- 구현 목적: 환경 변수 로드 및 기본 설정\n",
                "'''\n",
                "from dotenv import load_dotenv\n",
                "import os\n",
                "\n",
                "# API 키 정보 로드\n",
                "load_dotenv()\n",
                "\n",
                "print(\"✅ 환경 설정 완료\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. 데이터 준비\n",
                "\n",
                "벡터스토어에 저장할 문서를 준비합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from langchain_community.document_loaders import TextLoader\n",
                "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
                "\n",
                "# 텍스트 분할 설정\n",
                "text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)\n",
                "\n",
                "# 문서 로더 생성\n",
                "loader_nlp = TextLoader(\"data/nlp-keywords.txt\")\n",
                "loader_finance = TextLoader(\"data/finance-keywords.txt\")\n",
                "\n",
                "# 문서 로드 및 분할\n",
                "docs_nlp = loader_nlp.load_and_split(text_splitter)\n",
                "docs_finance = loader_finance.load_and_split(text_splitter)\n",
                "\n",
                "# 문서 개수 확인\n",
                "print(f\"NLP 문서: {len(docs_nlp)}개\")\n",
                "print(f\"금융 문서: {len(docs_finance)}개\")\n",
                "print(f\"\\n[NLP 문서 샘플]\\n{docs_nlp[0].page_content[:200]}...\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Milvus VectorStore 생성\n",
                "\n",
                "### 2.1 from_documents로 생성\n",
                "\n",
                "Milvus는 자동으로 컬렉션(테이블)을 생성하고 벡터를 저장합니다.\n",
                "\n",
                "**주요 매개변수:**\n",
                "- `documents`: Document 객체 리스트\n",
                "- `embedding`: 임베딩 모델\n",
                "- `collection_name`: 컬렉션 이름 (테이블 이름)\n",
                "- `connection_args`: 연결 정보 (URI 등)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "'''\n",
                "@TAG-MILVUS-CREATION\n",
                "- 구현 목적: Milvus VectorStore 생성 및 문서 임베딩\n",
                "- 구현 과정:\n",
                "  1. OpenAIEmbeddings 모델 초기화\n",
                "  2. Milvus.from_documents()로 벡터스토어 생성\n",
                "  3. Milvus Lite 사용 (connection_args 미지정)\n",
                "- 구현 결과: 자동으로 컬렉션 생성 및 벡터 저장\n",
                "'''\n",
                "\n",
                "from langchain_community.vectorstores import Milvus\n",
                "from langchain_openai import OpenAIEmbeddings\n",
                "\n",
                "# 임베딩 모델\n",
                "embeddings = OpenAIEmbeddings(model=\"text-embedding-3-small\")\n",
                "\n",
                "# Milvus VectorStore 생성 (NLP 문서)\n",
                "vectorstore_nlp = Milvus.from_documents(\n",
                "    documents=docs_nlp,\n",
                "    embedding=embeddings,\n",
                "    collection_name=\"nlp_collection\",\n",
                "    connection_args={\"uri\": \"./milvus_demo.db\"}  # Milvus Lite: 로컬 파일\n",
                ")\n",
                "\n",
                "print(\"✅ NLP VectorStore 생성 완료\")\n",
                "print(f\"- 컬렉션: nlp_collection\")\n",
                "print(f\"- 문서 수: {len(docs_nlp)}개\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.2 기존 컬렉션 연결\n",
                "\n",
                "이미 생성된 컬렉션에 연결할 수 있습니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 기존 컬렉션에 연결\n",
                "vectorstore_existing = Milvus(\n",
                "    embedding_function=embeddings,\n",
                "    collection_name=\"nlp_collection\",\n",
                "    connection_args={\"uri\": \"./milvus_demo.db\"}\n",
                ")\n",
                "\n",
                "print(\"✅ 기존 컬렉션 연결 완료\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. 유사도 검색\n",
                "\n",
                "### 3.1 기본 유사도 검색"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "'''\n",
                "@TAG-SIMILARITY-SEARCH\n",
                "- 구현 목적: 쿼리와 유사한 문서 검색\n",
                "- k=3: 상위 3개 결과 반환\n",
                "'''\n",
                "\n",
                "query = \"자연어 처리와 딥러닝\"\n",
                "results = vectorstore_nlp.similarity_search(query, k=3)\n",
                "\n",
                "print(f\"검색어: '{query}'\")\n",
                "print(\"=\" * 60)\n",
                "for i, doc in enumerate(results, 1):\n",
                "    print(f\"\\n[결과 {i}]\")\n",
                "    print(doc.page_content[:200])\n",
                "    print(f\"메타데이터: {doc.metadata}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.2 점수 포함 검색\n",
                "\n",
                "유사도 점수와 함께 결과를 반환합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "results_with_scores = vectorstore_nlp.similarity_search_with_score(query, k=3)\n",
                "\n",
                "print(f\"검색어: '{query}'\")\n",
                "print(\"=\" * 60)\n",
                "for i, (doc, score) in enumerate(results_with_scores, 1):\n",
                "    print(f\"\\n[결과 {i}] 유사도: {score:.4f}\")\n",
                "    print(doc.page_content[:150])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.3 MMR 검색 (다양성 고려)\n",
                "\n",
                "Maximum Marginal Relevance: 유사도와 다양성을 균형있게 고려합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "'''\n",
                "@TAG-MMR-SEARCH\n",
                "- 구현 목적: 관련성과 다양성을 모두 고려한 검색\n",
                "- fetch_k: 먼저 20개 후보를 가져옴\n",
                "- k: 최종 4개를 선택 (중복 제거)\n",
                "'''\n",
                "\n",
                "mmr_results = vectorstore_nlp.max_marginal_relevance_search(\n",
                "    query,\n",
                "    k=4,\n",
                "    fetch_k=20  # 20개 후보에서 4개 선택\n",
                ")\n",
                "\n",
                "print(f\"MMR 검색어: '{query}'\")\n",
                "print(\"=\" * 60)\n",
                "for i, doc in enumerate(mmr_results, 1):\n",
                "    print(f\"\\n[MMR 결과 {i}]\")\n",
                "    print(doc.page_content[:150])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. 메타데이터 필터링\n",
                "\n",
                "Milvus는 강력한 메타데이터 필터링을 지원합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 메타데이터와 함께 문서 추가\n",
                "from langchain.schema import Document\n",
                "\n",
                "docs_with_metadata = [\n",
                "    Document(\n",
                "        page_content=\"Transformer는 2017년에 발표된 딥러닝 모델입니다.\",\n",
                "        metadata={\"year\": 2017, \"category\": \"architecture\", \"importance\": \"high\"}\n",
                "    ),\n",
                "    Document(\n",
                "        page_content=\"BERT는 2018년 Google이 발표한 사전학습 모델입니다.\",\n",
                "        metadata={\"year\": 2018, \"category\": \"pretrained\", \"importance\": \"high\"}\n",
                "    ),\n",
                "    Document(\n",
                "        page_content=\"Word2Vec은 2013년 단어 임베딩 기법입니다.\",\n",
                "        metadata={\"year\": 2013, \"category\": \"embedding\", \"importance\": \"medium\"}\n",
                "    ),\n",
                "]\n",
                "\n",
                "# 새 컬렉션에 추가\n",
                "vectorstore_filtered = Milvus.from_documents(\n",
                "    documents=docs_with_metadata,\n",
                "    embedding=embeddings,\n",
                "    collection_name=\"nlp_with_metadata\",\n",
                "    connection_args={\"uri\": \"./milvus_demo.db\"}\n",
                ")\n",
                "\n",
                "print(\"✅ 메타데이터 포함 문서 추가 완료\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "'''\n",
                "@TAG-METADATA-FILTER\n",
                "- 구현 목적: 메타데이터 조건으로 검색 범위 제한\n",
                "- expr: Milvus 표현식 (SQL과 유사)\n",
                "'''\n",
                "\n",
                "# 2017년 이후 문서만 검색\n",
                "filtered_results = vectorstore_filtered.similarity_search(\n",
                "    \"딥러닝 모델\",\n",
                "    k=3,\n",
                "    expr=\"year >= 2017\"  # Milvus 필터 표현식\n",
                ")\n",
                "\n",
                "print(\"필터: year >= 2017\")\n",
                "print(\"=\" * 60)\n",
                "for i, doc in enumerate(filtered_results, 1):\n",
                "    print(f\"\\n[결과 {i}]\")\n",
                "    print(f\"내용: {doc.page_content}\")\n",
                "    print(f\"연도: {doc.metadata['year']}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. 문서 추가 및 삭제\n",
                "\n",
                "### 5.1 문서 추가"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 새 문서 추가\n",
                "new_docs = [\n",
                "    Document(\n",
                "        page_content=\"GPT-4는 2023년에 발표된 대규모 언어 모델입니다.\",\n",
                "        metadata={\"year\": 2023, \"category\": \"llm\"}\n",
                "    )\n",
                "]\n",
                "\n",
                "# 기존 컬렉션에 추가\n",
                "ids = vectorstore_filtered.add_documents(new_docs)\n",
                "\n",
                "print(f\"✅ 문서 추가 완료\")\n",
                "print(f\"추가된 문서 ID: {ids}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 추가 확인\n",
                "results = vectorstore_filtered.similarity_search(\"대규모 언어 모델\", k=2)\n",
                "print(\"검색 결과:\")\n",
                "for doc in results:\n",
                "    print(f\"- {doc.page_content}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.2 문서 삭제"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ID로 문서 삭제\n",
                "if ids:\n",
                "    vectorstore_filtered.delete(ids)\n",
                "    print(f\"✅ 문서 삭제 완료: {ids}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Retriever로 변환\n",
                "\n",
                "RAG 체인에서 사용하기 위해 Retriever로 변환합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "'''\n",
                "@TAG-RETRIEVER\n",
                "- 구현 목적: VectorStore를 Retriever로 변환하여 RAG 체인에서 사용\n",
                "'''\n",
                "\n",
                "# Retriever 생성\n",
                "retriever = vectorstore_nlp.as_retriever(\n",
                "    search_type=\"similarity\",\n",
                "    search_kwargs={\"k\": 4}\n",
                ")\n",
                "\n",
                "# Retriever 사용\n",
                "retrieved_docs = retriever.invoke(\"자연어 처리\")\n",
                "\n",
                "print(f\"✅ Retriever 생성 완료\")\n",
                "print(f\"검색된 문서 수: {len(retrieved_docs)}개\")\n",
                "print(f\"\\n첫 번째 문서:\\n{retrieved_docs[0].page_content[:150]}...\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. RAG 체인 구축\n",
                "\n",
                "Milvus를 사용한 완전한 RAG 시스템을 구축합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from langchain_core.prompts import PromptTemplate\n",
                "from langchain_openai import ChatOpenAI\n",
                "from langchain_core.output_parsers import StrOutputParser\n",
                "from langchain_core.runnables import RunnablePassthrough\n",
                "\n",
                "# 프롬프트\n",
                "prompt = PromptTemplate.from_template(\n",
                "    \"\"\"문맥을 참고하여 질문에 답하세요.\n",
                "\n",
                "문맥:\n",
                "{context}\n",
                "\n",
                "질문: {question}\n",
                "\n",
                "답변:\"\"\"\n",
                ")\n",
                "\n",
                "# LLM\n",
                "llm = ChatOpenAI(model=\"gpt-4o-mini\", temperature=0)\n",
                "\n",
                "# RAG 체인\n",
                "rag_chain = (\n",
                "    {\"context\": retriever, \"question\": RunnablePassthrough()}\n",
                "    | prompt\n",
                "    | llm\n",
                "    | StrOutputParser()\n",
                ")\n",
                "\n",
                "print(\"✅ RAG 체인 구축 완료\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# RAG 실행\n",
                "question = \"트랜스포머 아키텍처에 대해 설명해주세요.\"\n",
                "\n",
                "print(f\"질문: {question}\")\n",
                "print(\"=\" * 60)\n",
                "answer = rag_chain.invoke(question)\n",
                "print(f\"답변:\\n{answer}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Milvus 연결 설정 (Docker 사용 시)\n",
                "\n",
                "Docker로 Milvus를 실행하는 경우 연결 방법입니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Docker Milvus 연결 예시 (실제 실행하지 않음)\n",
                "'''\n",
                "# Docker로 Milvus 실행 시\n",
                "vectorstore_docker = Milvus.from_documents(\n",
                "    documents=docs_nlp,\n",
                "    embedding=embeddings,\n",
                "    collection_name=\"nlp_docker\",\n",
                "    connection_args={\n",
                "        \"host\": \"localhost\",  # Milvus 서버 주소\n",
                "        \"port\": \"19530\"  # 기본 포트\n",
                "    }\n",
                ")\n",
                "'''\n",
                "\n",
                "print(\"💡 Docker Milvus 연결 방법을 주석으로 확인하세요.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. 컬렉션 정보 확인"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Milvus 컬렉션 통계 (pymilvus 직접 사용)\n",
                "from pymilvus import connections, Collection\n",
                "\n",
                "# 연결\n",
                "connections.connect(uri=\"./milvus_demo.db\")\n",
                "\n",
                "# 컬렉션 가져오기\n",
                "collection = Collection(\"nlp_collection\")\n",
                "\n",
                "# 통계\n",
                "print(f\"컬렉션명: {collection.name}\")\n",
                "print(f\"저장된 벡터 수: {collection.num_entities}개\")\n",
                "print(f\"스키마: {collection.schema}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 💡 핵심 정리\n",
                "\n",
                "### Milvus의 장점\n",
                "\n",
                "1. **확장성**: 수십억 개 벡터 처리 가능\n",
                "2. **성능**: 밀리초 단위 검색 응답\n",
                "3. **유연성**: 다양한 인덱스 알고리즘 선택\n",
                "4. **오픈소스**: 무료, 커스터마이징 가능\n",
                "5. **분산 지원**: 클러스터 구성으로 고가용성\n",
                "\n",
                "### VectorDB 선택 가이드\n",
                "\n",
                "| 상황 | 추천 | 이유 |\n",
                "|------|------|------|\n",
                "| 개발/테스트 | Chroma | 설치 간편 |\n",
                "| 소규모 (< 100만) | FAISS | 빠르고 간단 |\n",
                "| 중대규모 (100만~10억) | **Milvus** | 성능 + 확장성 |\n",
                "| 초대규모 (10억+) | **Milvus 클러스터** | 분산 처리 |\n",
                "| 관리 부담 최소화 | Pinecone | 완전 관리형 |\n",
                "\n",
                "### Milvus vs 다른 VectorDB\n",
                "\n",
                "| 특징 | Chroma | FAISS | Milvus | Pinecone |\n",
                "|------|--------|-------|--------|----------|\n",
                "| 확장성 | 소규모 | 중규모 | **대규모** | 무제한 |\n",
                "| 설치 | 쉬움 | 쉬움 | 보통 | 가입만 |\n",
                "| 비용 | 무료 | 무료 | 무료 | 유료 |\n",
                "| 분산 | ❌ | ❌ | ✅ | ✅ |\n",
                "| GPU 지원 | ❌ | 제한적 | ✅ | ✅ |\n",
                "\n",
                "### 실전 활용 팁\n",
                "\n",
                "**개발 단계:**\n",
                "```python\n",
                "# Milvus Lite (파일 기반)\n",
                "connection_args={\"uri\": \"./milvus_dev.db\"}\n",
                "```\n",
                "\n",
                "**프로덕션:**\n",
                "```python\n",
                "# Docker Milvus\n",
                "connection_args={\"host\": \"milvus-server\", \"port\": \"19530\"}\n",
                "```\n",
                "\n",
                "**대규모:**\n",
                "```python\n",
                "# Milvus 클러스터\n",
                "connection_args={\n",
                "    \"host\": \"milvus-cluster-lb\",\n",
                "    \"port\": \"19530\",\n",
                "    \"secure\": True\n",
                "}\n",
                "```\n",
                "\n",
                "### 다음 단계\n",
                "\n",
                "- **인덱스 최적화**: HNSW, IVF 등 인덱스 타입 선택\n",
                "- **파티셔닝**: 대규모 데이터를 파티션으로 분할\n",
                "- **모니터링**: Attu (Milvus GUI) 사용\n",
                "- **백업/복구**: 데이터 백업 전략 수립"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 정리 (Cleanup)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 연결 종료\n",
                "connections.disconnect(\"default\")\n",
                "print(\"✅ Milvus 연결 종료\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "\n",
                "## @TAG-REVIEW-POINT\n",
                "\n",
                "### 주제\n",
                "Milvus VectorStore - 대규모 확장 가능한 오픈소스 벡터 데이터베이스\n",
                "\n",
                "### 구현 내용\n",
                "\n",
                "#### 1. Milvus 소개\n",
                "- 대규모 프로덕션급 벡터 DB\n",
                "- Docker 또는 Milvus Lite로 실행\n",
                "- 분산 아키텍처 지원\n",
                "\n",
                "#### 2. 핵심 기능\n",
                "- **VectorStore 생성**: from_documents(), 자동 컬렉션 생성\n",
                "- **유사도 검색**: similarity_search(), similarity_search_with_score()\n",
                "- **MMR 검색**: max_marginal_relevance_search() - 다양성 고려\n",
                "- **메타데이터 필터링**: expr 표현식으로 조건 검색\n",
                "- **CRUD**: add_documents(), delete()\n",
                "\n",
                "#### 3. RAG 통합\n",
                "- Retriever 변환\n",
                "- 완전한 RAG 체인 구축\n",
                "\n",
                "#### 4. 실전 활용\n",
                "- Milvus Lite (개발)\n",
                "- Docker (프로덕션)\n",
                "- 클러스터 (대규모)\n",
                "\n",
                "### 교재 작성 규칙 준수\n",
                "\n",
                "✅ 기존 VectorStore 노트북과 일관된 구조\n",
                "✅ 단계별 명확한 설명\n",
                "✅ 검수 태그 포함\n",
                "✅ 실행 가능한 예제 코드\n",
                "✅ VectorDB 비교 가이드 제공\n",
                "\n",
                "### 학습 목표 달성\n",
                "\n",
                "1. Milvus 설치 및 실행 방법\n",
                "2. 벡터 저장 및 검색\n",
                "3. 메타데이터 필터링\n",
                "4. RAG 시스템 통합\n",
                "5. 다른 VectorDB와 비교\n",
                "\n",
                "---"
            ]
        }
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('04-Milvus.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    
    print("✅ 04-Milvus.ipynb 생성 완료!")

if __name__ == "__main__":
    create_milvus_notebook()

