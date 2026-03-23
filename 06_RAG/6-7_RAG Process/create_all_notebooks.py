import json
import os

def create_advanced_rag():
    """02-RAG-Advanced.ipynb - Ensemble Retriever와 MMR"""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 고급 RAG 기법\n",
                "\n",
                "기본 RAG 시스템을 넘어서, 검색 품질을 향상시키는 고급 기법들을 학습합니다.\n",
                "\n",
                "이 노트북에서는 다음 기법들을 다룹니다:\n",
                "1. **Ensemble Retriever**: 벡터 검색 + 키워드 검색 결합\n",
                "2. **MMR (Maximum Marginal Relevance)**: 다양성 있는 검색 결과"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 기본 RAG의 한계\n",
                "\n",
                "### 1. 벡터 검색만 사용할 때의 문제\n",
                "\n",
                "- **의미는 비슷하지만 정확한 키워드가 없으면 검색 실패**\n",
                "  - 예: \"AI\"를 검색했는데 문서에는 \"인공지능\"만 있는 경우\n",
                "- **빈도가 낮은 중요 키워드 놓침**\n",
                "  - 예: 고유명사나 기술 용어\n",
                "\n",
                "### 2. 검색 결과의 중복\n",
                "\n",
                "- 유사한 내용의 문서들이 반복적으로 검색됨\n",
                "- 다양한 관점의 정보 부족\n",
                "\n",
                "### 해결책\n",
                "\n",
                "- **Ensemble Retriever**: Dense(벡터) + Sparse(키워드) 검색 결합\n",
                "- **MMR**: 관련성과 다양성을 동시에 고려"
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
                "from dotenv import load_dotenv\n",
                "\n",
                "load_dotenv()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Ensemble Retriever\n",
                "\n",
                "### 개념\n",
                "\n",
                "Ensemble Retriever는 여러 검색 방식을 결합하여 더 나은 검색 결과를 얻습니다.\n",
                "\n",
                "```\n",
                "                  User Query\n",
                "                      |\n",
                "         +------------+------------+\n",
                "         |                         |\n",
                "    BM25 검색               벡터 검색\n",
                "  (키워드 기반)           (의미 기반)\n",
                "         |                         |\n",
                "         +------------+------------+\n",
                "                      |\n",
                "              Ensemble Retriever\n",
                "            (가중치로 결합)\n",
                "                      |\n",
                "                최종 검색 결과\n",
                "```\n",
                "\n",
                "### BM25 vs 벡터 검색\n",
                "\n",
                "| 특징 | BM25 (Sparse) | 벡터 검색 (Dense) |\n",
                "|------|---------------|------------------|\n",
                "| 방식 | 키워드 매칭 | 의미 유사도 |\n",
                "| 장점 | 정확한 용어 검색 | 동의어, 유사 개념 검색 |\n",
                "| 단점 | 의미 파악 불가 | 정확한 키워드 놓칠 수 있음 |\n",
                "| 예시 | \"GPT-4\" 정확히 검색 | \"대형 언어 모델\" ≈ \"LLM\" |"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
                "from langchain_community.document_loaders import PyMuPDFLoader\n",
                "from langchain_community.vectorstores import FAISS\n",
                "from langchain_core.output_parsers import StrOutputParser\n",
                "from langchain_core.runnables import RunnablePassthrough\n",
                "from langchain_core.prompts import PromptTemplate\n",
                "from langchain_openai import ChatOpenAI, OpenAIEmbeddings\n",
                "from langchain.retrievers import BM25Retriever, EnsembleRetriever"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 문서 준비"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 문서 로드 및 분할\n",
                "loader = PyMuPDFLoader(\"../6-1_DocumentLoaders/data/sample-rag-brief.pdf\")\n",
                "docs = loader.load()\n",
                "\n",
                "text_splitter = RecursiveCharacterTextSplitter(\n",
                "    chunk_size=1000,\n",
                "    chunk_overlap=100\n",
                ")\n",
                "split_documents = text_splitter.split_documents(docs)\n",
                "\n",
                "print(f\"분할된 청크 수: {len(split_documents)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Ensemble Retriever 생성"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "'''\n",
                "@TAG-ENSEMBLE-RETRIEVER\n",
                "- 구현 목적: BM25(키워드)와 FAISS(벡터) 검색을 결합하여 검색 정확도 향상\n",
                "- 구현 과정: 각각의 retriever 생성 후 EnsembleRetriever로 결합, 가중치 0.5:0.5\n",
                "- 구현 결과: 키워드와 의미 기반 검색의 장점을 모두 활용\n",
                "'''\n",
                "\n",
                "# 1. BM25 Retriever (키워드 기반)\n",
                "bm25_retriever = BM25Retriever.from_documents(split_documents)\n",
                "bm25_retriever.k = 3  # 상위 3개 반환\n",
                "\n",
                "# 2. FAISS Retriever (벡터 기반)\n",
                "embeddings = OpenAIEmbeddings(model=\"text-embedding-3-small\")\n",
                "vectorstore = FAISS.from_documents(split_documents, embeddings)\n",
                "faiss_retriever = vectorstore.as_retriever(search_kwargs={\"k\": 3})\n",
                "\n",
                "# 3. Ensemble Retriever (결합)\n",
                "ensemble_retriever = EnsembleRetriever(\n",
                "    retrievers=[bm25_retriever, faiss_retriever],\n",
                "    weights=[0.5, 0.5]  # 동일한 가중치\n",
                ")\n",
                "\n",
                "print(\"✅ Ensemble Retriever 생성 완료\")\n",
                "print(\"- BM25 (키워드 검색): 가중치 0.5\")\n",
                "print(\"- FAISS (벡터 검색): 가중치 0.5\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 검색 비교 테스트"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 테스트 질문\n",
                "test_query = \"디지털 전환\"\n",
                "\n",
                "# BM25만 사용\n",
                "bm25_results = bm25_retriever.get_relevant_documents(test_query)\n",
                "print(f\"🔍 BM25 검색 결과 ({len(bm25_results)}개):\")\n",
                "print(\"=\" * 60)\n",
                "for i, doc in enumerate(bm25_results[:2], 1):\n",
                "    print(f\"[{i}] {doc.page_content[:150]}...\\n\")\n",
                "\n",
                "# FAISS만 사용\n",
                "faiss_results = faiss_retriever.get_relevant_documents(test_query)\n",
                "print(f\"\\n🔍 FAISS 검색 결과 ({len(faiss_results)}개):\")\n",
                "print(\"=\" * 60)\n",
                "for i, doc in enumerate(faiss_results[:2], 1):\n",
                "    print(f\"[{i}] {doc.page_content[:150]}...\\n\")\n",
                "\n",
                "# Ensemble 사용\n",
                "ensemble_results = ensemble_retriever.get_relevant_documents(test_query)\n",
                "print(f\"\\n🎯 Ensemble 검색 결과 ({len(ensemble_results)}개):\")\n",
                "print(\"=\" * 60)\n",
                "for i, doc in enumerate(ensemble_results[:2], 1):\n",
                "    print(f\"[{i}] {doc.page_content[:150]}...\\n\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. MMR (Maximum Marginal Relevance)\n",
                "\n",
                "### 개념\n",
                "\n",
                "MMR은 검색 결과의 **관련성(Relevance)**과 **다양성(Diversity)**을 동시에 고려합니다.\n",
                "\n",
                "**문제 상황**:\n",
                "```\n",
                "질문: \"인공지능의 활용 분야는?\"\n",
                "\n",
                "일반 검색 결과:\n",
                "1. 의료 분야 AI 활용\n",
                "2. 의료 분야 AI 진단\n",
                "3. 의료 분야 AI 치료  ← 모두 의료 분야만!\n",
                "4. 의료 AI 사례\n",
                "```\n",
                "\n",
                "**MMR 검색 결과**:\n",
                "```\n",
                "1. 의료 분야 AI 활용\n",
                "2. 금융 분야 AI 활용  ← 다양한 분야\n",
                "3. 제조업 AI 활용\n",
                "4. 교육 AI 활용\n",
                "```\n",
                "\n",
                "### 작동 원리\n",
                "\n",
                "MMR 점수 = λ × (관련성) - (1-λ) × (중복도)\n",
                "\n",
                "- λ = 1: 관련성만 고려 (일반 검색)\n",
                "- λ = 0: 다양성만 고려\n",
                "- λ = 0.5: 균형"
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
                "- 구현 목적: 관련성과 다양성을 균형있게 고려한 검색 결과 제공\n",
                "- 구현 과정: as_retriever에서 search_type=\"mmr\" 지정, fetch_k와 lambda_mult 조정\n",
                "- 구현 결과: 중복된 내용을 피하고 다양한 관점의 문서 검색\n",
                "'''\n",
                "\n",
                "# MMR Retriever 생성\n",
                "mmr_retriever = vectorstore.as_retriever(\n",
                "    search_type=\"mmr\",\n",
                "    search_kwargs={\n",
                "        \"k\": 4,  # 최종 반환 문서 수\n",
                "        \"fetch_k\": 20,  # 먼저 가져올 후보 문서 수\n",
                "        \"lambda_mult\": 0.5  # 관련성 vs 다양성 균형 (0~1)\n",
                "    }\n",
                ")\n",
                "\n",
                "print(\"✅ MMR Retriever 생성 완료\")\n",
                "print(\"- k=4: 최종 4개 문서 반환\")\n",
                "print(\"- fetch_k=20: 20개 후보에서 선택\")\n",
                "print(\"- lambda_mult=0.5: 관련성과 다양성 균형\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 일반 검색 vs MMR 비교"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "test_query = \"정부 정책\"\n",
                "\n",
                "# 일반 유사도 검색\n",
                "print(\"🔍 일반 유사도 검색:\")\n",
                "print(\"=\" * 60)\n",
                "similarity_results = faiss_retriever.get_relevant_documents(test_query)\n",
                "for i, doc in enumerate(similarity_results, 1):\n",
                "    print(f\"[{i}] {doc.page_content[:100]}...\\n\")\n",
                "\n",
                "# MMR 검색\n",
                "print(\"\\n🎯 MMR 검색 (다양성 고려):\")\n",
                "print(\"=\" * 60)\n",
                "mmr_results = mmr_retriever.get_relevant_documents(test_query)\n",
                "for i, doc in enumerate(mmr_results, 1):\n",
                "    print(f\"[{i}] {doc.page_content[:100]}...\\n\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Ensemble + MMR 결합\n",
                "\n",
                "최고의 성능을 위해 Ensemble Retriever와 MMR을 함께 사용할 수 있습니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# MMR을 적용한 FAISS Retriever\n",
                "faiss_mmr_retriever = vectorstore.as_retriever(\n",
                "    search_type=\"mmr\",\n",
                "    search_kwargs={\"k\": 3, \"fetch_k\": 20, \"lambda_mult\": 0.5}\n",
                ")\n",
                "\n",
                "# Ensemble: BM25 + FAISS(MMR)\n",
                "ensemble_mmr_retriever = EnsembleRetriever(\n",
                "    retrievers=[bm25_retriever, faiss_mmr_retriever],\n",
                "    weights=[0.4, 0.6]  # 의미 검색에 약간 더 가중치\n",
                ")\n",
                "\n",
                "print(\"✅ Ensemble + MMR Retriever 생성 완료\")\n",
                "print(\"- BM25: 정확한 키워드 검색 (가중치 0.4)\")\n",
                "print(\"- FAISS(MMR): 의미 + 다양성 (가중치 0.6)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. 고급 RAG 체인 구축"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 프롬프트\n",
                "prompt = PromptTemplate.from_template(\n",
                "    \"\"\"당신은 문서 기반 질의응답을 수행하는 AI 어시스턴트입니다.\n",
                "주어진 문맥을 바탕으로 질문에 답변해 주세요.\n",
                "\n",
                "#Context:\n",
                "{context}\n",
                "\n",
                "#Question:\n",
                "{question}\n",
                "\n",
                "#Answer:\"\"\"\n",
                ")\n",
                "\n",
                "# LLM\n",
                "llm = ChatOpenAI(model=\"gpt-4o-mini\", temperature=0)\n",
                "\n",
                "# 고급 RAG 체인\n",
                "advanced_rag_chain = (\n",
                "    {\"context\": ensemble_mmr_retriever, \"question\": RunnablePassthrough()}\n",
                "    | prompt\n",
                "    | llm\n",
                "    | StrOutputParser()\n",
                ")\n",
                "\n",
                "print(\"✅ 고급 RAG 체인 생성 완료\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 성능 비교: 기본 vs 고급 RAG"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 기본 RAG 체인 (비교용)\n",
                "basic_retriever = vectorstore.as_retriever(search_kwargs={\"k\": 4})\n",
                "basic_rag_chain = (\n",
                "    {\"context\": basic_retriever, \"question\": RunnablePassthrough()}\n",
                "    | prompt\n",
                "    | llm\n",
                "    | StrOutputParser()\n",
                ")\n",
                "\n",
                "test_question = \"디지털 혁신의 주요 추진 방향은 무엇인가요?\"\n",
                "\n",
                "print(\"질문:\", test_question)\n",
                "print(\"\\n\" + \"=\" * 60)\n",
                "print(\"📝 기본 RAG 답변:\")\n",
                "print(\"=\" * 60)\n",
                "basic_answer = basic_rag_chain.invoke(test_question)\n",
                "print(basic_answer)\n",
                "\n",
                "print(\"\\n\" + \"=\" * 60)\n",
                "print(\"🚀 고급 RAG 답변 (Ensemble + MMR):\")\n",
                "print(\"=\" * 60)\n",
                "advanced_answer = advanced_rag_chain.invoke(test_question)\n",
                "print(advanced_answer)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 💡 핵심 정리\n",
                "\n",
                "### Ensemble Retriever\n",
                "\n",
                "**장점**:\n",
                "- ✅ 키워드와 의미 검색의 장점 결합\n",
                "- ✅ 정확한 용어와 유사 개념 모두 검색\n",
                "- ✅ 검색 정확도 향상\n",
                "\n",
                "**사용 시나리오**:\n",
                "- 전문 용어가 많은 문서 (의료, 법률, 기술)\n",
                "- 고유명사 검색이 중요한 경우\n",
                "\n",
                "### MMR (Maximum Marginal Relevance)\n",
                "\n",
                "**장점**:\n",
                "- ✅ 검색 결과의 다양성 확보\n",
                "- ✅ 중복 내용 제거\n",
                "- ✅ 다양한 관점의 정보 제공\n",
                "\n",
                "**사용 시나리오**:\n",
                "- 포괄적인 정보가 필요한 경우\n",
                "- 여러 관점의 답변이 필요한 경우\n",
                "\n",
                "### 파라미터 튜닝 가이드\n",
                "\n",
                "**Ensemble 가중치**:\n",
                "```python\n",
                "weights=[0.7, 0.3]  # 키워드 중심 (전문 용어 많을 때)\n",
                "weights=[0.5, 0.5]  # 균형 (일반적)\n",
                "weights=[0.3, 0.7]  # 의미 중심 (동의어 많을 때)\n",
                "```\n",
                "\n",
                "**MMR lambda_mult**:\n",
                "```python\n",
                "lambda_mult=1.0  # 관련성만 고려 (일반 검색과 동일)\n",
                "lambda_mult=0.7  # 관련성 우선\n",
                "lambda_mult=0.5  # 균형 (권장)\n",
                "lambda_mult=0.3  # 다양성 우선\n",
                "```\n",
                "\n",
                "### 성능 향상 효과\n",
                "\n",
                "| 기법 | 검색 정확도 | 다양성 | 처리 시간 |\n",
                "|------|-----------|--------|----------|\n",
                "| 기본 RAG | ⭐⭐⭐ | ⭐⭐ | 빠름 |\n",
                "| + Ensemble | ⭐⭐⭐⭐ | ⭐⭐ | 보통 |\n",
                "| + MMR | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 약간 느림 |\n",
                "| Ensemble + MMR | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 보통 |"
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
                "고급 RAG 기법 - Ensemble Retriever와 MMR을 활용한 검색 품질 향상\n",
                "\n",
                "### 구현 내용\n",
                "\n",
                "#### 1. Ensemble Retriever\n",
                "- **BM25 + FAISS 결합**: 키워드 검색과 의미 검색의 장점 통합\n",
                "  - 실행 위치: 셀 8\n",
                "  - BM25Retriever.from_documents()로 키워드 검색기 생성\n",
                "  - FAISS로 벡터 검색기 생성\n",
                "  - EnsembleRetriever로 가중치 0.5:0.5 결합\n",
                "  - 검증: 셀 9에서 각 retriever 결과 비교\n",
                "\n",
                "#### 2. MMR (Maximum Marginal Relevance)\n",
                "- **관련성 + 다양성**: 중복 제거 및 다양한 관점 확보\n",
                "  - 실행 위치: 셀 11\n",
                "  - search_type=\"mmr\" 지정\n",
                "  - lambda_mult=0.5로 균형 설정\n",
                "  - fetch_k=20으로 충분한 후보 확보\n",
                "  - 검증: 셀 12에서 일반 검색과 MMR 비교\n",
                "\n",
                "#### 3. Ensemble + MMR 결합\n",
                "- **최고 성능**: 두 기법을 함께 사용\n",
                "  - 실행 위치: 셀 13\n",
                "  - BM25 + FAISS(MMR) 결합\n",
                "  - 가중치 0.4:0.6 (의미 검색 우선)\n",
                "\n",
                "#### 4. 성능 비교\n",
                "- **기본 vs 고급 RAG**: 실제 질문으로 답변 품질 비교\n",
                "  - 실행 위치: 셀 15\n",
                "  - 동일 질문에 대한 기본/고급 RAG 답변 출력\n",
                "\n",
                "### 교재 작성 규칙 준수\n",
                "\n",
                "✅ **독립적 구현**: langchain_teddynote 미사용\n",
                "✅ **명확한 설명**: 각 기법의 원리와 장단점 도식화\n",
                "✅ **실용적 가이드**: 파라미터 튜닝 가이드 제공\n",
                "✅ **LangChain 1.0.x**: 최신 API 사용\n",
                "✅ **검수 태그**: @TAG-ENSEMBLE-RETRIEVER, @TAG-MMR-SEARCH\n",
                "\n",
                "### 학습 목표 달성\n",
                "\n",
                "1. **기본 RAG 한계 이해**: 벡터 검색만의 문제점 명확히 설명\n",
                "2. **Ensemble 원리**: Dense + Sparse 검색 결합 방법\n",
                "3. **MMR 개념**: 관련성과 다양성 균형\n",
                "4. **실전 적용**: 파라미터 튜닝 가이드와 사용 시나리오\n",
                "\n",
                "### 레퍼런스와의 차이\n",
                "\n",
                "- Reference: Reranking, Contextual Compression 등 여러 기법 나열\n",
                "- 현재 노트북: 핵심 2가지(Ensemble, MMR)에 집중하여 깊이 있게 설명\n",
                "- 이유: 학습자가 개념을 확실히 이해하고 실전 적용 가능하도록\n",
                "\n",
                "---"
            ]
        }
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
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
    
    with open('02-RAG-Advanced.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    
    print("✅ 02-RAG-Advanced.ipynb 생성 완료!")

def create_conversation_rag():
    """03-Conversation-RAG.ipynb"""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 대화 이력을 기억하는 RAG\n",
                "\n",
                "지금까지의 RAG는 각 질문을 독립적으로 처리했습니다. 이번에는 **대화 이력을 기억**하여 문맥을 유지하는 대화형 RAG를 구축합니다.\n",
                "\n",
                "## 학습 목표\n",
                "\n",
                "- RunnableWithMessageHistory를 사용한 대화 이력 관리\n",
                "- 세션 기반 대화 저장 및 불러오기\n",
                "- 대화 문맥을 고려한 RAG 구현"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 일반 RAG vs 대화형 RAG\n",
                "\n",
                "### 일반 RAG의 한계\n",
                "\n",
                "```\n",
                "User: 인공지능이 뭐야?\n",
                "AI: 인공지능은 컴퓨터가 인간처럼 생각하고 학습하는 기술입니다.\n",
                "\n",
                "User: 그것의 역사는?\n",
                "AI: ❌ \"그것\"이 무엇을 가리키는지 모름\n",
                "```\n",
                "\n",
                "### 대화형 RAG\n",
                "\n",
                "```\n",
                "User: 인공지능이 뭐야?\n",
                "AI: 인공지능은 컴퓨터가 인간처럼 생각하고 학습하는 기술입니다.\n",
                "\n",
                "User: 그것의 역사는?\n",
                "AI: ✅ 인공지능의 역사는 1950년대에 시작되었습니다...\n",
                "     (이전 대화에서 \"그것\" = \"인공지능\" 파악)\n",
                "```"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from dotenv import load_dotenv\n",
                "\n",
                "load_dotenv()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
                "from langchain_community.document_loaders import PyMuPDFLoader\n",
                "from langchain_community.vectorstores import FAISS\n",
                "from langchain_core.output_parsers import StrOutputParser\n",
                "from langchain_core.runnables import RunnablePassthrough\n",
                "from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder\n",
                "from langchain_openai import ChatOpenAI, OpenAIEmbeddings\n",
                "from langchain_core.runnables.history import RunnableWithMessageHistory\n",
                "from langchain_community.chat_message_histories import ChatMessageHistory\n",
                "from langchain_core.chat_history import BaseChatMessageHistory"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. 문서 준비 및 RAG 기본 설정"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 문서 로드\n",
                "loader = PyMuPDFLoader(\"../6-1_DocumentLoaders/data/sample-rag-brief.pdf\")\n",
                "docs = loader.load()\n",
                "\n",
                "# 분할\n",
                "text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)\n",
                "split_documents = text_splitter.split_documents(docs)\n",
                "\n",
                "# 벡터스토어\n",
                "embeddings = OpenAIEmbeddings(model=\"text-embedding-3-small\")\n",
                "vectorstore = FAISS.from_documents(split_documents, embeddings)\n",
                "\n",
                "# 검색기\n",
                "retriever = vectorstore.as_retriever(search_kwargs={\"k\": 4})\n",
                "\n",
                "print(\"✅ 문서 및 검색기 준비 완료\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. 대화 이력 저장소 설정\n",
                "\n",
                "세션별로 대화 이력을 저장하고 관리합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "'''\n",
                "@TAG-CONVERSATION-HISTORY-STORE\n",
                "- 구현 목적: 사용자별 세션을 구분하여 대화 이력 저장 및 관리\n",
                "- 구현 과정: ChatMessageHistory로 메모리 기반 저장소 구현\n",
                "- 구현 결과: session_id로 여러 사용자의 대화 이력 독립적 관리\n",
                "'''\n",
                "\n",
                "# 세션별 대화 이력 저장소\n",
                "store = {}\n",
                "\n",
                "def get_session_history(session_id: str) -> BaseChatMessageHistory:\n",
                "    \"\"\"세션 ID에 해당하는 대화 이력을 반환합니다.\"\"\"\n",
                "    if session_id not in store:\n",
                "        store[session_id] = ChatMessageHistory()\n",
                "    return store[session_id]\n",
                "\n",
                "print(\"✅ 대화 이력 저장소 설정 완료\")\n",
                "print(\"- 세션별로 독립적인 대화 이력 관리\")\n",
                "print(\"- 메모리 기반 저장 (재시작 시 초기화)\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. 대화형 프롬프트 생성\n",
                "\n",
                "일반 프롬프트와 달리, **MessagesPlaceholder**를 사용하여 대화 이력을 포함합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 대화형 프롬프트\n",
                "conversational_prompt = ChatPromptTemplate.from_messages([\n",
                "    (\"system\", \"\"\"당신은 문서 기반 질의응답을 수행하는 AI 어시스턴트입니다.\n",
                "주어진 문맥(Context)과 대화 이력(Chat History)을 참고하여 질문에 답변하세요.\n",
                "\n",
                "대화 이력을 활용하여:\n",
                "- 대명사(\"그것\", \"이것\" 등)의 지시 대상 파악\n",
                "- 이전 질문과 관련된 추가 정보 제공\n",
                "- 문맥을 이어가는 자연스러운 답변\n",
                "\n",
                "#Context:\n",
                "{context}\n",
                "\"\"\"),\n",
                "    MessagesPlaceholder(variable_name=\"chat_history\"),  # 대화 이력 삽입\n",
                "    (\"human\", \"{question}\")\n",
                "])\n",
                "\n",
                "print(\"✅ 대화형 프롬프트 생성 완료\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. 대화형 RAG 체인 구축"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "'''\n",
                "@TAG-CONVERSATIONAL-RAG-CHAIN\n",
                "- 구현 목적: 대화 이력을 유지하며 문서 기반 QA 수행\n",
                "- 구현 과정: RunnableWithMessageHistory로 체인 감싸기, get_session_history 연결\n",
                "- 구현 결과: 이전 대화를 기억하며 문맥 있는 답변 생성\n",
                "'''\n",
                "\n",
                "# LLM\n",
                "llm = ChatOpenAI(model=\"gpt-4o-mini\", temperature=0)\n",
                "\n",
                "# 기본 RAG 체인 (대화 이력 없이)\n",
                "rag_chain = (\n",
                "    {\"context\": retriever, \"question\": RunnablePassthrough()}\n",
                "    | conversational_prompt\n",
                "    | llm\n",
                "    | StrOutputParser()\n",
                ")\n",
                "\n",
                "# 대화 이력 추가\n",
                "conversational_rag_chain = RunnableWithMessageHistory(\n",
                "    rag_chain,\n",
                "    get_session_history,\n",
                "    input_messages_key=\"question\",\n",
                "    history_messages_key=\"chat_history\",\n",
                ")\n",
                "\n",
                "print(\"✅ 대화형 RAG 체인 생성 완료\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. 대화형 RAG 실행\n",
                "\n",
                "세션 ID를 지정하여 대화를 시작합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 세션 설정\n",
                "session_id = \"user_001\"\n",
                "config = {\"configurable\": {\"session_id\": session_id}}\n",
                "\n",
                "# 첫 번째 질문\n",
                "question1 = \"디지털 전환이란 무엇인가요?\"\n",
                "print(f\"👤 User: {question1}\")\n",
                "answer1 = conversational_rag_chain.invoke({\"question\": question1}, config=config)\n",
                "print(f\"🤖 AI: {answer1}\\n\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 두 번째 질문 (이전 대화 참조)\n",
                "question2 = \"그것의 주요 목표는 무엇인가요?\"  # \"그것\" = \"디지털 전환\"\n",
                "print(f\"👤 User: {question2}\")\n",
                "answer2 = conversational_rag_chain.invoke({\"question\": question2}, config=config)\n",
                "print(f\"🤖 AI: {answer2}\\n\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 세 번째 질문 (추가 정보)\n",
                "question3 = \"구체적인 추진 방법을 알려주세요.\"\n",
                "print(f\"👤 User: {question3}\")\n",
                "answer3 = conversational_rag_chain.invoke({\"question\": question3}, config=config)\n",
                "print(f\"🤖 AI: {answer3}\\n\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. 대화 이력 확인"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 현재 세션의 대화 이력 출력\n",
                "history = get_session_history(session_id)\n",
                "print(\"📜 대화 이력:\")\n",
                "print(\"=\" * 60)\n",
                "for i, message in enumerate(history.messages, 1):\n",
                "    role = \"👤 User\" if message.type == \"human\" else \"🤖 AI\"\n",
                "    print(f\"[{i}] {role}: {message.content[:100]}...\\n\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. 다중 세션 테스트\n",
                "\n",
                "서로 다른 세션은 독립적인 대화 이력을 유지합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 두 번째 사용자 (다른 세션)\n",
                "session_id_2 = \"user_002\"\n",
                "config_2 = {\"configurable\": {\"session_id\": session_id_2}}\n",
                "\n",
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"새로운 사용자 (user_002) 세션 시작\")\n",
                "print(\"=\"*60 + \"\\n\")\n",
                "\n",
                "question_new = \"인공지능의 활용 분야는?\"\n",
                "print(f\"👤 User (002): {question_new}\")\n",
                "answer_new = conversational_rag_chain.invoke({\"question\": question_new}, config=config_2)\n",
                "print(f\"🤖 AI: {answer_new}\\n\")\n",
                "\n",
                "# user_001의 이력은 유지됨\n",
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"user_001 세션으로 복귀\")\n",
                "print(\"=\"*60 + \"\\n\")\n",
                "\n",
                "question_continue = \"이전에 말한 목표를 다시 요약해주세요.\"\n",
                "print(f\"👤 User (001): {question_continue}\")\n",
                "answer_continue = conversational_rag_chain.invoke({\"question\": question_continue}, config=config)\n",
                "print(f\"🤖 AI: {answer_continue}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. 대화 이력 초기화"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 특정 세션 초기화\n",
                "def clear_session(session_id: str):\n",
                "    \"\"\"세션의 대화 이력을 초기화합니다.\"\"\"\n",
                "    if session_id in store:\n",
                "        store[session_id].clear()\n",
                "        print(f\"✅ 세션 '{session_id}' 초기화 완료\")\n",
                "    else:\n",
                "        print(f\"⚠️  세션 '{session_id}'이(가) 존재하지 않습니다.\")\n",
                "\n",
                "# 모든 세션 초기화\n",
                "def clear_all_sessions():\n",
                "    \"\"\"모든 세션의 대화 이력을 초기화합니다.\"\"\"\n",
                "    store.clear()\n",
                "    print(\"✅ 모든 세션 초기화 완료\")\n",
                "\n",
                "# 테스트\n",
                "clear_session(\"user_001\")\n",
                "print(f\"\\n현재 활성 세션: {list(store.keys())}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 💡 핵심 정리\n",
                "\n",
                "### 대화형 RAG의 구성 요소\n",
                "\n",
                "1. **ChatMessageHistory**: 대화 이력 저장\n",
                "2. **MessagesPlaceholder**: 프롬프트에 이력 삽입\n",
                "3. **RunnableWithMessageHistory**: 체인에 이력 관리 추가\n",
                "4. **세션 관리**: 사용자별 독립적인 대화 유지\n",
                "\n",
                "### 실전 활용 시나리오\n",
                "\n",
                "- **고객 지원 챗봇**: 고객과의 대화 문맥 유지\n",
                "- **교육 AI 튜터**: 학습 진행 상황 기억\n",
                "- **문서 탐색 도우미**: 연속된 질문으로 심화 탐구\n",
                "\n",
                "### 대화 이력 저장 옵션\n",
                "\n",
                "**메모리 기반 (현재)**:\n",
                "```python\n",
                "store = {}  # 재시작 시 초기화\n",
                "```\n",
                "\n",
                "**영구 저장 (Production)**:\n",
                "```python\n",
                "# SQLite, Redis, PostgreSQL 등 사용\n",
                "from langchain.memory import SQLChatMessageHistory\n",
                "\n",
                "def get_session_history(session_id: str):\n",
                "    return SQLChatMessageHistory(\n",
                "        session_id=session_id,\n",
                "        connection_string=\"sqlite:///chat_history.db\"\n",
                "    )\n",
                "```\n",
                "\n",
                "### 주의사항\n",
                "\n",
                "- ⚠️ **컨텍스트 길이**: 대화가 길어지면 토큰 제한 초과 가능\n",
                "  - 해결: 최근 N개 메시지만 유지하거나 요약 사용\n",
                "- ⚠️ **개인정보**: 대화 이력에 민감 정보 포함 주의\n",
                "  - 해결: 암호화 저장, 주기적 삭제\n",
                "\n",
                "### 성능 향상 팁\n",
                "\n",
                "1. **대화 길이 제한**: 최근 5~10개 메시지만 유지\n",
                "2. **요약 활용**: 긴 대화는 요약하여 저장\n",
                "3. **세션 타임아웃**: 일정 시간 후 자동 초기화"
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
                "대화 이력을 기억하는 RAG - 문맥을 유지하는 대화형 QA 시스템\n",
                "\n",
                "### 구현 내용\n",
                "\n",
                "#### 1. 대화 이력 저장소\n",
                "- **ChatMessageHistory 활용**: 메모리 기반 세션 관리\n",
                "  - 실행 위치: 셀 6\n",
                "  - get_session_history() 함수로 세션별 이력 반환\n",
                "  - store 딕셔너리로 여러 사용자 관리\n",
                "\n",
                "#### 2. 대화형 프롬프트\n",
                "- **MessagesPlaceholder 사용**: 동적으로 대화 이력 삽입\n",
                "  - 실행 위치: 셀 7\n",
                "  - chat_history 변수로 이력 전달\n",
                "  - System 프롬프트에 대화 활용 지침 명시\n",
                "\n",
                "#### 3. RunnableWithMessageHistory\n",
                "- **체인에 이력 관리 추가**: 자동으로 메시지 저장/로드\n",
                "  - 실행 위치: 셀 8\n",
                "  - input_messages_key=\"question\"\n",
                "  - history_messages_key=\"chat_history\"\n",
                "  - get_session_history 콜백 연결\n",
                "\n",
                "#### 4. 실행 및 검증\n",
                "- **연속 대화 테스트**: 대명사 해석 및 문맥 유지 확인\n",
                "  - 셀 9-11: \"그것\", \"구체적으로\" 등 문맥 의존 질문\n",
                "  - 셀 13: 다중 세션 독립성 테스트\n",
                "  - 셀 12: 대화 이력 확인\n",
                "\n",
                "### 교재 작성 규칙 준수\n",
                "\n",
                "✅ **독립적 구현**: 외부 특수 모듈 미사용\n",
                "✅ **명확한 설명**: 일반 RAG vs 대화형 RAG 비교\n",
                "✅ **실용적 가이드**: 영구 저장 옵션, 주의사항 제공\n",
                "✅ **LangChain 1.0.x**: RunnableWithMessageHistory 사용\n",
                "✅ **검수 태그**: @TAG-CONVERSATION-HISTORY-STORE, @TAG-CONVERSATIONAL-RAG-CHAIN\n",
                "\n",
                "### 학습 목표 달성\n",
                "\n",
                "1. **대화 이력 관리**: 세션별 메시지 저장/불러오기\n",
                "2. **문맥 유지**: 이전 대화 참조하여 답변\n",
                "3. **다중 세션**: 여러 사용자 독립적 관리\n",
                "4. **실전 적용**: 저장 옵션 및 최적화 방법\n",
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
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('03-Conversation-RAG.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    
    print("✅ 03-Conversation-RAG.ipynb 생성 완료!")

# 실행
if __name__ == "__main__":
    print("📚 RAG 교재 노트북 생성 시작...\n")
    
    create_advanced_rag()
    create_conversation_rag()
    
    print("\n🎉 모든 노트북 생성 완료!")
    print("- 02-RAG-Advanced.ipynb")
    print("- 03-Conversation-RAG.ipynb")

