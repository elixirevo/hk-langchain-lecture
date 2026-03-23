import json

def create_summary_notebook():
    """05-Web-Summarization.ipynb - 웹 문서 요약"""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 웹 문서 요약 with RAG\n",
                "\n",
                "웹 페이지의 긴 내용을 요약하고, 요약본을 기반으로 질문에 답변하는 시스템을 구축합니다.\n",
                "\n",
                "## 학습 목표\n",
                "\n",
                "- 웹 문서 크롤링 및 요약\n",
                "- Map-Reduce 패턴을 활용한 긴 문서 요약\n",
                "- 요약본 기반 효율적인 RAG 구축"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 왜 요약이 필요한가?\n",
                "\n",
                "### 문제 상황\n",
                "\n",
                "- 웹 페이지는 수만 자의 긴 텍스트 포함\n",
                "- 모든 내용을 검색 대상으로 하면 노이즈 증가\n",
                "- LLM 컨텍스트 윈도우 제한\n",
                "\n",
                "### 해결책: 요약 기반 RAG\n",
                "\n",
                "```\n",
                "긴 웹 문서 (10,000자)\n",
                "    ↓\n",
                "자동 요약 (2,000자)\n",
                "    ↓\n",
                "벡터 검색 (효율적)\n",
                "    ↓\n",
                "정확한 답변\n",
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
                "import bs4\n",
                "from langchain_community.document_loaders import WebBaseLoader\n",
                "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
                "from langchain_core.prompts import PromptTemplate\n",
                "from langchain_openai import ChatOpenAI\n",
                "from langchain_core.output_parsers import StrOutputParser\n",
                "from langchain.chains.summarize import load_summarize_chain"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. 웹 문서 로드"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Wikipedia 긴 문서 로드\n",
                "loader = WebBaseLoader(\n",
                "    web_paths=(\"https://ko.wikipedia.org/wiki/딥_러닝\",),\n",
                "    bs_kwargs=dict(\n",
                "        parse_only=bs4.SoupStrainer(\n",
                "            \"div\",\n",
                "            attrs={\"class\": \"mw-parser-output\"}\n",
                "        )\n",
                "    ),\n",
                ")\n",
                "\n",
                "docs = loader.load()\n",
                "\n",
                "print(f\"문서 수: {len(docs)}\")\n",
                "print(f\"문서 길이: {len(docs[0].page_content):,} 문자\")\n",
                "print(f\"\\n문서 미리보기:\\n{docs[0].page_content[:300]}...\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. 문서 분할 (요약 준비)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 요약을 위한 분할 (청크 크기를 크게)\n",
                "text_splitter = RecursiveCharacterTextSplitter(\n",
                "    chunk_size=3000,  # 요약용으로 큰 청크\n",
                "    chunk_overlap=200\n",
                ")\n",
                "\n",
                "split_docs = text_splitter.split_documents(docs)\n",
                "\n",
                "print(f\"분할된 청크 수: {len(split_docs)}\")\n",
                "print(f\"첫 번째 청크 길이: {len(split_docs[0].page_content)} 문자\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Map-Reduce 요약\n",
                "\n",
                "### Map-Reduce 패턴\n",
                "\n",
                "```\n",
                "청크 1  →  요약 1  \\\n",
                "청크 2  →  요약 2   → 최종 통합 요약\n",
                "청크 3  →  요약 3  /\n",
                "```\n",
                "\n",
                "- **Map**: 각 청크를 독립적으로 요약\n",
                "- **Reduce**: 모든 요약을 하나로 통합"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "'''\n",
                "@TAG-MAP-REDUCE-SUMMARIZE\n",
                "- 구현 목적: 긴 문서를 청크별로 요약 후 통합하여 전체 요약 생성\n",
                "- 구현 과정: load_summarize_chain으로 map_reduce 체인 구성\n",
                "- 구현 결과: 긴 문서도 컨텍스트 제한 없이 요약 가능\n",
                "'''\n",
                "\n",
                "# LLM\n",
                "llm = ChatOpenAI(model=\"gpt-4o-mini\", temperature=0)\n",
                "\n",
                "# Map-Reduce 요약 체인\n",
                "summarize_chain = load_summarize_chain(\n",
                "    llm=llm,\n",
                "    chain_type=\"map_reduce\",\n",
                "    verbose=False\n",
                ")\n",
                "\n",
                "# 요약 실행\n",
                "summary = summarize_chain.run(split_docs)\n",
                "\n",
                "print(\"📄 전체 문서 요약:\")\n",
                "print(\"=\" * 60)\n",
                "print(summary)\n",
                "print(\"\\n\" + \"=\" * 60)\n",
                "print(f\"요약 길이: {len(summary)} 문자\")\n",
                "print(f\"압축률: {len(summary) / len(docs[0].page_content) * 100:.1f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. 맞춤 요약 프롬프트\n",
                "\n",
                "기본 요약 대신, 특정 관점으로 요약할 수 있습니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 커스텀 Map 프롬프트 (각 청크 요약)\n",
                "map_prompt = PromptTemplate.from_template(\n",
                "    \"\"\"다음 텍스트를 **핵심 내용 위주**로 간결하게 요약하세요.\n",
                "중요한 개념, 정의, 예시만 포함하세요.\n",
                "\n",
                "텍스트:\n",
                "{text}\n",
                "\n",
                "간결한 요약:\"\"\"\n",
                ")\n",
                "\n",
                "# 커스텀 Reduce 프롬프트 (통합 요약)\n",
                "reduce_prompt = PromptTemplate.from_template(\n",
                "    \"\"\"다음은 문서의 여러 부분을 요약한 내용입니다.\n",
                "이들을 하나의 일관된 요약으로 통합하세요.\n",
                "\n",
                "부분 요약들:\n",
                "{text}\n",
                "\n",
                "통합 요약:\"\"\"\n",
                ")\n",
                "\n",
                "# 커스텀 프롬프트 적용\n",
                "custom_chain = load_summarize_chain(\n",
                "    llm=llm,\n",
                "    chain_type=\"map_reduce\",\n",
                "    map_prompt=map_prompt,\n",
                "    combine_prompt=reduce_prompt,\n",
                "    verbose=False\n",
                ")\n",
                "\n",
                "custom_summary = custom_chain.run(split_docs)\n",
                "\n",
                "print(\"📝 커스텀 요약:\")\n",
                "print(\"=\" * 60)\n",
                "print(custom_summary)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. 요약 기반 RAG 구축\n",
                "\n",
                "요약된 내용을 벡터스토어에 저장하여 효율적인 검색을 수행합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from langchain_community.vectorstores import FAISS\n",
                "from langchain_openai import OpenAIEmbeddings\n",
                "from langchain_core.runnables import RunnablePassthrough\n",
                "from langchain.schema import Document\n",
                "\n",
                "# 요약을 Document 객체로 변환\n",
                "summary_doc = Document(\n",
                "    page_content=custom_summary,\n",
                "    metadata={\"source\": \"summary\", \"url\": \"https://ko.wikipedia.org/wiki/딥_러닝\"}\n",
                ")\n",
                "\n",
                "# 원본 문서 + 요약 함께 저장\n",
                "all_docs = split_docs + [summary_doc]\n",
                "\n",
                "# 벡터스토어 생성\n",
                "embeddings = OpenAIEmbeddings(model=\"text-embedding-3-small\")\n",
                "vectorstore = FAISS.from_documents(all_docs, embeddings)\n",
                "retriever = vectorstore.as_retriever(search_kwargs={\"k\": 3})\n",
                "\n",
                "# RAG 체인\n",
                "rag_prompt = PromptTemplate.from_template(\n",
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
                "rag_chain = (\n",
                "    {\"context\": retriever, \"question\": RunnablePassthrough()}\n",
                "    | rag_prompt\n",
                "    | llm\n",
                "    | StrOutputParser()\n",
                ")\n",
                "\n",
                "print(\"✅ 요약 기반 RAG 시스템 구축 완료\")\n",
                "print(f\"- 원본 청크: {len(split_docs)}개\")\n",
                "print(f\"- 요약 문서: 1개\")\n",
                "print(f\"- 총 검색 대상: {len(all_docs)}개\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. 질의응답 테스트"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 질문 1: 개념 질의\n",
                "question1 = \"딥러닝이란 무엇인가요?\"\n",
                "print(f\"질문: {question1}\")\n",
                "print(\"=\" * 60)\n",
                "answer1 = rag_chain.invoke(question1)\n",
                "print(f\"답변:\\n{answer1}\\n\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 질문 2: 구체적 정보\n",
                "question2 = \"딥러닝의 주요 활용 분야는 무엇인가요?\"\n",
                "print(f\"질문: {question2}\")\n",
                "print(\"=\" * 60)\n",
                "answer2 = rag_chain.invoke(question2)\n",
                "print(f\"답변:\\n{answer2}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 💡 핵심 정리\n",
                "\n",
                "### Map-Reduce 요약의 장점\n",
                "\n",
                "1. **긴 문서 처리**: LLM 컨텍스트 제한 극복\n",
                "2. **병렬 처리**: 각 청크를 독립적으로 요약 (속도 향상 가능)\n",
                "3. **유연성**: Map/Reduce 프롬프트 커스터마이징\n",
                "\n",
                "### 요약 체인 타입\n",
                "\n",
                "| 타입 | 설명 | 장점 | 단점 |\n",
                "|------|------|------|------|\n",
                "| **stuff** | 모든 문서를 한 번에 | 빠름 | 긴 문서 불가 |\n",
                "| **map_reduce** | 청크별 요약 후 통합 | 긴 문서 가능 | 느림 |\n",
                "| **refine** | 순차적 정제 요약 | 일관성 높음 | 매우 느림 |\n",
                "\n",
                "### 실전 활용\n",
                "\n",
                "- 뉴스 기사 요약 및 QA\n",
                "- 논문/보고서 분석\n",
                "- 웹 콘텐츠 요약 봇\n",
                "- 긴 회의록 요약"
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
                "웹 문서 요약 with RAG - Map-Reduce 패턴을 활용한 긴 문서 처리\n",
                "\n",
                "### 구현 내용\n",
                "\n",
                "#### 1. 웹 문서 로드\n",
                "- Wikipedia 딥러닝 페이지 크롤링\n",
                "- 긴 문서 (수천~수만 자)\n",
                "\n",
                "#### 2. Map-Reduce 요약\n",
                "- load_summarize_chain(chain_type=\"map_reduce\")\n",
                "- 각 청크를 독립적으로 요약\n",
                "- 모든 요약을 하나로 통합\n",
                "\n",
                "#### 3. 커스텀 프롬프트\n",
                "- map_prompt: 각 청크 요약 방식 지정\n",
                "- combine_prompt: 통합 요약 방식 지정\n",
                "\n",
                "#### 4. 요약 기반 RAG\n",
                "- 원본 청크 + 요약 문서 함께 벡터스토어에 저장\n",
                "- 요약으로 빠른 개요 파악\n",
                "- 세부 질문은 원본 청크에서 검색\n",
                "\n",
                "### 학습 목표 달성\n",
                "\n",
                "1. Map-Reduce 패턴 이해 및 구현\n",
                "2. 긴 문서 요약 전략\n",
                "3. 요약과 원본을 함께 활용하는 하이브리드 RAG\n",
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
    
    with open('05-Web-Summarization.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    
    print("✅ 05-Web-Summarization.ipynb 생성 완료!")

def create_simple_raptor():
    """04-RAPTOR.ipynb - 간소화된 버전"""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# RAPTOR: 긴 문서를 위한 계층적 RAG\n",
                "\n",
                "**RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)**는 긴 문서를 계층적으로 요약하여 다양한 추상화 레벨에서 검색할 수 있게 하는 고급 RAG 기법입니다.\n",
                "\n",
                "## 핵심 아이디어\n",
                "\n",
                "```\n",
                "레벨 2 (최상위 요약)     [전체 문서 요약]\n",
                "                            ↑\n",
                "레벨 1 (섹션 요약)    [요약1] [요약2] [요약3]\n",
                "                       ↑       ↑       ↑\n",
                "레벨 0 (원본 청크)  [청크1][청크2][청크3][청크4][청크5][청크6]\n",
                "```\n",
                "\n",
                "**질문의 추상화 레벨에 따라 적절한 레벨에서 검색**\n",
                "- 개괄적 질문 → 상위 레벨 요약 검색\n",
                "- 세부적 질문 → 하위 레벨 원본 검색"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 왜 RAPTOR가 필요한가?\n",
                "\n",
                "### 일반 RAG의 한계\n",
                "\n",
                "긴 문서(예: 100페이지 논문)에서:\n",
                "- \"이 논문의 주요 기여는?\" → 여러 청크를 종합해야 답변 가능\n",
                "- 하지만 청크는 독립적으로 검색됨\n",
                "- 전체적인 맥락 파악 어려움\n",
                "\n",
                "### RAPTOR의 해결책\n",
                "\n",
                "- 여러 레벨의 요약 생성\n",
                "- 질문에 따라 적절한 추상화 레벨 선택\n",
                "- 전체 맥락과 세부 정보 모두 활용\n",
                "\n",
                "## 학습 목표\n",
                "\n",
                "이 노트북에서는 **간소화된 RAPTOR**를 구현합니다:\n",
                "- 2레벨 계층 구조 (원본 + 요약)\n",
                "- 통합 검색 (모든 레벨에서 동시 검색)"
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
                "from langchain_openai import ChatOpenAI, OpenAIEmbeddings\n",
                "from langchain_core.prompts import PromptTemplate\n",
                "from langchain_core.output_parsers import StrOutputParser\n",
                "from langchain_core.runnables import RunnablePassthrough\n",
                "from langchain.schema import Document"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. 긴 문서 로드"
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
                "# 원본 청크로 분할\n",
                "text_splitter = RecursiveCharacterTextSplitter(\n",
                "    chunk_size=800,\n",
                "    chunk_overlap=100\n",
                ")\n",
                "level0_chunks = text_splitter.split_documents(docs)\n",
                "\n",
                "print(f\"레벨 0 (원본): {len(level0_chunks)}개 청크\")\n",
                "print(f\"평균 청크 크기: {sum(len(c.page_content) for c in level0_chunks) / len(level0_chunks):.0f} 문자\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. 레벨 1 요약 생성\n",
                "\n",
                "여러 청크를 그룹화하여 요약합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "'''\n",
                "@TAG-RAPTOR-HIERARCHICAL-SUMMARY\n",
                "- 구현 목적: 원본 청크를 그룹화하여 상위 레벨 요약 생성\n",
                "- 구현 과정: 5개 청크씩 묶어서 요약, 계층 구조 형성\n",
                "- 구현 결과: 다양한 추상화 레벨의 문서 표현\n",
                "'''\n",
                "\n",
                "llm = ChatOpenAI(model=\"gpt-4o-mini\", temperature=0)\n",
                "\n",
                "# 요약 프롬프트\n",
                "summary_prompt = PromptTemplate.from_template(\n",
                "    \"\"\"다음 텍스트들을 하나의 일관된 요약으로 통합하세요.\n",
                "핵심 개념과 주요 정보만 포함하세요.\n",
                "\n",
                "텍스트들:\n",
                "{text}\n",
                "\n",
                "통합 요약:\"\"\"\n",
                ")\n",
                "\n",
                "summary_chain = summary_prompt | llm | StrOutputParser()\n",
                "\n",
                "# 청크를 그룹으로 묶기 (5개씩)\n",
                "group_size = 5\n",
                "level1_summaries = []\n",
                "\n",
                "for i in range(0, len(level0_chunks), group_size):\n",
                "    group = level0_chunks[i:i+group_size]\n",
                "    combined_text = \"\\n\\n\".join([chunk.page_content for chunk in group])\n",
                "    \n",
                "    # 요약 생성\n",
                "    summary = summary_chain.invoke({\"text\": combined_text})\n",
                "    \n",
                "    # Document 객체로 저장\n",
                "    summary_doc = Document(\n",
                "        page_content=summary,\n",
                "        metadata={\n",
                "            \"level\": 1,\n",
                "            \"group_id\": i//group_size,\n",
                "            \"source\": \"summary\"\n",
                "        }\n",
                "    )\n",
                "    level1_summaries.append(summary_doc)\n",
                "    \n",
                "    print(f\"그룹 {i//group_size + 1} 요약 완료\")\n",
                "\n",
                "print(f\"\\n✅ 레벨 1 (요약): {len(level1_summaries)}개 생성\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. 계층적 벡터스토어 구축\n",
                "\n",
                "모든 레벨의 문서를 하나의 벡터스토어에 저장합니다."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 레벨 0에 메타데이터 추가\n",
                "for chunk in level0_chunks:\n",
                "    chunk.metadata[\"level\"] = 0\n",
                "\n",
                "# 모든 레벨 통합\n",
                "all_documents = level0_chunks + level1_summaries\n",
                "\n",
                "# 벡터스토어 생성\n",
                "embeddings = OpenAIEmbeddings(model=\"text-embedding-3-small\")\n",
                "vectorstore = FAISS.from_documents(all_documents, embeddings)\n",
                "\n",
                "print(f\"✅ 계층적 벡터스토어 구축 완료\")\n",
                "print(f\"- 레벨 0 (원본): {len(level0_chunks)}개\")\n",
                "print(f\"- 레벨 1 (요약): {len(level1_summaries)}개\")\n",
                "print(f\"- 총 문서: {len(all_documents)}개\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. RAPTOR RAG 체인"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Retriever (모든 레벨에서 검색)\n",
                "retriever = vectorstore.as_retriever(\n",
                "    search_type=\"similarity\",\n",
                "    search_kwargs={\"k\": 5}  # 다양한 레벨에서 검색\n",
                ")\n",
                "\n",
                "# 프롬프트\n",
                "raptor_prompt = PromptTemplate.from_template(\n",
                "    \"\"\"문맥을 참고하여 질문에 답하세요.\n",
                "문맥은 서로 다른 추상화 레벨에서 검색된 정보입니다.\n",
                "\n",
                "문맥:\n",
                "{context}\n",
                "\n",
                "질문: {question}\n",
                "\n",
                "답변:\"\"\"\n",
                ")\n",
                "\n",
                "# RAPTOR RAG 체인\n",
                "raptor_chain = (\n",
                "    {\"context\": retriever, \"question\": RunnablePassthrough()}\n",
                "    | raptor_prompt\n",
                "    | llm\n",
                "    | StrOutputParser()\n",
                ")\n",
                "\n",
                "print(\"✅ RAPTOR RAG 체인 생성 완료\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. 테스트: 추상화 레벨별 질문"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 질문 1: 개괄적 질문 (레벨 1 요약에서 답변 가능)\n",
                "question1 = \"이 문서의 전반적인 주제는 무엇인가요?\"\n",
                "\n",
                "print(f\"질문: {question1}\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "# 검색된 문서 확인\n",
                "retrieved = retriever.get_relevant_documents(question1)\n",
                "print(\"\\n검색된 문서 레벨:\")\n",
                "for i, doc in enumerate(retrieved[:3], 1):\n",
                "    level = doc.metadata.get('level', 0)\n",
                "    print(f\"  [{i}] 레벨 {level}: {doc.page_content[:80]}...\")\n",
                "\n",
                "print(\"\\n답변:\")\n",
                "answer1 = raptor_chain.invoke(question1)\n",
                "print(answer1)\n",
                "print()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 질문 2: 세부적 질문 (레벨 0 원본에서 답변)\n",
                "question2 = \"디지털 전환의 구체적인 추진 방법을 상세히 설명해주세요.\"\n",
                "\n",
                "print(f\"질문: {question2}\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "# 검색된 문서 확인\n",
                "retrieved = retriever.get_relevant_documents(question2)\n",
                "print(\"\\n검색된 문서 레벨:\")\n",
                "for i, doc in enumerate(retrieved[:3], 1):\n",
                "    level = doc.metadata.get('level', 0)\n",
                "    print(f\"  [{i}] 레벨 {level}: {doc.page_content[:80]}...\")\n",
                "\n",
                "print(\"\\n답변:\")\n",
                "answer2 = raptor_chain.invoke(question2)\n",
                "print(answer2)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 💡 핵심 정리\n",
                "\n",
                "### RAPTOR의 핵심 아이디어\n",
                "\n",
                "1. **계층적 요약**: 문서를 여러 추상화 레벨로 표현\n",
                "2. **통합 검색**: 모든 레벨에서 동시 검색\n",
                "3. **유연한 답변**: 질문의 특성에 따라 적절한 레벨 활용\n",
                "\n",
                "### 일반 RAG vs RAPTOR\n",
                "\n",
                "| 특징 | 일반 RAG | RAPTOR |\n",
                "|------|---------|--------|\n",
                "| 검색 대상 | 원본 청크만 | 원본 + 여러 레벨 요약 |\n",
                "| 개괄적 질문 | 어려움 | 상위 요약에서 답변 |\n",
                "| 세부적 질문 | 가능 | 원본 청크에서 답변 |\n",
                "| 긴 문서 처리 | 제한적 | 효과적 |\n",
                "\n",
                "### 실전 활용\n",
                "\n",
                "- **연구 논문 분석**: 전체 내용 파악 + 세부 실험 질의\n",
                "- **긴 보고서**: 요약 + 상세 데이터 검색\n",
                "- **법률 문서**: 전체 맥락 + 특정 조항 검색\n",
                "\n",
                "### 확장 가능성\n",
                "\n",
                "이 노트북은 2레벨 구조를 구현했지만, 실제 RAPTOR는:\n",
                "- 3~4개 레벨 사용 가능\n",
                "- 클러스터링으로 관련 청크 그룹화\n",
                "- 트리 구조로 계층 관리"
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
                "RAPTOR - 계층적 요약을 활용한 긴 문서 RAG\n",
                "\n",
                "### 구현 내용\n",
                "\n",
                "#### 1. 2레벨 계층 구조\n",
                "- 레벨 0: 원본 청크 (800자)\n",
                "- 레벨 1: 그룹 요약 (5개 청크 → 1개 요약)\n",
                "\n",
                "#### 2. 계층적 벡터스토어\n",
                "- 모든 레벨을 하나의 벡터스토어에 저장\n",
                "- 메타데이터로 레벨 구분\n",
                "- 통합 검색으로 적절한 레벨 자동 선택\n",
                "\n",
                "#### 3. 질문 유형별 검색\n",
                "- 개괄적 질문 → 상위 요약 우선 검색\n",
                "- 세부적 질문 → 원본 청크 우선 검색\n",
                "\n",
                "### 간소화 사항\n",
                "\n",
                "- 원본 RAPTOR: 클러스터링, 트리 구조, 3~4 레벨\n",
                "- 이 노트북: 간단한 그룹화, 2레벨로 핵심 개념 전달\n",
                "- 이유: 학습자가 계층적 RAG의 핵심 아이디어 이해\n",
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
    
    with open('04-RAPTOR.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    
    print("✅ 04-RAPTOR.ipynb 생성 완료!")

def create_readme():
    """README.md 생성"""
    readme_content = """# 6-7_RAG Process

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
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ README.md 생성 완료!")

# 실행
if __name__ == "__main__":
    print("📚 나머지 RAG 노트북 생성 시작...\n")
    
    create_simple_raptor()
    create_summary_notebook()
    create_readme()
    
    print("\n🎉 모든 노트북 생성 완료!")
    print("- 04-RAPTOR.ipynb")
    print("- 05-Web-Summarization.ipynb")
    print("- README.md")

