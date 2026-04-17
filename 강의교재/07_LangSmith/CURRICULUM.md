# LangSmith Curriculum v1

> LangChain 기초 과정(01-06) 수강 완료 학생 대상, LangSmith 관측성 및 디버깅 참고 교재

## Overview

| Item | Value |
|------|-------|
| Parts | 4 |
| Notebooks | 11 |
| Position | 06_RAG와 07_Evaluation 사이 참고 교재 |
| Prerequisites | 01_Basics ~ 06_RAG 완료 |

## Source

- **Official Docs** (PRIMARY) — LangSmith documentation at docs.langchain.com/langsmith/
- **Existing Course** (code patterns) — notebooks/01_Basics/, notebooks/04_Memory/, notebooks/06_RAG/

## Directory Structure

```
notebooks/06.5_LangSmith/
├── 01-Setup-and-Overview.ipynb
├── 02-First-Trace.ipynb
├── 03-Chat-Model-Tracing.ipynb
├── 04-Chain-and-LCEL-Tracing.ipynb
├── 05-Memory-Tracing.ipynb
├── 06-Metadata-Tags-Filtering.ipynb
├── 07-Document-Loading-Tracing.ipynb
├── 08-Retrieval-Analysis.ipynb
├── 09-Full-RAG-Pipeline-Debug.ipynb
├── 10-Playground.ipynb
├── 11-Prompt-Management.ipynb
└── data/
```

---

## Part 1: LangSmith 시작하기 (2 notebooks)

| # | Notebook | Content | Source |
|---|----------|---------|--------|
| 01 | 01-Setup-and-Overview | LangSmith 개요, 계정/API Key 설정, 환경변수 (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT), Project 생성, SDK 설치 | Official docs: observability-quickstart, create-account-api-key, env-var |
| 02 | 02-First-Trace | @traceable 데코레이터, 자동 추적(LANGCHAIN_TRACING_V2), LangSmith UI 탐색 (trace viewer, run tree), Trace/Run/Span 개념 | Official docs: annotate-code, observability-concepts |

## Part 2: LLM 호출 추적 (4 notebooks)

| # | Notebook | Content | Source |
|---|----------|---------|--------|
| 03 | 03-Chat-Model-Tracing | ChatOpenAI/ChatAnthropic 호출 추적, invoke/stream/batch Trace 차이, 토큰/비용 추적, 모델별 응답 시간 비교 | Official docs: log-llm-trace, cost-tracking, trace-with-langchain |
| 04 | 04-Chain-and-LCEL-Tracing | LCEL 체인 Trace 구조, 중간 단계별 입출력, RunnablePassthrough/Parallel 추적, 오류 지점 확인 | Official docs: trace-with-langchain, nest-traces, observability-llm-tutorial |
| 05 | 05-Memory-Tracing | ConversationBufferMemory/History 추적, 멀티턴 대화 Trace 구조, 메시지 누적 과정, thread_id별 추적 | Official docs: trace-with-langchain, access-current-span |
| 06 | 06-Metadata-Tags-Filtering | Metadata 추가(config), Tag 기반 분류, UI 필터링/검색, 프로젝트 관리 전략 | Official docs: add-metadata-tags, filter-traces-in-application, log-traces-to-project |

## Part 3: RAG 파이프라인 분석 (3 notebooks)

| # | Notebook | Content | Source |
|---|----------|---------|--------|
| 07 | 07-Document-Loading-Tracing | DocumentLoader, TextSplitter, Embedding 추적, VectorStore 저장/조회 추적 | Official docs: log-retriever-trace, trace-with-langchain + 06_RAG ref |
| 08 | 08-Retrieval-Analysis | Retriever 호출 추적, 검색 결과 품질 확인, 다양한 Retriever 비교 (VectorStore, Ensemble, MultiQuery), Reranker 전후 비교 | Official docs: log-retriever-trace + 06_RAG ref |
| 09 | 09-Full-RAG-Pipeline-Debug | 전체 RAG 체인 Trace 분석 (질문→검색→생성), 병목 식별, 원인 추적 (검색 vs 생성), Trace에서 데이터셋 생성 | Official docs: manage-trace, dashboards + 06_RAG ref |

## Part 4: 프롬프트 엔지니어링 (2 notebooks)

| # | Notebook | Content | Source |
|---|----------|---------|--------|
| 10 | 10-Playground | Playground 사용법, Trace에서 재실행, 모델/파라미터 변경 비교, 프롬프트 반복 개선 | Official docs: prompt-engineering-quickstart, prompt-engineering |
| 11 | 11-Prompt-Management | Prompt Hub 활용 (pull/push), 버전 관리, SDK로 프롬프트 관리, 팀 협업 | Official docs: create-a-prompt, manage-prompts, manage-prompts-programmatically |
