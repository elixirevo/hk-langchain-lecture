# 07_Evaluation: RAG 및 LLM 시스템 평가

RAG 시스템과 LLM 애플리케이션의 품질을 측정하고 개선하는 평가 방법론을 학습합니다.

## 📋 개요

RAG 시스템을 구축한 후에는 그 성능을 객관적으로 평가하고 개선해야 합니다. 이 장에서는 RAGAS와 LangSmith를 활용한 체계적인 평가 방법을 다룹니다.

## 🎯 학습 목표

- RAG 시스템의 성능 평가 메트릭 이해
- RAGAS를 활용한 자동 테스트 데이터셋 생성
- 4가지 핵심 메트릭으로 RAG 평가
- LangSmith를 활용한 프로덕션 평가
- LLM-as-Judge 패턴 적용

## 📚 노트북 구성

### 1. RAGAS 기반 평가

| 노트북 | 내용 | 난이도 |
|--------|------|--------|
| `01-Test-Dataset-Generator-RAGAS.ipynb` | 합성 테스트 데이터셋 자동 생성 | ⭐⭐ |
| `02-Evaluation-Using-RAGAS.ipynb` | RAGAS 4가지 메트릭으로 RAG 평가 | ⭐⭐⭐ |

### 2. LangSmith 기반 평가

| 노트북 | 내용 | 난이도 |
|--------|------|--------|
| `03-LangSmith-Dataset-and-Evaluation.ipynb` | LangSmith 데이터셋 관리 및 LLM-as-Judge | ⭐⭐⭐ |
| `04-LangSmith-Custom-Evaluators.ipynb` | 커스텀 평가자 구현 (규칙, LLM, 임베딩) | ⭐⭐⭐ |
| `05-LangSmith-Heuristic-Evaluation.ipynb` | Heuristic 메트릭 (ROUGE, BLEU, METEOR, SemScore) | ⭐⭐⭐ |
| `06-LangSmith-Groundedness-Evaluation.ipynb` | 근거성 평가 (Hallucination 방지) | ⭐⭐⭐ |
| `07-LangSmith-Model-Comparison.ipynb` | 여러 모델 성능 비교 | ⭐⭐ |
| `08-LangSmith-Online-Evaluation.ipynb` | 온라인 평가 (프로덕션 모니터링) | ⭐⭐⭐ |

## 🔑 핵심 개념

### RAGAS 평가 메트릭

1. **Context Precision** (컨텍스트 정확도)
   - 검색된 문서가 얼마나 정확한가?
   - 관련 문서가 상위에 배치되었는가?
   - 목표: > 0.8

2. **Context Recall** (컨텍스트 재현율)
   - 필요한 정보를 모두 검색했는가?
   - Ground truth의 모든 내용이 컨텍스트에 포함되었는가?
   - 목표: > 0.85

3. **Faithfulness** (충실도)
   - 답변이 검색된 문서에 근거하는가?
   - Hallucination이 없는가?
   - 목표: > 0.9

4. **Answer Relevancy** (답변 관련성)
   - 답변이 질문과 관련있는가?
   - 불필요한 정보가 포함되지 않았는가?
   - 목표: > 0.85

### 평가 워크플로우

```
1. 테스트 데이터셋 생성 (RAGAS)
   ├─ 문서 로드
   ├─ 자동 질문 생성 (simple, reasoning, multi_context, conditional)
   └─ CSV 저장
   
2. RAG 시스템 구축
   ├─ 문서 분할
   ├─ 벡터 DB 생성
   └─ RAG 체인 구축
   
3. 평가 실행
   ├─ RAGAS: 4가지 메트릭
   └─ LangSmith: LLM-as-Judge, 커스텀 평가자
   
4. 결과 분석 및 개선
   ├─ 메트릭별 점수 분석
   ├─ 저조한 케이스 파악
   └─ 개선 방향 도출
```

## 🛤️ 학습 경로

### 초급: RAGAS 기본

1. **01-Test-Dataset-Generator-RAGAS** (⭐⭐)
   - 합성 데이터 생성의 필요성 이해
   - RAGAS로 자동 테스트셋 생성
   - 질문 유형별 분포 설정

2. **02-Evaluation-Using-RAGAS** (⭐⭐⭐)
   - RAG 시스템 구축
   - 4가지 메트릭으로 평가
   - 결과 분석 및 개선 방향

### 중급: LangSmith 평가

3. **03-LangSmith-Dataset-and-Evaluation** (⭐⭐⭐)
   - LangSmith 데이터셋 관리
   - LLM-as-Judge 패턴
   - 프로덕션 환경 평가

4. **04-LangSmith-Custom-Evaluators** (⭐⭐⭐)
   - 규칙 기반 평가자 구현
   - LLM 기반 평가자 구현
   - 임베딩 거리 기반 평가자

5. **05-LangSmith-Heuristic-Evaluation** (⭐⭐⭐)
   - ROUGE, BLEU, METEOR 메트릭
   - SemScore (임베딩 기반 유사도)
   - Heuristic vs LLM-as-Judge 비교

### 고급: 프로덕션 평가

6. **06-LangSmith-Groundedness-Evaluation** (⭐⭐⭐)
   - 답변의 근거성 평가
   - Hallucination 방지
   - RAG 신뢰성 향상

7. **07-LangSmith-Model-Comparison** (⭐⭐)
   - 여러 모델 성능 비교
   - LangSmith Compare 기능
   - 최적 모델 선택

8. **08-LangSmith-Online-Evaluation** (⭐⭐⭐)
   - 프로덕션 모니터링
   - 실시간 평가
   - Latency, Token Usage, Error Rate 추적

## 💡 빠른 시작

### RAGAS 평가 예제

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)

# 평가 실행
result = evaluate(
    dataset=test_dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    ]
)

print(result)
```

### LangSmith 평가 예제

```python
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator

client = Client()

# LLM-as-Judge 평가자
evaluator = LangChainStringEvaluator(
    "labeled_criteria",
    config={
        "criteria": {
            "accuracy": "Is the answer factually correct?",
            "relevance": "Does it address the question?"
        }
    }
)

# 평가 실행
results = evaluate(
    lambda x: rag_chain.invoke(x["question"]),
    data="dataset-name",
    evaluators=[evaluator]
)
```

## 📊 평가 전략

### 개발 단계

```python
# 1. RAGAS로 빠른 평가
testset = generate_testset(docs, size=20)
results = evaluate(testset, metrics=[...])

# 2. 메트릭별 개선
if results["faithfulness"] < 0.9:
    # 프롬프트 개선: "문서 기반 답변" 강조
    # Temperature 낮추기
```

### 프로덕션 단계

```python
# 1. LangSmith로 지속적 모니터링
# 2. A/B 테스트
# 3. 사용자 피드백 수집
```

## 🎯 평가 메트릭 목표

| 메트릭 | 목표 점수 | 개선 방법 |
|--------|----------|----------|
| Context Precision | > 0.8 | 임베딩 모델 개선, Reranking |
| Context Recall | > 0.85 | 검색 문서 수 증가, 하이브리드 검색 |
| Faithfulness | > 0.9 | 프롬프트 개선, Temperature 조정 |
| Answer Relevancy | > 0.85 | Few-shot 예제, 프롬프트 최적화 |

## 🔗 관련 장

- **이전**: `06_RAG` - RAG 시스템 구축
- **다음**: `08_Agent` - Agent 시스템 구축

## 📖 참고 자료

- [RAGAS Documentation](https://docs.ragas.io/)
- [LangSmith](https://smith.langchain.com/)
- [RAGAS Paper](https://arxiv.org/abs/2309.15217)
- [LLM Evaluation Best Practices](https://www.anthropic.com/index/evaluating-ai-systems)

## ⚠️ 주의사항

1. **RAGAS 버전**: LangChain 0.2.16과 호환되는 RAGAS 0.1.19 사용
2. **비용**: LLM-as-Judge는 추가 LLM 호출 비용 발생
3. **LangSmith**: 계정 설정 및 API 키 필요
4. **평가 시간**: 대규모 데이터셋 평가는 시간이 오래 걸림

## 🚀 시작하기

1. 환경 설정
```bash
pip install langchain==0.2.16 ragas==0.1.19 langsmith
```

2. API 키 설정
```bash
export OPENAI_API_KEY="your-key"
export LANGCHAIN_API_KEY="your-key"
export LANGCHAIN_TRACING_V2="true"
```

3. 첫 번째 노트북 실행
```bash
jupyter notebook 01-Test-Dataset-Generator-RAGAS.ipynb
```

---

**학습 순서**: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08

- **필수 노트북**: 01, 02, 03 (RAGAS 기본, LangSmith 기본)
- **권장 노트북**: 04, 05, 06 (커스텀 평가자, Heuristic, Groundedness)
- **선택 노트북**: 07, 08 (모델 비교, 온라인 평가)

각 노트북은 독립적으로 실행 가능하지만, 순서대로 학습하는 것을 권장합니다.



