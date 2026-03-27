# 03-OutputParser: LLM 출력 구조화 및 파싱

## 📋 Overview
LLM의 비구조화된 텍스트 출력을 구조화된 데이터로 변환하는 다양한 파서를 학습합니다. Pydantic 모델부터 JSON, DataFrame까지 다양한 출력 형식을 다룹니다.

## 🎯 Prerequisites
- `01-Basic`, `02-Prompt` 완료
- Python 타입 힌팅 기본 이해
- Pydantic 기본 개념 (선택사항)

## 📚 Contents

| Notebook | Description | Use Case |
|----------|-------------|----------|
| `01-PydanticOutputParser.ipynb` | Pydantic 모델 기반 구조화 출력 | 복잡한 데이터 구조, 타입 검증 |
| `02-CommaSeparatedListOutputParser.ipynb` | 쉼표 구분 리스트 파싱 | 간단한 리스트 출력 |
| `03-StructuredOutputParser.ipynb` | 딕셔너리 형태 구조화 출력 | 키-값 쌍 데이터 |
| `04-JsonOutputParser.ipynb` | JSON 형식 출력 파싱 | JSON 데이터 처리 |
| `05-PandasDataFrameOutputParser.ipynb` | DataFrame 형식으로 변환 | 테이블 데이터 분석 |
| `06-DatetimeOutputParser.ipynb` | 날짜/시간 정보 파싱 | 시간 정보 추출 |
| `07-EnumOutputParser.ipynb` | 열거형(Enum) 출력 제한 | 선택지 제한 출력 |
| `08-OutputFixingParser.ipynb` | 파싱 오류 자동 수정 | 에러 복구, 재시도 로직 |

## 🔑 Key Concepts

### 1. Pydantic Integration
- **Type Safety**: 강력한 타입 검증
- **Nested Models**: 중첩된 데이터 구조 지원
- **Validation**: 자동 데이터 검증 및 에러 처리

### 2. Output Format Control
- **Format Instructions**: LLM에게 출력 형식 지시
- **Structured Output**: 일관된 구조화 데이터
- **Schema Definition**: 명확한 스키마 정의

### 3. Error Handling
- **Retry Logic**: 파싱 실패 시 재시도
- **Auto-fixing**: LLM을 사용한 자동 오류 수정
- **Fallback**: 대체 파서 체인

## 🛤️ Learning Path

```
1. Pydantic 기본 (01) ← 가장 중요
   ↓
2. 간단한 리스트/구조화 (02, 03)
   ↓
3. JSON/DataFrame (04, 05)
   ↓
4. 특수 타입 (06, 07)
   ↓
5. 에러 처리 (08)
```

## 💡 Quick Start Examples

### PydanticOutputParser
```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="person's name")
    age: int = Field(description="person's age")

parser = PydanticOutputParser(pydantic_object=Person)

# 프롬프트에 format instructions 추가
prompt = PromptTemplate(
    template="Extract person info.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

### JsonOutputParser
```python
from langchain.output_parsers import JsonOutputParser

parser = JsonOutputParser()
chain = prompt | llm | parser

result = chain.invoke({"query": "List top 3 programming languages"})
# Output: {"languages": ["Python", "JavaScript", "Java"]}
```

### OutputFixingParser
```python
from langchain.output_parsers import OutputFixingParser

# 기본 파서가 실패하면 LLM이 수정 시도
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI()
)
```

## 🎯 Parser Selection Guide

| Use Case | Recommended Parser |
|----------|-------------------|
| 복잡한 중첩 구조 | PydanticOutputParser |
| 간단한 리스트 | CommaSeparatedListOutputParser |
| API 응답 형식 | JsonOutputParser |
| 데이터 분석 | PandasDataFrameOutputParser |
| 날짜/시간 추출 | DatetimeOutputParser |
| 제한된 선택지 | EnumOutputParser |
| 신뢰성 향상 | OutputFixingParser (wrapper) |

## 🔗 Related Sections
- **Previous**: `02-Prompt` - 프롬프트 템플릿
- **Next**: `04-Model` - 다양한 LLM 모델
- **Related**: `14-Chains` - Structured Output Chain

## 📖 Resources
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [LangChain Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [Structured Output Guide](https://platform.openai.com/docs/guides/structured-outputs)
