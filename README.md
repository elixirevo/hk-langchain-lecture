# LangChain 실습

## 커맨드

```bash
# 파이썬 버전 고정
uv python pin 3.11


# 가상환경 생성
uv sync

# 가상환경 활성화
# 맥, 리눅스
source .venv/bin/activate

# 윈도우
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1

# 패키지 설치 (안해도됨)
uv add langchain langchain-openai python-dotenv
```
