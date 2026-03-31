import streamlit as st
from datetime import datetime as dt
import datetime

button = st.button("버튼을 눌러보세요!")

# 버튼을 눌렀을 때 수행할 행동
if button:
    st.write(":blue[버튼]이 눌렸습니당.")

# 체크 박스
agree = st.checkbox("동의?")

if agree:
    st.write("동의해 주셔서 감사합니다")

# 라디오 버튼
mbti = st.radio(
    "MBTI를 선택하세요",
    ("ISTJ", "ENFP", "선택지 없음")
)

if mbti == "ISTJ":
    st.write("아 ISTJ 시구나")
elif mbti == "ENFP":
    st.write("네네 ENFP 시군요")
else:
    st.write("선택 해주세요 ㅠㅠ")

# 선택 박스
mbti = st.selectbox(
    "MBTI를 선택하세요",
    ("ISTJ", "ENFP", "선택지 없음")
)

if mbti == "ISTJ":
    st.write("아 ISTJ 시구나")
elif mbti == "ENFP":
    st.write("네네 ENFP 시군요")
else:
    st.write("선택 해주세요 ㅠㅠ")

# 다중 선택 박스
options = st.multiselect(
    "좋아하는 과일 고르세요",
    ["망고", "오렌지", "사과", "바나나"], # 옵션 리스트
    ["망고", "사과"] # 기본 선택
)
st.write(f"선택한 내용 : {options}")

# 슬라이더
values = st.slider(
    "범위를 지정하기 위해 사용",
    min_value=0.0,
    max_value=100.0,
    value=(25.0, 75.0)
)
st.write(f"선택 범위 : {values}")

# 텍스트 입력
title = st.text_input(
    label="오늘 저녁 뭐먹음?",
    placeholder="여기에 메뉴를 적어주세요!"
)

st.write(f"오늘 저녁 --> {title}")