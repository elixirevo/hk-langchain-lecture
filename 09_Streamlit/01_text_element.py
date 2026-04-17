import streamlit as st

st.title("Hello Streamlit")
st.header("여기는 헤더입니다.")
st.subheader("여기는 서브헤더입니다.")


# 캡션
st.caption("여기는 캡션입니다.")

my_sample_code = """
import streamlit as st

st.title("Hello Streamlit")
st.header("여기는 헤더입니다.")
st.subheader("여기는 서브헤더입니다.")
st.caption("여기는 캡션입니다.")
"""

st.code(my_sample_code, language="python")

st.text("여기는 일반 텍스트")

# 마크다운
st.markdown("여기는 **마크다운**으로 입력된 텍스트")
st.markdown("여기는 :green[초록색] 입니다.")