import streamlit as st
import pandas as pd

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40],
})

st.dataframe(df, use_container_width=True)

st.table(df)

# matric
col1, col2 = st.columns(2)
col1.metric(label="삼성전자", value="100,000원", delta="1,000원")
col2.metric(label="카카오", value="50,000원", delta="-1,000원")