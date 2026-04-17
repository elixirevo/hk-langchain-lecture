"""
Ch06-2 Chunking - Streamlit 데모 앱

텍스트를 입력하거나 파일을 업로드하면, 여러 텍스트 분할기를 선택하여
청크 결과를 색상으로 시각화하고 분할기별 메트릭을 비교합니다.

실행 방법:
    cd notebooks/06_RAG/6-2_Chunking
    streamlit run streamlit_app.py
"""

import os
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from utils import (
    BREAKPOINT_TYPES,
    SPLITTER_DESCRIPTIONS,
    SPLITTER_KEYS,
    build_chunks_html,
    build_overlap_html,
    chunks_to_csv_bytes,
    chunks_to_json,
    compare_splitters,
    split_text,
)

load_dotenv()

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(
    page_title="Chunking 비교 데모",
    page_icon="✂️",
    layout="wide",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .chunk-container {
        max-height: 520px;
        overflow-y: auto;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 8px;
        background: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    # API 키
    st.subheader("API 키")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="SemanticChunker 사용 시 필요합니다.",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        st.success("API 키 적용됨", icon="✅")
    elif not os.getenv("OPENAI_API_KEY"):
        st.warning("SemanticChunker를 사용하려면 API 키를 입력하세요.", icon="⚠️")

    st.divider()

    # 분할기 선택
    st.subheader("분할기 선택")
    splitter_key = st.selectbox(
        "텍스트 분할기",
        SPLITTER_KEYS,
        index=1,
        help="사용할 텍스트 분할기를 선택합니다.",
    )
    st.caption(SPLITTER_DESCRIPTIONS.get(splitter_key, ""))

    st.divider()

    # 공통 파라미터
    st.subheader("공통 파라미터")
    chunk_size = st.slider(
        "chunk_size",
        min_value=50,
        max_value=2000,
        value=250,
        step=50,
        help="청크 하나의 최대 크기 (글자 수 또는 토큰 수)",
    )
    chunk_overlap = st.slider(
        "chunk_overlap",
        min_value=0,
        max_value=min(chunk_size // 2, 500),
        value=min(50, chunk_size // 5),
        step=10,
        help="인접 청크 간 중복 영역 크기",
    )

    # 분할기별 추가 옵션
    extra_kwargs: dict = {}

    if splitter_key == "CharacterTextSplitter":
        st.divider()
        st.subheader("CharacterTextSplitter 옵션")
        separator = st.selectbox(
            "구분자 (separator)",
            ["\\n\\n", "\\n", ". ", " ", ""],
            index=0,
            help="텍스트를 나눌 구분자. \\n\\n은 단락, \\n은 줄 단위입니다.",
        )
        sep_map = {"\\n\\n": "\n\n", "\\n": "\n"}
        extra_kwargs["separator"] = sep_map.get(separator, separator)

    elif splitter_key == "RecursiveCharacterTextSplitter":
        st.divider()
        st.subheader("RecursiveCharacterTextSplitter 옵션")
        use_korean = st.checkbox(
            "한글 문장 부호 구분자 추가",
            value=False,
            help="'. ', '? ', '! ' 를 구분자에 추가합니다.",
        )
        if use_korean:
            extra_kwargs["separators"] = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    elif splitter_key == "TokenTextSplitter":
        st.divider()
        st.subheader("TokenTextSplitter 옵션")
        encoding_name = st.selectbox(
            "인코딩",
            ["cl100k_base", "p50k_base", "r50k_base"],
            index=0,
            help="cl100k_base: GPT-4/GPT-3.5, p50k_base: GPT-3 계열",
        )
        extra_kwargs["encoding_name"] = encoding_name

    elif "SemanticChunker" in splitter_key:
        st.divider()
        st.subheader("SemanticChunker 옵션")
        breakpoint_label = st.selectbox(
            "Breakpoint 방식",
            list(BREAKPOINT_TYPES.keys()),
            index=0,
        )
        extra_kwargs["breakpoint_type"] = BREAKPOINT_TYPES[breakpoint_label]

        if extra_kwargs["breakpoint_type"] == "percentile":
            extra_kwargs["breakpoint_amount"] = st.slider(
                "Threshold (백분위수)",
                min_value=50.0, max_value=99.0, value=95.0, step=1.0,
                help="높을수록 분할이 적게 일어납니다.",
            )
        elif extra_kwargs["breakpoint_type"] == "standard_deviation":
            extra_kwargs["breakpoint_amount"] = st.slider(
                "Threshold (표준편차 배수)",
                min_value=0.5, max_value=3.0, value=1.25, step=0.25,
                help="높을수록 분할이 적게 일어납니다.",
            )
        else:
            extra_kwargs["breakpoint_amount"] = st.slider(
                "Threshold (IQR 배수)",
                min_value=0.5, max_value=3.0, value=1.5, step=0.25,
            )

    st.divider()
    st.subheader("RAG 파이프라인 위치")
    st.markdown(
        """
```
✅ Load
✅ Chunk  ← 지금 여기
⬜ Embed
⬜ Store
⬜ Retrieve
⬜ Generate
```
"""
    )

    st.divider()
    st.subheader("분할기 비교")
    st.markdown(
        """
| 분할기 | 속도 | 정확도 | 비용 |
|--------|------|--------|------|
| Character | 빠름 | 낮음 | 없음 |
| Recursive | 빠름 | 보통 | 없음 |
| Token | 빠름 | 보통 | 없음 |
| Semantic | 느림 | 높음 | API |
"""
    )

# ── 메인 헤더 ─────────────────────────────────────────────────
st.title("✂️ Chunking 비교 데모")
st.markdown(
    """
RAG 파이프라인의 **두 번째 단계** — 긴 문서를 LLM이 처리할 수 있는 크기로 분할합니다.

`📄 Load` → `✂️ Chunk` → `🔢 Embed` → `🗄️ Store` → `🔍 Retrieve` → `💬 Generate`
"""
)

st.divider()

# ── 텍스트 입력 ────────────────────────────────────────────────
st.subheader("텍스트 입력")
input_mode = st.radio(
    "입력 방식",
    ["직접 입력", "파일 업로드", "샘플 텍스트 사용"],
    horizontal=True,
)

input_text = ""

if input_mode == "직접 입력":
    input_text = st.text_area(
        "분할할 텍스트를 입력하세요",
        height=180,
        placeholder="여기에 텍스트를 붙여넣으세요...",
    )

elif input_mode == "파일 업로드":
    uploaded_file = st.file_uploader(
        "텍스트 파일 업로드 (TXT, PDF)",
        type=["txt", "pdf"],
        help="TXT 또는 PDF 파일을 업로드합니다.",
    )
    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, mode="wb") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            if file_ext == ".txt":
                for enc in ("utf-8", "cp949", "euc-kr"):
                    try:
                        with open(tmp_path, encoding=enc) as f:
                            input_text = f.read()
                        break
                    except Exception:
                        continue
            elif file_ext == ".pdf":
                from langchain_community.document_loaders import PyMuPDFLoader
                docs = PyMuPDFLoader(tmp_path).load()
                input_text = "\n\n".join(d.page_content for d in docs)
        finally:
            os.unlink(tmp_path)

        if input_text:
            st.success(f"파일 로드 완료: {len(input_text):,}자")
            with st.expander("파일 내용 미리보기 (처음 300자)"):
                st.text(input_text[:300] + ("..." if len(input_text) > 300 else ""))

else:  # 샘플 텍스트
    sample_data_path = os.path.join(
        os.path.dirname(__file__), "data", "appendix-keywords.txt"
    )
    if os.path.exists(sample_data_path):
        for enc in ("utf-8", "cp949"):
            try:
                with open(sample_data_path, encoding=enc) as f:
                    input_text = f.read()
                break
            except Exception:
                continue
        st.success(f"샘플 텍스트 로드 완료: {len(input_text):,}자 (appendix-keywords.txt)")
    else:
        # 내장 샘플
        input_text = (
            "Semantic Search\n\n"
            "정의: 의미론적 검색은 사용자의 질의를 단순한 키워드 매칭을 넘어서 그 의미를 파악하여 관련된 결과를 반환하는 검색 방식입니다.\n"
            "예시: 사용자가 '태양계 행성'이라고 검색하면, '목성', '화성' 등과 같이 관련된 행성에 대한 정보를 반환합니다.\n"
            "연관키워드: 자연어 처리, 검색 알고리즘, 데이터 마이닝\n\n"
            "Embedding\n\n"
            "정의: 임베딩은 단어나 문장 같은 텍스트 데이터를 저차원의 연속적인 벡터로 변환하는 과정입니다. 이를 통해 컴퓨터가 텍스트를 이해하고 처리할 수 있게 합니다.\n"
            "예시: '사과'라는 단어를 [0.65, -0.23, 0.17]과 같은 벡터로 표현합니다.\n"
            "연관키워드: 자연어 처리, 벡터화, 딥러닝\n\n"
            "Token\n\n"
            "정의: 토큰은 텍스트를 더 작은 단위로 분할하는 것을 의미합니다. 이는 일반적으로 단어, 문장, 또는 구절일 수 있습니다.\n"
            "예시: 문장 '나는 학교에 간다'를 '나는', '학교에', '간다'로 분할합니다.\n"
            "연관키워드: 토큰화, 자연어 처리, 구문 분석\n\n"
            "RAG\n\n"
            "정의: RAG(Retrieval-Augmented Generation)는 검색 기반 문서를 활용하여 LLM의 답변 품질을 향상시키는 기법입니다.\n"
            "예시: 질문에 대해 관련 문서를 먼저 검색한 후, 해당 문서를 컨텍스트로 제공하여 LLM이 더 정확한 답변을 생성합니다.\n"
            "연관키워드: 벡터 데이터베이스, 임베딩, LLM\n\n"
            "Vector Store\n\n"
            "정의: 벡터 스토어는 임베딩 벡터를 효율적으로 저장하고 검색할 수 있는 데이터베이스입니다.\n"
            "예시: FAISS, Chroma, Pinecone 등이 대표적인 벡터 스토어입니다.\n"
            "연관키워드: RAG, 유사도 검색, 벡터 인덱스"
        )
        st.info(f"내장 샘플 텍스트 사용 중: {len(input_text):,}자")

# ── 분할 실행 ─────────────────────────────────────────────────
if input_text.strip():
    st.divider()

    # 텍스트 정보
    col_ti1, col_ti2 = st.columns(2)
    col_ti1.metric("입력 텍스트 길이", f"{len(input_text):,}자")
    col_ti2.metric("선택된 분할기", splitter_key)

    # 분할 실행 버튼
    run_btn = st.button("분할 실행", type="primary", use_container_width=False)

    if run_btn or input_mode == "샘플 텍스트 사용":
        with st.spinner("텍스트 분할 중..."):
            result = split_text(
                input_text,
                splitter_key,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **extra_kwargs,
            )

        if result.error:
            st.error(f"분할 실패: {result.error}")
        elif result.success:
            chunks = result.chunks

            st.subheader("분할 결과")

            # 메트릭
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("청크 수", f"{result.count}개")
            m2.metric("평균 글자 수", f"{result.avg_chars:.0f}자")
            m3.metric("최솟값", f"{result.min_chars:,}자")
            m4.metric("최댓값", f"{result.max_chars:,}자")
            m5.metric("소요 시간", f"{result.elapsed_sec:.3f}초")

            st.divider()

            # ── 탭 ───────────────────────────────────────────────
            tab_visual, tab_overlap, tab_compare, tab_chart, tab_export = st.tabs(
                ["🎨 청크 시각화", "🔗 오버랩 확인", "📊 분할기 비교", "📈 분포 차트", "💾 내보내기"]
            )

            # 탭1: 청크 시각화
            with tab_visual:
                max_display = st.slider(
                    "표시할 청크 수",
                    min_value=1,
                    max_value=min(50, result.count),
                    value=min(15, result.count),
                    key="visual_slider",
                )
                st.markdown(
                    f'<div class="chunk-container">{build_chunks_html(chunks, max_display)}</div>',
                    unsafe_allow_html=True,
                )
                if result.count > max_display:
                    st.caption(f"전체 {result.count}개 중 {max_display}개 표시 중")

            # 탭2: 오버랩 확인
            with tab_overlap:
                if chunk_overlap > 0:
                    max_overlap_pairs = st.slider(
                        "표시할 인접 청크 쌍 수",
                        min_value=1,
                        max_value=min(10, result.count - 1),
                        value=min(3, result.count - 1),
                        key="overlap_slider",
                    )
                    st.markdown(
                        build_overlap_html(chunks, max_overlap_pairs),
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("chunk_overlap이 0으로 설정되어 있습니다. 사이드바에서 값을 높이면 오버랩 영역을 확인할 수 있습니다.")

            # 탭3: 분할기 비교
            with tab_compare:
                st.markdown(
                    "같은 텍스트와 파라미터로 Character / Recursive / Token 분할기를 비교합니다."
                    " (SemanticChunker는 API 비용 관계로 제외)"
                )
                with st.spinner("비교 중..."):
                    compare_df = compare_splitters(input_text, chunk_size, chunk_overlap)
                st.dataframe(compare_df, use_container_width=True, hide_index=True)

                # 청크 수 막대 그래프
                valid_rows = compare_df[compare_df["청크 수"] != "오류"].copy()
                if not valid_rows.empty:
                    valid_rows["청크 수"] = valid_rows["청크 수"].astype(int)
                    fig_bar = px.bar(
                        valid_rows,
                        x="분할기",
                        y="청크 수",
                        title="분할기별 청크 수 비교",
                        color="분할기",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig_bar.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            # 탭4: 분포 차트
            with tab_chart:
                char_counts = result.char_counts
                df_dist = pd.DataFrame(
                    {
                        "청크": [f"청크 {i+1}" for i in range(len(char_counts))],
                        "글자 수": char_counts,
                    }
                )

                # 히스토그램
                fig_hist = px.histogram(
                    df_dist,
                    x="글자 수",
                    nbins=min(30, result.count),
                    title=f"청크 글자 수 분포 ({splitter_key})",
                    labels={"글자 수": "글자 수 (chars)", "count": "빈도"},
                    color_discrete_sequence=["#22c55e"],
                )
                fig_hist.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
                fig_hist.add_vline(
                    x=result.avg_chars,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"평균 {result.avg_chars:.0f}자",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                # 청크별 글자 수 꺾은선 그래프
                fig_line = px.line(
                    df_dist,
                    x="청크",
                    y="글자 수",
                    title="청크별 글자 수",
                    markers=True,
                    color_discrete_sequence=["#3b82f6"],
                )
                fig_line.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis_tickangle=-45,
                )
                fig_line.add_hline(
                    y=chunk_size,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"chunk_size={chunk_size}",
                )
                st.plotly_chart(fig_line, use_container_width=True)

                # 기술 통계
                st.markdown("#### 기술 통계")
                st.dataframe(
                    result.to_stats_df(),
                    use_container_width=True,
                    hide_index=True,
                )

            # 탭5: 내보내기
            with tab_export:
                st.markdown("#### 분할 결과 내보내기")
                col_dl1, col_dl2 = st.columns(2)

                safe_name = splitter_key.replace(" ", "_").replace("(", "").replace(")", "")

                with col_dl1:
                    st.download_button(
                        label="JSON으로 내보내기",
                        data=chunks_to_json(chunks, result.splitter_name).encode("utf-8"),
                        file_name=f"chunks_{safe_name}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                with col_dl2:
                    st.download_button(
                        label="CSV로 내보내기",
                        data=chunks_to_csv_bytes(chunks),
                        file_name=f"chunks_{safe_name}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                st.markdown("#### JSON 미리보기 (처음 3개)")
                import json as _json
                preview = [
                    {"index": i, "content": c, "char_count": len(c)}
                    for i, c in enumerate(chunks[:3])
                ]
                st.json({"splitter": result.splitter_name, "chunks": preview})

        else:
            st.warning("분할 결과가 없습니다. 텍스트와 파라미터를 확인해 주세요.")

else:
    # 입력 전 안내
    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.subheader("CharacterTextSplitter")
        st.code(
            '''from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\\n\\n",   # 구분자
    chunk_size=250,        # 최대 크기
    chunk_overlap=50,      # 중복 영역
)
chunks = splitter.split_text(text)''',
            language="python",
        )

    with col_info2:
        st.subheader("RecursiveCharacterTextSplitter")
        st.code(
            '''from langchain_text_splitters import RecursiveCharacterTextSplitter

# 범용 권장 방식
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50,
    # 기본 구분자: ["\\n\\n", "\\n", " ", ""]
)
chunks = splitter.split_text(text)''',
            language="python",
        )

    st.info(
        "위에서 '샘플 텍스트 사용'을 선택하거나 텍스트를 입력하면 분할 결과를 확인할 수 있습니다."
    )
