"""
Ch06-1 Document Loaders - Streamlit 데모 앱

파일을 업로드하면 자동으로 적절한 Document Loader를 선택하여
LangChain Document 객체로 변환하고, 내용과 메타데이터를 시각화합니다.

실행 방법:
    cd notebooks/06_RAG/6-1_DocumentLoaders
    streamlit run streamlit_app.py
"""

import os
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from utils import (
    CSV_DELIMITERS,
    PDF_LOADERS,
    docs_to_csv_bytes,
    docs_to_json,
    get_metadata_df,
    get_stats_df,
    load_file,
)

load_dotenv()

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Loaders 데모",
    page_icon="📄",
    layout="wide",
)

# ── CSS 스타일 ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .doc-preview {
        background-color: #fafafa;
        border-left: 4px solid #1f77b4;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        word-break: break-word;
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
        "OpenAI API Key (선택)",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="현재 앱에서는 사용되지 않지만, 후속 RAG 파이프라인(Embed 단계)에서 필요합니다.",
    )
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        st.success("API 키 적용됨", icon="✅")

    st.divider()

    # PDF 로더 선택
    st.subheader("PDF 로더 선택")
    pdf_loader_key = st.radio(
        "PDF 파서",
        list(PDF_LOADERS.keys()),
        help="PDF 파일 업로드 시에만 적용됩니다.",
    )

    st.divider()

    # CSV 옵션
    st.subheader("CSV 옵션")
    csv_source_col = st.text_input(
        "source 컬럼명 (선택)",
        placeholder="예: Name",
        help="해당 컬럼 값이 Document의 source 메타데이터로 저장됩니다.",
    )
    csv_delimiter_label = st.selectbox(
        "구분자",
        list(CSV_DELIMITERS.keys()),
        index=0,
    )
    csv_delimiter = CSV_DELIMITERS[csv_delimiter_label]

    st.divider()

    # 미리보기 옵션
    st.subheader("미리보기 옵션")
    preview_chars = st.slider("미리보기 글자 수", 100, 1000, 300, 100)

    st.divider()

    st.subheader("RAG 파이프라인 위치")
    st.markdown(
        """
```
✅ Load  ← 지금 여기
⬜ Chunk
⬜ Embed
⬜ Store
⬜ Retrieve
⬜ Generate
```
"""
    )

    st.divider()
    st.subheader("로더 선택 가이드")
    st.markdown(
        """
| 상황 | 추천 로더 |
|------|----------|
| 일반 PDF | PyMuPDF |
| 테이블 PDF | PDFPlumber |
| 스캔 PDF | PyPDF+OCR |
| 데이터 CSV | CSVLoader |
| 일반 텍스트 | TextLoader |
| JSON 데이터 | JSONLoader |
"""
    )

# ── 메인 헤더 ─────────────────────────────────────────────────
st.title("📄 Document Loaders 데모")
st.markdown(
    """
RAG 파이프라인의 **첫 번째 단계** — 파일을 업로드하면 LangChain Document Loader가
자동으로 선택되어 `Document` 객체로 변환합니다.

`📄 Load` → `✂️ Chunk` → `🔢 Embed` → `🗄️ Store` → `🔍 Retrieve` → `💬 Generate`
"""
)

with st.expander("지원 파일 형식 및 로더 정보", expanded=False):
    st.markdown(
        """
| 형식 | 확장자 | 로더 | 특징 |
|------|--------|------|------|
| PDF | `.pdf` | PyMuPDFLoader / PyPDFLoader / PDFPlumberLoader | 페이지별 Document 생성 |
| CSV | `.csv` | CSVLoader | 행 단위로 Document 생성 |
| 텍스트 | `.txt` | TextLoader | 파일 전체를 하나의 Document로 |
| JSON | `.json` | JSONLoader (커스텀) | 항목별 Document 생성 |
"""
    )

st.divider()

# ── 파일 업로더 ────────────────────────────────────────────────
st.subheader("파일 업로드")
uploaded_file = st.file_uploader(
    "파일을 여기에 드래그하거나 클릭하여 선택하세요",
    type=["pdf", "csv", "txt", "json"],
    help="지원 형식: PDF, CSV, TXT, JSON",
)

# ── 로더 실행 ─────────────────────────────────────────────────
if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    file_size_kb = len(uploaded_file.getbuffer()) / 1024

    # 파일 정보 메트릭
    col_fi1, col_fi2, col_fi3 = st.columns(3)
    col_fi1.metric("파일명", uploaded_file.name)
    col_fi2.metric("파일 크기", f"{file_size_kb:.1f} KB")
    col_fi3.metric("파일 형식", file_ext.upper())

    # 임시 파일에 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, mode="wb") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    st.divider()

    try:
        # 진행 표시
        progress_bar = st.progress(0, text="로딩 준비 중...")

        progress_bar.progress(30, text="파일 로딩 중...")

        result = load_file(
            tmp_path,
            file_ext,
            pdf_loader_key=pdf_loader_key,
            delimiter=csv_delimiter,
            source_column=csv_source_col.strip() or None,
            source_name=uploaded_file.name,
        )

        progress_bar.progress(100, text="로딩 완료!")

        if result.error:
            st.error(f"로딩 실패: {result.error}")
        elif result.success:
            docs = result.docs

            st.subheader("로딩 결과")

            # 핵심 메트릭
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("로더", result.loader_name.split("(")[0].strip(), help=result.loader_name)
            m2.metric(
                "Document 수",
                f"{result.count:,}개",
                help="PDF: 페이지 수 / CSV: 행 수 / TXT: 1개 / JSON: 항목 수",
            )
            m3.metric("전체 글자 수", f"{result.total_chars:,}자")
            m4.metric("평균 글자 수", f"{result.avg_chars:,}자")
            m5.metric("로딩 시간", f"{result.elapsed_sec:.3f}초")

            st.divider()

            # ── 탭 ───────────────────────────────────────────────
            tab_preview, tab_meta, tab_stats, tab_export = st.tabs(
                ["📄 Document 미리보기", "📋 메타데이터", "📊 통계 차트", "💾 내보내기"]
            )

            # 탭1: 미리보기
            with tab_preview:
                show_count = st.slider(
                    "표시할 Document 수",
                    min_value=1,
                    max_value=min(20, result.count),
                    value=min(5, result.count),
                    key="preview_slider",
                )

                for i, doc in enumerate(docs[:show_count]):
                    if file_ext == ".pdf":
                        label = f"페이지 {doc.metadata.get('page', i) + 1}"
                    elif file_ext == ".csv":
                        label = f"행 {i + 1}"
                    else:
                        label = f"Document {i + 1}"

                    with st.expander(
                        f"{label}  |  {len(doc.page_content):,}자",
                        expanded=(i == 0),
                    ):
                        content = doc.page_content
                        display = (
                            content[:preview_chars]
                            + f"\n\n... ({len(content):,}자 중 {preview_chars}자 표시)"
                            if len(content) > preview_chars
                            else content
                        )
                        st.markdown(
                            f'<div class="doc-preview">{display}</div>',
                            unsafe_allow_html=True,
                        )
                        if doc.metadata:
                            st.caption("메타데이터:")
                            st.json(doc.metadata, expanded=False)

                if result.count > show_count:
                    st.info(f"전체 {result.count:,}개 중 {show_count}개 표시 중")

            # 탭2: 메타데이터
            with tab_meta:
                st.markdown("#### 첫 번째 Document 메타데이터")
                meta = docs[0].metadata
                if meta:
                    meta_df = pd.DataFrame(
                        [{"키": k, "값": str(v)} for k, v in meta.items()]
                    )
                    st.dataframe(meta_df, use_container_width=True, hide_index=True)
                else:
                    st.info("메타데이터가 없습니다.")

                if result.count > 1:
                    st.markdown("#### 전체 메타데이터 표 (최대 20개)")
                    st.dataframe(
                        get_metadata_df(docs, limit=20),
                        use_container_width=True,
                        hide_index=True,
                    )

            # 탭3: 통계 차트
            with tab_stats:
                if result.count > 1:
                    st.markdown("#### Document별 글자 수 분포")
                    char_counts = result.char_counts
                    df_chars = pd.DataFrame(
                        {
                            "Document": [f"Doc {i+1}" for i in range(len(char_counts))],
                            "글자 수": char_counts,
                        }
                    )
                    fig_hist = px.histogram(
                        df_chars,
                        x="글자 수",
                        nbins=min(30, len(char_counts)),
                        title="Document 글자 수 히스토그램",
                        labels={"글자 수": "글자 수 (chars)", "count": "빈도"},
                        color_discrete_sequence=["#1f77b4"],
                    )
                    fig_hist.update_layout(
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    st.markdown("#### 기술 통계")
                    st.dataframe(
                        get_stats_df(char_counts),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Document가 2개 이상이어야 통계 차트를 표시할 수 있습니다.")

            # 탭4: 내보내기
            with tab_export:
                st.markdown("#### 처리된 Document 내보내기")

                base_name = os.path.splitext(uploaded_file.name)[0]
                col_dl1, col_dl2 = st.columns(2)

                with col_dl1:
                    st.download_button(
                        label="JSON으로 내보내기",
                        data=docs_to_json(docs).encode("utf-8"),
                        file_name=f"{base_name}_documents.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                with col_dl2:
                    st.download_button(
                        label="CSV로 내보내기",
                        data=docs_to_csv_bytes(docs),
                        file_name=f"{base_name}_documents.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                st.markdown("#### JSON 미리보기 (처음 3개)")
                import json as _json
                preview_export = [
                    {"index": i, "page_content": d.page_content, "metadata": d.metadata}
                    for i, d in enumerate(docs[:3])
                ]
                st.json(preview_export)

        else:
            st.warning("문서를 로드할 수 없습니다. 파일 형식을 확인해 주세요.")
            progress_bar.empty()

    except ImportError as e:
        st.error(
            f"필요한 패키지가 설치되지 않았습니다.\n\n`{e}`\n\n"
            "설치 명령: `pip install langchain-community pymupdf pdfplumber pypdf`"
        )
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    # 업로드 전 안내 화면
    col_intro1, col_intro2 = st.columns(2)

    with col_intro1:
        st.subheader("Document 객체 구조")
        st.code(
            '''from langchain_core.documents import Document

document = Document(
    page_content="LangChain은 LLM 앱 개발 프레임워크입니다.",
    metadata={
        "source": "tutorial.pdf",
        "page": 0,
        "total_pages": 10,
    }
)''',
            language="python",
        )

    with col_intro2:
        st.subheader("주요 Document Loader")
        st.code(
            '''# PDF (빠른 속도 + 풍부한 메타데이터)
from langchain_community.document_loaders import PyMuPDFLoader
docs = PyMuPDFLoader("file.pdf").load()

# CSV (행 단위 Document)
from langchain_community.document_loaders import CSVLoader
docs = CSVLoader("file.csv").load()

# TXT
from langchain_community.document_loaders import TextLoader
docs = TextLoader("file.txt", encoding="utf-8").load()

# 로딩 방식 비교
loader.load()            # 전체 로딩
loader.load_and_split()  # 로딩 + 분할
loader.lazy_load()       # 제너레이터 (메모리 절약)''',
            language="python",
        )

    st.info("위에서 파일을 업로드하면 Document Loader 결과를 확인할 수 있습니다.")
