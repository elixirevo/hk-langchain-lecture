"""
Ch06-4 벡터스토어 데모 (VectorStore Demo)

텍스트 또는 파일을 FAISS에 저장하고 Similarity / MMR 검색을 수행하는 데모
"""

import os

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from utils import (
    SearchSession,
    build_faiss_vectorstore,
    build_mmr_diversity_heatmap,
    build_score_bar,
    compute_pairwise_diversity,
    load_openai_embeddings,
    search_mmr,
    search_similarity,
    split_text,
)

load_dotenv()

# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="벡터스토어 데모",
    page_icon="🗄️",
    layout="wide",
)

st.title("벡터스토어 데모")
st.markdown(
    "텍스트나 파일을 **FAISS** 벡터스토어에 저장하고, "
    "**Similarity(유사도)** 또는 **MMR(다양성)** 검색을 체험해 보세요."
)

# ---------------------------------------------------------------------------
# 세션 상태 초기화
# ---------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "last_session" not in st.session_state:
    st.session_state.last_session = None
if "search_history" not in st.session_state:
    st.session_state.search_history: list[SearchSession] = []

# ---------------------------------------------------------------------------
# 사이드바 — 설정
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("설정")

    api_key = st.text_input(
        "OpenAI API 키",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help=".env 파일에 OPENAI_API_KEY가 설정되어 있으면 자동 불러옵니다.",
    )

    openai_model = st.selectbox(
        "임베딩 모델",
        options=["text-embedding-3-small", "text-embedding-3-large"],
        index=0,
    )

    st.divider()
    st.subheader("청킹 설정")
    chunk_size = st.slider("청크 크기 (글자)", 100, 1200, 400, 50)
    chunk_overlap = st.slider("청크 겹침 (글자)", 0, 200, 50, 10)

    st.divider()
    st.subheader("검색 파라미터")

    search_type = st.radio(
        "검색 방식",
        options=["similarity", "mmr"],
        format_func=lambda x: "유사도 검색 (Similarity)" if x == "similarity" else "다양성 검색 (MMR)",
        index=0,
    )

    k = st.slider("반환 문서 수 (k)", min_value=1, max_value=10, value=3)

    if search_type == "mmr":
        fetch_k = st.slider(
            "MMR 후보 수 (fetch_k)",
            min_value=k,
            max_value=30,
            value=max(k * 3, 10),
            help="후보 fetch_k개에서 다양성을 고려해 k개를 최종 선택합니다.",
        )
        lambda_mult = st.slider(
            "유사도-다양성 균형 (λ)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 = 최대 다양성  |  1 = 최대 유사도",
        )
    else:
        fetch_k = None
        lambda_mult = None

    st.divider()
    st.markdown(
        "**검색 방식 비교**\n\n"
        "- **Similarity**: 코사인 유사도 순서로 반환 (L2 거리 기준)\n"
        "- **MMR**: 유사도 + 결과 간 다양성을 함께 고려"
    )


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def get_embeddings():
    if not api_key:
        st.error("사이드바에서 OpenAI API 키를 입력해 주세요.")
        st.stop()
    return load_openai_embeddings(openai_model, api_key)


# ===========================================================================
# 섹션 1: 문서 입력
# ===========================================================================
st.header("1단계: 문서 입력")

input_method = st.radio(
    "입력 방식",
    options=["직접 입력", "파일 업로드"],
    horizontal=True,
)

default_text = """\
경복궁은 조선시대의 대표적인 궁궐로, 서울의 랜드마크입니다.
광화문을 통해 입장할 수 있으며, 아름다운 건축물과 넓은 정원이 특징입니다.

제주도는 한국의 대표적인 관광지로, 아름다운 자연경관과 독특한 문화를 자랑합니다.
한라산, 성산일출봉, 만장굴 등이 유명합니다.

부산은 한국 제2의 도시로, 해운대 해수욕장과 자갈치 시장이 유명합니다.
신선한 해산물과 아름다운 해변을 즐길 수 있습니다.

인사동은 서울의 전통 문화거리로, 한국 전통 공예품과 차를 판매하는 상점들이 있습니다.
주말에는 차 없는 거리로 운영됩니다.

N서울타워는 남산에 위치한 전망대로, 서울 전경을 한눈에 볼 수 있습니다.
야경이 특히 아름답고 연인들의 명소로도 유명합니다.

FAISS는 Facebook AI Research에서 개발한 고속 벡터 검색 라이브러리입니다.
대규모 문서 컬렉션에서 밀리초 단위의 빠른 유사도 검색이 가능합니다.

임베딩(Embedding)은 텍스트를 숫자 배열(벡터)로 변환하는 과정입니다.
비슷한 의미의 문장은 벡터 공간에서 가까이 위치하여 의미 기반 검색이 가능합니다.

RAG(Retrieval-Augmented Generation)는 검색과 생성을 결합한 AI 기법입니다.
외부 문서를 검색하여 LLM에 컨텍스트로 제공함으로써 정확도를 높입니다.

트랜스포머(Transformer)는 어텐션 메커니즘을 기반으로 한 딥러닝 아키텍처입니다.
BERT, GPT 등 대부분의 최신 NLP 모델의 기반이 됩니다.

자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 생성하는 기술입니다.
텍스트 분류, 기계 번역, 감성 분석 등 다양한 분야에 응용됩니다.
"""

if input_method == "직접 입력":
    user_text = st.text_area("텍스트 입력", value=default_text, height=250)
    source_label = "직접 입력"

    if st.button("벡터스토어 구축", type="primary", key="build_text"):
        if not user_text.strip():
            st.warning("텍스트를 입력해 주세요.")
        else:
            prog = st.progress(0, text="청킹 중...")
            try:
                chunks = split_text(user_text, source_label, chunk_size, chunk_overlap)
                prog.progress(40, text="임베딩 생성 중...")
                emb = get_embeddings()
                prog.progress(70, text="FAISS 인덱스 구축 중...")
                vs = build_faiss_vectorstore(chunks, emb)
                prog.progress(100, text="완료")
                prog.empty()
                st.session_state.vectorstore = vs
                st.session_state.doc_chunks = chunks
                st.session_state.last_session = None
                st.session_state.search_history = []
                st.success(f"벡터스토어 구축 완료 — {len(chunks)}개 청크 저장됨")
            except Exception as e:
                prog.empty()
                st.error(f"오류: {e}")

else:
    uploaded_files = st.file_uploader(
        "파일 업로드 (.txt, .md)",
        type=["txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("벡터스토어 구축", type="primary", key="build_file"):
        if not uploaded_files:
            st.warning("파일을 업로드해 주세요.")
        else:
            prog = st.progress(0, text="파일 처리 중...")
            try:
                all_chunks = []
                for i, f in enumerate(uploaded_files):
                    content = f.read().decode("utf-8", errors="ignore")
                    chunks = split_text(content, f.name, chunk_size, chunk_overlap)
                    all_chunks.extend(chunks)
                    prog.progress(int(30 * (i + 1) / len(uploaded_files)), text=f"{f.name} 처리 중...")

                prog.progress(40, text="임베딩 생성 중...")
                emb = get_embeddings()
                prog.progress(70, text="FAISS 인덱스 구축 중...")
                vs = build_faiss_vectorstore(all_chunks, emb)
                prog.progress(100, text="완료")
                prog.empty()
                st.session_state.vectorstore = vs
                st.session_state.doc_chunks = all_chunks
                st.session_state.last_session = None
                st.session_state.search_history = []
                st.success(
                    f"벡터스토어 구축 완료 — {len(uploaded_files)}개 파일, {len(all_chunks)}개 청크 저장됨"
                )
            except Exception as e:
                prog.empty()
                st.error(f"오류: {e}")


# ===========================================================================
# 섹션 2: 저장된 문서 현황
# ===========================================================================
if st.session_state.vectorstore is not None:
    st.divider()
    st.header("2단계: 저장된 문서 현황")

    chunks = st.session_state.doc_chunks
    sources = list({c.metadata.get("source", "알 수 없음") for c in chunks})
    avg_len = int(np.mean([len(c.page_content) for c in chunks]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 청크 수", len(chunks))
    c2.metric("고유 소스 수", len(sources))
    c3.metric("평균 청크 길이", f"{avg_len} 글자")
    c4.metric("청크 크기 설정", f"{chunk_size} / 겹침 {chunk_overlap}")

    with st.expander("저장된 청크 미리보기 (최대 5개)"):
        for i, chunk in enumerate(chunks[:5]):
            st.markdown(f"**청크 {i + 1}** — 출처: `{chunk.metadata.get('source', '?')}`")
            st.text(chunk.page_content[:250] + ("..." if len(chunk.page_content) > 250 else ""))
            if i < 4:
                st.divider()


    # ===========================================================================
    # 섹션 3: 유사 문서 검색
    # ===========================================================================
    st.divider()
    st.header("3단계: 유사 문서 검색")

    query = st.text_input(
        "검색어를 입력하세요",
        placeholder="예: 서울에서 전통 문화를 체험할 수 있는 곳",
    )

    col_run, col_clear = st.columns([1, 5])
    with col_run:
        run_search = st.button("검색", type="primary", key="search_btn")
    with col_clear:
        if st.button("검색 기록 초기화", key="clear_history"):
            st.session_state.search_history = []
            st.session_state.last_session = None

    if run_search:
        if not query.strip():
            st.warning("검색어를 입력해 주세요.")
        else:
            with st.spinner("검색 중..."):
                try:
                    emb = get_embeddings()
                    if search_type == "similarity":
                        session = search_similarity(
                            st.session_state.vectorstore, emb, query, k
                        )
                    else:
                        session = search_mmr(
                            st.session_state.vectorstore, emb, query, k, fetch_k, lambda_mult
                        )
                    st.session_state.last_session = session
                    st.session_state.search_history.insert(0, session)
                    if len(st.session_state.search_history) > 10:
                        st.session_state.search_history.pop()
                except Exception as e:
                    st.error(f"검색 오류: {e}")

    # 결과 표시
    session: SearchSession | None = st.session_state.last_session
    if session:
        results = session.results
        st.subheader(
            f"검색 결과 — {session.search_type.upper()}, 상위 {len(results)}개"
        )

        # 요약 메트릭
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("검색 방식", session.search_type.upper())
        m2.metric("결과 수", len(results))
        m3.metric("검색 시간", f"{session.latency_ms:.1f} ms")
        score_type_label = "L2 거리 (낮을수록 유사)" if session.search_type == "similarity" else "코사인 유사도 (높을수록 유사)"
        m4.metric("점수 유형", score_type_label.split(" ")[0])

        st.caption(f"점수 기준: {score_type_label}")

        # 개별 결과 점수 메트릭
        score_cols = st.columns(min(len(results), 5))
        for i, r in enumerate(results[:5]):
            score_cols[i].metric(f"#{r.rank} 점수", f"{r.score:.4f}")

        st.divider()

        # 결과 카드
        for r in results:
            with st.container():
                h_col, s_col = st.columns([3, 1])
                with h_col:
                    st.markdown(f"**#{r.rank}** — 출처: `{r.source}`")
                with s_col:
                    label = "L2 거리" if r.score_type == "l2_distance" else "코사인 유사도"
                    st.metric(label, f"{r.score:.4f}")
                st.text_area(
                    "",
                    value=r.content,
                    height=90,
                    key=f"res_{r.rank}_{session.query[:10]}",
                    disabled=True,
                    label_visibility="collapsed",
                )
                st.divider()

        # -----------------------------------------------------------------------
        # 시각화 탭: 점수 막대 / MMR 다양성
        # -----------------------------------------------------------------------
        if len(results) > 1:
            viz_tab1, viz_tab2 = st.tabs(["점수 막대그래프", "결과 다양성 히트맵"])

            with viz_tab1:
                fig_bar = build_score_bar(session)
                st.plotly_chart(fig_bar, use_container_width=True)

            with viz_tab2:
                with st.spinner("결과 문서 간 유사도 계산 중..."):
                    try:
                        emb = get_embeddings()
                        sim_mat = compute_pairwise_diversity(results, emb)
                        labels = [f"#{r.rank}: {r.content[:20]}..." for r in results]
                        avg_diversity = 1.0 - float(np.mean(sim_mat[np.triu_indices(len(results), k=1)]))

                        col_info, col_metric = st.columns([3, 1])
                        col_info.markdown(
                            "결과 문서 간의 유사도를 보여줍니다. "
                            "**값이 낮을수록 결과가 다양합니다** (MMR의 목표)."
                        )
                        col_metric.metric("평균 다양성 점수", f"{avg_diversity:.4f}")

                        fig_heat = build_mmr_diversity_heatmap(sim_mat, labels)
                        st.plotly_chart(fig_heat, use_container_width=True)
                    except Exception as e:
                        st.warning(f"다양성 히트맵 생성 실패: {e}")

    # ===========================================================================
    # 섹션 4: 검색 기록
    # ===========================================================================
    if st.session_state.search_history:
        st.divider()
        st.header("검색 기록")
        history = st.session_state.search_history

        with st.expander(f"최근 검색 {len(history)}건 보기"):
            for i, s in enumerate(history):
                col_q, col_t, col_n, col_lat = st.columns([3, 1, 1, 1])
                col_q.markdown(f"**{i + 1}.** {s.query}")
                col_t.markdown(f"`{s.search_type.upper()}`")
                col_n.markdown(f"결과 {len(s.results)}개")
                col_lat.markdown(f"{s.latency_ms:.1f} ms")

else:
    st.info(
        "먼저 텍스트를 입력하거나 파일을 업로드한 뒤 "
        "**'벡터스토어 구축'** 버튼을 눌러 주세요."
    )
