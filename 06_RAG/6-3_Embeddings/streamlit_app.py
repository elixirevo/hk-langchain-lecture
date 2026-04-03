"""
Ch06-3 임베딩 탐색기 (Embedding Explorer)

문장 간 코사인 유사도 계산, 유사도 매트릭스 히트맵,
PCA/t-SNE 기반 임베딩 공간 2D/3D 시각화 데모
"""

import time

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from utils import (
    build_scatter_2d,
    build_scatter_3d,
    build_similarity_heatmap,
    cosine_similarity_matrix,
    cosine_similarity_pair,
    load_hf_embeddings,
    load_openai_embeddings,
    reduce_pca,
    reduce_tsne,
    similarity_level,
)

load_dotenv()

# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="임베딩 탐색기",
    page_icon="🔢",
    layout="wide",
)

st.title("임베딩 탐색기")
st.markdown(
    "텍스트를 벡터로 변환하고 의미적 유사도를 탐색해 보세요. "
    "OpenAI 또는 HuggingFace 모델을 선택하여 임베딩을 생성하고 시각화할 수 있습니다."
)

# ---------------------------------------------------------------------------
# 사이드바 — 설정
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("설정")

    embedding_model = st.selectbox(
        "임베딩 모델 종류",
        options=["openai", "huggingface"],
        format_func=lambda x: "OpenAI" if x == "openai" else "HuggingFace (로컬)",
        index=0,
    )

    if embedding_model == "openai":
        import os
        openai_model = st.selectbox(
            "OpenAI 모델",
            options=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            index=0,
            help="text-embedding-3-small — 비용 효율적\ntext-embedding-3-large — 고품질",
        )
        api_key = st.text_input(
            "OpenAI API 키",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help=".env 파일에 OPENAI_API_KEY가 설정되어 있으면 자동 불러옵니다.",
        )
        hf_model = None
    else:
        hf_model = st.selectbox(
            "HuggingFace 모델",
            options=[
                "intfloat/multilingual-e5-large-instruct",
                "intfloat/multilingual-e5-base",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ],
            index=0,
        )
        api_key = None
        openai_model = None

    st.divider()
    st.subheader("시각화 설정")
    viz_dim = st.radio("차원 축소 방향", options=["2D", "3D"], horizontal=True)
    viz_method = st.radio(
        "차원 축소 방법",
        options=["PCA", "t-SNE"],
        horizontal=True,
        help="PCA — 빠름, t-SNE — 군집 구조 파악에 유리",
    )

    st.divider()
    st.markdown(
        "**코사인 유사도 해석**\n"
        "- `0.9+` : 매우 높음 (거의 동일)\n"
        "- `0.75+` : 높음 (유사한 주제)\n"
        "- `0.55+` : 보통 (부분 관련)\n"
        "- `0.55-` : 낮음 (의미 다름)"
    )


# ---------------------------------------------------------------------------
# 헬퍼 — 임베딩 모델 선택
# ---------------------------------------------------------------------------
def get_embeddings():
    if embedding_model == "openai":
        if not api_key:
            st.error("사이드바에서 OpenAI API 키를 입력해 주세요.")
            st.stop()
        return load_openai_embeddings(openai_model, api_key)
    else:
        return load_hf_embeddings(hf_model)


# ---------------------------------------------------------------------------
# 탭 구성
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["두 문장 비교", "유사도 매트릭스", "임베딩 공간 시각화"])


# ===========================================================================
# 탭 1: 두 문장 유사도 비교
# ===========================================================================
with tab1:
    st.subheader("두 문장 간 코사인 유사도")
    st.caption("두 문장을 입력하고 의미적 유사도를 수치로 확인해 보세요.")

    col_a, col_b = st.columns(2)
    with col_a:
        sentence_a = st.text_area(
            "문장 A",
            value="우주 탐사의 미래는 어떻게 될까요?",
            height=110,
            key="s_a",
        )
    with col_b:
        sentence_b = st.text_area(
            "문장 B",
            value="화성 탐사 로봇이 물의 흔적을 발견했습니다.",
            height=110,
            key="s_b",
        )

    if st.button("유사도 계산", type="primary", key="btn_pair"):
        if not sentence_a.strip() or not sentence_b.strip():
            st.warning("두 문장을 모두 입력해 주세요.")
        else:
            with st.spinner("임베딩 생성 중..."):
                try:
                    t0 = time.perf_counter()
                    emb = get_embeddings()
                    vec_a = emb.embed_query(sentence_a)
                    vec_b = emb.embed_query(sentence_b)
                    latency = (time.perf_counter() - t0) * 1000
                    score = cosine_similarity_pair(vec_a, vec_b)
                    level, color = similarity_level(score)

                    st.divider()

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("코사인 유사도", f"{score:.4f}")
                    m2.metric("벡터 차원", f"{len(vec_a):,}")
                    m3.metric("임베딩 시간", f"{latency:.1f} ms")
                    m4.metric("유사도 수준", level.split(" — ")[0])

                    st.markdown(f"**분석**: :{color}[{level}]")
                    st.progress(float(np.clip(score, 0, 1)))

                    with st.expander("벡터 미리보기 (처음 10개 값)"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("문장 A 벡터")
                            st.json([round(v, 5) for v in vec_a[:10]])
                        with c2:
                            st.caption("문장 B 벡터")
                            st.json([round(v, 5) for v in vec_b[:10]])

                except Exception as e:
                    st.error(f"오류: {e}")


# ===========================================================================
# 탭 2: 유사도 매트릭스
# ===========================================================================
with tab2:
    st.subheader("여러 문장 간 코사인 유사도 매트릭스")
    st.caption("여러 문장을 한 줄에 하나씩 입력하면 쌍별 유사도 히트맵을 생성합니다.")

    default_sentences = (
        "우주 탐사의 미래는 어떻게 될까요?\n"
        "화성 탐사 로봇이 물의 흔적을 발견했습니다.\n"
        "SpaceX가 재사용 가능한 로켓 기술을 개발했습니다.\n"
        "오늘 점심 메뉴는 무엇인가요?\n"
        "인공지능이 의료 진단을 혁신하고 있습니다.\n"
        "자율주행차 기술이 교통 시스템을 변화시키고 있습니다.\n"
        "제임스 웹 망원경이 초기 우주의 모습을 촬영했습니다.\n"
        "새로운 딥러닝 모델이 이미지 인식 성능을 높였습니다."
    )

    sentences_input = st.text_area(
        "문장 목록 (한 줄에 하나씩, 최대 20개)",
        value=default_sentences,
        height=200,
    )

    viz_type = st.radio(
        "시각화 방식",
        options=["plotly 히트맵", "데이터프레임"],
        horizontal=True,
        key="matrix_viz",
    )

    if st.button("매트릭스 생성", type="primary", key="btn_matrix"):
        sentences = [s.strip() for s in sentences_input.splitlines() if s.strip()]
        if len(sentences) < 2:
            st.warning("최소 2개 이상의 문장을 입력해 주세요.")
        elif len(sentences) > 20:
            st.warning("최대 20개 문장까지 지원합니다.")
        else:
            prog = st.progress(0, text="임베딩 생성 중...")
            try:
                t0 = time.perf_counter()
                emb = get_embeddings()
                prog.progress(30, text="벡터 변환 중...")
                vectors = emb.embed_documents(sentences)
                prog.progress(70, text="유사도 매트릭스 계산 중...")
                sim_matrix = cosine_similarity_matrix(vectors)
                latency = (time.perf_counter() - t0) * 1000
                prog.progress(100, text="완료")
                prog.empty()

                labels = [s[:30] + "..." if len(s) > 30 else s for s in sentences]

                # 요약 메트릭
                n = len(sentences)
                upper = sim_matrix[np.triu_indices(n, k=1)]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("문장 수", n)
                m2.metric("평균 유사도", f"{np.mean(upper):.4f}")
                m3.metric("최대 유사도", f"{np.max(upper):.4f}")
                m4.metric("임베딩 시간", f"{latency:.0f} ms")

                st.divider()

                if viz_type == "plotly 히트맵":
                    fig = build_similarity_heatmap(sim_matrix, labels)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    import pandas as pd
                    df = pd.DataFrame(np.round(sim_matrix, 4), index=labels, columns=labels)
                    st.dataframe(
                        df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1)
                          .format("{:.4f}"),
                        use_container_width=True,
                    )

                # 상위/하위 쌍 표시
                idx_sorted = np.argsort(upper)
                st.subheader("유사도 순위")
                top_col, bot_col = st.columns(2)
                with top_col:
                    st.markdown("**가장 유사한 쌍 (상위 3개)**")
                    for i in idx_sorted[-3:][::-1]:
                        r, c = np.triu_indices(n, k=1)
                        st.markdown(
                            f"- `{sentences[r[i]][:25]}...` ↔ `{sentences[c[i]][:25]}...` → **{upper[i]:.4f}**"
                        )
                with bot_col:
                    st.markdown("**가장 다른 쌍 (하위 3개)**")
                    for i in idx_sorted[:3]:
                        r, c = np.triu_indices(n, k=1)
                        st.markdown(
                            f"- `{sentences[r[i]][:25]}...` ↔ `{sentences[c[i]][:25]}...` → **{upper[i]:.4f}**"
                        )

            except Exception as e:
                prog.empty()
                st.error(f"오류: {e}")


# ===========================================================================
# 탭 3: 임베딩 공간 시각화 (PCA / t-SNE)
# ===========================================================================
with tab3:
    st.subheader("임베딩 공간 시각화")
    st.caption(
        "여러 문장의 임베딩을 PCA 또는 t-SNE로 차원 축소하여 2D/3D 공간에 시각화합니다. "
        "의미가 비슷한 문장끼리 가까이 모이는지 확인해 보세요."
    )

    default_viz = (
        "우주 탐사의 미래는 어떻게 될까요?\n"
        "화성 탐사 로봇이 물의 흔적을 발견했습니다.\n"
        "SpaceX가 재사용 가능한 로켓 기술을 개발했습니다.\n"
        "제임스 웹 망원경이 초기 우주의 모습을 촬영했습니다.\n"
        "오늘 점심 메뉴는 무엇인가요?\n"
        "저녁에 치킨을 먹으러 갔습니다.\n"
        "인공지능이 의료 진단을 혁신하고 있습니다.\n"
        "딥러닝 모델이 이미지 분류 성능을 높였습니다.\n"
        "자율주행차 기술이 교통 시스템을 변화시키고 있습니다.\n"
        "전기차 배터리 기술이 빠르게 발전하고 있습니다."
    )

    viz_sentences_input = st.text_area(
        "문장 목록 (한 줄에 하나씩, 최소 4개 이상 권장)",
        value=default_viz,
        height=220,
        key="viz_text",
    )

    if st.button("시각화 생성", type="primary", key="btn_viz"):
        viz_sentences = [s.strip() for s in viz_sentences_input.splitlines() if s.strip()]
        n_dim = int(viz_dim[0])  # "2D" -> 2, "3D" -> 3

        if len(viz_sentences) < 2:
            st.warning("최소 2개 이상의 문장을 입력해 주세요.")
        elif n_dim == 3 and len(viz_sentences) < 4:
            st.warning("3D 시각화는 최소 4개 이상의 문장이 필요합니다.")
        elif len(viz_sentences) > 50:
            st.warning("최대 50개 문장까지 지원합니다.")
        else:
            prog = st.progress(0, text="임베딩 생성 중...")
            try:
                t0 = time.perf_counter()
                emb = get_embeddings()
                prog.progress(30, text="벡터 변환 중...")
                vectors = emb.embed_documents(viz_sentences)
                prog.progress(60, text=f"{viz_method} 차원 축소 중...")

                if viz_method == "PCA":
                    reduced = reduce_pca(vectors, n_components=n_dim)
                else:
                    perplexity = min(5, len(viz_sentences) - 1)
                    reduced = reduce_tsne(vectors, n_components=n_dim, perplexity=perplexity)

                latency = (time.perf_counter() - t0) * 1000
                prog.progress(100, text="완료")
                prog.empty()

                m1, m2, m3 = st.columns(3)
                m1.metric("문장 수", len(viz_sentences))
                m2.metric("원본 차원", f"{len(vectors[0]):,}")
                m3.metric("처리 시간", f"{latency:.0f} ms")

                title = f"임베딩 공간 ({viz_dim}) — {viz_method}"
                if n_dim == 2:
                    fig = build_scatter_2d(reduced, viz_sentences, title)
                else:
                    fig = build_scatter_3d(reduced, viz_sentences, title)

                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    f"**{viz_method} 설명**: "
                    + (
                        "주성분 분석으로 분산이 가장 큰 방향을 기준으로 좌표계를 재구성합니다. "
                        "전역 구조 파악에 적합합니다."
                        if viz_method == "PCA"
                        else "비선형 차원 축소 알고리즘으로 지역 군집 구조를 잘 보존합니다. "
                        "의미적 클러스터 확인에 적합합니다."
                    )
                )

            except Exception as e:
                prog.empty()
                st.error(f"오류: {e}")
