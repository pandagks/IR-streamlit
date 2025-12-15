import streamlit as st
import pandas as pd
from module.pattern_mining import direct_pattern_mining

st.set_page_config(
    page_title="선배들이 들었던 교과, 비교과 확인",
    layout="wide",
    initial_sidebar_state="expanded")
st.title("선배와 역량 비교")

tab0, tab1, tab2, tab3 = st.tabs(['연관성 분석 설명','교과', '비교과', '교과+비교과'])


with tab0:
    st.markdown("""
### 📘 패턴마이닝 파라미터 설명  
여기에서 찾고 싶은 것은 간단히 말하면 다음과 같습니다.

  **“대기업에 간 선배들은 어떤 교과·비교과 조합을 많이 들었을까?”**  
  **“그 조합이 대기업 취업과 실제로 관련이 있을까?”**

이걸 자동으로 찾아주는 분석이 바로 **패턴마이닝**입니다.  
아래 값들은 “어떤 조합을 의미 있게 인정할지” 기준을 정하는 역할을 합니다.

---

### 🔹 max_items (조합 최대 길이)
**한 번에 몇 개의 활동을 묶어서 볼 것인가?**

예를 들어 max_items = 3이면 다음 같은 조합까지 탐색합니다:

- "직무기초수업"
- "직무기초수업 + 캡스톤디자인"
- "직무기초수업 + 비교과(멘토링) + 자격증 준비"

→ *조합이 길수록 더 구체적인 “패턴”을 발견할 수 있지만, 너무 길면 해당 학생 수가 적어져 의미가 줄어들 수 있습니다.*

---

### 🔹 min_support (최소 지지도)
**“이 조합이 실제로 학생들 중 얼마나 자주 등장했는가?”**  
예: min_support = 0.0001 → 전체 학생 중 0.01% 이상 등장해야 표시됨.

예시  
- “직무기초 + 멘토링 프로그램 참여”를 2명만 들었다면  
  → 너무 적어서 일반적인 패턴이라고 보기 어려움  
- “직무기초 + 캡스톤디자인”을 80명이 들었다면  
  → 충분히 의미 있는 패턴이라고 판단

→ *즉, 이 설정은 ‘너무 희귀한 조합은 빼자’는 기준입니다.*

---

### 🔹 min_confidence (최소 신뢰도)
**“이 조합을 가진 학생이 실제로 대기업에 간 비율이 얼마나 될까?”**

예를 들어 “직무기초 + 멘토링” 조합에 대한 신뢰도가 0.70이라면?

→ 이 조합을 들은 선배 중 **70%가 대기업에 갔다**는 의미입니다.

그래서:

- confidence가 높다 = **대기업 취업과 강하게 관련 있다**
- confidence가 낮다 = **그 조합을 들어도 대기업에 잘 안 갈 수 있다**

→ *즉, 신뢰도는 “대기업에 간 사람들과 정말 관련 있는지”를 확인하는 지표입니다.*

---

### 🔹 min_count (최소 출현 횟수)
**“최소 몇 명 이상이 들은 조합만 의미 있는 패턴으로 인정할까?”**

예:
- min_count = 10 → 최소 10명 이상이 실제로 들은 조합만 결과에 표시  
- min_count = 3  → 3명만 들어도 결과에 표시됨

→ *너무 적게 들은 조합은 일반화하기 어렵기 때문에, 최소 몇 명 이상일 때만 결과에 포함할지 결정하는 것입니다.*

---

### 요약 (대기업 취업 예시 기준으로만 정리)
- **max_items**  
  → “대기업에 간 선배들이 동시에 가진 활동 조합을 최대 몇 개 묶어서 볼까?”

- **min_support**  
  → “그 조합이 전체 학생 중 최소 어느 정도 등장해야 의미 있다고 볼까?”

- **min_confidence**  
  → “그 조합을 가진 선배 중 실제로 대기업에 간 비율이 최소 어느 정도여야 할까?”

- **min_count**  
  → “최소 몇 명 이상이 들은 조합만 패턴으로 인정할까?”

---

### 예시로 이해하기
예를 들어, 분석 결과에서 이런 패턴이 나왔다고 해본다면,

| 활동 조합 | count | confidence |
|----------|--------|--------------|
| 직무기초수업 + 멘토링 | 35명 | 0.72 |

이 의미는?

35명의 선배가 “직무기초 + 멘토링”을 들었고  
그 중 **72%가 실제로 대기업에 갔다**  
즉, 대기업에 가려는 후배에게 **의미 있는 조합**이라는 의미

이걸 자동으로 찾아주는 분석이 바로 지금 화면에서 실행하는 기능입니다.
"""
)


# ==============================================================
# ======================== TAB 1: 교과 ==========================
# ==============================================================
with tab1:
    st.title('교과 기반 연관 패턴 분석')
    
    df = pd.read_csv('./data/연관성분석_데이터셋(교과만).csv.gz')
    st.success("데이터 로딩 완료")

    filter_cols = ["학부(대학)", "직무", "기업구분"]

    filter_choice = st.selectbox(
        "데이터 필터 기준을 선택하세요",
        filter_cols,
        key="t1_filter_choice"
    )

    filter_values = df[filter_choice].dropna().unique()

    selected_filter_value = st.selectbox(
        f"{filter_choice} 값 선택",
        filter_values,
        key="t1_filter_value"
    )

    df_filtered = df[df[filter_choice] == selected_filter_value]
    st.write(f"선택된 데이터 수: **{len(df_filtered)}개**")


    # 타겟 컬럼 설정
    target_cols = [c for c in filter_cols if c != filter_choice]

    target_col = st.selectbox(
        "타겟 컬럼을 선택하세요",
        target_cols,
        key="t1_target_col"
    )

    target_value = st.selectbox(
        f"타겟 값 선택 ({target_col})",
        df_filtered[target_col].dropna().unique(),
        key="t1_target_value"
    )

    df_filtered["target"] = (df_filtered[target_col] == target_value).astype(int)

    st.markdown("---")  # 구분선

    # Binary feature 탐색
    st.subheader("선배들의 교과 탐지")

    binary_cols = [
        col for col in df_filtered.columns
        if df_filtered[col].dropna().isin([0, 1]).all()
    ]
    binary_cols = [col for col in binary_cols if col != "target"]

    all_df_binary = df_filtered[binary_cols]
    col_sum = all_df_binary.sum(axis=0)
    all_df_binary = all_df_binary.loc[:, col_sum >= 1]

    st.write(f"탐지된 교과 수: **{len(all_df_binary.columns)}개**")


    # 파라미터
    st.subheader("⚙️ 패턴마이닝 파라미터 설정")
    st.info("결과가 나오지 않으면, 밑의 파라미터를 조종하세요.")

    max_items = st.slider(
        "규칙 최대 항목 수(max_items)",
        1, 6, 3,
        key="t1_max_items"
    )

    min_count = st.slider(
        "최소 count 기준 (결과 필터)",
        1, 50, 10,
        key="t1_min_count"
    )
    min_conf = st.slider("min_confidence", 0.0, 1.0, 0.3, step=0.01)
    
    min_supp_raw = st.slider("min_support (x 0.0001)", 1, 50, 1)
    min_supp = min_supp_raw * 0.0001



    # 실행 버튼
    if st.button("🚀 패턴마이닝 실행", key="t1_run"):
        with st.spinner("분석 중..."):
            df_input = all_df_binary.join(df_filtered["target"])

            try:
                results = direct_pattern_mining(
                    data=df_input,
                    target_col="target",
                    min_confidence=min_conf,
                    min_support=min_supp,
                    max_items=max_items
                )
            except:
                st.warning("⚠️ 패턴을 생성할 수 없습니다.")
                st.stop()
        st.write(f"min_support = {min_supp:.4f}")
        st.success("완료!")

        st.subheader("📊 최종 결과")

        if results is None or len(results) == 0:
            st.warning("⚠️ 유효한 패턴이 없습니다.")
            st.stop()

        required_cols = {"count", "lift", "confidence"}
        if not required_cols.issubset(results.columns):
            st.warning("⚠️ 패턴 생성 데이터가 부족합니다.")
            st.stop()

        post_df = (
            results[results["count"] >= min_count]
            .sort_values("lift", ascending=False)
            .head(50)
        )
        # ======================
        # A. 조합 자체가 없는 경우
        # ======================
        if all_df_binary.sum().sum() == 0:
            st.error("""
            ### 😢 분석이 불가능합니다.
            현재 선택한 조건에서는 선배들의 교과/비교과 수강 기록(1로 표시된 항목)이 거의 없습니다.  
            즉, **데이터 자체에 조합이 존재하지 않아** 패턴을 만들 수 없습니다.

            👉 다른 조건(학부, 직무, 기업구분)을 선택해서 다시 시도해주세요.
            """)
            st.stop()

        # ======================
        # B. 패턴은 가능하지만 파라미터 문제로 탈락한 경우
        # ======================
        if post_df.empty:
            st.error("""
            ### 😢 결과가 없습니다.

            데이터는 존재하지만, 설정한 값이 너무 엄격해서  
            **모든 조합이 필터링된 상태**입니다.

            해결 방법:
            - min_support(지지도) ↓ 낮추기
            - min_confidence(신뢰도) ↓ 낮추기
            - min_count(최소 출현 수) ↓ 낮추기
            - max_items(조합 길이) ↑ 늘리기

            위 설정을 조정하면 결과가 나타날 수 있어요.
            """)
            st.stop()
        else:
            st.dataframe(post_df)

        st.markdown("---")  # 구분선

# ==============================================================
# ======================== TAB 2: 비교과 ========================
# ==============================================================


with tab2:
    st.title("📘 비교과 기반 연관 패턴 분석")
    
    df = pd.read_csv('./data/연관성분석_데이터셋(비교과만).csv.gz')
    st.success("데이터 로딩 완료")

    filter_cols = ["학부(대학)", "직무", "기업구분"]

    filter_choice = st.selectbox(
        "📌 분석 기준 컬럼을 선택하세요",
        filter_cols,
        key="t2_filter_choice"
    )

    filter_values = df[filter_choice].dropna().unique()

    selected_filter_value = st.selectbox(
        f"{filter_choice} 값 선택",
        filter_values,
        key="t2_filter_value"
    )

    df_filtered = df[df[filter_choice] == selected_filter_value]
    st.write(f"선택된 데이터 수: **{len(df_filtered)}개**")

    # 타겟 설정
    target_cols = [c for c in filter_cols if c != filter_choice]
    target_col = st.selectbox(
        "🎯 타겟 컬럼을 선택하세요",
        target_cols,
        key="t2_target_col"
    )

    target_value = st.selectbox(
        f"🎯 타겟 값 선택 ({target_col})",
        df_filtered[target_col].dropna().unique(),
        key="t2_target_value"
    )

    df_filtered["target"] = (df_filtered[target_col] == target_value).astype(int)

    st.markdown("---")

    # Binary 탐색
    st.subheader("선배들의 비교과 탐지")

    binary_cols = [
        col for col in df_filtered.columns
        if df_filtered[col].dropna().isin([0, 1]).all()
    ]
    binary_cols = [col for col in binary_cols if col != "target"]

    all_df_binary = df_filtered[binary_cols]
    col_sum = all_df_binary.sum(axis=0)
    all_df_binary = all_df_binary.loc[:, col_sum >= 1]

    st.write(f"감지된 비교과 항목 수: **{len(all_df_binary.columns)}개**")

    # 파라미터
    st.subheader("⚙️ 패턴마이닝 파라미터 설정")
    st.info("결과가 나오지 않으면 파라미터 값을 조정하세요.")

    max_items = st.slider("규칙 최대 항목 수(max_items)", 1, 6, 3, key="t2_max_items")
    min_count = st.slider("최소 count 기준", 1, 50, 10, key="t2_min_count")
    min_conf = st.slider("min_confidence", 0.0, 1.0, 0.3, step=0.01, key="t2_min_conf")
    min_supp_raw = st.slider("min_support (x0.0001)", 1, 50, 1, key="t2_min_supp")
    min_supp = min_supp_raw * 0.0001

    # 실행
    if st.button("🚀 패턴마이닝 실행", key="t2_run"):
        with st.spinner("분석 중..."):
            df_input = all_df_binary.join(df_filtered["target"])

            try:
                results = direct_pattern_mining(
                    data=df_input,
                    target_col="target",
                    min_confidence=min_conf,
                    min_support=min_supp,
                    max_items=max_items
                )
            except:
                st.warning("⚠️ 패턴 생성 불가")
                st.stop()

        st.write(f"min_support = {min_supp:.4f}")
        st.success("완료!")

        st.subheader("📊 최종 결과")

        if results is None or len(results) == 0:
            st.warning("⚠️ 유효한 패턴이 없습니다.")
            st.stop()

        required_cols = {"count", "lift", "confidence"}
        if not required_cols.issubset(results.columns):
            st.warning("⚠️ 패턴 생성 데이터 부족")
            st.stop()

        post_df = (
            results[results["count"] >= min_count]
            .sort_values("lift", ascending=False)
            .head(50)
        )
        # ======================
        # A. 조합 자체가 없는 경우
        # ======================
        if all_df_binary.sum().sum() == 0:
            st.error("""
            ### 😢 분석이 불가능합니다.
            현재 선택한 조건에서는 선배들의 교과/비교과 수강 기록(1로 표시된 항목)이 거의 없습니다.  
            즉, **데이터 자체에 조합이 존재하지 않아** 패턴을 만들 수 없습니다.

            👉 다른 조건(학부, 직무, 기업구분)을 선택해서 다시 시도해주세요.
            """)
            st.stop()

        # ======================
        # B. 패턴은 가능하지만 파라미터 문제로 탈락한 경우
        # ======================
        if post_df.empty:
            st.error("""
            ### 😢 결과가 없습니다.

            데이터는 존재하지만, 설정한 값이 너무 엄격해서  
            **모든 조합이 필터링된 상태**입니다.

            해결 방법:
            - min_support(지지도) ↓ 낮추기
            - min_confidence(신뢰도) ↓ 낮추기
            - min_count(최소 출현 수) ↓ 낮추기
            - max_items(조합 길이) ↑ 늘리기

            위 설정을 조정하면 결과가 나타날 수 있어요.
            """)
            st.stop()
        else:
            st.dataframe(post_df)

        st.markdown("---")


# ==============================================================
# ======================== TAB 3: 교과 + 비교과 =================
# ==============================================================

with tab3:
    st.title("📘 교과 + 비교과 기반 연관 패턴 분석")

    df = pd.read_csv('./data/연관성분석_데이터셋.csv.gz')
    
    st.success("데이터 로딩 완료")

    filter_cols = ["학부(대학)", "직무", "기업구분"]

    filter_choice = st.selectbox(
        "📌 분석 기준 컬럼을 선택하세요",
        filter_cols,
        key="t3_filter_choice"
    )

    filter_values = df[filter_choice].dropna().unique()

    selected_filter_value = st.selectbox(
        f"{filter_choice} 값 선택",
        filter_values,
        key="t3_filter_value"
    )

    df_filtered = df[df[filter_choice] == selected_filter_value]
    st.write(f"선택된 데이터 수: **{len(df_filtered)}개**")

    # 타겟 설정
    target_cols = [c for c in filter_cols if c != filter_choice]
    target_col = st.selectbox(
        "🎯 타겟 컬럼 선택",
        target_cols,
        key="t3_target_col"
    )

    target_value = st.selectbox(
        f"🎯 타겟 값 선택 ({target_col})",
        df_filtered[target_col].dropna().unique(),
        key="t3_target_value"
    )

    df_filtered["target"] = (df_filtered[target_col] == target_value).astype(int)

    st.markdown("---")

    # binary feature 탐색
    st.subheader("선배들의 교과 + 비교과 탐지")

    binary_cols = [
        col for col in df_filtered.columns
        if df_filtered[col].dropna().isin([0, 1]).all()
    ]
    binary_cols = [col for col in binary_cols if col != "target"]

    all_df_binary = df_filtered[binary_cols]
    col_sum = all_df_binary.sum(axis=0)
    all_df_binary = all_df_binary.loc[:, col_sum >= 1]

    st.write(f"감지된 항목 수: **{len(all_df_binary.columns)}개**")

    # 파라미터
    st.subheader("⚙️ 패턴마이닝 파라미터 설정")
    st.info("결과가 나오지 않으면 파라미터 값을 조정하세요.")

    max_items = st.slider("규칙 최대 항목 수(max_items)", 1, 6, 3, key="t3_max_items")
    min_count = st.slider("최소 count 기준", 1, 50, 10, key="t3_min_count")
    min_conf = st.slider("min_confidence", 0.0, 1.0, 0.3, step=0.01, key="t3_min_conf")
    min_supp_raw = st.slider("min_support (x0.0001)", 1, 50, 1, key="t3_min_supp")
    min_supp = min_supp_raw * 0.0001

    # 실행
    if st.button("🚀 패턴마이닝 실행", key="t3_run"):
        with st.spinner("분석 중..."):
            df_input = all_df_binary.join(df_filtered["target"])

            try:
                results = direct_pattern_mining(
                    data=df_input,
                    target_col="target",
                    min_confidence=min_conf,
                    min_support=min_supp,
                    max_items=max_items
                )
            except:
                st.warning("⚠️ 패턴 생성 불가")
                st.stop()

        st.write(f"min_support = {min_supp:.4f}")
        st.success("완료!")

        st.subheader("📊 최종 결과")

        if results is None or len(results) == 0:
            st.warning("⚠️ 유효한 패턴이 없습니다.")
            st.stop()

        required_cols = {"count", "lift", "confidence"}
        if not required_cols.issubset(results.columns):
            st.warning("⚠️ 패턴 생성 데이터 부족")
            st.stop()

        post_df = (
            results[results["count"] >= min_count]
            .sort_values("lift", ascending=False)
            .head(50)
        )
        # ======================
        # A. 조합 자체가 없는 경우
        # ======================
        if all_df_binary.sum().sum() == 0:
            st.error("""
            ### 😢 분석이 불가능합니다.
            현재 선택한 조건에서는 선배들의 교과/비교과 수강 기록(1로 표시된 항목)이 거의 없습니다.  
            즉, **데이터 자체에 조합이 존재하지 않아** 패턴을 만들 수 없습니다.

            👉 다른 조건(학부, 직무, 기업구분)을 선택해서 다시 시도해주세요.
            """)
            st.stop()

        # ======================
        # B. 패턴은 가능하지만 파라미터 문제로 탈락한 경우
        # ======================
        if post_df.empty:
            st.error("""
            ### 😢 결과가 없습니다.

            데이터는 존재하지만, 설정한 값이 너무 엄격해서  
            **모든 조합이 필터링된 상태**입니다.

            해결 방법:
            - min_support(지지도) ↓ 낮추기
            - min_confidence(신뢰도) ↓ 낮추기
            - min_count(최소 출현 수) ↓ 낮추기
            - max_items(조합 길이) ↑ 늘리기

            위 설정을 조정하면 결과가 나타날 수 있어요.
            """)
            st.stop()
        else:
            st.dataframe(post_df)

        st.markdown("---")




