import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ======================================================
# 데이터 로딩
# ======================================================
url = "https://drive.google.com/uc?export=download&id=1WyOr53Lf52zVG79obWfHsPzterKHIJcl"

df = pd.read_csv(url)
df = df.dropna(subset=["직무", "기업구분", "평점"])
df1 = pd.read_csv(url)  

st.set_page_config(
    page_title="선배와 역량비교",
    layout="wide",
    initial_sidebar_state="expanded")
st.title("선배와 역량 비교")

# ======================================================
# 사용자 입력 UI
# ======================================================
st.subheader("알고 싶은 선배들의 정보 입력")

# 1) 학부 선택
직무 = st.selectbox("1️⃣ 직무 선택", df["직무"].unique())

# 3) 희망기업 선택
희망기업 = st.selectbox("2️⃣ 희망기업 선택", df["기업구분"].unique())

# ======================================================
# 학번 입력 → df1에서 자동 조회
# ======================================================
st.markdown("---")
st.subheader("📘 3️⃣ 본인 정보 입력 방식 선택")

학번 = st.text_input("학번을 입력하세요 (예: 2005171009)", "")

# 학번 기반 자동 입력값 (기본값 None)
auto_gpa = None
auto_eng = None
auto_extra = None

if 학번:
    df_student = df1[df1["학번"].astype(str) == str(학번)]  # ★ df1 사용

    if len(df_student) == 1:
        st.success("학번 정보를 불러왔습니다!")

        auto_gpa = df_student["평점"].values[0]
        auto_eng = df_student["영어학점"].values[0]
        auto_extra = df_student["비교과 종합점수"].values[0]

        st.write("불러온 학생 정보:")
        st.write(df_student[["학번", "평점", "영어학점", "비교과 종합점수"]])

    else:
        st.error("해당 학번을 찾을 수 없습니다. 직접 입력해주세요.")
# ======================================================
# 본인 역량 입력 (자동 입력 + 수정 가능)
# ======================================================
st.subheader("📘 4️⃣ 본인 역량 입력")

평점 = st.number_input(
    "평점 (0~4.5)",
    0.0, 4.5,
    auto_gpa if auto_gpa is not None else 3.0
)

영어학점 = st.number_input(
    "영어학점 (0~4.5)",
    0.0, 4.5,
    auto_eng if auto_eng is not None else 2.0
)

비교과점수 = st.number_input(
    "비교과 종합점수 (0~100)",
    0.0, 100.0,
    auto_extra if auto_extra is not None else 30.0
)

대외활동_input = st.number_input("대외활동 (0~10)", 0, 10, 2)

user_input = {
    "평점": 평점,
    "영어학점": 영어학점,
    "비교과 종합점수": 비교과점수,
    "대외활동": 대외활동_input
}

# ======================================================
# 스케일 변환 함수 (0~100)
# ======================================================
MAX_GPA = 4.5
MAX_ACT = 10

def to_0_100(val, name):
    if name in ['평점','영어학점']:
        return (val / MAX_GPA) * 100
    elif name == '대외활동':
        return (val / MAX_ACT) * 100
    elif name == '비교과 종합점수':
        return float(val)   # 이미 0~100
    else:
        return float(val)

# ======================================================
# 기업규모별 대외활동 기본값
# ======================================================
COMP_BASE = {
    '대기업': 10,
    '외국계기업' : 10,
    '공공기관':8,
    '중견기업': 7,
    '중소기업': 4,
    '기타': 4
}

# ======================================================
# Plotly Radar Chart
# ======================================================
def plot_radar_plotly(user_vals, avg_vals, labels, title):
    user_close = np.append(user_vals, user_vals[0])
    avg_close = np.append(avg_vals, avg_vals[0])
    labels_close = labels + [labels[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=user_close,
        theta=labels_close,
        fill='toself',
        name="내 점수",
        line=dict(width=3, color="#1f77b4"),      # 파랑
        fillcolor="rgba(31,119,180,0.3)"
    ))

    fig.add_trace(go.Scatterpolar(
        r=avg_close,
        theta=labels_close,
        fill='toself',
        name="비교군 평균",
        line=dict(width=3, color="orange"),     
        fillcolor="rgba(214,39,40,0.3)"
    ))

    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=600
    )
    return fig

# ======================================================
# 리포트 생성 버튼
# ======================================================
if st.button("리포트 생성"):

    # --------------------------------------------------
    # 비교군 생성
    # --------------------------------------------------
    df_major_job = df[(df["직무"] == 직무)]
    df_company = df[df["기업구분"] == 희망기업]

    st.write(f"직무 비교군 수: {len(df_major_job)}명")
    st.write(f"희망기업 비교군 수: {len(df_company)}명")

    # --------------------------------------------------
    # 레이더 차트 항목 정의
    # --------------------------------------------------
    label_keys = ['평점','영어학점','비교과 종합점수','대외활동']
    display_labels = ['학점','영어','비교과','대외활동']

    # --------------------------------------------------
    # 사용자 스케일링
    # --------------------------------------------------
    user_scaled = np.array([to_0_100(user_input[k], k) for k in label_keys])
    st.markdown("---")  # 구분선
    # --------------------------------------------------
    # 비교군 스케일링
    # --------------------------------------------------
    major_scaled = []
    comp_scaled = []

    for k in label_keys:
        if k == '대외활동':  
            major_val = 5   # ← 직무 비교 기준에서는 무조건 5로 고정
            comp_val  = COMP_BASE.get(희망기업, 4)  # 기업 비교군은 기존대로
        else:
            major_val = df_major_job[k].mean()
            comp_val  = df_company[k].mean()

        major_scaled.append(to_0_100(major_val, k))
        comp_scaled.append(to_0_100(comp_val, k))

    major_scaled = np.array(major_scaled)
    comp_scaled = np.array(comp_scaled)


   

        # ===============================
    # 레이아웃: 2개 가로 배치(columns)
    # ===============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"직무 기준 레이더 차트 ({직무})")
        fig1 = plot_radar_plotly(user_scaled, major_scaled, display_labels, f"직무 비교: {직무}")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader(f"직무 기준 비교 (0~100) - {직무}")
        st.dataframe(pd.DataFrame({
            "항목": display_labels,
            "내 점수": user_scaled,
            "학부·직무 평균": major_scaled,
            "차이": major_scaled - user_scaled
        }))

    with col2:
        st.subheader(f"희망기업 기준 레이더 차트 ({희망기업})")
        fig2 = plot_radar_plotly(user_scaled, comp_scaled, display_labels, f"기업 비교: {희망기업}")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader(f"희망기업 기준 비교 (0~100) - {희망기업}")
        st.dataframe(pd.DataFrame({
            "항목": display_labels,
            "내 점수": user_scaled,
            "희망기업 평균": comp_scaled,
            "차이": comp_scaled - user_scaled
        }))



