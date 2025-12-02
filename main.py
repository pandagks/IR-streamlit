import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

url = "https://drive.google.com/uc?export=download&id=1WyOr53Lf52zVG79obWfHsPzterKHIJcl"

df = pd.read_csv(url)
df = df.dropna(subset=["직무", "기업구분", "평점"])

st.set_page_config(
    page_title="한국공학대 IR센터 데이터 분석 플랫폼",
    layout="wide",
    initial_sidebar_state="expanded")



# CSS 넣기
st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif;
}

.hero-section {
    width: 100%;
    height: 95vh;
    background: linear-gradient(rgba(0,0,0,0.60), rgba(0,0,0,0.60)),
                url('https://ypzxxdrj8709.edge.naverncp.com/data2/content/image/2022/04/18/.cache/512/20220418580036.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    display: flex;
    justify-content: center;
    align-items: center;
    color: white;
    padding: 2rem;
}

.hero-box {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(8px);
    border-radius: 15px;
    padding: 3rem 4rem;
    max-width: 900px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

# HTML 섹션 구성
st.markdown("""
<div class="hero-section">
    <div class="hero-box">
        <h1 style="font-size:2.75rem; font-weight:600;">
            한국공학대 IR센터 데이터 분석 플랫폼
        </h1>
        <p style="font-size:1.4rem; line-height:1.6; margin-top:1rem;">
            희망하는 기업·직무 진출 선배들의 역량 데이터를 기반으로,<br>
            현재 나의 역량을 비교 분석하고 선배들이 실제로 이수한 교과 및 비교과 활동을 <br>
            확인할 수 있는 IR센터 맞춤형 추천 시스템입니다.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


st.image('./images/한국공학대.png', width=210)
st.header('한국공학대 IR센터 데이터 분석 플랫폼')

with st.spinner('Updating Report...'):
    with st.expander('플랫폼 프레임워크'):
        st.markdown('<h2 class = "center-text"> IR센터 추천시스템 프레임워크 </h2>', unsafe_allow_html=True)
        # st.markdown('<img src="./images/로드맵.png" class="center-image">', unsafe_allow_html=True)
        left_co, cent_co, last_co = st.columns([1,8,1])
        with cent_co:
            st.image('./images/image1.png', output_format='auto', caption='플랫폼 설명', use_container_width=True)
    with st.expander('플랫폼 목적'):
        st.markdown('<h2 class = "center-text"> 시스템 제공 예시 </h2>', unsafe_allow_html=True)
        # st.markdown('<img src="./images/로드맵.png" class="center-image">', unsafe_allow_html=True)
        left_co, cent_co, last_co = st.columns([1,8,1])
        with cent_co:
            st.image('./images/image2.png', output_format='auto', caption='제공 예시', use_container_width=True)        



        


st.markdown("---")  # 구분선

st.markdown("## EDA 결과")

valid_groups = ['중소기업','중견기업','대기업','외국계기업']

df_plot = df[df['기업구분'].isin(valid_groups)][['기업구분','비교과 종합점수','영어학점','평점']].copy()


# 통계값 계산
stats_ncs = df_plot.groupby("기업구분")["비교과 종합점수"].agg(['mean','std','count']).reindex(valid_groups)
stats_eng = df_plot.groupby("기업구분")["영어학점"].agg(['mean','std','count']).reindex(valid_groups)

col1, col2 = st.columns(2)


# 비교과 종합점수 BOXPLOT

with col1:
    fig = px.box(
        df_plot,
        x="기업구분",
        y="비교과 종합점수",
        color="기업구분",
        title="기업구분별 비교과 종합점수 분포"
    )

    # 기업별 평균(X 표시)
    for group in valid_groups:
        mean_val = stats_ncs.loc[group, 'mean']
        fig.add_trace(go.Scatter(
            x=[group],
            y=[mean_val],
            mode='markers+text',
            marker=dict(size=12, color="red", symbol="x"),
            text=[f"{mean_val:.1f}"],
            textposition="top center",
            showlegend=False
        ))

    # HoverTemplate에 stats 표시
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "값: %{y}<br>" +
            "평균: %{customdata[0]:.2f}<br>" +
            "표준편차: %{customdata[1]:.2f}<br>" +
            "N: %{customdata[2]}<extra></extra>"
        ),
        customdata=stats_ncs.loc[df_plot['기업구분']].values
    )

    st.plotly_chart(fig, use_container_width=True)



# 영어학점 BOXPLOT

with col2:
    fig1 = px.box(
        df_plot,
        x="기업구분",
        y="영어학점",
        color="기업구분",
        title="기업구분별 영어학점 분포"
    )

    # 기업별 평균(X 표시)
    for group in valid_groups:
        mean_val = stats_eng.loc[group, 'mean']
        fig1.add_trace(go.Scatter(
            x=[group],
            y=[mean_val],
            mode='markers+text',
            marker=dict(size=12, color="blue", symbol="x"),
            text=[f"{mean_val:.1f}"],
            textposition="top center",
            showlegend=False
        ))

    fig1.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "값: %{y}<br>" +
            "평균: %{customdata[0]:.2f}<br>" +
            "표준편차: %{customdata[1]:.2f}<br>" +
            "N: %{customdata[2]}<extra></extra>"
        ),
        customdata=stats_eng.loc[df_plot['기업구분']].values
    )

    st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")  # 구분선

st.markdown("## 학점별 취업 분석")

url2 = "https://drive.google.com/uc?export=download&id=d/1KPrtxnvYca9_DMB3kUiCI-nbxapop_r2"
df = pd.read_csv(url2)

replace_map = {
    '보호대상중견기업': '중견기업',
    '한시성중소기업': '중소기업',
    '공공/공직(공무원,공공기관,공기업)': '공공기관',
    '소기업': '중소기업',
    '중기업': '중소기업',
    '기타(비영리단체)': '비영리단체',
}
df['기업구분'] = df['기업구분'].replace(replace_map)

def add_gpa_bin(df):
    bins = [0, 2.5, 3.0, 3.5, 4.0, 4.5]
    labels = ['≤2.5','2.5~3.0','3.0~3.5','3.5~4.0','4.0~4.5']
    df['학점구간'] = pd.cut(df['평점'], bins=bins, labels=labels, include_lowest=True)
    return df

df = add_gpa_bin(df)

st.title("학점구간별 기업구분 Heatmap")

학부 = st.selectbox("학과 선택", df['학부(대학)'].unique())

df_filtered = df[df['학부(대학)'] == 학부]

# 모든 학점구간 × 기업구분 강제 생성
gpa_levels = ['≤2.5','2.5~3.0','3.0~3.5','3.5~4.0','4.0~4.5']
ctab = pd.crosstab(df_filtered['학점구간'], df_filtered['기업구분'])

# 기업구분 레벨을 crosstab의 실제 컬럼 순서로 가져오기 (이게 핵심!!)
company_levels = ctab.columns.tolist()

ctab = ctab.reindex(index=gpa_levels, columns=company_levels, fill_value=0)

# percent 계산 (학점 기준 100%)
percent_tab = ctab.div(ctab.sum(axis=1).replace(0,1), axis=0) * 100

# text 구성
custom_text = percent_tab.copy()
for r in range(percent_tab.shape[0]):
    for c in range(percent_tab.shape[1]):
        custom_text.iloc[r,c] = f"{percent_tab.iloc[r,c]:.1f}%\n({ctab.iloc[r,c]}명)"


col1, col2 = st.columns([1, 3])


with col1:
    st.subheader("기업구분별 전체 비율")

    total_company_count = df_filtered['기업구분'].value_counts().sort_index()

    fig_pie = go.Figure(data=[go.Pie(
        labels=total_company_count.index,
        values=total_company_count.values,
        hole=0.3,
        marker=dict(colors=px.colors.qualitative.Set3),

        #  여기서 percent + 명수 둘 다 표시
        texttemplate="%{percent:.1%}\n(%{value}명)",
        textposition="inside",
    )])

    fig_pie.update_layout(height=600)

    st.plotly_chart(fig_pie, use_container_width=True)



# 3) 오른쪽: 기존 Heatmap

with col2:
    st.subheader("학점구간별 기업구분 비율")

    # Heatmap 생성
    fig = go.Figure(data=go.Heatmap(
        z=percent_tab.T.values,       # z transpose
        x=gpa_levels,
        y=company_levels,
        text=custom_text.T.values,
        texttemplate="%{text}",
        colorscale="YlOrRd",
        colorbar=dict(title="비율 (%)")
    ))

    fig.update_layout(
        title=f"{학부} - 학점구간별 기업구분 비율 Heatmap",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)





st.markdown("---")  # 구분선


