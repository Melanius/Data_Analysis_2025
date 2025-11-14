"""
낙찰하한가 예측 웹 애플리케이션
- 3가지 ML 모델을 통한 예측
- Supabase 연동 예측 이력 관리
- 정확도 추적 및 분석
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from supabase import create_client, Client
import os
import uuid

# 페이지 설정
st.set_page_config(
    page_title="낙찰하한가 예측 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card-selected {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        border: 3px solid #fff;
    }
    .input-section {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* number input 증감 버튼 숨기기 - Streamlit 전용 */
    /* Streamlit의 NumberInput 컴포넌트 타겟 */
    div[data-testid="stNumberInput"] input::-webkit-outer-spin-button,
    div[data-testid="stNumberInput"] input::-webkit-inner-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
        display: none !important;
    }

    div[data-testid="stNumberInput"] input[type=number] {
        -moz-appearance: textfield !important;
    }

    /* BaseWeb Input 컴포넌트 (Streamlit이 사용하는 UI 라이브러리) */
    [data-baseweb="input"] input::-webkit-outer-spin-button,
    [data-baseweb="input"] input::-webkit-inner-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
        display: none !important;
    }

    [data-baseweb="input"] input[type=number] {
        -moz-appearance: textfield !important;
    }

    /* 모든 number input에 대한 백업 규칙 */
    input::-webkit-outer-spin-button,
    input::-webkit-inner-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
        display: none !important;
    }

    input[type=number] {
        -moz-appearance: textfield !important;
    }
</style>

<script>
    // JavaScript로 추가 보장 - number input 스피너 제거
    const removeSpinners = () => {
        const inputs = document.querySelectorAll('input[type=number]');
        inputs.forEach(input => {
            input.style.MozAppearance = 'textfield';
            input.style.webkitAppearance = 'none';
            input.style.appearance = 'none';
        });
    };

    // 페이지 로드 시 실행
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', removeSpinners);
    } else {
        removeSpinners();
    }

    // Streamlit 리렌더링 감지 및 재실행
    const observer = new MutationObserver(removeSpinners);
    observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# Supabase 설정
@st.cache_resource
def init_supabase():
    """Supabase 클라이언트 초기화"""
    url = os.getenv("SUPABASE_URL", "https://kkaltyxehwupjxijpwtv.supabase.co")
    key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtrYWx0eXhlaHd1cGp4aWpwd3R2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI5NDcxNzQsImV4cCI6MjA3ODUyMzE3NH0.Nl4yC6xqvbAN3KOJfcj7CsVJhp79CQlKc16qnhJRroo")
    return create_client(url, key)

# 모델 로드
@st.cache_resource
def load_models():
    """저장된 3가지 모델 로드"""
    model_dir = Path("models/production")

    # 가장 최근 모델 파일 찾기
    linear_files = sorted(model_dir.glob("linear_model_*.pkl"), reverse=True)
    ridge_files = sorted(model_dir.glob("ridge_model_*.pkl"), reverse=True)
    ensemble_files = sorted(model_dir.glob("ensemble_model_*.pkl"), reverse=True)

    if not (linear_files and ridge_files and ensemble_files):
        st.error("⚠️ 모델 파일을 찾을 수 없습니다.")
        return None

    # 모델 로드
    linear_data = joblib.load(linear_files[0])
    ridge_data = joblib.load(ridge_files[0])
    ensemble_data = joblib.load(ensemble_files[0])

    return {
        'linear': {
            'model': linear_data['model'],
            'name': 'Linear Regression',
            'description': '빠르고 안정적',
            'metrics': linear_data['metrics']
        },
        'ridge': {
            'model': ridge_data['model'],
            'scaler': ridge_data['scaler'],
            'name': 'Ridge Regression',
            'description': '추천 모델',
            'metrics': ridge_data['metrics']
        },
        'ensemble': {
            'lr_model': ensemble_data['lr_model'],
            'xgb_model': ensemble_data['xgb_model'],
            'lr_weight': ensemble_data['lr_weight'],
            'xgb_weight': ensemble_data['xgb_weight'],
            'name': 'Weighted Ensemble',
            'description': '정교한 예측',
            'metrics': ensemble_data['metrics']
        }
    }

def predict_yega(models, input_data):
    """3가지 모델로 예가 예측"""
    predictions = {}

    # 입력 데이터 준비
    feature_names = ['예정가격', '기초금액', 'A값', '낙찰하한율', '예가변동폭_범위']
    X = pd.DataFrame([input_data], columns=feature_names)

    # 1. Linear Regression
    yega_linear = models['linear']['model'].predict(X)[0]
    predictions['linear'] = yega_linear

    # 2. Ridge Regression (스케일링 필요)
    X_scaled = models['ridge']['scaler'].transform(X)
    yega_ridge = models['ridge']['model'].predict(X_scaled)[0]
    predictions['ridge'] = yega_ridge

    # 3. Weighted Ensemble
    yega_lr = models['ensemble']['lr_model'].predict(X)[0]
    yega_xgb = models['ensemble']['xgb_model'].predict(X)[0]
    yega_ensemble = (models['ensemble']['lr_weight'] * yega_lr +
                     models['ensemble']['xgb_weight'] * yega_xgb)
    predictions['ensemble'] = yega_ensemble

    return predictions

def calculate_bid_price(gichogeum, yega_ratio, a_value, nakchalhahan_rate):
    """낙찰하한가 계산"""
    yejeong_price = gichogeum * (yega_ratio / 100)
    nakchalhahan_price = (yejeong_price - a_value) * (nakchalhahan_rate / 100) + a_value
    return yejeong_price, nakchalhahan_price

def format_currency(value):
    """통화 포맷 (천 단위 쉼표)"""
    return f"{value:,.0f}"

def main():
    # 헤더
    st.markdown('<div class="main-header">📊 낙찰하한가 예측 시스템</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">3가지 머신러닝 모델을 통한 정확한 낙찰하한가 예측</div>', unsafe_allow_html=True)

    # Supabase 초기화
    supabase = init_supabase()

    # 모델 로드
    models = load_models()
    if models is None:
        return

    # 사이드바 - 메뉴
    st.sidebar.title("📋 메뉴")
    menu = st.sidebar.radio(
        "기능 선택",
        ["🎯 예측하기", "📊 예측 이력", "📈 정확도 분석"],
        label_visibility="collapsed"
    )

    # ========== 예측하기 ==========
    if menu == "🎯 예측하기":
        st.markdown("## 🎯 낙찰하한가 예측")

        # 입력 섹션
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 📝 입력 정보")

        # 공사명 입력 (전체 너비, 최상단)
        project_name = st.text_input(
            "공사명 *",
            placeholder="예: 서울시 강남구 테헤란로 도로 보수공사",
            help="예측하려는 공사의 이름을 입력하세요 (필수)",
            max_chars=200
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            chujeong_price = st.number_input(
                "추정가격 (원)",
                min_value=0,
                value=100000000,
                step=1000000,
                format="%d",
                help="입찰 공고에 명시된 추정가격"
            )

            gichogeum = st.number_input(
                "기초금액 (원)",
                min_value=0,
                value=95000000,
                step=1000000,
                format="%d",
                help="예가 산정의 기준이 되는 금액"
            )

            a_value = st.number_input(
                "A값 (원)",
                min_value=0,
                value=0,
                step=100000,
                format="%d",
                help="낙찰하한율 계산에 사용되는 A값 (없으면 0)"
            )

        with col2:
            nakchalhahan_rate = st.number_input(
                "낙찰하한율 (%)",
                min_value=0.0,
                max_value=100.0,
                value=87.745,
                step=0.001,
                format="%.3f",
                help="낙찰 가능한 최저 비율"
            )

            yega_range = st.selectbox(
                "예가변동폭",
                options=[2, 2.5, 3],
                index=1,
                help="예정가격의 변동 범위 (±%)"
            )

            # 예가변동폭_범위 계산 (2→4, 2.5→5, 3→6)
            yega_range_value = yega_range * 2

        st.markdown('</div>', unsafe_allow_html=True)

        # 예측 버튼
        if st.button("🔮 예측 실행", type="primary", use_container_width=True):
            # 공사명 필수 검증
            if not project_name or project_name.strip() == "":
                st.error("⚠️ 공사명을 입력해주세요!")
                st.stop()

            # 입력 데이터 준비
            input_data = [
                chujeong_price,
                gichogeum,
                a_value,
                nakchalhahan_rate,
                yega_range_value
            ]

            # 예가 예측
            with st.spinner("예측 중..."):
                yega_predictions = predict_yega(models, input_data)

            # 세션에 예측 결과 저장 (저장 버튼을 위해)
            st.session_state.yega_predictions = yega_predictions
            st.session_state.input_data = {
                'project_name': project_name,
                'chujeong_price': chujeong_price,
                'gichogeum': gichogeum,
                'a_value': a_value,
                'nakchalhahan_rate': nakchalhahan_rate,
                'yega_range': yega_range
            }

            # 결과 표시
            st.markdown("---")
            st.markdown("## 🎯 예측 결과")
            st.markdown("##### 3가지 모델의 예측 결과입니다. '저장하기' 버튼으로 모두 저장됩니다.")

            cols = st.columns(3)

            model_keys = ['ridge', 'linear', 'ensemble']
            model_icons = ['⭐', '⚡', '🎯']

            for idx, (col, model_key, icon) in enumerate(zip(cols, model_keys, model_icons)):
                with col:
                    model = models[model_key]
                    yega_ratio = yega_predictions[model_key]
                    yejeong_price, nakchalhahan_price = calculate_bid_price(
                        gichogeum, yega_ratio, a_value, nakchalhahan_rate
                    )

                    # 카드 스타일 (선택 기능 제거, 모두 동일한 스타일)
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.markdown(f"### {icon} {model['name']}")
                    st.markdown(f"**{model['description']}**")
                    st.markdown("---")
                    st.markdown(f"**예가**: {yega_ratio:.4f}%")
                    st.markdown(f"**예정가격**: {format_currency(yejeong_price)}원")
                    st.markdown(f"**낙찰하한가**: {format_currency(nakchalhahan_price)}원")
                    st.markdown("---")
                    test_mape = model['metrics']['test']['MAPE']
                    st.markdown(f"*평균 오차: {test_mape:.2f}%*")
                    st.markdown('</div>', unsafe_allow_html=True)

        # 저장 섹션 (예측 실행 후에만 표시)
        if 'yega_predictions' in st.session_state and st.session_state.yega_predictions:
            st.markdown("---")
            st.markdown("### 💾 예측 저장")

            st.info("💡 저장하기 버튼을 클릭하면 위의 3가지 모델 예측 결과가 모두 저장됩니다.")

            if st.button("💾 저장하기 (3개 모델 모두)", type="primary", use_container_width=True):
                try:
                    # 그룹 ID 생성
                    group_id = str(uuid.uuid4())

                    # 3개 모델의 결과를 모두 저장
                    for model_key in ['ridge', 'linear', 'ensemble']:
                        yega = st.session_state.yega_predictions[model_key]
                        yejeong, nakchalhahan = calculate_bid_price(
                            st.session_state.input_data['gichogeum'],
                            yega,
                            st.session_state.input_data['a_value'],
                            st.session_state.input_data['nakchalhahan_rate']
                        )

                        data = {
                            "prediction_group_id": group_id,
                            "project_name": st.session_state.input_data['project_name'],
                            "chujeong_price": float(st.session_state.input_data['chujeong_price']),
                            "gichogeum": float(st.session_state.input_data['gichogeum']),
                            "a_value": float(st.session_state.input_data['a_value']),
                            "nakchalhahan_rate": float(st.session_state.input_data['nakchalhahan_rate']),
                            "yega_range": float(st.session_state.input_data['yega_range']),
                            "model_type": model_key,
                            "model_name": models[model_key]['name'],
                            "predicted_yega": float(yega),
                            "predicted_yejeong_price": float(yejeong),
                            "predicted_nakchalhahan_price": float(nakchalhahan),
                            "created_at": datetime.now().isoformat()
                        }

                        supabase.table("predictions").insert(data).execute()

                    st.success("✅ 3개 모델의 예측이 모두 저장되었습니다!")

                    # 세션 상태 초기화
                    st.session_state.yega_predictions = None
                    st.session_state.input_data = None

                except Exception as e:
                    st.error(f"❌ 저장 실패: {str(e)}")

    # ========== 예측 이력 ==========
    elif menu == "📊 예측 이력":
        st.markdown("## 📊 예측 이력")

        try:
            # 모든 예측 조회
            response = supabase.table("predictions").select("*").order("created_at", desc=True).execute()

            if response.data:
                df = pd.DataFrame(response.data)

                # 날짜 포맷
                df['created_at_formatted'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')

                # 그룹별로 정리
                unique_groups = df['prediction_group_id'].unique()

                st.markdown(f"### 📋 총 {len(unique_groups)}개의 예측 그룹")
                st.markdown("---")

                # 그룹별로 표시
                for group_id in unique_groups:
                    group_df = df[df['prediction_group_id'] == group_id].copy()

                    # 그룹 정보
                    first_row = group_df.iloc[0]
                    created_at = first_row['created_at_formatted']
                    project_name = first_row.get('project_name', None)
                    chujeong = format_currency(first_row['chujeong_price'])
                    gichogeum = format_currency(first_row['gichogeum'])
                    actual_value = first_row['actual_nakchalhahan_price']

                    # 실제값 입력 여부 표시
                    status_icon = "✅" if pd.notna(actual_value) else "⏳"
                    actual_text = format_currency(actual_value) + "원" if pd.notna(actual_value) else "미입력"

                    # 공사명 표시 (NULL인 경우 대체 텍스트)
                    project_display = project_name if pd.notna(project_name) and project_name else "공사명 미입력"

                    # Expander로 그룹별 상세 정보 표시
                    with st.expander(
                        f"{status_icon} {created_at} | 📌 {project_display} | 기초금액: {gichogeum}원 | 실제: {actual_text}",
                        expanded=False
                    ):
                        # 공사명 표시
                        st.info(f"**🏗️ 공사명**: {project_display}")

                        # 입력 정보 표시
                        st.markdown("#### 📝 입력 정보")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("추정가격", chujeong + "원")
                        col2.metric("기초금액", gichogeum + "원")
                        col3.metric("A값", format_currency(first_row['a_value']) + "원")
                        col4.metric("낙찰하한율", f"{first_row['nakchalhahan_rate']:.3f}%")
                        col5.metric("예가변동폭", f"{first_row['yega_range']}")

                        st.markdown("---")
                        st.markdown("#### 🎯 모델별 예측 결과")

                        # 3개 모델 결과 표시
                        cols = st.columns(3)
                        model_order = ['ridge', 'linear', 'ensemble']
                        model_icons = {'ridge': '⭐', 'linear': '⚡', 'ensemble': '🎯'}

                        for idx, model_type in enumerate(model_order):
                            model_row = group_df[group_df['model_type'] == model_type].iloc[0]

                            with cols[idx]:
                                icon = model_icons.get(model_type, '📊')
                                st.markdown(f"**{icon} {model_row['model_name']}**")
                                st.metric("예측 낙찰하한가", format_currency(model_row['predicted_nakchalhahan_price']) + "원")
                                st.caption(f"예가: {model_row['predicted_yega']:.4f}%")

                        # 실제값 입력 섹션
                        st.markdown("---")
                        st.markdown("#### ✏️ 실제 낙찰하한가 입력")

                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            actual_input = st.number_input(
                                "실제 낙찰하한가 (원)",
                                min_value=0,
                                value=int(actual_value) if pd.notna(actual_value) else 0,
                                step=100000,
                                format="%d",
                                key=f"actual_{group_id}"
                            )

                        with col2:
                            # 기존 날짜 값 가져오기
                            existing_date = first_row.get('bid_announcement_date', None)
                            date_value = None
                            if pd.notna(existing_date) and existing_date:
                                try:
                                    date_value = pd.to_datetime(existing_date).date()
                                except:
                                    date_value = None

                            actual_date = st.date_input(
                                "낙찰 발표일 (선택사항)",
                                value=date_value,
                                min_value=datetime(2020, 1, 1).date(),
                                max_value=datetime.now().date(),
                                help="낙찰하한가가 발표된 날짜",
                                key=f"date_{group_id}"
                            )

                        with col3:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("💾 저장", type="primary", use_container_width=True, key=f"save_{group_id}"):
                                try:
                                    # 해당 그룹의 모든 행 업데이트
                                    update_data = {
                                        "actual_nakchalhahan_price": float(actual_input),
                                        "updated_at": datetime.now().isoformat()
                                    }

                                    # 날짜가 선택된 경우에만 추가
                                    if actual_date:
                                        update_data["bid_announcement_date"] = actual_date.isoformat()

                                    supabase.table("predictions").update(update_data).eq("prediction_group_id", group_id).execute()

                                    st.success("✅ 실제값이 저장되었습니다!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ 저장 실패: {str(e)}")

            else:
                st.info("📭 아직 저장된 예측이 없습니다.")

        except Exception as e:
            st.error(f"⚠️ 데이터 조회 실패: {str(e)}")

    # ========== 정확도 분석 ==========
    elif menu == "📈 정확도 분석":
        st.markdown("## 📈 정확도 분석")

        try:
            # 실제값이 입력된 예측만 조회
            response = supabase.table("predictions").select("*").not_.is_("actual_nakchalhahan_price", "null").execute()

            if response.data and len(response.data) > 0:
                df = pd.DataFrame(response.data)

                # 오차 계산
                df['error'] = df['actual_nakchalhahan_price'] - df['predicted_nakchalhahan_price']
                df['error_pct'] = (df['error'] / df['actual_nakchalhahan_price']) * 100
                df['abs_error_pct'] = df['error_pct'].abs()

                # 모델별 통계
                st.markdown("### 📊 모델별 성능")

                model_stats = df.groupby('model_name').agg({
                    'abs_error_pct': ['mean', 'std', 'count'],
                    'error': ['mean', 'std']
                }).round(2)

                model_stats.columns = ['MAPE (%)', '표준편차 (%)', '예측 건수', '평균 오차 (원)', '오차 표준편차 (원)']

                st.dataframe(model_stats, use_container_width=True)

                # 시각화
                col1, col2 = st.columns(2)

                with col1:
                    # 모델별 MAPE 비교
                    fig = go.Figure(data=[
                        go.Bar(
                            x=model_stats.index,
                            y=model_stats['MAPE (%)'],
                            text=model_stats['MAPE (%)'],
                            textposition='auto',
                            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
                        )
                    ])
                    fig.update_layout(
                        title="모델별 평균 오차율 (MAPE)",
                        xaxis_title="모델",
                        yaxis_title="MAPE (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # 예측 vs 실제 산점도
                    fig = go.Figure()

                    for model in df['model_name'].unique():
                        model_df = df[df['model_name'] == model]
                        fig.add_trace(go.Scatter(
                            x=model_df['predicted_nakchalhahan_price'],
                            y=model_df['actual_nakchalhahan_price'],
                            mode='markers',
                            name=model,
                            marker=dict(size=10)
                        ))

                    # 대각선 (완벽한 예측선)
                    min_val = min(df['predicted_nakchalhahan_price'].min(), df['actual_nakchalhahan_price'].min())
                    max_val = max(df['predicted_nakchalhahan_price'].max(), df['actual_nakchalhahan_price'].max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='완벽한 예측',
                        line=dict(dash='dash', color='gray')
                    ))

                    fig.update_layout(
                        title="예측 vs 실제",
                        xaxis_title="예측 낙찰하한가",
                        yaxis_title="실제 낙찰하한가",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # 최근 10건 상세
                st.markdown("---")
                st.markdown("### 📋 최근 예측 상세")

                recent = df.sort_values('created_at', ascending=False).head(10)[[
                    'created_at', 'model_name', 'predicted_nakchalhahan_price',
                    'actual_nakchalhahan_price', 'error', 'error_pct'
                ]].copy()

                recent.columns = ['예측일시', '모델', '예측값', '실제값', '오차 (원)', '오차율 (%)']
                recent['예측일시'] = pd.to_datetime(recent['예측일시']).dt.strftime('%Y-%m-%d %H:%M')

                for col in ['예측값', '실제값', '오차 (원)']:
                    recent[col] = recent[col].apply(format_currency)

                recent['오차율 (%)'] = recent['오차율 (%)'].apply(lambda x: f"{x:.2f}%")

                st.dataframe(recent, use_container_width=True)

            else:
                st.info("📭 실제값이 입력된 예측이 아직 없습니다.")

        except Exception as e:
            st.error(f"⚠️ 데이터 분석 실패: {str(e)}")

if __name__ == "__main__":
    main()
