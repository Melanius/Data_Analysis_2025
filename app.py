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
import re
import secrets
import string

# bcrypt 설치 확인 및 자동 설치
try:
    import bcrypt
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "bcrypt", "--break-system-packages"])
    import bcrypt

# .env 파일 로드 (python-dotenv 없이)
def load_env():
    """간단한 .env 파일 로더"""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

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

# Supabase 설정 - Streamlit Cloud Secrets 우선, 로컬 .env 폴백
try:
    # Streamlit Cloud 환경: st.secrets 사용
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except (KeyError, FileNotFoundError, AttributeError):
    # 로컬 환경: .env 파일 사용
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

@st.cache_resource
def init_supabase():
    """Supabase 클라이언트 초기화"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.error("⚠️ 데이터베이스 연결 정보가 설정되지 않았습니다. 관리자에게 문의하세요.")
        st.info("💡 환경변수 SUPABASE_URL과 SUPABASE_KEY를 설정해주세요.")
        st.stop()
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ================================================================
# 인증 관련 함수들
# ================================================================

def hash_password(password: str) -> str:
    """비밀번호를 bcrypt로 해싱"""
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    password_hash = bcrypt.hashpw(password_bytes, salt)
    return password_hash.decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    """비밀번호 검증"""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except Exception:
        return False

def login_user(supabase: Client, username: str, password: str) -> dict | None:
    """
    사용자 로그인 검증

    Returns:
        성공 시: {'id': UUID, 'username': str, 'email': str, 'is_admin': bool}
        실패 시: None
    """
    try:
        # username으로 사용자 조회
        response = supabase.table('users').select('*').eq('username', username).execute()

        if not response.data or len(response.data) == 0:
            return None

        user = response.data[0]

        # 비밀번호 검증
        if verify_password(password, user['password_hash']):
            return {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'is_admin': user['is_admin']
            }

        return None
    except Exception as e:
        st.error(f"로그인 오류: {e}")
        return None

def register_user(supabase: Client, username: str, password: str, email: str) -> bool:
    """
    신규 사용자 등록

    Returns:
        성공 시: True
        실패 시: False
    """
    try:
        # username 중복 체크
        response = supabase.table('users').select('id').eq('username', username).execute()
        if response.data and len(response.data) > 0:
            st.error(f"⚠️ 이미 존재하는 아이디입니다: {username}")
            return False

        # email 중복 체크
        response = supabase.table('users').select('id').eq('email', email).execute()
        if response.data and len(response.data) > 0:
            st.error(f"⚠️ 이미 등록된 이메일입니다: {email}")
            return False

        # 비밀번호 길이 검증
        if len(password) < 6:
            st.error("⚠️ 비밀번호는 최소 6자 이상이어야 합니다.")
            return False

        # 비밀번호 해싱
        password_hash = hash_password(password)

        # 사용자 데이터 삽입
        user_data = {
            'username': username,
            'password_hash': password_hash,
            'email': email,
            'is_admin': False  # 일반 사용자로 등록
        }

        response = supabase.table('users').insert(user_data).execute()

        if response.data:
            st.success(f"✅ 회원가입이 완료되었습니다! ({username})")
            return True
        else:
            st.error("❌ 회원가입 실패")
            return False

    except Exception as e:
        st.error(f"회원가입 오류: {e}")
        return False

def generate_temp_password(length: int = 8) -> str:
    """임시 비밀번호 생성 (영문 대소문자 + 숫자)"""
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

def reset_password(supabase: Client, email: str) -> str | None:
    """
    비밀번호 재설정 (임시 비밀번호 발급)

    Returns:
        성공 시: 임시 비밀번호 (str)
        실패 시: None
    """
    try:
        # 이메일로 사용자 조회
        response = supabase.table('users').select('id, username').eq('email', email).execute()

        if not response.data or len(response.data) == 0:
            st.error(f"⚠️ 등록되지 않은 이메일입니다: {email}")
            return None

        user = response.data[0]

        # 임시 비밀번호 생성
        temp_password = generate_temp_password()
        temp_password_hash = hash_password(temp_password)

        # 비밀번호 업데이트
        response = supabase.table('users').update({
            'password_hash': temp_password_hash
        }).eq('id', user['id']).execute()

        if response.data:
            st.success(f"✅ 임시 비밀번호가 발급되었습니다!")
            st.info(f"📧 이메일: {email}")
            st.info(f"👤 아이디: {user['username']}")
            return temp_password
        else:
            st.error("❌ 비밀번호 재설정 실패")
            return None

    except Exception as e:
        st.error(f"비밀번호 재설정 오류: {e}")
        return None

def logout_user():
    """로그아웃 (세션 초기화)"""
    # 세션에서 사용자 정보 제거
    if 'user' in st.session_state:
        del st.session_state.user
    if 'logged_in' in st.session_state:
        del st.session_state.logged_in
    st.success("✅ 로그아웃되었습니다.")
    st.rerun()

# ================================================================
# 기존 함수들
# ================================================================

def search_bid_list(supabase, query):
    """
    입찰 공고 검색
    - 공고번호 또는 공고명 부분 매칭 (대소문자 구분 없음)
    - 최대 5개 결과 반환
    """
    if not query or len(query) < 2:
        return []

    try:
        # 공고번호 또는 공고명에 검색어가 포함된 경우 모두 검색
        result = supabase.table('bid_list')\
            .select('*')\
            .or_(f'bid_number.ilike.%{query}%,bid_name.ilike.%{query}%')\
            .order('opening_date', desc=True)\
            .limit(5)\
            .execute()

        return result.data
    except Exception as e:
        st.error(f"검색 오류: {str(e)}")
        return []

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

def parse_number_input(input_str):
    """쉼표가 포함된 문자열을 숫자로 변환"""
    if not input_str or input_str.strip() == "":
        return 0
    try:
        # 쉼표 제거 후 숫자 변환
        return int(input_str.replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0

def format_number_for_display(value):
    """숫자를 쉼표 포맷 문자열로 변환"""
    if value == 0:
        return ""
    return f"{value:,}"

def create_bid_template():
    """입찰공고 양식 엑셀 파일 생성"""
    from io import BytesIO

    # 샘플 데이터 (다양한 형식 예시 포함)
    sample_data = {
        '공고번호': ['20250101-001', '20250102-002', '20250103-003'],
        '공고명': ['도로 포장 공사', '건물 신축 공사', '교량 보수 공사'],
        '발주기관': ['서울시', '국토부', '경기도'],
        '업종': ['토목', '건축', '토목'],
        '지역': ['서울', '경기', '인천'],
        '추정가격': [100000000, 500000000, 250000000],
        '기초금액': [95000000, 475000000, 237500000],
        'A값': [5000000, '', 0],  # 예시: 정상값, 빈값(→0), 0
        '낙찰하한율': [87.745, 88.145, 89.500],
        '예가변동폭': ['-2.0/+2.0', '-3/+3', '±3.0'],  # 예시: 다양한 형식
        '개찰일': ['25.01.15 (10:00)', '25.01.20 (14:00)', '25.01.25 (11:00)'],
        '투찰마감': ['25.01.14 (17:00)', '25.01.19 (17:00)', '25.01.24 (17:00)'],
        '입력일': ['25.01.10', '25.01.12', '25.01.15'],
        '순공사원가': [90000000, 450000000, 225000000],
        'G2B물품분류': ['토목-도로', '건축-일반건축', '토목-교량']
    }

    df = pd.DataFrame(sample_data)

    # 엑셀 생성
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 데이터 입력 시트
        df.to_excel(writer, sheet_name='데이터 입력', index=False)

        # 가이드 시트
        guide_data = {
            '컬럼명': list(sample_data.keys()),
            '필수여부': ['✅ 필수', '✅ 필수', '선택', '선택', '선택',
                        '✅ 필수', '✅ 필수', '선택', '✅ 필수', '선택',
                        '선택', '선택', '선택', '선택', '선택'],
            '형식': ['텍스트', '텍스트', '텍스트', '텍스트', '텍스트',
                    '숫자', '숫자', '숫자', '숫자', '숫자/텍스트',
                    '날짜(YY.MM.DD)', '텍스트', '날짜(YY.MM.DD)', '숫자', '텍스트'],
            '설명': [
                '공고 고유번호 (중복 시 업데이트)',
                '공사명',
                '발주 기관명',
                '공사 업종',
                '공사 지역',
                '추정가격(원) - 쉼표 없이 입력',
                '기초금액(원) - 쉼표 없이 입력',
                'A값(원) - 빈 값, 0, "-" 모두 0으로 처리됨',
                '낙찰하한율(%) - 소수점 포함',
                '예가변동폭 - 다양한 형식 지원: -2.0/+2.0, -3/+3, ±3, ±3.0% 등',
                '개찰일시 - 25.01.15 형식',
                '투찰 마감시간',
                '입력일 - 25.01.10 형식',
                '순공사원가(원)',
                'G2B 물품분류'
            ],
            '예시': [
                '20250101-001',
                '도로 포장 공사',
                '서울시',
                '토목',
                '서울',
                '100000000',
                '95000000',
                '5000000 또는 빈칸 또는 0',
                '87.745',
                '-2.0/+2.0 또는 ±3 또는 -3/+3',
                '25.01.15 (10:00)',
                '25.01.14 (17:00)',
                '25.01.10',
                '90000000',
                '토목-도로'
            ]
        }
        guide_df = pd.DataFrame(guide_data)
        guide_df.to_excel(writer, sheet_name='작성 가이드', index=False)

    output.seek(0)
    return output

def clean_a_value(a_value_input):
    """A값 전처리 - 빈 값, 특수문자 등을 0으로 변환"""
    # None, NaN 체크
    if pd.isna(a_value_input):
        return 0.0

    # 문자열로 변환 후 정리
    value_str = str(a_value_input).strip()

    # 빈 문자열, 특수문자만 있는 경우
    if not value_str or value_str in ['', '-', 'N/A', 'NA', 'n/a', 'null', 'NULL', '없음']:
        return 0.0

    # 숫자로 변환 시도
    try:
        # 쉼표 제거 후 변환
        cleaned = value_str.replace(',', '').replace(' ', '')
        return float(cleaned)
    except (ValueError, AttributeError):
        return 0.0

def parse_yega_range_app(yega_input):
    """
    예가변동폭 전처리

    지원 형식:
    - '-3/+3' → 3.0
    - '-3.0/+3.0' → 3.0
    - '±3' → 3.0
    - '±3.0%' → 3.0
    - '-2.0/2.0' → 2.0
    - '3' → 3.0
    - 빈 값 → 2.0 (기본값)
    """
    # None, NaN 체크
    if pd.isna(yega_input):
        return 2.0

    # 문자열로 변환
    yega_str = str(yega_input).strip()

    # 빈 문자열 체크
    if not yega_str:
        return 2.0

    try:
        # %, 공백 제거
        cleaned = yega_str.replace('%', '').replace(' ', '')

        # 숫자 추출 (소수점 포함)
        # [+\-±]? : 부호 (선택)
        # (\d+\.?\d*) : 정수.소수 형태
        matches = re.findall(r'[+\-±]?(\d+\.?\d*)', cleaned)

        if matches:
            # 모든 숫자를 float으로 변환 후 절대값의 최대값 사용
            numbers = [abs(float(m)) for m in matches]
            return max(numbers)

        # 매칭 실패 시 기본값
        return 2.0

    except Exception as e:
        # 파싱 실패 시 기본값
        return 2.0

def parse_date_app(date_str):
    """날짜 파싱 (upload_bid_list.py와 동일)"""
    if pd.isna(date_str):
        return None
    try:
        date_part = str(date_str).split('(')[0].strip()
        parts = date_part.split('.')
        if len(parts) == 3:
            year = int(parts[0])
            if year < 100:
                year = 2000 + year
            month = int(parts[1])
            day = int(parts[2])
            return f"{year:04d}-{month:02d}-{day:02d}"
        return None
    except (ValueError, AttributeError, IndexError):
        return None

@st.cache_data(ttl=60)  # 1분 캐시
def get_bid_list_stats(supabase_url, supabase_key):
    """입찰공고 DB 통계 정보 조회"""
    from supabase import create_client
    import pytz
    supabase = create_client(supabase_url, supabase_key)

    stats = {
        'latest_input_date': None,
        'total_count': 0,
        'oldest_input_date': None,
        'latest_created_at': None
    }

    try:
        # 합리적인 날짜 범위 설정 (오늘 ±2년)
        from datetime import datetime, timedelta
        today = datetime.now().date()
        two_years_ago = (today - timedelta(days=730)).isoformat()
        two_years_later = (today + timedelta(days=730)).isoformat()

        # 최신 입력일 (날짜 범위 필터 적용)
        latest_result = supabase.table('bid_list')\
            .select('input_date')\
            .gte('input_date', two_years_ago)\
            .lte('input_date', two_years_later)\
            .order('input_date', desc=True)\
            .not_.is_('input_date', 'null')\
            .limit(1).execute()

        if latest_result.data and latest_result.data[0]['input_date']:
            stats['latest_input_date'] = latest_result.data[0]['input_date']

        # 가장 오래된 입력일 (날짜 범위 필터 적용)
        oldest_result = supabase.table('bid_list')\
            .select('input_date')\
            .gte('input_date', two_years_ago)\
            .lte('input_date', two_years_later)\
            .order('input_date', desc=False)\
            .not_.is_('input_date', 'null')\
            .limit(1).execute()

        if oldest_result.data and oldest_result.data[0]['input_date']:
            stats['oldest_input_date'] = oldest_result.data[0]['input_date']

        # 총 건수
        count_result = supabase.table('bid_list')\
            .select('id', count='exact').execute()
        stats['total_count'] = count_result.count or 0

        # 최신 created_at (실제 DB 입력 시간)
        created_result = supabase.table('bid_list')\
            .select('created_at')\
            .order('created_at', desc=True)\
            .not_.is_('created_at', 'null')\
            .limit(1).execute()

        if created_result.data and created_result.data[0]['created_at']:
            # UTC → KST 변환
            utc_str = created_result.data[0]['created_at']
            utc_dt = datetime.fromisoformat(utc_str.replace('Z', '+00:00'))
            kst = pytz.timezone('Asia/Seoul')
            kst_dt = utc_dt.astimezone(kst)
            stats['latest_created_at'] = kst_dt

    except Exception as e:
        # 에러 발생 시 기본값 반환
        pass

    return stats

def get_latest_bids(supabase_url, supabase_key, page=1, per_page=10):
    """최신 입찰 공고 목록 조회 (페이지네이션)"""
    from supabase import create_client
    from datetime import datetime, timedelta

    supabase = create_client(supabase_url, supabase_key)

    try:
        # 날짜 범위 설정 (오늘 ±2년)
        today = datetime.now().date()
        two_years_ago = (today - timedelta(days=730)).isoformat()
        two_years_later = (today + timedelta(days=730)).isoformat()

        # 총 개수 조회 (날짜 범위 필터 적용)
        count_result = supabase.table('bid_list')\
            .select('id', count='exact')\
            .gte('input_date', two_years_ago)\
            .lte('input_date', two_years_later)\
            .not_.is_('input_date', 'null')\
            .execute()

        total_count = count_result.count or 0

        # 페이지네이션 계산
        offset = (page - 1) * per_page

        # 데이터 조회 (input_date 기준 최신순)
        result = supabase.table('bid_list')\
            .select('*')\
            .gte('input_date', two_years_ago)\
            .lte('input_date', two_years_later)\
            .not_.is_('input_date', 'null')\
            .order('input_date', desc=True)\
            .range(offset, offset + per_page - 1)\
            .execute()

        return result.data, total_count

    except Exception as e:
        print(f"❌ 최신 공고 조회 실패: {e}")
        return [], 0

def safe_value_compare(val1, val2):
    """None/NaN 안전 비교"""
    # 둘 다 None/NaN인 경우 동일
    if pd.isna(val1) and pd.isna(val2):
        return True
    # 하나만 None/NaN인 경우 다름
    if pd.isna(val1) or pd.isna(val2):
        return False
    # 둘 다 값이 있는 경우 비교
    return val1 == val2

def safe_float_compare(val1, val2, tolerance=0.0001):
    """부동소수점 안전 비교 (tolerance 허용)"""
    # 둘 다 None/NaN인 경우 동일
    if pd.isna(val1) and pd.isna(val2):
        return True
    # 하나만 None/NaN인 경우 다름
    if pd.isna(val1) or pd.isna(val2):
        return False
    # 부동소수점 비교 (오차 허용)
    try:
        return abs(float(val1) - float(val2)) < tolerance
    except (ValueError, TypeError):
        return val1 == val2

def compare_bid_records(existing, new):
    """두 입찰 레코드 비교 (모든 비즈니스 필드)"""
    # 비교할 필드 목록 (메타데이터 제외)
    fields_to_compare = [
        ('bid_name', safe_value_compare),
        ('ordering_agency', safe_value_compare),
        ('industry', safe_value_compare),
        ('region', safe_value_compare),
        ('estimated_price', safe_value_compare),
        ('base_price', safe_value_compare),
        ('a_value', safe_float_compare),
        ('bid_lower_limit_rate', safe_float_compare),
        ('yega_range', safe_float_compare),
        ('opening_date', safe_value_compare),
        ('submission_deadline', safe_value_compare),
        ('input_date', safe_value_compare),
        ('construction_cost', safe_float_compare),
        ('g2b_category', safe_value_compare)
    ]

    # 모든 필드가 동일한지 확인
    for field, compare_func in fields_to_compare:
        existing_val = existing.get(field)
        new_val = new.get(field)

        if not compare_func(existing_val, new_val):
            return False  # 하나라도 다르면 다른 레코드

    return True  # 모든 필드가 동일

def upload_bid_data_from_app(supabase, df):
    """업로드된 파일 데이터를 DB에 저장 (중복 판별 포함)"""

    # ID 컬럼 제거 (있다면)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # 필수 컬럼 확인
    required_cols = ['공고번호', '공고명', '추정가격', '기초금액', '낙찰하한율']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"필수 컬럼 누락: {', '.join(missing_cols)}", 0, 0, 0, 0

    # NaN 안전 처리 함수
    def safe_float(value, default=None):
        if pd.isna(value):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # 1단계: 업로드할 공고번호 추출 및 유효성 검사
    valid_data = []
    error_rows = []

    for idx, row in df.iterrows():
        try:
            # 필수 필드 검증
            bid_lower_limit_rate_value = safe_float(row['낙찰하한율'])
            if bid_lower_limit_rate_value is None:
                error_rows.append({'행번호': idx+2, '공고번호': str(row.get('공고번호', 'N/A')), '오류': '낙찰하한율 누락'})
                continue

            # 데이터 전처리
            data = {
                'bid_number': str(row['공고번호']),
                'bid_name': str(row['공고명']),
                'ordering_agency': str(row['발주기관']) if pd.notna(row.get('발주기관')) else None,
                'industry': str(row['업종']) if pd.notna(row.get('업종')) else None,
                'region': str(row['지역']) if pd.notna(row.get('지역')) else None,
                'estimated_price': int(row['추정가격']),
                'base_price': int(row['기초금액']),
                'a_value': clean_a_value(row.get('A값')),  # 개선된 A값 전처리
                'bid_lower_limit_rate': bid_lower_limit_rate_value,
                'yega_range': parse_yega_range_app(row.get('예가변동폭')),  # 개선된 예가변동폭 전처리
                'opening_date': parse_date_app(row.get('개찰일')),
                'submission_deadline': str(row['투찰마감']) if pd.notna(row.get('투찰마감')) else None,
                'input_date': parse_date_app(row.get('입력일')),
                'construction_cost': safe_float(row.get('순공사원가'), None),
                'g2b_category': str(row['G2B물품분류']) if pd.notna(row.get('G2B물품분류')) else None
            }

            valid_data.append((idx+2, data))  # (엑셀 행번호, 데이터)

        except Exception as e:
            error_rows.append({'행번호': idx+2, '공고번호': str(row.get('공고번호', 'N/A')), '오류': str(e)[:50]})

    if not valid_data:
        return False, "유효한 데이터가 없습니다", 0, 0, 0, error_rows

    # 2단계: 기존 데이터 일괄 조회
    bid_numbers = [data['bid_number'] for _, data in valid_data]

    try:
        existing_response = supabase.table('bid_list')\
            .select('*')\
            .in_('bid_number', bid_numbers)\
            .execute()

        # 딕셔너리로 변환 (O(1) 검색)
        existing_dict = {item['bid_number']: item for item in existing_response.data}

    except Exception as e:
        return False, f"DB 조회 실패: {str(e)}", 0, 0, 0, error_rows

    # 3단계: 신규/업데이트/동일 분류
    new_records = []
    update_records = []
    skip_records = []

    for row_num, data in valid_data:
        bid_num = data['bid_number']

        if bid_num not in existing_dict:
            # 신규 레코드
            new_records.append((row_num, data))
        else:
            # 기존 레코드 존재 - 비교
            existing = existing_dict[bid_num]

            if compare_bid_records(existing, data):
                # 완전히 동일 - 스킵
                skip_records.append((row_num, data))
            else:
                # 다름 - 업데이트 필요
                update_records.append((row_num, data))

    # 4단계: DB 반영
    insert_success = 0
    update_success = 0

    # 신규 삽입
    for row_num, data in new_records:
        try:
            supabase.table('bid_list').insert(data).execute()
            insert_success += 1
        except Exception as e:
            error_rows.append({'행번호': row_num, '공고번호': data['bid_number'], '오류': f"삽입 실패: {str(e)[:30]}"})

    # 업데이트
    for row_num, data in update_records:
        try:
            supabase.table('bid_list')\
                .update(data)\
                .eq('bid_number', data['bid_number'])\
                .execute()
            update_success += 1
        except Exception as e:
            error_rows.append({'행번호': row_num, '공고번호': data['bid_number'], '오류': f"업데이트 실패: {str(e)[:30]}"})

    # 5단계: 결과 리포트
    skip_count = len(skip_records)
    error_count = len(error_rows)

    return True, "업로드 완료", insert_success, update_success, skip_count, error_rows

@st.cache_data(ttl=3600)
def get_training_model_info():
    """학습 모델 정보 조회 (캐시 1시간)"""
    try:
        model_dir = Path("models/production")

        # 최신 메타데이터 파일 찾기
        metadata_files = sorted(model_dir.glob("metadata_*.json"), reverse=True)
        if not metadata_files:
            return None

        # 메타데이터 로드
        import json
        with open(metadata_files[0], 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # 학습 데이터에서 날짜 범위 조회
        try:
            data_path = Path("data/processed/features_engineered.xlsx")
            if data_path.exists():
                df = pd.read_excel(data_path)

                # 날짜 컬럼 찾기 (입력일 또는 개찰일)
                date_col = None
                for col in ['입력일', '개찰일']:
                    if col in df.columns:
                        date_col = col
                        break

                if date_col:
                    # YY.MM.DD 형식 (예: 25.11.17, 24.01.02)을 올바르게 파싱
                    dates = pd.to_datetime(df[date_col], format='%y.%m.%d', errors='coerce')
                    date_min = dates.min()
                    date_max = dates.max()
                else:
                    date_min = date_max = None
            else:
                date_min = date_max = None
        except Exception:
            date_min = date_max = None

        # 이전 모델과 비교 (성능 개선율 계산)
        old_metadata_files = [f for f in metadata_files if f != metadata_files[0]]
        improvement = None
        if old_metadata_files:
            try:
                with open(old_metadata_files[0], 'r', encoding='utf-8') as f:
                    old_metadata = json.load(f)

                old_mape = old_metadata['models']['ensemble']['test_mape']
                new_mape = metadata['models']['ensemble']['test_mape']

                if old_mape > 0:
                    improvement = old_mape / new_mape
            except Exception:
                pass

        return {
            'timestamp': metadata['timestamp'],
            'train_size': metadata['train_size'],
            'test_size': metadata['test_size'],
            'total_size': metadata['train_size'] + metadata['test_size'],
            'date_min': date_min,
            'date_max': date_max,
            'features': metadata['features'],
            'models': metadata['models'],
            'improvement': improvement
        }

    except Exception as e:
        st.error(f"모델 정보 로드 실패: {e}")
        return None


def render_training_model_status():
    """학습 모델 현황 섹션 렌더링 (옵션 2: 콤팩트 카드 + 펼치기)"""
    info = get_training_model_info()

    if not info:
        st.warning("⚠️ 학습 모델 정보를 불러올 수 없습니다.")
        return

    # 타임스탬프 파싱 (YYYYMMDD_HHMMSS)
    timestamp_str = info['timestamp']
    try:
        dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        formatted_time = dt.strftime('%Y-%m-%d %H:%M')
    except:
        formatted_time = timestamp_str

    # 데이터 기간 포맷
    if info['date_min'] and info['date_max']:
        date_range_str = f"{info['date_min'].strftime('%Y-%m')} ~ {info['date_max'].strftime('%Y-%m')}"
    else:
        date_range_str = "알 수 없음"

    # Ensemble 모델 성능
    ensemble = info['models']['ensemble']
    mape = ensemble['test_mape'] * 100  # 백분율로 변환
    r2 = ensemble['test_r2']

    # 개선율 계산
    improvement_text = ""
    if info['improvement']:
        improvement_text = f" ({info['improvement']:.0f}배 개선)"

    # 정보 박스 표시 (데이터베이스 현황과 동일한 디자인, 주황색 계열)
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #ff9a56 0%, #ff6b6b 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <div style="color: white; font-size: 14px; font-weight: 600; margin-bottom: 12px;">
            📊 학습 모델 현황
        </div>
        <div style="
            background: rgba(255, 255, 255, 0.95);
            padding: 16px;
            border-radius: 8px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        ">
            <div>
                <div style="color: #ff6b6b; font-size: 12px; font-weight: 600; margin-bottom: 4px;">
                    ✅ 최신 재학습
                </div>
                <div style="color: #1a202c; font-size: 18px; font-weight: 700;">
                    {formatted_time}
                </div>
            </div>
            <div>
                <div style="color: #ff6b6b; font-size: 12px; font-weight: 600; margin-bottom: 4px;">
                    📈 학습 데이터
                </div>
                <div style="color: #1a202c; font-size: 18px; font-weight: 700;">
                    {info['total_size']:,}개
                </div>
                <div style="color: #718096; font-size: 11px; margin-top: 2px;">
                    {date_range_str}
                </div>
            </div>
            <div>
                <div style="color: #ff6b6b; font-size: 12px; font-weight: 600; margin-bottom: 4px;">
                    🎯 예측 정확도
                </div>
                <div style="color: #1a202c; font-size: 18px; font-weight: 700;">
                    MAPE {mape:.2f}%
                </div>
                <div style="color: #718096; font-size: 11px; margin-top: 2px;">
                    R² {r2:.3f}{improvement_text}
                </div>
            </div>
        </div>
        <div style="color: rgba(255, 255, 255, 0.9); font-size: 11px; margin-top: 10px; text-align: right;">
            💡 Ensemble 모델 (Linear 10% + XGBoost 90%)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 상세 정보 (펼치기) - 기존 유지
    with st.expander("📋 상세 정보 보기", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 데이터 분할")
                st.metric("학습 데이터", f"{info['train_size']:,}개", delta="80%", delta_color="off")
                st.metric("검증 데이터", f"{info['test_size']:,}개", delta="20%", delta_color="off")

            with col2:
                st.markdown("#### 🔧 모델 구성")
                st.markdown(f"""
                **사용 피처** ({len(info['features'])}개):
                {', '.join(info['features'])}

                **Weighted Ensemble**:
                - Linear Regression: {ensemble['lr_weight']*100:.0f}%
                - XGBoost: {ensemble['xgb_weight']*100:.0f}%
                """)

            # 성능 개선 정보
            if info['improvement']:
                st.success(f"🚀 이전 모델 대비 MAPE {info['improvement']:.0f}배 개선")

            # 모델별 성능 비교
            st.markdown("#### 📈 모델별 성능")
            perf_data = {
                '모델': ['Linear Regression', 'Ridge Regression', 'Weighted Ensemble'],
                'Test R²': [
                    info['models']['linear']['test_r2'],
                    info['models']['ridge']['test_r2'],
                    info['models']['ensemble']['test_r2']
                ],
                'Test MAPE (%)': [
                    info['models']['linear']['test_mape'] * 100,
                    info['models']['ridge']['test_mape'] * 100,
                    info['models']['ensemble']['test_mape'] * 100
                ]
            }
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(
                perf_df.style.format({
                    'Test R²': '{:.3f}',
                    'Test MAPE (%)': '{:.2f}%'
                }).highlight_max(subset=['Test R²'], color='lightgreen')
                  .highlight_min(subset=['Test MAPE (%)'], color='lightgreen'),
                use_container_width=True,
                hide_index=True
            )


# ================================================================
# 인증 UI 함수들
# ================================================================

def show_login_page(supabase: Client):
    """로그인 페이지"""
    st.markdown('<div class="main-header">🔐 로그인</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">낙찰하한가 예측 시스템</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 📝 로그인")

        username = st.text_input("아이디", key="login_username", placeholder="아이디를 입력하세요")
        password = st.text_input("비밀번호", type="password", key="login_password", placeholder="비밀번호를 입력하세요")

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("🔓 로그인", use_container_width=True, type="primary"):
                if not username or not password:
                    st.error("⚠️ 아이디와 비밀번호를 모두 입력해주세요.")
                else:
                    user_info = login_user(supabase, username, password)
                    if user_info:
                        st.session_state.user = user_info
                        st.session_state.logged_in = True
                        st.success(f"✅ 환영합니다, {user_info['username']}님!")
                        st.rerun()
                    else:
                        st.error("❌ 아이디 또는 비밀번호가 일치하지 않습니다.")

        with col_btn2:
            if st.button("📝 회원가입", use_container_width=True):
                st.session_state.auth_mode = "register"
                st.rerun()

        st.markdown("---")

        if st.button("🔑 비밀번호 찾기"):
            st.session_state.auth_mode = "reset_password"
            st.rerun()


def show_register_page(supabase: Client):
    """회원가입 페이지"""
    st.markdown('<div class="main-header">📝 회원가입</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">낙찰하한가 예측 시스템</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 👤 회원 정보 입력")

        username = st.text_input("아이디", key="register_username", placeholder="영문, 숫자 조합 (3자 이상)")
        password = st.text_input("비밀번호", type="password", key="register_password", placeholder="최소 6자 이상")
        password_confirm = st.text_input("비밀번호 확인", type="password", key="register_password_confirm", placeholder="비밀번호를 다시 입력하세요")
        email = st.text_input("이메일", key="register_email", placeholder="example@email.com")

        st.markdown("---")

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("✅ 가입하기", use_container_width=True, type="primary"):
                # 유효성 검사
                if not username or not password or not password_confirm or not email:
                    st.error("⚠️ 모든 항목을 입력해주세요.")
                elif len(username) < 3:
                    st.error("⚠️ 아이디는 최소 3자 이상이어야 합니다.")
                elif password != password_confirm:
                    st.error("⚠️ 비밀번호가 일치하지 않습니다.")
                elif len(password) < 6:
                    st.error("⚠️ 비밀번호는 최소 6자 이상이어야 합니다.")
                elif '@' not in email or '.' not in email:
                    st.error("⚠️ 올바른 이메일 형식이 아닙니다.")
                else:
                    # 회원가입 처리
                    if register_user(supabase, username, password, email):
                        st.success("🎉 회원가입이 완료되었습니다! 로그인 페이지로 이동합니다.")
                        st.session_state.auth_mode = "login"
                        st.balloons()
                        st.rerun()

        with col_btn2:
            if st.button("← 로그인으로", use_container_width=True):
                st.session_state.auth_mode = "login"
                st.rerun()


def show_password_reset_page(supabase: Client):
    """비밀번호 찾기 페이지"""
    st.markdown('<div class="main-header">🔑 비밀번호 찾기</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">낙찰하한가 예측 시스템</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 📧 이메일 인증")
        st.info("💡 등록된 이메일을 입력하시면 임시 비밀번호를 발급해드립니다.")

        email = st.text_input("등록된 이메일", key="reset_email", placeholder="example@email.com")

        st.markdown("---")

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("🔄 비밀번호 재설정", use_container_width=True, type="primary"):
                if not email:
                    st.error("⚠️ 이메일을 입력해주세요.")
                elif '@' not in email or '.' not in email:
                    st.error("⚠️ 올바른 이메일 형식이 아닙니다.")
                else:
                    temp_password = reset_password(supabase, email)
                    if temp_password:
                        st.markdown("---")
                        st.markdown("### 🔐 임시 비밀번호")
                        st.code(temp_password, language=None)
                        st.warning("⚠️ 임시 비밀번호를 복사하여 로그인 후 비밀번호를 변경해주세요!")
                        st.info("💡 보안을 위해 이 페이지를 새로고침하면 임시 비밀번호가 사라집니다.")

        with col_btn2:
            if st.button("← 로그인으로", use_container_width=True):
                st.session_state.auth_mode = "login"
                st.rerun()


def main():
    """애플리케이션 진입점 (로그인 게이트)"""
    # Supabase 초기화
    supabase = init_supabase()

    # 세션 상태 초기화
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = "login"  # login, register, reset_password

    # 로그인 여부 확인
    if st.session_state.logged_in and st.session_state.user:
        # 로그인 상태 - 메인 앱 표시
        # 사이드바에 사용자 정보와 로그아웃 버튼 추가
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"### 👤 {st.session_state.user['username']}")
            if st.session_state.user['is_admin']:
                st.caption("🔑 관리자")
            else:
                st.caption("👥 일반 사용자")
            st.caption(f"📧 {st.session_state.user['email']}")

            if st.button("🚪 로그아웃", use_container_width=True):
                logout_user()

        # 메인 애플리케이션 실행
        main_app()
    else:
        # 비로그인 상태 - 인증 페이지 표시
        if st.session_state.auth_mode == "login":
            show_login_page(supabase)
        elif st.session_state.auth_mode == "register":
            show_register_page(supabase)
        elif st.session_state.auth_mode == "reset_password":
            show_password_reset_page(supabase)


# ================================================================
# 메인 애플리케이션
# ================================================================

def main_app():
    """메인 애플리케이션 (로그인 후 진입)"""
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

        # ========== 입찰 공고 검색 섹션 ==========
        st.markdown('<div class="input-section">', unsafe_allow_html=True)

        # DB 통계 정보 표시
        try:
            stats = get_bid_list_stats(SUPABASE_URL, SUPABASE_KEY)

            if stats['total_count'] > 0:
                # 날짜 포맷팅
                latest_date_str = "정보 없음"
                date_range_str = ""

                if stats['latest_input_date']:
                    try:
                        latest_date = pd.to_datetime(stats['latest_input_date'])
                        latest_date_str = latest_date.strftime('%Y년 %m월 %d일')
                    except (ValueError, AttributeError):
                        latest_date_str = stats['latest_input_date']

                if stats['oldest_input_date'] and stats['latest_input_date']:
                    try:
                        oldest_date = pd.to_datetime(stats['oldest_input_date'])
                        date_range_str = f" (DB 범위: {oldest_date.strftime('%Y.%m.%d')} ~ {latest_date.strftime('%Y.%m.%d')})"
                    except (ValueError, AttributeError):
                        pass

                # 최근 데이터 입력 시간 포맷팅 (KST 기준 상대 시간)
                recent_input_str = "정보 없음"
                if stats['latest_created_at']:
                    try:
                        import pytz

                        # created_at은 이미 KST datetime 객체
                        created_kst = stats['latest_created_at']

                        # 현재 KST 시간
                        now_kst = datetime.now(pytz.timezone('Asia/Seoul'))

                        # 시간 차이 계산
                        time_diff = now_kst - created_kst

                        if time_diff.total_seconds() < 60:
                            recent_input_str = "방금 전"
                        elif time_diff.total_seconds() < 3600:
                            minutes = int(time_diff.total_seconds() / 60)
                            recent_input_str = f"{minutes}분 전"
                        elif time_diff.total_seconds() < 86400:
                            hours = int(time_diff.total_seconds() / 3600)
                            recent_input_str = f"{hours}시간 전"
                        else:
                            days = int(time_diff.total_seconds() / 86400)
                            recent_input_str = f"{days}일 전"
                    except Exception as e:
                        recent_input_str = "알 수 없음"

                # 정보 박스 표시
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                ">
                    <div style="color: white; font-size: 14px; font-weight: 600; margin-bottom: 12px;">
                        📊 입찰공고 데이터베이스 현황
                    </div>
                    <div style="
                        background: rgba(255, 255, 255, 0.95);
                        padding: 16px;
                        border-radius: 8px;
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 16px;
                    ">
                        <div>
                            <div style="color: #667eea; font-size: 12px; font-weight: 600; margin-bottom: 4px;">
                                📅 최신 입력일
                            </div>
                            <div style="color: #1a202c; font-size: 18px; font-weight: 700;">
                                {latest_date_str}
                            </div>
                        </div>
                        <div>
                            <div style="color: #667eea; font-size: 12px; font-weight: 600; margin-bottom: 4px;">
                                📈 총 공고 수
                            </div>
                            <div style="color: #1a202c; font-size: 18px; font-weight: 700;">
                                {stats['total_count']:,}건
                            </div>
                        </div>
                        <div>
                            <div style="color: #667eea; font-size: 12px; font-weight: 600; margin-bottom: 4px;">
                                🔄 최근 데이터 입력
                            </div>
                            <div style="color: #1a202c; font-size: 18px; font-weight: 700;">
                                {recent_input_str}
                            </div>
                        </div>
                    </div>
                    <div style="color: rgba(255, 255, 255, 0.9); font-size: 11px; margin-top: 10px; text-align: right;">
                        💡 {latest_date_str} 이후 공고를 업로드하세요{date_range_str}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            # 에러 로깅 (디버깅용)
            import traceback
            print(f"❌ DB 통계 로드 실패: {e}")
            print(traceback.format_exc())
            # 사용자에게는 표시하지 않음

        st.markdown("### 🔍 입찰 공고 검색")
        st.markdown("공고번호 또는 공고명으로 검색하거나, 최신 공고 목록에서 선택하세요.")

        # 검색어 입력 + 최신 공고 버튼
        col_search1, col_search2 = st.columns([6, 1])

        with col_search1:
            # 검색어 입력
            search_query = st.text_input(
                "검색어",
                placeholder="공고번호 또는 공고명을 입력하세요 (최소 2자)",
                help="공고번호로 정확 매칭 또는 공고명으로 부분 검색",
                label_visibility="collapsed"
            )

        with col_search2:
            # 최신 공고 버튼
            latest_btn_label = "📋 최신 공고 ▼" if st.session_state.get('show_latest_bids', False) else "📋 최신 공고"
            if st.button(latest_btn_label, use_container_width=True, help="최신 입력일 기준 공고 목록 보기"):
                st.session_state.show_latest_bids = not st.session_state.get('show_latest_bids', False)
                if 'latest_bids_page' not in st.session_state:
                    st.session_state.latest_bids_page = 1
                st.rerun()

        # ========== 최신 공고 목록 (조건부 표시) ==========
        if st.session_state.get('show_latest_bids', False):
            st.markdown("---")
            st.markdown("#### 📋 최신 입찰 공고")

            # 페이지 가져오기
            page = st.session_state.get('latest_bids_page', 1)

            # 데이터 조회
            bids, total_count = get_latest_bids(SUPABASE_URL, SUPABASE_KEY, page, 10)

            if total_count > 0:
                total_pages = (total_count + 9) // 10  # 올림

                st.markdown(f"총 **{total_count:,}**건 중 **{(page-1)*10+1}~{min(page*10, total_count)}**번째 (페이지 {page}/{total_pages})")

                # 카드 표시
                for bid in bids:
                    with st.container():
                        # 날짜 포맷팅
                        input_date_str = "-"
                        if bid.get('input_date'):
                            try:
                                input_date_str = pd.to_datetime(bid['input_date']).strftime('%Y-%m-%d')
                            except (ValueError, AttributeError):
                                input_date_str = str(bid['input_date'])

                        # 카드 디자인
                        st.markdown(f"""
                        <div style="
                            background: white;
                            padding: 15px;
                            border-radius: 8px;
                            border-left: 4px solid #667eea;
                            margin-bottom: 10px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        ">
                            <div style="font-weight: 600; color: #667eea; font-size: 13px;">📌 {bid['bid_number']}</div>
                            <div style="font-size: 16px; font-weight: 700; margin: 5px 0; color: #1a202c;">{bid['bid_name'][:80]}{'...' if len(bid['bid_name']) > 80 else ''}</div>
                            <div style="font-size: 13px; color: #666;">
                                {bid.get('ordering_agency', '-')[:20] if bid.get('ordering_agency') else '-'} •
                                {input_date_str} •
                                {bid.get('estimated_price', 0):,}원
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # 선택 버튼
                        col_btn1, col_btn2 = st.columns([5, 1])
                        with col_btn2:
                            if st.button("선택하기 →", key=f"select_bid_{bid['id']}", use_container_width=True, type="primary"):
                                # 자동 기입
                                st.session_state.selected_bid = bid
                                st.session_state.show_latest_bids = False
                                st.rerun()

                # 페이지네이션
                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                col_prev, col_info, col_next = st.columns([1, 2, 1])

                with col_prev:
                    if page > 1:
                        if st.button("← 이전", key="prev_page", use_container_width=True):
                            st.session_state.latest_bids_page = page - 1
                            st.rerun()
                    else:
                        st.button("← 이전", key="prev_page_disabled", use_container_width=True, disabled=True)

                with col_info:
                    st.markdown(f"<div style='text-align: center; padding-top: 8px; font-size: 14px;'>페이지 {page} / {total_pages}</div>", unsafe_allow_html=True)

                with col_next:
                    if page < total_pages:
                        if st.button("다음 →", key="next_page", use_container_width=True):
                            st.session_state.latest_bids_page = page + 1
                            st.rerun()
                    else:
                        st.button("다음 →", key="next_page_disabled", use_container_width=True, disabled=True)
            else:
                st.info("데이터가 없습니다.")

            st.markdown("---")

        # ========== 양식 다운로드 및 공고 업로드 ==========
        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown("")  # 공간 유지

        with col2:
            # 양식 다운로드 버튼
            template_file = create_bid_template()
            st.download_button(
                label="📥 양식 다운로드",
                data=template_file,
                file_name=f"입찰공고_업로드_양식_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help="입찰공고 업로드용 엑셀 양식 다운로드"
            )

        with col3:
            # 업로드 버튼 (모달 트리거)
            if st.button("📤 공고 업로드", use_container_width=True, help="엑셀/CSV 파일로 입찰공고 일괄 업로드"):
                st.session_state.show_upload_modal = True

        # 업로드 모달
        if st.session_state.get('show_upload_modal', False):
            with st.container():
                st.markdown("---")
                st.markdown("#### 📤 입찰공고 일괄 업로드")

                uploaded_file = st.file_uploader(
                    "엑셀 또는 CSV 파일을 선택하세요",
                    type=['xlsx', 'csv'],
                    help="양식에 맞게 작성된 입찰공고 파일을 업로드하세요 (.xlsx 또는 .csv 파일만 지원)"
                )

                col_btn1, col_btn2 = st.columns(2)

                with col_btn1:
                    if st.button("❌ 취소", use_container_width=True):
                        st.session_state.show_upload_modal = False
                        st.rerun()

                with col_btn2:
                    if st.button("✅ 업로드 실행", type="primary", use_container_width=True, disabled=uploaded_file is None):
                        if uploaded_file is not None:
                            try:
                                # 파일 읽기
                                if uploaded_file.name.endswith('.csv'):
                                    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                                else:
                                    df = pd.read_excel(uploaded_file)

                                # 업로드 실행
                                with st.spinner('📊 데이터 분석 및 업로드 중...'):
                                    success, message, insert_count, update_count, skip_count, error_rows = upload_bid_data_from_app(supabase, df)

                                if success:
                                    total_processed = insert_count + update_count + skip_count
                                    error_count = len(error_rows)

                                    # 성공 메시지
                                    st.success(f"""
                                    ✅ {message}

                                    **📊 처리 결과**
                                    - 🆕 신규 등록: {insert_count}건
                                    - 🔄 업데이트: {update_count}건
                                    - ⏭️ 동일 (스킵): {skip_count}건
                                    - ✔️ 총 처리: {total_processed}건
                                    - ❌ 오류: {error_count}건
                                    """)

                                    # 캐시 무효화 (DB 통계 즉시 갱신)
                                    st.cache_data.clear()

                                    # 통계 시각화
                                    if total_processed > 0:
                                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                                        with col_stat1:
                                            st.metric("신규", f"{insert_count}건",
                                                     delta=f"{insert_count/total_processed*100:.1f}%")
                                        with col_stat2:
                                            st.metric("업데이트", f"{update_count}건",
                                                     delta=f"{update_count/total_processed*100:.1f}%")
                                        with col_stat3:
                                            st.metric("동일(스킵)", f"{skip_count}건",
                                                     delta=f"{skip_count/total_processed*100:.1f}%")

                                    # 오류 상세
                                    if error_count > 0 and error_rows:
                                        with st.expander("❌ 오류 상세 보기"):
                                            error_df = pd.DataFrame(error_rows)
                                            st.dataframe(error_df, use_container_width=True)

                                            # CSV 다운로드
                                            csv = error_df.to_csv(index=False, encoding='utf-8-sig')
                                            st.download_button(
                                                label="📥 오류 목록 다운로드",
                                                data=csv,
                                                file_name=f"upload_errors_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv"
                                            )

                                    st.session_state.show_upload_modal = False
                                    if insert_count > 0 or update_count > 0:
                                        st.balloons()
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")

                            except Exception as e:
                                st.error(f"❌ 파일 처리 실패: {str(e)}")
                                import traceback
                                with st.expander("상세 오류"):
                                    st.code(traceback.format_exc())

                st.markdown("---")

        # 검색 실행 (2자 이상 입력시 자동 검색)
        if search_query and len(search_query) >= 2:
            search_results = search_bid_list(supabase, search_query)

            if search_results:
                st.markdown(f"**검색 결과: {len(search_results)}건**")

                # 검색 결과를 카드 형태로 표시
                for idx, result in enumerate(search_results):
                    with st.container():
                        # 날짜 포맷 (None 처리)
                        opening_date_str = result['opening_date'] if result.get('opening_date') else '미정'
                        if opening_date_str != '미정':
                            try:
                                opening_date_str = pd.to_datetime(opening_date_str).strftime('%Y-%m-%d')
                            except (ValueError, AttributeError):
                                opening_date_str = str(result['opening_date'])

                        # 선택적 필드 처리 (None일 경우 대체 텍스트)
                        industry_str = result.get('industry') if result.get('industry') else '-'
                        region_str = result.get('region') if result.get('region') else '-'
                        yega_range_str = f"±{result['yega_range']}%" if result.get('yega_range') else '-'

                        st.markdown(
                            f"""
                            <div style="
                                border: 1px solid #e0e0e0;
                                border-radius: 8px;
                                padding: 15px;
                                margin: 10px 0;
                                background-color: #f9f9f9;
                            ">
                                <div style="font-weight: bold; font-size: 16px; margin-bottom: 8px;">
                                    📋 {result['bid_name']}
                                </div>
                                <div style="color: #666; font-size: 14px; line-height: 1.6;">
                                    <span style="font-weight: 600;">공고번호:</span> {result['bid_number']}<br>
                                    <span style="font-weight: 600;">개찰일:</span> {opening_date_str} |
                                    <span style="font-weight: 600;">예가변동폭:</span> {yega_range_str}<br>
                                    <span style="font-weight: 600;">업종:</span> {industry_str} |
                                    <span style="font-weight: 600;">지역:</span> {region_str}<br>
                                    <span style="font-weight: 600;">추정가격:</span> {format_currency(result['estimated_price'])}원 |
                                    <span style="font-weight: 600;">기초금액:</span> {format_currency(result['base_price'])}원<br>
                                    <span style="font-weight: 600;">낙찰하한율:</span> {result['bid_lower_limit_rate']}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # 선택 버튼
                        if st.button(f"✅ 이 공고 선택", key=f"select_bid_{idx}"):
                            # session_state에 선택된 입찰 정보 저장
                            st.session_state.selected_bid = result
                            st.success(f"✅ '{result['bid_name']}' 공고가 선택되었습니다. 아래 입력란이 자동으로 채워집니다.")
                            st.rerun()
            else:
                st.info("검색 결과가 없습니다. 다른 검색어를 입력해보세요.")

        st.markdown('</div>', unsafe_allow_html=True)

        # 구분선
        st.markdown("---")

        # ========== 수동 입력 섹션 ==========
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 📝 입력 정보")

        # 선택된 입찰이 있으면 자동 입력, 없으면 기본값 사용
        selected_bid = st.session_state.get('selected_bid', None)

        default_bid_number = selected_bid['bid_number'] if selected_bid else ""
        default_project_name = selected_bid['bid_name'] if selected_bid else ""
        default_chujeong = selected_bid['estimated_price'] if selected_bid else 0
        default_gichogeum = selected_bid['base_price'] if selected_bid else 0
        default_a_value = float(selected_bid['a_value']) if selected_bid and selected_bid['a_value'] else 0
        default_nakchalhahan = float(selected_bid['bid_lower_limit_rate']) if selected_bid else 87.745
        default_yega_range = float(selected_bid['yega_range']) if selected_bid else 2.5

        # 공사명 입력 (전체 너비, 최상단)
        project_name = st.text_input(
            "공사명 *",
            value=default_project_name,
            placeholder="예: 서울시 강남구 테헤란로 도로 보수공사",
            help="예측하려는 공사의 이름을 입력하세요 (필수)",
            max_chars=200
        )

        # 공고번호 입력 (선택사항)
        bid_number = st.text_input(
            "공고번호",
            value=default_bid_number,
            placeholder="예: R25BK00554395-000",
            help="입찰 공고번호 (선택사항)",
            max_chars=100
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            chujeong_price_str = st.text_input(
                "추정가격 (원)",
                value=format_number_for_display(default_chujeong),
                placeholder="예: 100,000,000",
                help="입찰 공고에 명시된 추정가격 (쉼표 입력 가능)"
            )
            chujeong_price = parse_number_input(chujeong_price_str)

            gichogeum_str = st.text_input(
                "기초금액 (원)",
                value=format_number_for_display(default_gichogeum),
                placeholder="예: 95,000,000",
                help="예가 산정의 기준이 되는 금액 (쉼표 입력 가능)"
            )
            gichogeum = parse_number_input(gichogeum_str)

            a_value_str = st.text_input(
                "A값 (원)",
                value=format_number_for_display(int(default_a_value)),
                placeholder="예: 5,000,000",
                help="낙찰하한율 계산에 사용되는 A값 (쉼표 입력 가능, 없으면 0)"
            )
            a_value = parse_number_input(a_value_str)

        with col2:
            nakchalhahan_rate = st.number_input(
                "낙찰하한율 (%)",
                min_value=0.0,
                max_value=100.0,
                value=default_nakchalhahan,
                step=0.001,
                format="%.3f",
                help="낙찰 가능한 최저 비율"
            )

            # 예가변동폭 selectbox의 인덱스 계산
            yega_options = [2, 2.5, 3]
            default_yega_idx = yega_options.index(default_yega_range) if default_yega_range in yega_options else 1

            yega_range = st.selectbox(
                "예가변동폭",
                options=yega_options,
                index=default_yega_idx,
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

            # 금액 입력 검증
            if chujeong_price == 0:
                st.error("⚠️ 추정가격을 입력해주세요!")
                st.stop()
            if gichogeum == 0:
                st.error("⚠️ 기초금액을 입력해주세요!")
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
                'bid_number': bid_number,
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

                        # 공고번호 처리 (빈 문자열이면 None)
                        bid_number_value = st.session_state.input_data['bid_number']
                        bid_number_value = bid_number_value if bid_number_value.strip() else None

                        data = {
                            "prediction_group_id": group_id,
                            "bid_number": bid_number_value,
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
                            "user_id": st.session_state.user['id'],  # 현재 로그인한 사용자 ID
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

        # 학습 모델 현황 표시
        render_training_model_status()

        try:
            # 사용자 권한에 따라 예측 조회
            if st.session_state.user['is_admin']:
                # 관리자: 모든 데이터 조회
                response = supabase.table("predictions").select("*").order("created_at", desc=True).execute()
            else:
                # 일반 사용자: 본인 데이터만 조회
                user_id = st.session_state.user['id']
                response = (supabase.table("predictions")
                    .select("*")
                    .eq("user_id", user_id)
                    .order("created_at", desc=True)
                    .execute())

            if response.data:
                df = pd.DataFrame(response.data)

                # 날짜 포맷
                df['created_at_formatted'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')

                # 그룹별로 정리
                unique_groups = df['prediction_group_id'].unique()

                # ========== 필터 섹션 ==========
                st.markdown("### 🔍 필터 및 정렬")
                filter_col1, filter_col2, filter_col3 = st.columns(3)

                with filter_col1:
                    sort_option = st.selectbox(
                        "정렬 기준",
                        options=["최신순", "오래된순", "오차율 낮은순", "오차율 높은순"],
                        index=0,
                        key="sort_option"
                    )

                with filter_col2:
                    items_per_page = st.selectbox(
                        "페이지당 표시",
                        options=[10, 20, 50, "전체"],
                        index=1,  # 기본값: 20개
                        key="items_per_page"
                    )

                with filter_col3:
                    status_filter = st.selectbox(
                        "상태 필터",
                        options=["전체", "완료 (실제값 입력됨)", "대기 (실제값 미입력)"],
                        index=0,
                        key="status_filter"
                    )

                st.markdown("---")

                # ========== 데이터 필터링 및 정렬 ==========
                # 그룹별 데이터 준비
                group_data_list = []
                for group_id in unique_groups:
                    group_df = df[df['prediction_group_id'] == group_id].copy()
                    first_row = group_df.iloc[0]

                    # 실제값 및 오차 정보
                    actual_value = first_row['actual_nakchalhahan_price']
                    has_actual = pd.notna(actual_value)

                    # 오차율 계산 (Ensemble 모델 기준)
                    ensemble_row = group_df[group_df['model_type'] == 'ensemble']
                    avg_error_rate = None
                    if not ensemble_row.empty and pd.notna(ensemble_row.iloc[0].get('error_rate')):
                        avg_error_rate = abs(ensemble_row.iloc[0]['error_rate'])

                    group_data_list.append({
                        'group_id': group_id,
                        'created_at': pd.to_datetime(first_row['created_at']),
                        'created_at_formatted': first_row['created_at_formatted'],
                        'has_actual': has_actual,
                        'avg_error_rate': avg_error_rate,
                        'group_df': group_df
                    })

                # 상태 필터링
                if status_filter == "완료 (실제값 입력됨)":
                    group_data_list = [g for g in group_data_list if g['has_actual']]
                elif status_filter == "대기 (실제값 미입력)":
                    group_data_list = [g for g in group_data_list if not g['has_actual']]

                # 정렬
                if sort_option == "최신순":
                    group_data_list.sort(key=lambda x: x['created_at'], reverse=True)
                elif sort_option == "오래된순":
                    group_data_list.sort(key=lambda x: x['created_at'], reverse=False)
                elif sort_option == "오차율 낮은순":
                    # 오차율이 있는 것만 먼저, 그 다음 오차율 오름차순
                    group_data_list.sort(key=lambda x: (x['avg_error_rate'] is None, x['avg_error_rate'] if x['avg_error_rate'] is not None else float('inf')))
                elif sort_option == "오차율 높은순":
                    # 오차율이 있는 것만 먼저, 그 다음 오차율 내림차순
                    group_data_list.sort(key=lambda x: (x['avg_error_rate'] is None, -(x['avg_error_rate'] if x['avg_error_rate'] is not None else 0)), reverse=False)

                total_groups = len(group_data_list)

                # ========== 페이지네이션 ==========
                if items_per_page == "전체":
                    items_per_page_int = total_groups
                else:
                    items_per_page_int = int(items_per_page)

                # 세션 상태 초기화
                if 'current_page' not in st.session_state:
                    st.session_state.current_page = 1

                # 총 페이지 수 계산
                total_pages = max(1, (total_groups + items_per_page_int - 1) // items_per_page_int)

                # 현재 페이지가 총 페이지 수를 초과하는 경우 조정
                if st.session_state.current_page > total_pages:
                    st.session_state.current_page = total_pages

                # 현재 페이지 데이터 추출
                start_idx = (st.session_state.current_page - 1) * items_per_page_int
                end_idx = min(start_idx + items_per_page_int, total_groups)
                current_page_data = group_data_list[start_idx:end_idx]

                # 페이지 정보 표시
                if total_groups > 0:
                    st.markdown(f"### 📋 총 {total_groups}개의 예측 그룹 (현재: {start_idx + 1}-{end_idx})")
                else:
                    st.markdown(f"### 📋 총 0개의 예측 그룹")

                st.markdown("---")

                # ========== 예측 그룹 표시 (현재 페이지만) ==========
                for group_data in current_page_data:
                    group_id = group_data['group_id']
                    group_df = group_data['group_df']
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

                    # 오차 정보 (Ensemble 모델 기준)
                    error_info = ""
                    if pd.notna(actual_value):
                        ensemble_row = group_df[group_df['model_type'] == 'ensemble']
                        if not ensemble_row.empty and pd.notna(ensemble_row.iloc[0].get('error_rate')):
                            error_rate = ensemble_row.iloc[0]['error_rate']
                            error_sign = "+" if error_rate >= 0 else ""

                            # 최소 오차 모델 찾기
                            min_abs_error = float('inf')
                            best_model_name = None
                            for model_type in ['ridge', 'linear', 'ensemble']:
                                model_row = group_df[group_df['model_type'] == model_type]
                                if not model_row.empty and pd.notna(model_row.iloc[0].get('error_rate')):
                                    abs_err = abs(model_row.iloc[0]['error_rate'])
                                    if abs_err < min_abs_error:
                                        min_abs_error = abs_err
                                        best_model_name = model_row.iloc[0]['model_name']

                            error_info = f" | 오차: {error_sign}{error_rate:.2f}% ({best_model_name} ✅)"

                    # 컴팩트 Expander 제목 (2줄)
                    expander_title = f"{status_icon} {created_at} | {project_display}\n   기초: {gichogeum}원{error_info if error_info else ' | 실제: ' + actual_text}"

                    # Expander로 그룹별 상세 정보 표시
                    with st.expander(
                        expander_title,
                        expanded=False
                    ):
                        # 공사명 및 공고번호 표시
                        bid_number_display = first_row.get('bid_number', None)
                        bid_number_text = f" | **📋 공고번호**: {bid_number_display}" if pd.notna(bid_number_display) and bid_number_display else ""
                        st.info(f"**🏗️ 공사명**: {project_display}{bid_number_text}")

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

                        # 오차가 있는 경우 최소 오차 모델 찾기
                        best_model = None
                        if pd.notna(actual_value):
                            min_abs_error = float('inf')
                            for model_type in model_order:
                                model_row = group_df[group_df['model_type'] == model_type].iloc[0]
                                if pd.notna(model_row.get('error_rate')):
                                    abs_error = abs(model_row['error_rate'])
                                    if abs_error < min_abs_error:
                                        min_abs_error = abs_error
                                        best_model = model_type

                        for idx, model_type in enumerate(model_order):
                            model_row = group_df[group_df['model_type'] == model_type].iloc[0]

                            with cols[idx]:
                                icon = model_icons.get(model_type, '📊')
                                is_best = (model_type == best_model)
                                best_badge = " ✅ 최소오차" if is_best else ""

                                st.markdown(f"**{icon} {model_row['model_name']}{best_badge}**")
                                st.metric("예측 낙찰하한가", format_currency(model_row['predicted_nakchalhahan_price']) + "원")
                                st.caption(f"예가: {model_row['predicted_yega']:.4f}%")

                                # 오차 정보 표시 (실제값이 입력된 경우)
                                if pd.notna(actual_value) and pd.notna(model_row.get('error_amount')):
                                    st.markdown("---")
                                    st.markdown(f"**실제**: {format_currency(actual_value)}원")

                                    error_amount = model_row['error_amount']
                                    error_rate = model_row['error_rate']

                                    # 오차 부호에 따른 표시
                                    error_sign = "+" if error_amount >= 0 else ""

                                    st.markdown(f"**오차**: {error_sign}{format_currency(abs(error_amount))}원")
                                    st.markdown(f"**오차율**: {error_sign}{error_rate:.2f}%")

                                    # 프로그레스 바 (오차율 시각화)
                                    abs_error_rate = abs(error_rate)
                                    # 0-5% 범위로 정규화 (5% 이상은 100%로 표시)
                                    progress_value = min(abs_error_rate / 5.0, 1.0)

                                    # 색상 선택 (중급 시각화)
                                    if abs_error_rate <= 1:
                                        color = "🟢"  # 매우 우수
                                    elif abs_error_rate <= 3:
                                        color = "🟡"  # 우수
                                    else:
                                        color = "🟠"  # 보통

                                    st.progress(progress_value)
                                    st.caption(f"{color} 정확도: {100 - abs_error_rate:.1f}%")

                        # 실제값 입력 섹션
                        st.markdown("---")
                        st.markdown("#### ✏️ 실제 낙찰하한가 입력")

                        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                        with col1:
                            # 기존 값 포맷팅
                            default_actual_value = int(actual_value) if pd.notna(actual_value) else 0
                            actual_input_str = st.text_input(
                                "실제 낙찰하한가 (원)",
                                value=format_number_for_display(default_actual_value),
                                placeholder="예: 85,000,000",
                                help="실제 낙찰하한가 (쉼표 입력 가능)",
                                key=f"actual_{group_id}"
                            )
                            actual_input = parse_number_input(actual_input_str)

                        with col2:
                            # 기존 날짜 값 가져오기
                            existing_date = first_row.get('bid_announcement_date', None)
                            date_value = None
                            if pd.notna(existing_date) and existing_date:
                                try:
                                    date_value = pd.to_datetime(existing_date).date()
                                except (ValueError, AttributeError):
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
                                    # 해당 그룹의 각 모델별로 오차 계산 및 업데이트
                                    for _, row in group_df.iterrows():
                                        # 오차 계산
                                        predicted_value = row['predicted_nakchalhahan_price']
                                        error_amount = float(actual_input) - predicted_value
                                        error_rate = (error_amount / float(actual_input)) * 100 if actual_input > 0 else 0

                                        update_data = {
                                            "actual_nakchalhahan_price": float(actual_input),
                                            "error_amount": float(error_amount),
                                            "error_rate": float(error_rate),
                                            "updated_at": datetime.now().isoformat()
                                        }

                                        # 날짜가 선택된 경우에만 추가
                                        if actual_date:
                                            update_data["bid_announcement_date"] = actual_date.isoformat()

                                        # 각 레코드 개별 업데이트 (id 기준)
                                        supabase.table("predictions").update(update_data).eq("id", row['id']).execute()

                                    st.success("✅ 실제값 및 오차 분석이 저장되었습니다!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ 저장 실패: {str(e)}")

                        with col4:
                            st.markdown("<br>", unsafe_allow_html=True)
                            confirm_key = f"confirm_delete_{group_id}"

                            if st.button("🗑️ 삭제", type="secondary", use_container_width=True, key=f"delete_{group_id}"):
                                if st.session_state.get(confirm_key, False):
                                    # 2차 클릭 - 실제 삭제 실행
                                    try:
                                        delete_count = len(group_df)
                                        for _, row in group_df.iterrows():
                                            supabase.table("predictions").delete().eq("id", row['id']).execute()

                                        # 확인 상태 초기화
                                        if confirm_key in st.session_state:
                                            del st.session_state[confirm_key]

                                        st.success(f"✅ {delete_count}건의 예측 이력이 삭제되었습니다!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ 삭제 실패: {str(e)}")
                                else:
                                    # 1차 클릭 - 확인 요청
                                    st.session_state[confirm_key] = True
                                    st.warning(f"⚠️ 한 번 더 클릭하면 {len(group_df)}건이 삭제됩니다 (되돌릴 수 없음)")
                                    st.rerun()

                # ========== 페이지네이션 버튼 ==========
                if total_groups > 0 and items_per_page != "전체":
                    st.markdown("---")

                    # 페이지 버튼 배치
                    page_cols = st.columns([1, 3, 1])

                    with page_cols[0]:
                        if st.session_state.current_page > 1:
                            if st.button("◀ 이전", use_container_width=True, key="prev_page"):
                                st.session_state.current_page -= 1
                                st.rerun()

                    with page_cols[1]:
                        # 페이지 번호 버튼
                        max_buttons = 10  # 최대 표시할 페이지 버튼 수

                        # 현재 페이지 기준으로 표시할 페이지 범위 계산
                        if total_pages <= max_buttons:
                            page_range = range(1, total_pages + 1)
                        else:
                            # 현재 페이지를 중심으로 표시
                            half = max_buttons // 2
                            start = max(1, st.session_state.current_page - half)
                            end = min(total_pages, start + max_buttons - 1)

                            # 끝에 도달한 경우 시작 조정
                            if end - start < max_buttons - 1:
                                start = max(1, end - max_buttons + 1)

                            page_range = range(start, end + 1)

                        # 페이지 버튼 생성
                        button_cols = st.columns(len(page_range))
                        for idx, page_num in enumerate(page_range):
                            with button_cols[idx]:
                                if page_num == st.session_state.current_page:
                                    # 현재 페이지 (비활성화)
                                    st.button(
                                        f"**{page_num}**",
                                        disabled=True,
                                        use_container_width=True,
                                        key=f"page_btn_{page_num}"
                                    )
                                else:
                                    # 다른 페이지 (클릭 가능)
                                    if st.button(
                                        str(page_num),
                                        use_container_width=True,
                                        key=f"page_btn_{page_num}"
                                    ):
                                        st.session_state.current_page = page_num
                                        st.rerun()

                    with page_cols[2]:
                        if st.session_state.current_page < total_pages:
                            if st.button("다음 ▶", use_container_width=True, key="next_page"):
                                st.session_state.current_page += 1
                                st.rerun()

                    # 페이지 정보 표시
                    st.caption(f"페이지 {st.session_state.current_page} / {total_pages}")

            else:
                st.info("📭 아직 저장된 예측이 없습니다.")

        except Exception as e:
            st.error(f"⚠️ 데이터 조회 실패: {str(e)}")

    # ========== 정확도 분석 ==========
    elif menu == "📈 정확도 분석":
        st.markdown("## 📈 정확도 분석")

        try:
            # 사용자 권한에 따라 실제값이 입력된 예측만 조회
            if st.session_state.user['is_admin']:
                # 관리자: 모든 데이터 조회
                response = supabase.table("predictions").select("*").not_.is_("actual_nakchalhahan_price", "null").execute()
            else:
                # 일반 사용자: 본인 데이터만 조회
                user_id = st.session_state.user['id']
                response = (supabase.table("predictions")
                    .select("*")
                    .eq("user_id", user_id)
                    .not_.is_("actual_nakchalhahan_price", "null")
                    .execute())

            if response.data and len(response.data) > 0:
                df = pd.DataFrame(response.data)

                # 오차 계산 (DB에 error_rate가 있지만 호환성을 위해 계산)
                df['error'] = df['actual_nakchalhahan_price'] - df['predicted_nakchalhahan_price']
                df['error_pct'] = (df['error'] / df['actual_nakchalhahan_price']) * 100
                df['abs_error_pct'] = df['error_pct'].abs()

                # 날짜 변환
                df['created_at'] = pd.to_datetime(df['created_at'])
                df['date'] = df['created_at'].dt.date

                # ============ 필터링 UI ============
                st.markdown("### 🔍 필터 옵션")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    # 모델 선택
                    all_models = df['model_name'].unique().tolist()
                    selected_models = st.multiselect(
                        "모델 선택",
                        options=all_models,
                        default=all_models,
                        help="분석할 모델을 선택하세요"
                    )

                with col2:
                    # 기간 선택
                    min_date = df['date'].min()
                    max_date = df['date'].max()
                    date_range = st.date_input(
                        "예측 기간",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        help="분석 기간을 선택하세요"
                    )

                with col3:
                    # 오차율 범위
                    error_range = st.slider(
                        "오차율 범위 (%)",
                        min_value=-20.0,
                        max_value=20.0,
                        value=(-10.0, 10.0),
                        step=0.5,
                        help="표시할 오차율 범위를 선택하세요"
                    )

                with col4:
                    # 공사명 검색
                    search_term = st.text_input(
                        "공사명 검색",
                        placeholder="예: 도로",
                        help="공사명에 포함된 키워드로 검색"
                    )

                # 필터 적용
                filtered_df = df.copy()

                # 모델 필터
                if selected_models:
                    filtered_df = filtered_df[filtered_df['model_name'].isin(selected_models)]

                # 날짜 필터
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]

                # 오차율 필터
                filtered_df = filtered_df[(filtered_df['error_pct'] >= error_range[0]) & (filtered_df['error_pct'] <= error_range[1])]

                # 공사명 검색
                if search_term:
                    filtered_df = filtered_df[filtered_df['project_name'].str.contains(search_term, case=False, na=False)]

                # 필터 결과 표시
                st.caption(f"📊 총 {len(df)}건 중 {len(filtered_df)}건 표시")

                if len(filtered_df) == 0:
                    st.warning("⚠️ 필터 조건에 맞는 데이터가 없습니다.")
                    st.stop()

                # ============ 모델별 성능 요약 ============
                st.markdown("---")
                st.markdown("### 📊 모델별 성능 요약")

                model_stats = filtered_df.groupby('model_name').agg({
                    'abs_error_pct': ['mean', 'std', 'count']
                }).round(2)

                model_stats.columns = ['MAPE (%)', '표준편차 (%)', '예측 건수']
                st.dataframe(model_stats, use_container_width=True)

                # ============ 차트 섹션 ============
                st.markdown("---")
                st.markdown("### 📈 상세 분석 차트")

                # Row 1: 시간별 정확도 추이 & 오차율 분포
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📅 시간에 따른 모델 정확도 추이")
                    # 날짜별 평균 오차율
                    time_stats = filtered_df.groupby(['date', 'model_name'])['abs_error_pct'].mean().reset_index()

                    fig = go.Figure()
                    for model in selected_models:
                        model_data = time_stats[time_stats['model_name'] == model]
                        fig.add_trace(go.Scatter(
                            x=model_data['date'],
                            y=model_data['abs_error_pct'],
                            mode='lines+markers',
                            name=model,
                            line=dict(width=2)
                        ))

                    fig.update_layout(
                        xaxis_title="날짜",
                        yaxis_title="평균 절대 오차율 (%)",
                        height=350,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("#### 📊 오차율 분포 히스토그램")
                    fig = go.Figure()

                    for model in selected_models:
                        model_data = filtered_df[filtered_df['model_name'] == model]
                        fig.add_trace(go.Histogram(
                            x=model_data['error_pct'],
                            name=model,
                            opacity=0.7,
                            nbinsx=20
                        ))

                    fig.update_layout(
                        barmode='overlay',
                        xaxis_title="오차율 (%)",
                        yaxis_title="빈도",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Row 2: 금액별 정확도 & 모델 승률
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 💰 금액 구간별 예측 정확도")
                    # 추정가격 구간 생성
                    filtered_df['price_range'] = pd.cut(
                        filtered_df['chujeong_price'],
                        bins=[0, 100000000, 500000000, 1000000000, float('inf')],
                        labels=['0~1억', '1~5억', '5~10억', '10억 이상']
                    )

                    price_stats = filtered_df.groupby(['price_range', 'model_name'])['abs_error_pct'].mean().reset_index()

                    fig = go.Figure()
                    for model in selected_models:
                        model_data = price_stats[price_stats['model_name'] == model]
                        fig.add_trace(go.Bar(
                            x=model_data['price_range'],
                            y=model_data['abs_error_pct'],
                            name=model,
                            text=model_data['abs_error_pct'].round(2),
                            textposition='auto'
                        ))

                    fig.update_layout(
                        barmode='group',
                        xaxis_title="추정가격 구간",
                        yaxis_title="평균 MAPE (%)",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("#### 🏆 모델별 승률 (최고 정확도 달성)")
                    # 각 예측 그룹에서 최고 성능 모델 찾기
                    best_models = filtered_df.loc[filtered_df.groupby('prediction_group_id')['abs_error_pct'].idxmin()]
                    win_counts = best_models['model_name'].value_counts()

                    fig = go.Figure(data=[go.Pie(
                        labels=win_counts.index,
                        values=win_counts.values,
                        hole=0.4,
                        textinfo='label+percent',
                        marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
                    )])

                    fig.update_layout(
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Row 3: 박스플롯 (전체 폭)
                st.markdown("#### 📦 모델별 오차율 분포 (박스플롯)")
                fig = go.Figure()

                for model in selected_models:
                    model_data = filtered_df[filtered_df['model_name'] == model]
                    fig.add_trace(go.Box(
                        y=model_data['error_pct'],
                        name=model,
                        boxmean='sd'  # 평균과 표준편차 표시
                    ))

                fig.update_layout(
                    yaxis_title="오차율 (%)",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

                # Row 4: 예측 vs 실제 & 월별 활동
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 🎯 예측 vs 실제")
                    fig = go.Figure()

                    for model in selected_models:
                        model_df = filtered_df[filtered_df['model_name'] == model]
                        fig.add_trace(go.Scatter(
                            x=model_df['predicted_nakchalhahan_price'],
                            y=model_df['actual_nakchalhahan_price'],
                            mode='markers',
                            name=model,
                            marker=dict(size=8)
                        ))

                    # 대각선 (완벽한 예측선)
                    min_val = min(filtered_df['predicted_nakchalhahan_price'].min(), filtered_df['actual_nakchalhahan_price'].min())
                    max_val = max(filtered_df['predicted_nakchalhahan_price'].max(), filtered_df['actual_nakchalhahan_price'].max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='완벽한 예측',
                        line=dict(dash='dash', color='gray')
                    ))

                    fig.update_layout(
                        xaxis_title="예측 낙찰하한가",
                        yaxis_title="실제 낙찰하한가",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("#### 📅 월별 예측 활동 및 정확도")
                    # 월별 데이터 집계
                    filtered_df['month'] = filtered_df['created_at'].dt.to_period('M').astype(str)
                    monthly_stats = filtered_df.groupby('month').agg({
                        'prediction_group_id': 'nunique',  # 예측 건수
                        'abs_error_pct': 'mean'  # 평균 MAPE
                    }).reset_index()
                    monthly_stats.columns = ['month', 'count', 'mape']

                    # 이중 Y축 차트
                    fig = go.Figure()

                    # 예측 건수 (바)
                    fig.add_trace(go.Bar(
                        x=monthly_stats['month'],
                        y=monthly_stats['count'],
                        name='예측 건수',
                        yaxis='y',
                        marker_color='lightblue'
                    ))

                    # 평균 MAPE (라인)
                    fig.add_trace(go.Scatter(
                        x=monthly_stats['month'],
                        y=monthly_stats['mape'],
                        name='평균 MAPE',
                        yaxis='y2',
                        mode='lines+markers',
                        line=dict(color='red', width=2)
                    ))

                    fig.update_layout(
                        xaxis_title="월",
                        yaxis=dict(title="예측 건수", side='left'),
                        yaxis2=dict(title="평균 MAPE (%)", overlaying='y', side='right'),
                        height=350,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # ============ 전체 데이터 테이블 ============
                st.markdown("---")
                st.markdown("### 📋 전체 예측 내역")

                # 데이터 준비
                display_df = filtered_df[[
                    'created_at', 'project_name', 'model_name',
                    'predicted_nakchalhahan_price', 'actual_nakchalhahan_price',
                    'error', 'error_pct', 'abs_error_pct'
                ]].copy()

                display_df.columns = [
                    '예측일시', '공사명', '모델',
                    '예측값', '실제값', '오차(원)', '오차율(%)', '절대오차율(%)'
                ]

                # 포맷팅
                display_df['예측일시'] = pd.to_datetime(display_df['예측일시']).dt.strftime('%Y-%m-%d %H:%M')
                display_df['예측값'] = display_df['예측값'].apply(lambda x: f"{x:,.0f}")
                display_df['실제값'] = display_df['실제값'].apply(lambda x: f"{x:,.0f}")
                display_df['오차(원)'] = display_df['오차(원)'].apply(lambda x: f"{x:+,.0f}")
                display_df['오차율(%)'] = display_df['오차율(%)'].apply(lambda x: f"{x:+.2f}%")
                display_df['절대오차율(%)'] = display_df['절대오차율(%)'].apply(lambda x: f"{x:.2f}%")

                # 다운로드 버튼
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    csv = display_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 CSV 다운로드",
                        data=csv,
                        file_name=f"prediction_accuracy_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

                # 테이블 표시
                st.dataframe(display_df, use_container_width=True, height=400)

            else:
                st.info("📭 실제값이 입력된 예측이 아직 없습니다.")

        except Exception as e:
            st.error(f"⚠️ 데이터 분석 실패: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
