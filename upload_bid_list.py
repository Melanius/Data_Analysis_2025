"""
Bid list Excel 데이터를 Supabase에 업로드하는 스크립트
"""
import pandas as pd
import re
from supabase import create_client
import os
from datetime import datetime

# 환경 변수 로드 (dotenv 없이 직접 읽기)
def load_env_file(env_path='.env'):
    """간단한 .env 파일 파서"""
    env_vars = {}
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return env_vars

env_vars = load_env_file('/mnt/c/Users/star/claude/bid-prediction/.env')
SUPABASE_URL = env_vars.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = env_vars.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

def parse_yega_range(yega_str):
    """
    예가변동폭 문자열을 숫자로 파싱
    예: "-2.0/2.0" -> 2.0, "-3/+3" -> 3.0
    """
    if pd.isna(yega_str):
        return 2.0  # 기본값

    try:
        # 문자열에서 숫자 추출
        match = re.findall(r'[+\-]?(\d+\.?\d*)', str(yega_str))
        if match:
            # 양수 값 반환 (상한값)
            numbers = [float(m) for m in match]
            return max(numbers)
        return 2.0
    except:
        return 2.0

def parse_date(date_str):
    """
    날짜 문자열을 파싱
    예: "25.01.08 (10:30)" -> "2025-01-08"
    """
    if pd.isna(date_str):
        return None

    try:
        # "25.01.08 (10:30)" 형식 파싱
        date_part = str(date_str).split('(')[0].strip()
        # YY.MM.DD 형식
        parts = date_part.split('.')
        if len(parts) == 3:
            year = int(parts[0])
            # 25 -> 2025
            if year < 100:
                year = 2000 + year
            month = int(parts[1])
            day = int(parts[2])
            return f"{year:04d}-{month:02d}-{day:02d}"
        return None
    except:
        return None

def upload_bid_data(excel_file):
    """Excel 데이터를 Supabase에 업로드"""
    print(f"📂 Excel 파일 읽기: {excel_file}")
    df = pd.read_excel(excel_file)

    print(f"📊 총 {len(df)}개 레코드 발견")

    # Supabase 클라이언트 초기화
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    success_count = 0
    error_count = 0

    for idx, row in df.iterrows():
        try:
            # NaN 안전 처리 함수
            def safe_float(value, default=None):
                """NaN 값을 안전하게 처리"""
                if pd.isna(value):
                    return default
                try:
                    return float(value)
                except:
                    return default

            # 필수 필드 검증 (낙찰하한율이 없으면 스킵)
            bid_lower_limit_rate_value = safe_float(row['낙찰하한율'])
            if bid_lower_limit_rate_value is None:
                error_count += 1
                continue  # 낙찰하한율이 없는 레코드는 예측에 사용 불가능하므로 스킵

            # 데이터 전처리
            data = {
                'bid_number': str(row['공고번호']),
                'bid_name': str(row['공고명']),
                'ordering_agency': str(row['발주기관']) if pd.notna(row['발주기관']) else None,
                'industry': str(row['업종']) if pd.notna(row['업종']) else None,
                'region': str(row['지역']) if pd.notna(row['지역']) else None,
                'estimated_price': int(row['추정가격']),
                'base_price': int(row['기초금액']),
                'a_value': safe_float(row['A값'], 0.0),
                'bid_lower_limit_rate': bid_lower_limit_rate_value,
                'yega_range': parse_yega_range(row['예가변동폭']),
                'opening_date': parse_date(row['개찰일']),
                'submission_deadline': str(row['투찰마감']) if pd.notna(row['투찰마감']) else None,
                'input_date': parse_date(row['입력일']),
                'construction_cost': safe_float(row['순공사원가'], None),
                'g2b_category': str(row['G2B물품분류']) if pd.notna(row['G2B물품분류']) else None
            }

            # UPSERT (공고번호가 있으면 업데이트, 없으면 삽입)
            result = supabase.table('bid_list').upsert(data, on_conflict='bid_number').execute()

            success_count += 1
            if (idx + 1) % 100 == 0:
                print(f"⏳ 진행 중... {idx + 1}/{len(df)} ({success_count} 성공, {error_count} 실패)")

        except Exception as e:
            error_count += 1
            print(f"❌ 오류 발생 (행 {idx + 1}): {str(e)[:100]}")
            continue

    print(f"\n✅ 업로드 완료!")
    print(f"   성공: {success_count}개")
    print(f"   실패: {error_count}개")

    return success_count, error_count

def verify_data(supabase):
    """업로드된 데이터 검증"""
    print("\n🔍 데이터 검증 중...")

    # 총 레코드 수 확인
    result = supabase.table('bid_list').select('id', count='exact').execute()
    print(f"   총 레코드 수: {result.count}개")

    # 샘플 데이터 확인
    sample = supabase.table('bid_list').select('*').limit(3).execute()
    print(f"\n📋 샘플 데이터 (최근 3개):")
    for i, record in enumerate(sample.data, 1):
        print(f"\n   {i}. {record['bid_number']} - {record['bid_name'][:50]}...")
        print(f"      추정가격: {record['estimated_price']:,}원")
        print(f"      기초금액: {record['base_price']:,}원")
        print(f"      낙찰하한율: {record['bid_lower_limit_rate']}%")
        print(f"      예가변동폭: {record['yega_range']}")

if __name__ == "__main__":
    # Excel 파일 경로
    excel_file = "/mnt/c/Users/star/claude/bid-prediction/data/raw/Bid list_merged.xlsx"

    # 업로드 실행
    success, error = upload_bid_data(excel_file)

    # 검증
    if success > 0:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        verify_data(supabase)
