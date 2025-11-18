"""
학습 데이터 전처리 스크립트
List of Successful Bidders_merged.xlsx → features_engineered.xlsx

누락된 피처 생성:
1. 낙찰하한율 = (낙찰하한가 / 기초금액) * 100
2. 예가변동폭_범위 = Supabase bid_list에서 조회 (없으면 기본값 2.0)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
from supabase import create_client

# 환경 변수 로드
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


def clean_numeric_string(value):
    """숫자가 포함된 문자열을 float로 변환"""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    # 문자열에서 쉼표 제거
    value_str = str(value).replace(',', '').strip()

    try:
        return float(value_str)
    except:
        return np.nan


def preprocess_training_data():
    """학습 데이터 전처리"""
    print("=" * 70)
    print("학습 데이터 전처리")
    print("=" * 70)

    # 1. 데이터 로드
    input_path = Path("/mnt/c/Users/star/claude/bid-prediction/data/raw/List of Successful Bidders_merged.xlsx")
    output_path = Path("/mnt/c/Users/star/claude/bid-prediction/data/processed/features_engineered.xlsx")

    print(f"\n📂 입력 파일: {input_path.name}")
    df = pd.read_excel(input_path)
    print(f"✓ 로드 완료: {df.shape[0]:,}행 × {df.shape[1]}열")

    # 2. 낙찰하한가 숫자 변환 (문자열 → float)
    print(f"\n🔧 낙찰하한가 전처리 중...")
    df['낙찰하한가_clean'] = df['낙찰하한가'].apply(clean_numeric_string)

    # 3. 낙찰하한율 계산
    print(f"🔧 낙찰하한율 계산 중...")
    df['낙찰하한율'] = (df['낙찰하한가_clean'] / df['기초금액']) * 100

    # 계산 결과 검증
    valid_rate = df['낙찰하한율'].notna().sum()
    print(f"  ✓ 낙찰하한율 계산 완료: {valid_rate:,}/{len(df):,}개")

    # 4. Supabase에서 예가변동폭 조회
    print(f"\n🔧 예가변동폭 조회 중...")

    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

            # bid_list 테이블에서 bid_number와 yega_range 가져오기
            response = supabase.table('bid_list').select('bid_number,yega_range').execute()

            if response.data:
                # DataFrame으로 변환
                yega_df = pd.DataFrame(response.data)
                print(f"  ✓ Supabase에서 {len(yega_df):,}개 레코드 조회")

                # 공고번호 기준으로 병합
                df = df.merge(
                    yega_df,
                    left_on='공고번호',
                    right_on='bid_number',
                    how='left'
                )

                # 예가변동폭_범위 컬럼 생성 (yega_range 값 사용, 없으면 기본값 2.0)
                df['예가변동폭_범위'] = df['yega_range'].fillna(2.0)

                matched = df['yega_range'].notna().sum()
                print(f"  ✓ 매칭 성공: {matched:,}개 (매칭 실패: {len(df)-matched:,}개 → 기본값 2.0 사용)")

                # 불필요한 컬럼 제거
                df.drop(columns=['bid_number', 'yega_range'], inplace=True, errors='ignore')

            else:
                print(f"  ⚠️ Supabase 데이터 없음, 기본값 2.0 사용")
                df['예가변동폭_범위'] = 2.0

        except Exception as e:
            print(f"  ⚠️ Supabase 조회 실패: {e}")
            print(f"  → 모든 레코드에 기본값 2.0 사용")
            df['예가변동폭_범위'] = 2.0
    else:
        print(f"  ⚠️ Supabase 설정 없음, 기본값 2.0 사용")
        df['예가변동폭_범위'] = 2.0

    # 5. 필수 컬럼 확인
    required_features = ['예정가격', '기초금액', 'A값', '낙찰하한율', '예가변동폭_범위']
    target_col = '예가/기초(100%)'

    print(f"\n🔍 필수 피처 확인:")
    for feature in required_features:
        exists = feature in df.columns
        print(f"  {'✓' if exists else '❌'} {feature}")

    print(f"\n🎯 타겟 변수:")
    print(f"  {'✓' if target_col in df.columns else '❌'} {target_col}")

    # 6. 결측치 제거
    print(f"\n🧹 결측치 제거 중...")
    print(f"  원본: {len(df):,}개")

    df_clean = df.copy()

    # 타겟 변수 결측치 제거
    df_clean = df_clean.dropna(subset=[target_col])
    print(f"  타겟 결측 제거 후: {len(df_clean):,}개")

    # 필수 피처 결측치 제거
    for feature in required_features:
        if feature in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean.dropna(subset=[feature])
            if before > len(df_clean):
                print(f"  {feature} 결측 제거: {before-len(df_clean):,}개")

    print(f"  최종: {len(df_clean):,}개")

    # 7. 피처 선택 및 저장
    save_columns = required_features + [target_col] + [
        '번호', '공고명', '공고번호', '발주기관', '추정가격',
        '1순위업체', '개찰일', '입력일', '업종', '지역'
    ]

    # 존재하는 컬럼만 선택
    available_columns = [col for col in save_columns if col in df_clean.columns]
    df_save = df_clean[available_columns].copy()

    # 8. 파일 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_save.to_excel(output_path, index=False)

    print(f"\n💾 저장 완료:")
    print(f"  경로: {output_path}")
    print(f"  크기: {len(df_save):,}행 × {len(df_save.columns)}열")

    # 9. 통계 정보
    print(f"\n📊 피처 통계:")
    print(df_save[required_features].describe())

    print(f"\n📈 타겟 변수 통계 ({target_col}):")
    print(df_save[target_col].describe())

    # 10. 요약
    print("\n" + "=" * 70)
    print("전처리 결과 요약")
    print("=" * 70)
    print(f"✅ 원본 데이터: {len(df):,}개")
    print(f"✅ 유효 데이터: {len(df_clean):,}개 ({len(df_clean)/len(df)*100:.1f}%)")
    print(f"✅ 필수 피처: {len(required_features)}개")
    print(f"✅ 저장 경로: {output_path}")

    # 80/20 분할 예상
    train_size = int(len(df_clean) * 0.8)
    val_size = len(df_clean) - train_size
    print(f"\n📊 80/20 분할 예상:")
    print(f"  학습: {train_size:,}개")
    print(f"  검증: {val_size:,}개")

    print("=" * 70)

    return df_save


if __name__ == "__main__":
    df = preprocess_training_data()
