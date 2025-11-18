"""
새로운 학습 데이터 검증 스크립트
List of Successful Bidders_merged.xlsx 데이터 구조 확인 및 전처리
"""
import pandas as pd
import numpy as np
from pathlib import Path

def validate_training_data():
    """새로운 학습 데이터 검증"""
    print("=" * 70)
    print("새로운 학습 데이터 검증")
    print("=" * 70)

    # 데이터 로드
    data_path = Path("/mnt/c/Users/star/claude/bid-prediction/data/raw/List of Successful Bidders_merged.xlsx")
    print(f"\n📂 데이터 로드: {data_path.name}")

    try:
        df = pd.read_excel(data_path)
        print(f"✓ 로드 성공: {df.shape[0]:,}행 × {df.shape[1]}열")
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        return None

    # 컬럼 확인
    print(f"\n📋 컬럼 목록 ({len(df.columns)}개):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    # 필수 피처 확인
    required_features = [
        '예정가격',
        '기초금액',
        'A값',
        '낙찰하한율',
        '예가변동폭'
    ]

    # 타겟 변수
    target_col = '예가/기초(100%)'

    print(f"\n🔍 필수 피처 존재 여부:")
    missing_features = []
    for feature in required_features:
        exists = feature in df.columns
        status = "✓" if exists else "❌"
        print(f"  {status} {feature}")
        if not exists:
            missing_features.append(feature)

    # 타겟 변수 확인
    target_exists = target_col in df.columns
    print(f"\n🎯 타겟 변수:")
    print(f"  {'✓' if target_exists else '❌'} {target_col}")

    # 데이터 타입 확인
    print(f"\n📊 데이터 타입:")
    print(df.dtypes)

    # 결측치 확인
    print(f"\n🔍 결측치 현황:")
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)

    missing_df = pd.DataFrame({
        '컬럼': missing_count.index,
        '결측 개수': missing_count.values,
        '결측 비율(%)': missing_pct.values
    })
    missing_df = missing_df[missing_df['결측 개수'] > 0].sort_values('결측 개수', ascending=False)

    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("  ✓ 결측치 없음")

    # 기본 통계
    if target_exists:
        print(f"\n📈 타겟 변수 통계 ({target_col}):")
        print(df[target_col].describe())

    # 사용 가능한 레코드 수 확인
    if target_exists and not missing_features:
        # 타겟과 필수 피처 모두 있는 레코드
        valid_mask = df[target_col].notna()
        for feature in required_features:
            if feature in df.columns:
                valid_mask &= df[feature].notna()

        valid_count = valid_mask.sum()
        print(f"\n✅ 사용 가능한 레코드:")
        print(f"  전체: {len(df):,}개")
        print(f"  유효: {valid_count:,}개 ({valid_count/len(df)*100:.1f}%)")
        print(f"  제외: {len(df)-valid_count:,}개")

        # 80/20 분할 시 예상 크기
        train_size = int(valid_count * 0.8)
        val_size = valid_count - train_size
        print(f"\n📊 80/20 분할 예상:")
        print(f"  학습 데이터: {train_size:,}개")
        print(f"  검증 데이터: {val_size:,}개")

    # 샘플 데이터 출력
    print(f"\n📋 샘플 데이터 (상위 5개):")
    if not missing_features and target_exists:
        display_cols = required_features + [target_col]
        available_cols = [col for col in display_cols if col in df.columns]
        print(df[available_cols].head().to_string(index=False))
    else:
        print(df.head().to_string())

    # 요약
    print("\n" + "=" * 70)
    print("검증 결과 요약")
    print("=" * 70)

    if missing_features:
        print(f"❌ 누락된 필수 피처: {', '.join(missing_features)}")
        print("   → 데이터 전처리 필요")
    elif not target_exists:
        print(f"❌ 타겟 변수 누락: {target_col}")
        print("   → 타겟 변수 생성 필요")
    else:
        print("✅ 모든 필수 피처 존재")
        print("✅ 타겟 변수 존재")
        print(f"✅ 사용 가능한 레코드: {valid_count:,}개")
        print("\n→ 모델 재학습 준비 완료!")

    print("=" * 70)

    return df

if __name__ == "__main__":
    df = validate_training_data()
