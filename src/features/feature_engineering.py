#!/usr/bin/env python3
"""
낙찰하한가 예측 모델 - Feature Engineering
예가/기초(100%) 예측을 위한 피처 생성
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import warnings
import re

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    피처 엔지니어링 클래스

    주요 기능:
    1. 기본 전처리 (A값 결측치 → 0)
    2. 시간 피처 생성 (개찰일 기반)
    3. 예가변동폭 파싱 및 피처화
    4. 비율/스케일 피처
    5. 집계 피처 (Target Encoding)
    6. 상호작용 피처
    7. 범주형 인코딩
    """

    def __init__(self):
        """초기화"""
        self.target_col = '예가/기초(100%)'

        # 집계값 저장 (fit 시 계산, transform 시 사용)
        self.aggregations = {}

        # 전역 평균 (fallback용)
        self.global_mean = None

        # Frequency 저장
        self.frequencies = {}

        logger.info("FeatureEngineer 초기화 완료")

    def fit(self, df: pd.DataFrame, target_col: str = None) -> 'FeatureEngineer':
        """
        집계값 계산 (Training data에서만 호출)

        Args:
            df: Training 데이터프레임
            target_col: 타겟 컬럼명

        Returns:
            self
        """
        if target_col:
            self.target_col = target_col

        logger.info("=" * 70)
        logger.info("피처 엔지니어링 fit 시작")
        logger.info("=" * 70)

        # 전역 평균 계산
        self.global_mean = df[self.target_col].mean()
        logger.info(f"전역 평균 예가: {self.global_mean:.4f}")

        # 집계값 계산 (Target Encoding)
        self._calculate_aggregations(df)

        # Frequency 계산
        self._calculate_frequencies(df)

        logger.info("✓ fit 완료")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        피처 변환 적용

        Args:
            df: 변환할 데이터프레임

        Returns:
            변환된 데이터프레임
        """
        logger.info("=" * 70)
        logger.info("피처 변환 시작")
        logger.info("=" * 70)

        df = df.copy()

        # 1. 기본 전처리
        df = self._basic_preprocessing(df)

        # 2. 시간 피처
        df = self._create_time_features(df)

        # 3. 예가변동폭 피처
        df = self._parse_price_range(df)

        # 4. 비율/스케일 피처
        df = self._create_ratio_features(df)

        # 5. 집계 피처 (Target Encoding)
        df = self._create_aggregation_features(df)

        # 6. 상호작용 피처
        df = self._create_interaction_features(df)

        # 7. 범주형 인코딩
        df = self._encode_categorical(df)

        logger.info("=" * 70)
        logger.info("✓ 피처 변환 완료")
        logger.info(f"최종 피처 수: {len(df.columns)}")
        logger.info("=" * 70)

        return df

    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        fit + transform 한 번에 실행

        Args:
            df: 데이터프레임
            target_col: 타겟 컬럼명

        Returns:
            변환된 데이터프레임
        """
        return self.fit(df, target_col).transform(df)

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 전처리"""
        logger.info("\n1. 기본 전처리 중...")

        # A값 결측치 → 0
        if df['A값'].isnull().sum() > 0:
            n_null = df['A값'].isnull().sum()
            df['A값'] = df['A값'].fillna(0)
            logger.info(f"  ✓ A값 결측치 {n_null}개를 0으로 채움")

        # 순공사원가 제거
        if '순공사원가' in df.columns:
            df = df.drop(columns=['순공사원가'])
            logger.info(f"  ✓ 순공사원가 컬럼 제거 (68.89% 결측)")

        # A값_비율 생성 (없으면)
        if 'A값_비율' not in df.columns:
            df['A값_비율'] = df['A값'] / df['기초금액']
            logger.info(f"  ✓ A값_비율 생성 (A값 / 기초금액)")

        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간 피처 생성"""
        logger.info("\n2. 시간 피처 생성 중...")

        # 개찰일 파싱 (YY.MM.DD → datetime)
        def parse_date(date_str):
            """날짜 파싱 함수"""
            if pd.isna(date_str):
                return pd.NaT

            try:
                # "25.09.26" 형식
                parts = str(date_str).split('.')
                if len(parts) == 3:
                    year, month, day = parts
                    # 2자리 연도 → 4자리 연도
                    year = int(year)
                    if year < 50:  # 00-49 → 2000-2049
                        year += 2000
                    else:  # 50-99 → 1950-1999
                        year += 1900

                    return pd.Timestamp(year=year, month=int(month), day=int(day))
            except:
                return pd.NaT

            return pd.NaT

        df['개찰일_dt'] = df['개찰일'].apply(parse_date)

        # 시간 피처들
        df['개찰_연도'] = df['개찰일_dt'].dt.year
        df['개찰_월'] = df['개찰일_dt'].dt.month
        df['개찰_분기'] = df['개찰일_dt'].dt.quarter
        df['개찰_요일'] = df['개찰일_dt'].dt.dayofweek
        df['개찰_일'] = df['개찰일_dt'].dt.day
        df['개찰_연중일자'] = df['개찰일_dt'].dt.dayofyear

        # 월의 주차 (1-5)
        df['개찰_월주차'] = ((df['개찰_일'] - 1) // 7) + 1

        # 계절 (봄:3-5, 여름:6-8, 가을:9-11, 겨울:12-2)
        df['개찰_계절'] = df['개찰_월'].apply(lambda x:
            1 if x in [3,4,5] else
            2 if x in [6,7,8] else
            3 if x in [9,10,11] else 4
        )

        logger.info(f"  ✓ 시간 피처 8개 생성")
        logger.info(f"    - 개찰일 범위: {df['개찰일_dt'].min()} ~ {df['개찰일_dt'].max()}")

        return df

    def _parse_price_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """예가변동폭 파싱 및 피처화"""
        logger.info("\n3. 예가변동폭 피처 생성 중...")

        def parse_range(range_str):
            """예가변동폭 파싱: "-3/+3" → (min=-3, max=+3)"""
            if pd.isna(range_str):
                return pd.Series([np.nan, np.nan])

            try:
                # "-3/+3" 또는 "-2.0/2.0" 형식
                parts = str(range_str).split('/')
                if len(parts) == 2:
                    min_val = float(parts[0].replace('+', ''))
                    max_val = float(parts[1].replace('+', ''))
                    return pd.Series([min_val, max_val])
            except:
                pass

            return pd.Series([np.nan, np.nan])

        # 파싱
        df[['예가변동폭_min', '예가변동폭_max']] = df['예가변동폭'].apply(parse_range)

        # 파생 피처
        df['예가변동폭_범위'] = df['예가변동폭_max'] - df['예가변동폭_min']
        df['예가변동폭_중앙'] = (df['예가변동폭_min'] + df['예가변동폭_max']) / 2
        df['예가변동폭_절대평균'] = (abs(df['예가변동폭_min']) + abs(df['예가변동폭_max'])) / 2

        # 예가변동폭 결측치 처리 (가장 흔한 값으로)
        most_common = df['예가변동폭'].mode()[0] if len(df['예가변동폭'].mode()) > 0 else '-3/+3'
        logger.info(f"  ✓ 예가변동폭 파싱 완료")
        logger.info(f"    - 가장 흔한 범위: {most_common}")
        logger.info(f"    - 고유값: {df['예가변동폭'].nunique()}개")

        return df

    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """비율/스케일 피처 생성"""
        logger.info("\n4. 비율/스케일 피처 생성 중...")

        # 로그 변환
        df['기초금액_로그'] = np.log1p(df['기초금액'])
        df['A값_로그'] = np.log1p(df['A값'])

        # A값_비율 제곱
        df['A값_비율_제곱'] = df['A값_비율'] ** 2

        # 기초금액 구간
        bins = [0, 1e8, 5e8, 10e8, 50e8, float('inf')]
        labels = ['0-1억', '1-5억', '5-10억', '10-50억', '50억+']
        df['기초금액_구간'] = pd.cut(df['기초금액'], bins=bins, labels=labels)

        # 낙찰하한율 결측치가 있을 수 있으므로 확인
        if '낙찰하한율' in df.columns and df['낙찰하한율'].notna().sum() > 0:
            # 낙찰하한율 구간 (실제 데이터 범위: 79.99~89.75%, 평균 88.31%)
            rate_bins = [0, 85, 87.5, 88.5, 90, 100]
            rate_labels = ['<85%', '85-87.5%', '87.5-88.5%', '88.5-90%', '90%+']
            df['낙찰하한율_구간'] = pd.cut(df['낙찰하한율'], bins=rate_bins, labels=rate_labels)

        logger.info(f"  ✓ 비율/스케일 피처 6개 생성")

        return df

    def _calculate_aggregations(self, df: pd.DataFrame):
        """집계값 계산 (fit 시에만 호출)"""
        logger.info("\n집계값 계산 중 (Target Encoding)...")

        # 타겟이 있는 데이터만 사용
        df_valid = df[df[self.target_col].notna()].copy()

        # 1. 업종별 평균
        self.aggregations['업종'] = df_valid.groupby('업종')[self.target_col].mean().to_dict()
        logger.info(f"  - 업종별 평균: {len(self.aggregations['업종'])}개")

        # 2. 지역별 평균
        self.aggregations['지역'] = df_valid.groupby('지역')[self.target_col].mean().to_dict()
        logger.info(f"  - 지역별 평균: {len(self.aggregations['지역'])}개")

        # 3. 발주기관별 평균
        self.aggregations['발주기관'] = df_valid.groupby('발주기관')[self.target_col].mean().to_dict()
        logger.info(f"  - 발주기관별 평균: {len(self.aggregations['발주기관'])}개")

        # 4. 예가변동폭별 평균
        self.aggregations['예가변동폭'] = df_valid.groupby('예가변동폭')[self.target_col].mean().to_dict()
        logger.info(f"  - 예가변동폭별 평균: {len(self.aggregations['예가변동폭'])}개")

        # 5. 업종×지역 평균
        df_valid['업종_지역'] = df_valid['업종'] + '_' + df_valid['지역']
        self.aggregations['업종_지역'] = df_valid.groupby('업종_지역')[self.target_col].mean().to_dict()
        logger.info(f"  - 업종×지역 평균: {len(self.aggregations['업종_지역'])}개")

    def _calculate_frequencies(self, df: pd.DataFrame):
        """빈도 계산 (fit 시에만 호출)"""
        logger.info("\n빈도 계산 중 (Frequency Encoding)...")

        # 업종 빈도
        self.frequencies['업종'] = df['업종'].value_counts().to_dict()
        logger.info(f"  - 업종 빈도: {len(self.frequencies['업종'])}개")

        # 지역 빈도
        self.frequencies['지역'] = df['지역'].value_counts().to_dict()
        logger.info(f"  - 지역 빈도: {len(self.frequencies['지역'])}개")

        # 발주기관 빈도
        self.frequencies['발주기관'] = df['발주기관'].value_counts().to_dict()
        logger.info(f"  - 발주기관 빈도: {len(self.frequencies['발주기관'])}개")

    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """집계 피처 생성 (transform 시 호출)"""
        logger.info("\n5. 집계 피처 생성 중 (Target Encoding)...")

        # 업종×지역 조합 (먼저 생성)
        df['업종_지역'] = df['업종'] + '_' + df['지역']

        # Target Encoding
        df['업종_평균예가'] = df['업종'].map(self.aggregations.get('업종', {})).fillna(self.global_mean)
        df['지역_평균예가'] = df['지역'].map(self.aggregations.get('지역', {})).fillna(self.global_mean)
        df['발주기관_평균예가'] = df['발주기관'].map(self.aggregations.get('발주기관', {})).fillna(self.global_mean)
        df['예가변동폭_평균예가'] = df['예가변동폭'].map(self.aggregations.get('예가변동폭', {})).fillna(self.global_mean)
        df['업종지역_평균예가'] = df['업종_지역'].map(self.aggregations.get('업종_지역', {})).fillna(self.global_mean)

        logger.info(f"  ✓ 집계 피처 5개 생성")

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """상호작용 피처 생성"""
        logger.info("\n6. 상호작용 피처 생성 중...")

        # 업종_지역은 이미 생성됨

        # 기초금액구간 × 낙찰하한율구간
        if '기초금액_구간' in df.columns and '낙찰하한율_구간' in df.columns:
            df['기초구간_낙찰구간'] = df['기초금액_구간'].astype(str) + '_' + df['낙찰하한율_구간'].astype(str)
            logger.info(f"  ✓ 기초구간_낙찰구간 생성")

        # 기초금액_로그 × 예가변동폭_범위
        df['기초로그_변동범위'] = df['기초금액_로그'] * df['예가변동폭_범위'].fillna(0)

        logger.info(f"  ✓ 상호작용 피처 생성 완료")

        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """범주형 인코딩"""
        logger.info("\n7. 범주형 인코딩 중...")

        # Frequency Encoding
        df['업종_빈도'] = df['업종'].map(self.frequencies.get('업종', {})).fillna(0)
        df['지역_빈도'] = df['지역'].map(self.frequencies.get('지역', {})).fillna(0)
        df['발주기관_빈도'] = df['발주기관'].map(self.frequencies.get('발주기관', {})).fillna(0)

        logger.info(f"  ✓ Frequency Encoding 3개 생성")

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """생성된 피처명 리스트 반환"""
        # 원본에 없던 새로운 피처들
        new_features = [
            # 시간
            '개찰_연도', '개찰_월', '개찰_분기', '개찰_요일', '개찰_일',
            '개찰_연중일자', '개찰_월주차', '개찰_계절',
            # 예가변동폭
            '예가변동폭_min', '예가변동폭_max', '예가변동폭_범위',
            '예가변동폭_중앙', '예가변동폭_절대평균',
            # 비율/스케일
            '기초금액_로그', 'A값_로그', 'A값_비율_제곱',
            # 집계
            '업종_평균예가', '지역_평균예가', '발주기관_평균예가',
            '예가변동폭_평균예가', '업종지역_평균예가',
            # 빈도
            '업종_빈도', '지역_빈도', '발주기관_빈도',
            # 상호작용
            '기초로그_변동범위'
        ]

        return [f for f in new_features if f in df.columns]


def main():
    """테스트 실행"""
    print("=" * 70)
    print("피처 엔지니어링 테스트")
    print("=" * 70)

    # 데이터 로드
    data_path = Path("data/processed/List_of_Successful_Bidders_processed_20251107_232514.xlsx")
    df = pd.read_excel(data_path)

    print(f"\n원본 데이터: {df.shape}")

    # 피처 엔지니어링
    fe = FeatureEngineer()
    df_transformed = fe.fit_transform(df)

    print(f"변환 후 데이터: {df_transformed.shape}")

    # 생성된 피처 확인
    new_features = fe.get_feature_names(df_transformed)
    print(f"\n생성된 피처 ({len(new_features)}개):")
    for i, feat in enumerate(new_features, 1):
        print(f"  {i:2d}. {feat}")

    # 결과 저장
    output_path = Path("data/processed/features_engineered.xlsx")
    df_transformed.to_excel(output_path, index=False)
    print(f"\n✓ 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
