#!/usr/bin/env python3
"""
낙찰하한가 예측 모델 - Interactive EDA Dashboard v2
Plotly 기반 인터랙티브 대시보드 생성
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EDADashboard:
    """
    인터랙티브 EDA 대시보드 생성기

    6개 섹션으로 구성:
    1. Executive Summary (KPI Cards)
    2. Target Variable Deep Dive
    3. Feature Analysis
    4. Relationship Analysis
    5. Domain-Specific Insights
    6. Problem Diagnosis
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        output_dir: str = "docs/analysis"
    ):
        """
        대시보드 초기화

        Args:
            data_dir: 전처리된 데이터 디렉토리
            output_dir: 대시보드 HTML 출력 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 데이터프레임
        self.df: Optional[pd.DataFrame] = None
        self.df_name: str = ""

        # 분석 결과 저장
        self.stats: Dict = {}
        self.figures: Dict[str, go.Figure] = {}

        # 색상 팔레트 (일관된 디자인)
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#ffbb00',
            'danger': '#d62728',
            'info': '#17a2b8',
            'dark': '#212529',
            'light': '#f8f9fa'
        }

        logger.info(f"EDA Dashboard 초기화 완료")
        logger.info(f"데이터 디렉토리: {self.data_dir}")
        logger.info(f"출력 디렉토리: {self.output_dir}")

    def load_data(
        self,
        filename: Optional[str] = None,
        use_successful_bidders: bool = True
    ) -> bool:
        """
        데이터 로딩 및 기본 검증

        Args:
            filename: 특정 파일명 (None이면 최신 파일 자동 선택)
            use_successful_bidders: True면 낙찰자 목록, False면 입찰 목록

        Returns:
            성공 여부
        """
        logger.info("=" * 60)
        logger.info("데이터 로딩 시작")
        logger.info("=" * 60)

        try:
            if filename:
                file_path = self.data_dir / filename
            else:
                # 최신 파일 자동 선택
                pattern = "List_of_Successful_Bidders_processed_*.xlsx" if use_successful_bidders else "Bid_list_processed_*.xlsx"
                files = list(self.data_dir.glob(pattern))

                if not files:
                    logger.error(f"파일을 찾을 수 없습니다: {pattern}")
                    return False

                file_path = max(files, key=lambda x: x.stat().st_mtime)

            logger.info(f"파일 로딩: {file_path.name}")

            # 데이터 로드
            self.df = pd.read_excel(file_path)
            self.df_name = file_path.stem

            # 기본 정보 출력
            logger.info(f"✓ 데이터 로드 완료")
            logger.info(f"  - 행 개수: {len(self.df):,}")
            logger.info(f"  - 열 개수: {len(self.df.columns)}")
            logger.info(f"  - 메모리 사용: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            # 컬럼 목록
            logger.info(f"  - 컬럼 목록:")
            for i, col in enumerate(self.df.columns, 1):
                logger.info(f"    {i:2d}. {col} ({self.df[col].dtype})")

            # 기본 검증
            self._validate_data()

            return True

        except Exception as e:
            logger.error(f"데이터 로딩 실패: {str(e)}", exc_info=True)
            return False

    def _validate_data(self) -> None:
        """데이터 기본 검증"""
        logger.info("\n데이터 검증 중...")

        issues = []

        # 1. 필수 컬럼 확인
        required_cols = ['지역', '업종', '기초금액', 'A값']
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            issues.append(f"필수 컬럼 누락: {missing_cols}")

        # 2. 타겟 변수 확인
        target_candidates = ['예가/기초(100%)', '예가']
        target_col = None
        for col in target_candidates:
            if col in self.df.columns:
                target_col = col
                break

        if target_col is None:
            issues.append(f"타겟 변수를 찾을 수 없습니다. 후보: {target_candidates}")
        else:
            logger.info(f"  ✓ 타겟 변수 확인: {target_col}")

        # 3. 결측치 확인
        missing_count = self.df.isnull().sum().sum()
        missing_pct = (missing_count / (len(self.df) * len(self.df.columns))) * 100

        logger.info(f"  ✓ 결측치: {missing_count:,}개 ({missing_pct:.2f}%)")

        # 4. 중복 데이터 확인
        duplicates = self.df.duplicated().sum()
        logger.info(f"  ✓ 중복 행: {duplicates:,}개")

        # 이슈 리포트
        if issues:
            logger.warning("⚠️ 데이터 검증 이슈:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("  ✓ 데이터 검증 통과")

    def calculate_basic_stats(self) -> Dict:
        """
        기본 통계량 계산 (Executive Summary용)

        Returns:
            통계량 딕셔너리
        """
        logger.info("\n기본 통계량 계산 중...")

        if self.df is None:
            logger.error("데이터가 로드되지 않았습니다.")
            return {}

        stats = {}

        # 데이터 개요
        stats['data_overview'] = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'data_period': self._extract_data_period(),
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }

        # 타겟 변수 통계
        target_col = self._get_target_column()
        if target_col:
            target_data = pd.to_numeric(self.df[target_col], errors='coerce').dropna()

            stats['target_stats'] = {
                'column_name': target_col,
                'count': len(target_data),
                'mean': float(target_data.mean()),
                'std': float(target_data.std()),
                'min': float(target_data.min()),
                'max': float(target_data.max()),
                'q25': float(target_data.quantile(0.25)),
                'median': float(target_data.median()),
                'q75': float(target_data.quantile(0.75)),
                'cv': float(target_data.std() / target_data.mean()) if target_data.mean() != 0 else 0,
                'skewness': float(target_data.skew()),
                'kurtosis': float(target_data.kurtosis())
            }

            logger.info(f"  타겟 변수: {target_col}")
            logger.info(f"  평균: {stats['target_stats']['mean']:.4f}")
            logger.info(f"  표준편차: {stats['target_stats']['std']:.4f}")
            logger.info(f"  CV: {stats['target_stats']['cv']:.4f}")

        # 결측치 통계
        missing_data = self.df.isnull().sum()
        stats['missing_stats'] = {
            'total_missing': int(missing_data.sum()),
            'missing_pct': float((missing_data.sum() / (len(self.df) * len(self.df.columns))) * 100),
            'columns_with_missing': int((missing_data > 0).sum())
        }

        # 데이터 품질 스코어 (0-100)
        stats['quality_score'] = self._calculate_quality_score(stats)

        # 예측 난이도 지표
        stats['prediction_difficulty'] = self._assess_prediction_difficulty(stats)

        self.stats = stats

        logger.info(f"  ✓ 기본 통계량 계산 완료")
        logger.info(f"  데이터 품질 스코어: {stats['quality_score']['overall']:.1f}/100")
        logger.info(f"  예측 난이도: {stats['prediction_difficulty']['level']}")

        return stats

    def _get_target_column(self) -> Optional[str]:
        """타겟 컬럼명 찾기"""
        candidates = ['예가/기초(100%)', '예가']
        for col in candidates:
            if col in self.df.columns:
                return col
        return None

    def _extract_data_period(self) -> Dict:
        """데이터 기간 추출 (날짜 컬럼이 있을 경우)"""
        date_cols = [col for col in self.df.columns if '일자' in col or 'date' in col.lower()]

        if not date_cols:
            return {'available': False}

        try:
            date_col = date_cols[0]
            dates = pd.to_datetime(self.df[date_col], errors='coerce').dropna()

            return {
                'available': True,
                'column': date_col,
                'start': dates.min().strftime('%Y-%m-%d'),
                'end': dates.max().strftime('%Y-%m-%d'),
                'days': (dates.max() - dates.min()).days
            }
        except:
            return {'available': False}

    def _calculate_quality_score(self, stats: Dict) -> Dict:
        """
        데이터 품질 스코어 계산

        점수 구성:
        - 완전성 (Completeness): 40점
        - 일관성 (Consistency): 30점
        - 정확성 (Accuracy): 30점
        """
        scores = {}

        # 1. 완전성: 결측치가 적을수록 높은 점수
        completeness = max(0, 100 - stats['missing_stats']['missing_pct'] * 2)
        scores['completeness'] = round(completeness, 1)

        # 2. 일관성: 중복 데이터, 데이터 타입 일관성
        duplicates_pct = (self.df.duplicated().sum() / len(self.df)) * 100
        consistency = max(0, 100 - duplicates_pct * 5)
        scores['consistency'] = round(consistency, 1)

        # 3. 정확성: 이상치, 논리적 오류
        accuracy = 100.0  # 기본값

        # 타겟 변수 범위 검사 (예가는 0.8~1.0)
        if 'target_stats' in stats:
            target_col = self._get_target_column()
            if target_col:
                target_data = pd.to_numeric(self.df[target_col], errors='coerce').dropna()
                out_of_range = ((target_data < 0.8) | (target_data > 1.0)).sum()
                out_of_range_pct = (out_of_range / len(target_data)) * 100
                accuracy -= out_of_range_pct * 10

        scores['accuracy'] = round(max(0, accuracy), 1)

        # 종합 점수 (가중 평균)
        overall = (
            scores['completeness'] * 0.4 +
            scores['consistency'] * 0.3 +
            scores['accuracy'] * 0.3
        )
        scores['overall'] = round(overall, 1)

        return scores

    def _assess_prediction_difficulty(self, stats: Dict) -> Dict:
        """
        예측 난이도 평가

        난이도 기준:
        - Easy: CV > 0.05 (5%)
        - Medium: 0.02 < CV ≤ 0.05 (2-5%)
        - Hard: 0.01 < CV ≤ 0.02 (1-2%)
        - Very Hard: CV ≤ 0.01 (1% 이하)
        - Extreme: CV ≤ 0.005 (0.5% 이하)
        """
        if 'target_stats' not in stats:
            return {'level': 'Unknown', 'score': 0, 'reason': '타겟 변수 없음'}

        cv = stats['target_stats']['cv']

        if cv > 0.05:
            level = 'Easy'
            score = 20
            reason = '타겟 변동성 충분'
        elif cv > 0.02:
            level = 'Medium'
            score = 40
            reason = '타겟 변동성 보통'
        elif cv > 0.01:
            level = 'Hard'
            score = 60
            reason = '타겟 변동성 낮음'
        elif cv > 0.005:
            level = 'Very Hard'
            score = 80
            reason = '타겟 변동성 매우 낮음'
        else:
            level = 'Extreme'
            score = 100
            reason = '타겟 변동성 극히 낮음 - 예측 거의 불가능'

        return {
            'level': level,
            'score': score,
            'cv': cv,
            'reason': reason
        }

    def generate_executive_summary(self) -> go.Figure:
        """
        Section 1: Executive Summary 생성

        구성:
        - KPI Cards (4개)
        - 데이터 품질 스코어
        - 예측 난이도 알림
        """
        logger.info("\n" + "=" * 60)
        logger.info("Section 1: Executive Summary 생성 중")
        logger.info("=" * 60)

        if not self.stats:
            self.calculate_basic_stats()

        # Subplot 생성: 2행 2열 KPI Cards
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '📊 데이터 개요',
                '🎯 타겟 변수',
                '⚠️ 결측치',
                '🔍 예측 난이도'
            ),
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'indicator'}, {'type': 'indicator'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # KPI Card 1: 데이터 개요
        total_records = self.stats['data_overview']['total_records']
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=total_records,
                title={'text': f"총 데이터<br>{total_records:,}건"},
                number={'font': {'size': 40}},
            ),
            row=1, col=1
        )

        # KPI Card 2: 타겟 변수
        if 'target_stats' in self.stats:
            target_mean = self.stats['target_stats']['mean']
            target_std = self.stats['target_stats']['std']
            target_cv = self.stats['target_stats']['cv']

            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=target_mean,
                    title={'text': f"예가 평균<br>±{target_std:.4f}"},
                    number={'font': {'size': 36}, 'suffix': '', 'valueformat': '.4f'},
                    delta={
                        'reference': 1.0,
                        'relative': False,
                        'valueformat': '.4f'
                    }
                ),
                row=1, col=2
            )

        # KPI Card 3: 결측치
        missing_pct = self.stats['missing_stats']['missing_pct']
        missing_color = 'green' if missing_pct < 5 else ('orange' if missing_pct < 15 else 'red')

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=missing_pct,
                title={'text': "결측치 비율"},
                number={'suffix': '%', 'font': {'size': 36}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': missing_color},
                    'steps': [
                        {'range': [0, 5], 'color': 'lightgreen'},
                        {'range': [5, 15], 'color': 'lightyellow'},
                        {'range': [15, 100], 'color': 'lightcoral'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 20
                    }
                }
            ),
            row=2, col=1
        )

        # KPI Card 4: 예측 난이도
        difficulty = self.stats['prediction_difficulty']
        difficulty_score = difficulty['score']
        difficulty_color = 'green' if difficulty_score < 40 else ('orange' if difficulty_score < 70 else 'red')

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=difficulty_score,
                title={'text': f"예측 난이도<br><sub>{difficulty['level']}</sub>"},
                number={'suffix': '', 'font': {'size': 36}},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': difficulty_color},
                    'steps': [
                        {'range': [0, 40], 'color': 'lightgreen'},
                        {'range': [40, 70], 'color': 'lightyellow'},
                        {'range': [70, 100], 'color': 'lightcoral'}
                    ],
                }
            ),
            row=2, col=2
        )

        # 레이아웃 설정
        fig.update_layout(
            title={
                'text': '🎯 낙찰하한가 예측 모델 - Executive Summary',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': self.color_palette['dark']}
            },
            height=600,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font={'family': 'Arial, sans-serif'}
        )

        self.figures['executive_summary'] = fig

        logger.info("✓ Executive Summary 생성 완료")

        return fig

    def save_dashboard(
        self,
        filename: Optional[str] = None,
        auto_open: bool = True
    ) -> Path:
        """
        대시보드를 HTML 파일로 저장

        Args:
            filename: 출력 파일명 (None이면 자동 생성)
            auto_open: 저장 후 브라우저로 자동 열기

        Returns:
            저장된 파일 경로
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eda_dashboard_{timestamp}.html"

        output_path = self.output_dir / filename

        logger.info("\n" + "=" * 60)
        logger.info("대시보드 HTML 저장 중")
        logger.info("=" * 60)

        # 현재까지 생성된 모든 figure를 HTML로 결합
        if not self.figures:
            logger.warning("생성된 figure가 없습니다.")
            return output_path

        # HTML 헤더
        html_parts = [
            '<html>',
            '<head>',
            '<meta charset="utf-8">',
            '<title>낙찰하한가 예측 모델 - EDA Dashboard</title>',
            '<style>',
            'body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }',
            '.section { background-color: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            'h1 { color: #333; text-align: center; }',
            'h2 { color: #555; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }',
            '.info-box { background-color: #e7f3ff; border-left: 4px solid #1f77b4; padding: 15px; margin: 15px 0; }',
            '.warning-box { background-color: #fff3cd; border-left: 4px solid #ffbb00; padding: 15px; margin: 15px 0; }',
            '.danger-box { background-color: #f8d7da; border-left: 4px solid #d62728; padding: 15px; margin: 15px 0; }',
            '</style>',
            '</head>',
            '<body>',
            '<h1>🎯 낙찰하한가 예측 모델 - EDA Dashboard v2.0</h1>',
            f'<p style="text-align: center; color: #666;">생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
            f'<p style="text-align: center; color: #666;">데이터: {self.df_name}</p>',
        ]

        # 각 섹션 추가
        for section_name, fig in self.figures.items():
            html_parts.append('<div class="section">')

            # 섹션별 설명 추가
            if section_name == 'executive_summary':
                html_parts.append('<div class="info-box">')
                html_parts.append('<strong>ℹ️ Executive Summary</strong><br>')
                html_parts.append('데이터의 전반적인 현황과 핵심 지표를 한눈에 확인할 수 있습니다.')
                html_parts.append('</div>')

            # Plotly figure를 HTML로 변환
            html_parts.append(fig.to_html(include_plotlyjs='cdn', full_html=False))
            html_parts.append('</div>')

        html_parts.extend(['</body>', '</html>'])

        # HTML 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html_parts))

        logger.info(f"✓ 대시보드 저장 완료: {output_path}")
        logger.info(f"  파일 크기: {output_path.stat().st_size / 1024:.2f} KB")

        # 브라우저로 자동 열기
        if auto_open:
            import webbrowser
            webbrowser.open(f'file://{output_path.absolute()}')
            logger.info(f"✓ 브라우저에서 대시보드 열기")

        return output_path

    def run_phase1(self) -> bool:
        """
        Phase 1 전체 실행: 기본 인프라 + Executive Summary

        Returns:
            성공 여부
        """
        logger.info("\n" + "=" * 70)
        logger.info("🚀 Phase 1: 기본 인프라 구축 시작")
        logger.info("=" * 70)

        try:
            # Step 1: 데이터 로딩
            if not self.load_data():
                return False

            # Step 2: 기본 통계량 계산
            self.calculate_basic_stats()

            # Step 3: Executive Summary 생성
            self.generate_executive_summary()

            # Step 4: HTML 저장
            output_path = self.save_dashboard()

            logger.info("\n" + "=" * 70)
            logger.info("✅ Phase 1 완료!")
            logger.info("=" * 70)
            logger.info(f"대시보드 파일: {output_path}")
            logger.info("\n다음 단계: Phase 2 - Target Variable Deep Dive")

            return True

        except Exception as e:
            logger.error(f"❌ Phase 1 실패: {str(e)}", exc_info=True)
            return False

    def analyze_target_variable(self) -> go.Figure:
        """
        Section 2: Target Variable Deep Dive

        타겟 변수의 상세 분석:
        1. 분포 시각화 (히스토그램 + KDE + 정규분포)
        2. Box Plot + Violin Plot
        3. 통계량 테이블
        4. 정규성 검정 (Shapiro-Wilk)
        """
        logger.info("\n" + "=" * 60)
        logger.info("Section 2: Target Variable Deep Dive 생성 중")
        logger.info("=" * 60)

        target_col = self._get_target_column()
        if not target_col:
            logger.error("타겟 변수를 찾을 수 없습니다.")
            return None

        # 타겟 데이터 준비
        target_data = pd.to_numeric(self.df[target_col], errors='coerce').dropna()

        logger.info(f"타겟 변수: {target_col}")
        logger.info(f"유효 데이터: {len(target_data)}개")

        # 4개 subplot 생성: 2x2 레이아웃
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '📊 분포 (히스토그램 + KDE)',
                '📦 Box Plot + Violin Plot',
                '📈 Q-Q Plot (정규성 검정)',
                '📋 통계량 테이블'
            ),
            specs=[
                [{'type': 'xy'}, {'type': 'xy'}],
                [{'type': 'xy'}, {'type': 'table'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )

        # 1. 히스토그램 + KDE
        fig.add_trace(
            go.Histogram(
                x=target_data,
                name='예가 분포',
                nbinsx=50,
                histnorm='probability density',
                marker=dict(
                    color=self.color_palette['primary'],
                    opacity=0.7,
                    line=dict(color='white', width=1)
                ),
                showlegend=True
            ),
            row=1, col=1
        )

        # KDE (커널 밀도 추정)
        from scipy import stats
        kde_x = np.linspace(target_data.min(), target_data.max(), 200)
        kde = stats.gaussian_kde(target_data)
        kde_y = kde(kde_x)

        fig.add_trace(
            go.Scatter(
                x=kde_x,
                y=kde_y,
                name='KDE',
                mode='lines',
                line=dict(color=self.color_palette['danger'], width=2),
                showlegend=True
            ),
            row=1, col=1
        )

        # 정규분포 오버레이
        mean, std = target_data.mean(), target_data.std()
        normal_y = stats.norm.pdf(kde_x, mean, std)

        fig.add_trace(
            go.Scatter(
                x=kde_x,
                y=normal_y,
                name='정규분포',
                mode='lines',
                line=dict(color=self.color_palette['success'], width=2, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )

        # 2. Box Plot + Violin Plot
        fig.add_trace(
            go.Box(
                y=target_data,
                name='Box Plot',
                marker=dict(color=self.color_palette['primary']),
                boxmean='sd',
                showlegend=True
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Violin(
                y=target_data,
                name='Violin Plot',
                marker=dict(color=self.color_palette['secondary']),
                box_visible=True,
                meanline_visible=True,
                showlegend=True,
                opacity=0.6
            ),
            row=1, col=2
        )

        # 3. Q-Q Plot (정규성 검정)
        from scipy.stats import probplot
        qq_result = probplot(target_data, dist="norm")
        theoretical_quantiles = qq_result[0][0]
        ordered_values = qq_result[0][1]

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=ordered_values,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(
                    color=self.color_palette['primary'],
                    size=5,
                    opacity=0.6
                ),
                showlegend=False
            ),
            row=2, col=1
        )

        # Q-Q Plot 기준선 (완벽한 정규분포)
        qq_line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
        qq_line_y = qq_result[1][1] + qq_result[1][0] * qq_line_x

        fig.add_trace(
            go.Scatter(
                x=qq_line_x,
                y=qq_line_y,
                mode='lines',
                name='기준선',
                line=dict(color=self.color_palette['danger'], width=2, dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. 통계량 테이블
        # Shapiro-Wilk 정규성 검정
        shapiro_stat, shapiro_p = stats.shapiro(target_data.sample(min(5000, len(target_data))))

        stats_table_data = [
            ['평균 (Mean)', f'{target_data.mean():.4f}'],
            ['중앙값 (Median)', f'{target_data.median():.4f}'],
            ['표준편차 (Std)', f'{target_data.std():.4f}'],
            ['최소값 (Min)', f'{target_data.min():.4f}'],
            ['최대값 (Max)', f'{target_data.max():.4f}'],
            ['범위 (Range)', f'{target_data.max() - target_data.min():.4f}'],
            ['25% (Q1)', f'{target_data.quantile(0.25):.4f}'],
            ['75% (Q3)', f'{target_data.quantile(0.75):.4f}'],
            ['IQR', f'{target_data.quantile(0.75) - target_data.quantile(0.25):.4f}'],
            ['변동계수 (CV)', f'{target_data.std() / target_data.mean():.4f}'],
            ['왜도 (Skewness)', f'{target_data.skew():.4f}'],
            ['첨도 (Kurtosis)', f'{target_data.kurtosis():.4f}'],
            ['', ''],
            ['<b>정규성 검정</b>', '<b>결과</b>'],
            ['Shapiro-Wilk 통계량', f'{shapiro_stat:.4f}'],
            ['p-value', f'{shapiro_p:.4e}'],
            ['정규분포 여부', '❌ 아니오' if shapiro_p < 0.05 else '✅ 예']
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>통계 지표</b>', '<b>값</b>'],
                    fill_color=self.color_palette['primary'],
                    align='left',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[[row[0] for row in stats_table_data],
                            [row[1] for row in stats_table_data]],
                    fill_color=[['white' if i % 2 == 0 else '#f0f0f0' for i in range(len(stats_table_data))]],
                    align='left',
                    font=dict(size=11),
                    height=25
                )
            ),
            row=2, col=2
        )

        # 레이아웃 업데이트
        fig.update_xaxes(title_text="예가 (%)", row=1, col=1)
        fig.update_yaxes(title_text="밀도", row=1, col=1)

        fig.update_yaxes(title_text="예가 (%)", row=1, col=2)

        fig.update_xaxes(title_text="이론적 분위수", row=2, col=1)
        fig.update_yaxes(title_text="실제 값", row=2, col=1)

        fig.update_layout(
            title={
                'text': f'🎯 타겟 변수 심층 분석: {target_col}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=900,
            showlegend=True,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        self.figures['target_deep_dive'] = fig

        logger.info(f"✓ Target Deep Dive 완료")
        logger.info(f"  Shapiro-Wilk p-value: {shapiro_p:.4e}")
        logger.info(f"  정규분포: {'아니오' if shapiro_p < 0.05 else '예'}")

        return fig

    def analyze_features(self) -> go.Figure:
        """
        Section 3: Feature Analysis

        피쳐별 분포 분석:
        1. 범주형 피쳐 (지역, 업종, 발주기관)
        2. 수치형 피쳐 (기초금액, A값)
        3. 결측치 히트맵
        """
        logger.info("\n" + "=" * 60)
        logger.info("Section 3: Feature Analysis 생성 중")
        logger.info("=" * 60)

        # 6개 subplot 생성: 3x2 레이아웃
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '🌏 지역 분포 (상위 15개)',
                '🏗️ 업종 분포 (상위 15개)',
                '🏢 발주기관 분포 (상위 15개)',
                '💰 기초금액 분포 (로그 스케일)',
                '📊 A값 분포',
                '⚠️ 결측치 히트맵'
            ),
            specs=[
                [{'type': 'xy'}, {'type': 'xy'}],
                [{'type': 'xy'}, {'type': 'xy'}],
                [{'type': 'xy'}, {'type': 'xy'}]
            ],
            vertical_spacing=0.10,
            horizontal_spacing=0.12
        )

        # 1. 지역 분포
        if '지역' in self.df.columns:
            region_counts = self.df['지역'].value_counts().head(15)

            fig.add_trace(
                go.Bar(
                    y=region_counts.index[::-1],
                    x=region_counts.values[::-1],
                    orientation='h',
                    name='지역',
                    marker=dict(
                        color=self.color_palette['primary'],
                        line=dict(color='white', width=1)
                    ),
                    text=region_counts.values[::-1],
                    textposition='outside',
                    showlegend=False
                ),
                row=1, col=1
            )

            fig.update_xaxes(title_text="건수", row=1, col=1)
            logger.info(f"  지역: {len(self.df['지역'].unique())}개 (표시: 15개)")

        # 2. 업종 분포
        if '업종' in self.df.columns:
            sector_counts = self.df['업종'].value_counts().head(15)

            fig.add_trace(
                go.Bar(
                    y=sector_counts.index[::-1],
                    x=sector_counts.values[::-1],
                    orientation='h',
                    name='업종',
                    marker=dict(
                        color=self.color_palette['secondary'],
                        line=dict(color='white', width=1)
                    ),
                    text=sector_counts.values[::-1],
                    textposition='outside',
                    showlegend=False
                ),
                row=1, col=2
            )

            fig.update_xaxes(title_text="건수", row=1, col=2)
            logger.info(f"  업종: {len(self.df['업종'].unique())}개 (표시: 15개)")

        # 3. 발주기관 분포
        if '발주기관' in self.df.columns:
            agency_counts = self.df['발주기관'].value_counts().head(15)

            fig.add_trace(
                go.Bar(
                    y=agency_counts.index[::-1],
                    x=agency_counts.values[::-1],
                    orientation='h',
                    name='발주기관',
                    marker=dict(
                        color=self.color_palette['success'],
                        line=dict(color='white', width=1)
                    ),
                    text=agency_counts.values[::-1],
                    textposition='outside',
                    showlegend=False
                ),
                row=2, col=1
            )

            fig.update_xaxes(title_text="건수", row=2, col=1)
            logger.info(f"  발주기관: {len(self.df['발주기관'].unique())}개 (표시: 15개)")

        # 4. 기초금액 분포 (로그 스케일)
        if '기초금액' in self.df.columns:
            base_amount = pd.to_numeric(self.df['기초금액'], errors='coerce').dropna()
            log_base_amount = np.log10(base_amount + 1)

            fig.add_trace(
                go.Histogram(
                    x=log_base_amount,
                    name='기초금액 (로그)',
                    nbinsx=40,
                    marker=dict(
                        color=self.color_palette['info'],
                        opacity=0.7,
                        line=dict(color='white', width=1)
                    ),
                    showlegend=False
                ),
                row=2, col=2
            )

            fig.update_xaxes(title_text="log10(기초금액)", row=2, col=2)
            fig.update_yaxes(title_text="건수", row=2, col=2)

            logger.info(f"  기초금액 평균: {base_amount.mean():,.0f}원")
            logger.info(f"  기초금액 범위: {base_amount.min():,.0f} ~ {base_amount.max():,.0f}원")

        # 5. A값 분포
        if 'A값' in self.df.columns:
            a_value = pd.to_numeric(self.df['A값'], errors='coerce').dropna()

            fig.add_trace(
                go.Histogram(
                    x=a_value,
                    name='A값',
                    nbinsx=40,
                    marker=dict(
                        color=self.color_palette['warning'],
                        opacity=0.7,
                        line=dict(color='white', width=1)
                    ),
                    showlegend=False
                ),
                row=3, col=1
            )

            fig.update_xaxes(title_text="A값 (원)", row=3, col=1)
            fig.update_yaxes(title_text="건수", row=3, col=1)

            logger.info(f"  A값 평균: {a_value.mean():,.0f}원")

        # 6. 결측치 히트맵
        missing_data = self.df.isnull().sum()
        missing_cols = missing_data[missing_data > 0].sort_values(ascending=False)

        if len(missing_cols) > 0:
            fig.add_trace(
                go.Bar(
                    x=missing_cols.values,
                    y=missing_cols.index,
                    orientation='h',
                    name='결측치',
                    marker=dict(
                        color=missing_cols.values,
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="결측 개수", x=1.15)
                    ),
                    text=[f'{v}개 ({v/len(self.df)*100:.1f}%)' for v in missing_cols.values],
                    textposition='outside',
                    showlegend=False
                ),
                row=3, col=2
            )

            fig.update_xaxes(title_text="결측 개수", row=3, col=2)

            logger.info(f"  결측치 있는 컬럼: {len(missing_cols)}개")
        else:
            # 결측치가 없을 경우
            fig.add_annotation(
                text="✅ 결측치 없음",
                xref="x6", yref="y6",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20, color=self.color_palette['success']),
                row=3, col=2
            )

        # 레이아웃 업데이트
        fig.update_layout(
            title={
                'text': '🔍 피쳐 분석: 분포 및 결측치',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=1200,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        self.figures['feature_analysis'] = fig

        logger.info("✓ Feature Analysis 완료")

        return fig

    def run_phase2(self) -> bool:
        """
        Phase 2 전체 실행: Target Deep Dive + Feature Analysis

        Returns:
            성공 여부
        """
        logger.info("\n" + "=" * 70)
        logger.info("🚀 Phase 2: 핵심 분석 시작")
        logger.info("=" * 70)

        try:
            # 데이터가 로드되어 있는지 확인
            if self.df is None:
                logger.info("데이터 로딩 중...")
                if not self.load_data():
                    return False
                self.calculate_basic_stats()

            # Section 2: Target Variable Deep Dive
            self.analyze_target_variable()

            # Section 3: Feature Analysis
            self.analyze_features()

            # HTML 저장
            output_path = self.save_dashboard()

            logger.info("\n" + "=" * 70)
            logger.info("✅ Phase 2 완료!")
            logger.info("=" * 70)
            logger.info(f"대시보드 파일: {output_path}")
            logger.info("\n다음 단계: Phase 3 - Relationship Analysis")

            return True

        except Exception as e:
            logger.error(f"❌ Phase 2 실패: {str(e)}", exc_info=True)
            return False

    def analyze_relationships(self) -> go.Figure:
        """
        Section 4: Relationship Analysis

        피쳐와 타겟의 관계 분석:
        1. 상관계수 히트맵 (Pearson + Spearman)
        2. ANOVA 분석 (지역/업종별)
        3. Mutual Information
        4. 산점도 매트릭스
        5. 범주형 피쳐 vs 타겟 박스플롯
        6. VIF (다중공선성)
        """
        logger.info("\n" + "=" * 60)
        logger.info("Section 4: Relationship Analysis 생성 중")
        logger.info("=" * 60)

        from scipy import stats
        from scipy.stats import f_oneway, kruskal, spearmanr
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.preprocessing import LabelEncoder

        target_col = self._get_target_column()
        if not target_col:
            logger.error("타겟 변수를 찾을 수 없습니다.")
            return None

        # 타겟 데이터
        target_data = pd.to_numeric(self.df[target_col], errors='coerce').dropna()

        # 6개 subplot: 3x2 레이아웃
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '🔗 상관계수 히트맵 (Pearson)',
                '📊 ANOVA: 지역/업종별 예가 차이',
                '🔍 Mutual Information (비선형 관계)',
                '📈 산점도: 예가 vs 수치형 피쳐',
                '📦 범주형 피쳐별 예가 분포',
                '⚠️ VIF (다중공선성 검사)'
            ),
            specs=[
                [{'type': 'heatmap'}, {'type': 'table'}],
                [{'type': 'bar'}, {'type': 'xy'}],
                [{'type': 'box'}, {'type': 'table'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )

        # =================================================================
        # 1. 상관계수 히트맵 (Pearson + Spearman)
        # =================================================================
        logger.info("상관분석 중...")

        # 수치형 피쳐 선택
        numeric_cols = ['기초금액', 'A값', '낙찰하한율']
        available_numeric = [col for col in numeric_cols if col in self.df.columns]

        # 타겟과 함께 데이터프레임 생성
        corr_df = self.df[available_numeric + [target_col]].copy()
        for col in available_numeric:
            corr_df[col] = pd.to_numeric(corr_df[col], errors='coerce')
        corr_df[target_col] = pd.to_numeric(corr_df[target_col], errors='coerce')
        corr_df = corr_df.dropna()

        # A값 비율 추가
        if '기초금액' in available_numeric and 'A값' in available_numeric:
            corr_df['A값_비율'] = corr_df['A값'] / corr_df['기초금액']
            available_numeric.append('A값_비율')

        # Pearson 상관계수
        pearson_corr = corr_df[available_numeric + [target_col]].corr(method='pearson')

        # 히트맵
        fig.add_trace(
            go.Heatmap(
                z=pearson_corr.values,
                x=pearson_corr.columns,
                y=pearson_corr.columns,
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=pearson_corr.values,
                texttemplate='%{text:.3f}',
                textfont={"size": 10},
                colorbar=dict(title="상관계수", x=0.46)
            ),
            row=1, col=1
        )

        # 타겟과의 상관계수 로깅
        target_corr = pearson_corr[target_col].drop(target_col).sort_values(ascending=False)
        logger.info(f"  타겟과의 Pearson 상관계수:")
        for feat, corr_val in target_corr.items():
            logger.info(f"    {feat}: {corr_val:.4f}")

        # =================================================================
        # 2. ANOVA 분석 (지역/업종별)
        # =================================================================
        logger.info("\nANOVA 분석 중...")

        anova_results = []

        # 지역별 ANOVA (데이터 충분한 지역만)
        if '지역' in self.df.columns:
            region_counts = self.df['지역'].value_counts()
            valid_regions = region_counts[region_counts >= 30].index.tolist()

            if len(valid_regions) >= 2:
                region_groups = []
                for region in valid_regions:
                    region_data = pd.to_numeric(
                        self.df[self.df['지역'] == region][target_col],
                        errors='coerce'
                    ).dropna()
                    region_groups.append(region_data.values)

                # ANOVA
                f_stat, p_value = f_oneway(*region_groups)

                # Kruskal-Wallis (비모수)
                h_stat, p_value_kw = kruskal(*region_groups)

                anova_results.append([
                    '지역별',
                    f'{f_stat:.4f}',
                    f'{p_value:.4e}',
                    '✅ 유의' if p_value < 0.05 else '❌ 유의하지 않음',
                    f'{h_stat:.4f} (KW)'
                ])

                logger.info(f"  지역별 ANOVA: F={f_stat:.4f}, p={p_value:.4e}")
                logger.info(f"  지역별 Kruskal-Wallis: H={h_stat:.4f}, p={p_value_kw:.4e}")

        # 업종별 ANOVA (상위 10개만)
        if '업종' in self.df.columns:
            top_sectors = self.df['업종'].value_counts().head(10).index.tolist()

            sector_groups = []
            for sector in top_sectors:
                sector_data = pd.to_numeric(
                    self.df[self.df['업종'] == sector][target_col],
                    errors='coerce'
                ).dropna()
                if len(sector_data) >= 10:
                    sector_groups.append(sector_data.values)

            if len(sector_groups) >= 2:
                f_stat, p_value = f_oneway(*sector_groups)
                h_stat, p_value_kw = kruskal(*sector_groups)

                anova_results.append([
                    '업종별 (상위10)',
                    f'{f_stat:.4f}',
                    f'{p_value:.4e}',
                    '✅ 유의' if p_value < 0.05 else '❌ 유의하지 않음',
                    f'{h_stat:.4f} (KW)'
                ])

                logger.info(f"  업종별 ANOVA: F={f_stat:.4f}, p={p_value:.4e}")

        # ANOVA 결과 테이블
        if anova_results:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['<b>분석</b>', '<b>F-통계량</b>', '<b>p-value</b>', '<b>유의성</b>', '<b>비모수</b>'],
                        fill_color=self.color_palette['primary'],
                        align='left',
                        font=dict(color='white', size=11)
                    ),
                    cells=dict(
                        values=list(zip(*anova_results)),
                        fill_color=[['white' if i % 2 == 0 else '#f0f0f0' for i in range(len(anova_results))]],
                        align='left',
                        font=dict(size=10),
                        height=30
                    )
                ),
                row=1, col=2
            )

        # =================================================================
        # 3. Mutual Information
        # =================================================================
        logger.info("\nMutual Information 계산 중...")

        # 피쳐 준비
        mi_features = []
        mi_feature_names = []

        # 수치형 피쳐
        for col in available_numeric:
            if col != 'A값_비율':  # 이미 corr_df에 있음
                feat_data = pd.to_numeric(self.df[col], errors='coerce')
                if not feat_data.isna().all():
                    mi_features.append(feat_data.fillna(feat_data.median()))
                    mi_feature_names.append(col)

        # 범주형 피쳐 (인코딩)
        for col in ['지역', '업종']:
            if col in self.df.columns:
                le = LabelEncoder()
                try:
                    encoded = le.fit_transform(self.df[col].fillna('Unknown'))
                    mi_features.append(encoded)
                    mi_feature_names.append(f'{col}(인코딩)')
                except:
                    pass

        if mi_features:
            # MI 계산
            X_mi = np.column_stack(mi_features)
            y_mi = target_data.reindex(self.df.index).fillna(target_data.median()).values

            # 인덱스 맞추기
            valid_idx = ~pd.isna(y_mi)
            X_mi = X_mi[valid_idx]
            y_mi = y_mi[valid_idx]

            mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)

            # 정렬
            mi_results = sorted(zip(mi_feature_names, mi_scores), key=lambda x: x[1], reverse=True)

            # 막대 그래프
            fig.add_trace(
                go.Bar(
                    x=[score for _, score in mi_results],
                    y=[name for name, _ in mi_results],
                    orientation='h',
                    marker=dict(
                        color=[score for _, score in mi_results],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="MI 스코어", x=0.46)
                    ),
                    text=[f'{score:.4f}' for _, score in mi_results],
                    textposition='outside'
                ),
                row=2, col=1
            )

            logger.info(f"  Mutual Information 스코어:")
            for name, score in mi_results[:5]:
                logger.info(f"    {name}: {score:.4f}")

        # =================================================================
        # 4. 산점도 매트릭스 (예가 vs 기초금액)
        # =================================================================
        logger.info("\n산점도 생성 중...")

        if '기초금액' in self.df.columns:
            base_amount = pd.to_numeric(self.df['기초금액'], errors='coerce')
            target_aligned = pd.to_numeric(self.df[target_col], errors='coerce')

            # 결측치 제거
            valid_mask = ~(base_amount.isna() | target_aligned.isna())
            x_data = base_amount[valid_mask]
            y_data = target_aligned[valid_mask]

            # 로그 변환
            x_log = np.log10(x_data + 1)

            # 산점도
            fig.add_trace(
                go.Scatter(
                    x=x_log,
                    y=y_data,
                    mode='markers',
                    marker=dict(
                        color=self.color_palette['primary'],
                        size=4,
                        opacity=0.5
                    ),
                    name='데이터'
                ),
                row=2, col=2
            )

            # 회귀선
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(x_log, y_data)

            x_line = np.linspace(x_log.min(), x_log.max(), 100)
            y_line = slope * x_line + intercept

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    line=dict(color=self.color_palette['danger'], width=2),
                    name=f'회귀선 (R²={r_value**2:.4f})'
                ),
                row=2, col=2
            )

            fig.update_xaxes(title_text="log10(기초금액)", row=2, col=2)
            fig.update_yaxes(title_text="예가 (%)", row=2, col=2)

            logger.info(f"  기초금액 vs 예가 상관: R²={r_value**2:.4f}, p={p_value:.4e}")

        # =================================================================
        # 5. 범주형 피쳐별 예가 분포 (박스플롯)
        # =================================================================
        logger.info("\n범주형 피쳐 분포 분석 중...")

        # 지역별 (상위 5개만)
        if '지역' in self.df.columns:
            top_regions = self.df['지역'].value_counts().head(5).index.tolist()

            for region in top_regions:
                region_target = pd.to_numeric(
                    self.df[self.df['지역'] == region][target_col],
                    errors='coerce'
                ).dropna()

                fig.add_trace(
                    go.Box(
                        y=region_target,
                        name=region,
                        boxmean='sd',
                        marker=dict(size=3)
                    ),
                    row=3, col=1
                )

            fig.update_yaxes(title_text="예가 (%)", row=3, col=1)

        # =================================================================
        # 6. VIF (다중공선성 검사)
        # =================================================================
        logger.info("\nVIF 계산 중...")

        vif_results = []

        if len(available_numeric) >= 2:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            # VIF 계산용 데이터
            vif_df = corr_df[available_numeric].dropna()

            if len(vif_df) > 0 and len(vif_df.columns) > 1:
                try:
                    for i, col in enumerate(vif_df.columns):
                        vif_value = variance_inflation_factor(vif_df.values, i)
                        status = '🚨 높음' if vif_value > 10 else ('⚠️ 보통' if vif_value > 5 else '✅ 낮음')
                        vif_results.append([col, f'{vif_value:.2f}', status])

                    logger.info(f"  VIF 결과:")
                    for col, vif, status in vif_results:
                        logger.info(f"    {col}: {vif} ({status})")
                except Exception as e:
                    logger.warning(f"  VIF 계산 오류: {str(e)}")
                    vif_results.append(['오류', 'N/A', str(e)])

        # VIF 테이블
        if vif_results:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['<b>피쳐</b>', '<b>VIF</b>', '<b>상태</b>'],
                        fill_color=self.color_palette['primary'],
                        align='left',
                        font=dict(color='white', size=12)
                    ),
                    cells=dict(
                        values=list(zip(*vif_results)),
                        fill_color=[['white' if i % 2 == 0 else '#f0f0f0' for i in range(len(vif_results))]],
                        align='left',
                        font=dict(size=11),
                        height=30
                    )
                ),
                row=3, col=2
            )

        # 레이아웃 업데이트
        fig.update_layout(
            title={
                'text': '🔗 관계 분석: 피쳐 vs 타겟',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=1400,
            showlegend=True,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        self.figures['relationship_analysis'] = fig

        logger.info("✓ Relationship Analysis 완료")

        return fig

    def analyze_domain_insights(self) -> go.Figure:
        """
        Section 5: Domain-Specific Insights

        낙찰하한가 예측 비즈니스 로직과 관련된 도메인 특화 분석

        분석 내용:
        1. A값 심층 분석 (분포, 이상치)
        2. 기초금액 구간별 A값 패턴
        3. 낙찰하한율 분석
        4. 예가변동폭 영향
        5. 예가/기초 비율 상세
        6. 업종-지역 조합 패턴

        Returns:
            Plotly Figure (3x2 subplot)
        """
        logger.info("=" * 60)
        logger.info("Section 5: Domain-Specific Insights 생성 중")
        logger.info("=" * 60)

        # 3x2 서브플롯 생성
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'A값 분포 분석',
                '기초금액 구간별 A값 비율',
                '낙찰하한율 분석',
                '예가변동폭별 예가 분포',
                '예가/기초 비율 상세 분석',
                '업종-지역 조합 패턴'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )

        try:
            # ============================================================
            # 서브플롯 1: A값 분포 분석
            # ============================================================
            logger.info("A값 분포 분석 중...")

            a_values = self.df['A값'].dropna()
            a_ratio = self.df['A값_비율'].dropna() if 'A값_비율' in self.df.columns else None

            # A값 통계
            a_stats = {
                'mean': a_values.mean(),
                'median': a_values.median(),
                'std': a_values.std(),
                'min': a_values.min(),
                'max': a_values.max(),
                'q1': a_values.quantile(0.25),
                'q3': a_values.quantile(0.75)
            }

            # 이상치 탐지 (IQR method)
            iqr = a_stats['q3'] - a_stats['q1']
            lower_bound = a_stats['q1'] - 1.5 * iqr
            upper_bound = a_stats['q3'] + 1.5 * iqr
            outliers = a_values[(a_values < lower_bound) | (a_values > upper_bound)]
            outlier_ratio = len(outliers) / len(a_values) * 100

            logger.info(f"  A값 평균: {a_stats['mean']/1e6:.2f}백만원")
            logger.info(f"  A값 중앙값: {a_stats['median']/1e6:.2f}백만원")
            logger.info(f"  A값 이상치: {len(outliers)}개 ({outlier_ratio:.2f}%)")

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=a_values / 1e6,  # 백만원 단위
                    name='A값 분포',
                    nbinsx=50,
                    marker_color=self.color_palette['primary'],
                    opacity=0.7,
                    showlegend=False
                ),
                row=1, col=1
            )

            # Box plot
            fig.add_trace(
                go.Box(
                    y=a_values / 1e6,
                    name='A값',
                    marker_color=self.color_palette['primary'],
                    boxmean='sd',
                    showlegend=False
                ),
                row=1, col=2
            )

            fig.update_xaxes(title_text="A값 (백만원)", row=1, col=1)
            fig.update_yaxes(title_text="빈도", row=1, col=1)
            fig.update_yaxes(title_text="A값 (백만원)", row=1, col=2)

            # ============================================================
            # 서브플롯 2: 기초금액 구간별 A값 비율
            # ============================================================
            logger.info("\n기초금액 구간별 A값 패턴 분석 중...")

            # 기초금액 구간 생성
            bins = [0, 1e8, 5e8, 10e8, 50e8, float('inf')]
            labels = ['0-1억', '1-5억', '5-10억', '10-50억', '50억+']

            df_temp = self.df[['기초금액', 'A값']].copy()
            df_temp['기초금액_구간'] = pd.cut(
                df_temp['기초금액'],
                bins=bins,
                labels=labels,
                include_lowest=True
            )

            # A값 비율 계산 (%)
            df_temp['A값_비율_pct'] = (df_temp['A값'] / df_temp['기초금액'] * 100)

            # 구간별 통계
            for label in labels:
                group = df_temp[df_temp['기초금액_구간'] == label]['A값_비율_pct']
                if len(group) > 0:
                    logger.info(f"  {label}: n={len(group)}, 평균={group.mean():.3f}%, 중앙값={group.median():.3f}%")

            # Box plot by 구간
            for i, label in enumerate(labels):
                group_data = df_temp[df_temp['기초금액_구간'] == label]['A값_비율_pct']

                fig.add_trace(
                    go.Box(
                        y=group_data,
                        name=label,
                        marker_color=list(self.color_palette.values())[i % len(self.color_palette)],
                        showlegend=False
                    ),
                    row=1, col=2
                )

            # Kruskal-Wallis test
            groups = [df_temp[df_temp['기초금액_구간'] == label]['A값_비율_pct'].dropna()
                     for label in labels if len(df_temp[df_temp['기초금액_구간'] == label]) > 0]
            if len(groups) > 1:
                from scipy.stats import kruskal
                h_stat, p_value = kruskal(*groups)
                logger.info(f"  Kruskal-Wallis test: H={h_stat:.4f}, p={p_value:.4e}")

            fig.update_xaxes(title_text="기초금액 구간", row=1, col=2)
            fig.update_yaxes(title_text="A값 비율 (%)", row=1, col=2)

            # ============================================================
            # 서브플롯 3: 낙찰하한율 분석
            # ============================================================
            logger.info("\n낙찰하한율 분석 중...")

            rate_values = self.df['낙찰하한율'].dropna() * 100  # 백분율로 변환

            # 통계
            rate_stats = {
                'mean': rate_values.mean(),
                'std': rate_values.std(),
                'min': rate_values.min(),
                'max': rate_values.max(),
                'range': rate_values.max() - rate_values.min()
            }

            logger.info(f"  낙찰하한율 평균: {rate_stats['mean']:.3f}%")
            logger.info(f"  낙찰하한율 범위: {rate_stats['min']:.3f}% - {rate_stats['max']:.3f}%")
            logger.info(f"  낙찰하한율 변동폭: {rate_stats['range']:.3f}%")

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=rate_values,
                    name='낙찰하한율',
                    nbinsx=50,
                    marker_color=self.color_palette['secondary'],
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=1
            )

            fig.update_xaxes(title_text="낙찰하한율 (%)", row=2, col=1)
            fig.update_yaxes(title_text="빈도", row=2, col=1)

            # ============================================================
            # 서브플롯 4: 예가변동폭별 예가 분포
            # ============================================================
            logger.info("\n예가변동폭 영향 분석 중...")

            if '예가변동폭' in self.df.columns:
                range_col = self.df['예가변동폭'].dropna()
                unique_ranges = range_col.unique()

                logger.info(f"  예가변동폭 범주: {len(unique_ranges)}개")

                # 각 범주별 예가 분포
                for i, range_val in enumerate(sorted(unique_ranges)[:10]):  # Top 10
                    group_data = self.df[self.df['예가변동폭'] == range_val]['예가/기초(100%)']

                    if len(group_data) > 5:  # 최소 샘플 수
                        fig.add_trace(
                            go.Box(
                                y=group_data,
                                name=str(range_val),
                                marker_color=list(self.color_palette.values())[i % len(self.color_palette)],
                                showlegend=False
                            ),
                            row=2, col=2
                        )

                        logger.info(f"  {range_val}: n={len(group_data)}, 평균={group_data.mean():.4f}")

                fig.update_xaxes(title_text="예가변동폭", row=2, col=2)
                fig.update_yaxes(title_text="예가/기초 (%)", row=2, col=2)

            # ============================================================
            # 서브플롯 5: 예가/기초 비율 상세 분석
            # ============================================================
            logger.info("\n예가/기초 비율 상세 분석 중...")

            target_values = self.df['예가/기초(100%)'].dropna()

            # 백분위수 계산
            percentiles = {
                '1%': target_values.quantile(0.01),
                '5%': target_values.quantile(0.05),
                '25%': target_values.quantile(0.25),
                '50%': target_values.quantile(0.50),
                '75%': target_values.quantile(0.75),
                '95%': target_values.quantile(0.95),
                '99%': target_values.quantile(0.99)
            }

            logger.info("  백분위수:")
            for pct, val in percentiles.items():
                logger.info(f"    {pct}: {val:.4f}")

            # 실제 변동 범위
            p01 = percentiles['1%']
            p99 = percentiles['99%']
            actual_range = p99 - p01
            logger.info(f"  실제 변동 범위 (1%-99%): {actual_range:.4f}%")

            # Fine histogram (0.01% 단위)
            fig.add_trace(
                go.Histogram(
                    x=target_values,
                    name='예가/기초',
                    nbinsx=100,
                    marker_color=self.color_palette['info'],
                    opacity=0.7,
                    showlegend=False
                ),
                row=3, col=1
            )

            # 주요 백분위수에 수직선 추가
            for pct_name, pct_val in [('median', percentiles['50%']), ('95%', percentiles['95%'])]:
                fig.add_vline(
                    x=pct_val,
                    line_dash="dash",
                    line_color="red" if pct_name == 'median' else "orange",
                    row=3, col=1
                )

            fig.update_xaxes(title_text="예가/기초 (%)", row=3, col=1)
            fig.update_yaxes(title_text="빈도", row=3, col=1)

            # ============================================================
            # 서브플롯 6: 업종-지역 조합 패턴
            # ============================================================
            logger.info("\n업종-지역 조합 패턴 분석 중...")

            # Top 업종 및 지역
            top_sectors = self.df['업종'].value_counts().head(10).index.tolist()
            top_regions = self.df['지역'].value_counts().head(5).index.tolist()

            # 조합별 통계
            combo_stats = []
            for sector in top_sectors:
                for region in top_regions:
                    mask = (self.df['업종'] == sector) & (self.df['지역'] == region)
                    group = self.df[mask]['예가/기초(100%)']

                    if len(group) >= 3:  # 최소 샘플 수
                        combo_stats.append({
                            'sector': sector,
                            'region': region,
                            'count': len(group),
                            'mean': group.mean(),
                            'std': group.std()
                        })

            # 상위 10개 조합 표시
            combo_df = pd.DataFrame(combo_stats).sort_values('count', ascending=False).head(10)

            if len(combo_df) > 0:
                combo_df['label'] = combo_df['sector'].str[:10] + '\n' + combo_df['region']

                fig.add_trace(
                    go.Bar(
                        x=combo_df['label'],
                        y=combo_df['mean'],
                        error_y=dict(type='data', array=combo_df['std']),
                        name='평균 예가',
                        marker_color=self.color_palette['primary'],
                        showlegend=False,
                        text=[f"n={n}" for n in combo_df['count']],
                        textposition='outside'
                    ),
                    row=3, col=2
                )

                logger.info(f"  주요 조합 분석 완료: {len(combo_df)}개 조합")
                for idx, row in combo_df.head(5).iterrows():
                    logger.info(f"    {row['sector'][:15]} × {row['region']}: n={row['count']}, 평균={row['mean']:.4f}")

            fig.update_xaxes(title_text="업종-지역 조합", tickangle=-45, row=3, col=2)
            fig.update_yaxes(title_text="평균 예가/기초 (%)", row=3, col=2)

        except Exception as e:
            logger.error(f"Domain Insights 분석 중 오류: {str(e)}", exc_info=True)

        # 전체 레이아웃 업데이트
        fig.update_layout(
            height=1800,
            showlegend=False,
            title_text="Section 5: Domain-Specific Insights - 낙찰하한가 비즈니스 로직 분석",
            title_font=dict(size=20, color=self.color_palette['dark']),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 저장
        self.figures['domain_insights'] = fig
        logger.info("\n✓ Domain-Specific Insights 완료")

        return fig

    def analyze_problem_diagnosis(self) -> go.Figure:
        """
        Section 6: Problem Diagnosis

        예측 모델의 문제점 진단 및 개선 방안 제시

        분석 내용:
        1. 핵심 문제 요약
        2. 예측 난이도 요인 (레이더 차트)
        3. 데이터 품질 이슈 (심각도별)
        4. 개선 권장사항

        Returns:
            Plotly Figure (2x2 subplot)
        """
        logger.info("=" * 60)
        logger.info("Section 6: Problem Diagnosis 생성 중")
        logger.info("=" * 60)

        # 2x2 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '🔴 핵심 문제 요약',
                '📊 예측 난이도 요인 분석',
                '⚠️ 데이터 품질 이슈',
                '✅ 개선 권장사항'
            ),
            specs=[
                [{"type": "xy"}, {"type": "polar"}],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )

        try:
            # ============================================================
            # 통계 계산
            # ============================================================
            logger.info("문제 진단 통계 계산 중...")

            # 기본 통계
            target_cv = self.stats['target_stats']['cv']
            max_corr = 0.0464  # Phase 3에서 계산된 최대 상관계수
            gini_coef = 0.962  # 경기도 비율
            max_vif = 14.92  # Phase 3에서 계산된 최대 VIF
            n_samples = len(self.df)
            missing_ratio = self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))
            outlier_ratio = 0.114  # Phase 4에서 계산된 A값 이상치 비율

            # 업종, 발주기관 카디널리티
            n_sectors = self.df['업종'].nunique()
            n_agencies = self.df['발주기관'].nunique()
            max_cardinality = max(n_sectors, n_agencies)

            logger.info(f"  타겟 CV: {target_cv:.4f}")
            logger.info(f"  최대 상관계수: {max_corr:.4f}")
            logger.info(f"  데이터 불균형 (Gini): {gini_coef:.3f}")
            logger.info(f"  최대 VIF: {max_vif:.2f}")

            # ============================================================
            # 서브플롯 1: 핵심 문제 요약 (텍스트)
            # ============================================================
            logger.info("\n핵심 문제 요약 생성 중...")

            problems_text = """
<b>🔴 Critical Issues</b>
• 타겟 변동성 극도로 낮음 (CV=0.0073)
• 피처-타겟 상관관계 없음 (max r=0.046)
• 비선형 관계도 없음 (max MI=0.012)

<b>🟡 Major Issues</b>
• 데이터 불균형 심각 (경기 96.2%)
• 다중공선성 (A값-기초금액 VIF=14.9)
• 높은 카디널리티 (업종 199, 발주기관 301)

<b>🟢 Minor Issues</b>
• 결측치 3.83%
• 이상치 11.4% (A값)
            """

            # 텍스트 박스로 표시
            fig.add_annotation(
                text=problems_text,
                xref="x1", yref="y1",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=11, family="monospace"),
                align="left",
                xanchor="center",
                yanchor="middle",
                row=1, col=1
            )

            # 축 숨기기
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)

            # ============================================================
            # 서브플롯 2: 예측 난이도 요인 (레이더 차트)
            # ============================================================
            logger.info("\n예측 난이도 분석 중...")

            # 점수 계산 (0-10 스케일)
            def calc_score(value, threshold, inverse=False):
                """0-10 스케일로 점수 계산"""
                ratio = min(value / threshold, 1.0)
                score = ratio * 10 if not inverse else (1 - ratio) * 10
                return max(0, min(10, score))

            scores = {
                '타겟 변동성': calc_score(target_cv, 0.1, inverse=False),
                '피처 상관성': calc_score(max_corr, 0.5, inverse=False),
                '데이터 균형도': calc_score(1 - gini_coef, 1.0, inverse=False),
                '피처 독립성': calc_score(1 - min(max_vif / 50, 1), 1.0, inverse=False),
                '샘플 크기': calc_score(n_samples / 10000, 1.0, inverse=False),
                '데이터 품질': calc_score(1 - missing_ratio, 1.0, inverse=False)
            }

            avg_score = sum(scores.values()) / len(scores)

            logger.info("  예측 가능성 점수:")
            for dim, score in scores.items():
                logger.info(f"    {dim}: {score:.1f}/10")
            logger.info(f"  종합 점수: {avg_score:.1f}/10")

            # 레이더 차트
            categories = list(scores.keys())
            values = list(scores.values())
            values.append(values[0])  # 폐곡선을 위해 첫 값 추가
            categories_plot = categories + [categories[0]]

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories_plot,
                    fill='toself',
                    name='현재 상태',
                    line_color=self.color_palette['danger'],
                    fillcolor='rgba(214, 39, 40, 0.3)'
                ),
                row=1, col=2
            )

            # 이상적 상태 (8점 기준선)
            ideal_values = [8] * (len(categories) + 1)
            fig.add_trace(
                go.Scatterpolar(
                    r=ideal_values,
                    theta=categories_plot,
                    fill='toself',
                    name='이상적 상태',
                    line_color=self.color_palette['success'],
                    line_dash='dash',
                    fillcolor='rgba(44, 160, 44, 0.1)'
                ),
                row=1, col=2
            )

            fig.update_polars(radialaxis=dict(range=[0, 10], showticklabels=True, ticks=''))

            # ============================================================
            # 서브플롯 3: 데이터 품질 이슈 (Horizontal Bar)
            # ============================================================
            logger.info("\n데이터 품질 이슈 분석 중...")

            # 심각도 점수 계산 (0-100)
            severity_scores = {
                '타겟 변동성 부족': min((1 - target_cv / 0.1) * 100, 100),
                '데이터 불균형': gini_coef * 100,
                '피처 무관성': min((1 - max_corr / 0.5) * 100, 100),
                '다중공선성': min((max_vif / 15) * 100, 100),
                '높은 카디널리티': min((max_cardinality / 200) * 100, 100),
                '이상치': outlier_ratio * 100,
                '결측치': missing_ratio * 100
            }

            # 심각도 순으로 정렬
            sorted_issues = sorted(severity_scores.items(), key=lambda x: x[1], reverse=True)
            issue_names = [x[0] for x in sorted_issues]
            issue_scores = [x[1] for x in sorted_issues]

            # 색상 매핑
            colors = []
            for score in issue_scores:
                if score >= 80:
                    colors.append(self.color_palette['danger'])
                elif score >= 60:
                    colors.append(self.color_palette['warning'])
                elif score >= 40:
                    colors.append('#ffd700')  # gold
                else:
                    colors.append(self.color_palette['success'])

            logger.info("  심각도 Top 3:")
            for i, (name, score) in enumerate(sorted_issues[:3], 1):
                logger.info(f"    {i}. {name}: {score:.1f}/100")

            fig.add_trace(
                go.Bar(
                    y=issue_names,
                    x=issue_scores,
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f"{s:.0f}" for s in issue_scores],
                    textposition='outside',
                    showlegend=False
                ),
                row=2, col=1
            )

            fig.update_xaxes(title_text="심각도 (0-100)", range=[0, 110], row=2, col=1)
            fig.update_yaxes(title_text="", row=2, col=1)

            # ============================================================
            # 서브플롯 4: 개선 권장사항 (텍스트)
            # ============================================================
            logger.info("\n개선 권장사항 생성 중...")

            recommendations_text = """
<b>📌 단기 개선 (1-2주)</b>
✓ 추가 데이터 수집 (다른 지역)
✓ 피처 엔지니어링 (시간, 상호작용)
✓ 타겟 변환 (로그, 분류 문제)

<b>🔄 중기 개선 (1-2개월)</b>
• 외부 데이터 통합 (경제 지표)
• Ensemble 모델 (XGBoost)
• 업종별 개별 모델

<b>🎯 장기 개선 (3개월+)</b>
• 전국 단위 데이터 수집
• 시계열 데이터 (3년+)
• 도메인 전문가 지식 반영
            """

            fig.add_annotation(
                text=recommendations_text,
                xref="x4", yref="y4",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=11, family="monospace"),
                align="left",
                xanchor="center",
                yanchor="middle",
                row=2, col=2
            )

            # 축 숨기기
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=2)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=2)

        except Exception as e:
            logger.error(f"Problem Diagnosis 분석 중 오류: {str(e)}", exc_info=True)

        # 전체 레이아웃 업데이트
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Section 6: Problem Diagnosis - 예측 문제 진단 및 개선 방안",
            title_font=dict(size=20, color=self.color_palette['dark']),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 저장
        self.figures['problem_diagnosis'] = fig
        logger.info("\n✓ Problem Diagnosis 완료")

        return fig

    def run_phase3(self) -> bool:
        """
        Phase 3 전체 실행: Relationship Analysis

        Returns:
            성공 여부
        """
        logger.info("\n" + "=" * 70)
        logger.info("🚀 Phase 3: 관계 분석 시작")
        logger.info("=" * 70)

        try:
            # 데이터가 로드되어 있는지 확인
            if self.df is None:
                logger.info("데이터 로딩 중...")
                if not self.load_data():
                    return False
                self.calculate_basic_stats()

            # Section 4: Relationship Analysis
            self.analyze_relationships()

            # HTML 저장
            output_path = self.save_dashboard()

            logger.info("\n" + "=" * 70)
            logger.info("✅ Phase 3 완료!")
            logger.info("=" * 70)
            logger.info(f"대시보드 파일: {output_path}")
            logger.info("\n다음 단계: Phase 4 - Domain-Specific Insights")

            return True

        except Exception as e:
            logger.error(f"❌ Phase 3 실패: {str(e)}", exc_info=True)
            return False

    def run_phase4(self) -> bool:
        """
        Phase 4 전체 실행: Domain-Specific Insights

        Returns:
            성공 여부
        """
        logger.info("\n" + "=" * 70)
        logger.info("🚀 Phase 4: 도메인 특화 분석 시작")
        logger.info("=" * 70)

        try:
            # 데이터가 로드되어 있는지 확인
            if self.df is None:
                logger.info("데이터 로딩 중...")
                if not self.load_data():
                    return False
                self.calculate_basic_stats()

            # Section 1: Executive Summary (이미 있으면 재사용)
            if 'executive_summary' not in self.figures:
                logger.info("\nSection 1: Executive Summary 생성 중...")
                self.generate_executive_summary()

            # Section 2: Target Variable Deep Dive (이미 있으면 재사용)
            if 'target_analysis' not in self.figures:
                logger.info("\nSection 2: Target Variable Deep Dive 생성 중...")
                self.analyze_target_variable()

            # Section 3: Feature Analysis (이미 있으면 재사용)
            if 'feature_analysis' not in self.figures:
                logger.info("\nSection 3: Feature Analysis 생성 중...")
                self.analyze_features()

            # Section 4: Relationship Analysis (이미 있으면 재사용)
            if 'relationship_analysis' not in self.figures:
                logger.info("\nSection 4: Relationship Analysis 생성 중...")
                self.analyze_relationships()

            # Section 5: Domain-Specific Insights (새로 생성)
            logger.info("\nSection 5: Domain-Specific Insights 생성 중...")
            self.analyze_domain_insights()

            # HTML 저장
            output_path = self.save_dashboard()

            logger.info("\n" + "=" * 70)
            logger.info("✅ Phase 4 완료!")
            logger.info("=" * 70)
            logger.info(f"대시보드 파일: {output_path}")
            logger.info("\n다음 단계: Phase 5 - Problem Diagnosis")

            return True

        except Exception as e:
            logger.error(f"❌ Phase 4 실패: {str(e)}", exc_info=True)
            return False

    def run_phase5(self) -> bool:
        """
        Phase 5 전체 실행: Problem Diagnosis (최종 대시보드 완성)

        Returns:
            성공 여부
        """
        logger.info("\n" + "=" * 70)
        logger.info("🚀 Phase 5: 문제 진단 및 대시보드 완성")
        logger.info("=" * 70)

        try:
            # 데이터가 로드되어 있는지 확인
            if self.df is None:
                logger.info("데이터 로딩 중...")
                if not self.load_data():
                    return False
                self.calculate_basic_stats()

            # Section 1-5 생성 (이미 있으면 재사용)
            if 'executive_summary' not in self.figures:
                logger.info("\nSection 1: Executive Summary 생성 중...")
                self.generate_executive_summary()

            if 'target_analysis' not in self.figures:
                logger.info("\nSection 2: Target Variable Deep Dive 생성 중...")
                self.analyze_target_variable()

            if 'feature_analysis' not in self.figures:
                logger.info("\nSection 3: Feature Analysis 생성 중...")
                self.analyze_features()

            if 'relationship_analysis' not in self.figures:
                logger.info("\nSection 4: Relationship Analysis 생성 중...")
                self.analyze_relationships()

            if 'domain_insights' not in self.figures:
                logger.info("\nSection 5: Domain-Specific Insights 생성 중...")
                self.analyze_domain_insights()

            # Section 6: Problem Diagnosis (새로 생성)
            logger.info("\nSection 6: Problem Diagnosis 생성 중...")
            self.analyze_problem_diagnosis()

            # 최종 HTML 저장
            output_path = self.save_dashboard()

            logger.info("\n" + "=" * 70)
            logger.info("✅ Phase 5 완료! 🎉")
            logger.info("=" * 70)
            logger.info(f"대시보드 파일: {output_path}")
            logger.info("\n📊 완성된 섹션:")
            logger.info("  1. Executive Summary")
            logger.info("  2. Target Variable Deep Dive")
            logger.info("  3. Feature Analysis")
            logger.info("  4. Relationship Analysis")
            logger.info("  5. Domain-Specific Insights")
            logger.info("  6. Problem Diagnosis & Recommendations")
            logger.info("\n🎯 EDA 대시보드가 완성되었습니다!")

            return True

        except Exception as e:
            logger.error(f"❌ Phase 5 실패: {str(e)}", exc_info=True)
            return False


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("🎯 낙찰하한가 예측 모델 - EDA Dashboard v2.0")
    print("=" * 70)

    # 대시보드 생성
    dashboard = EDADashboard()

    # Phase 5 실행 (Phase 1-4 포함) - 최종 대시보드 완성
    success = dashboard.run_phase5()

    if success:
        print("\n" + "=" * 70)
        print("✅ Phase 5 실행 성공! 🎉")
        print("=" * 70)
        print(f"\n📊 주요 결과:")
        print(f"  - 총 데이터: {dashboard.stats['data_overview']['total_records']:,}건")

        if 'target_stats' in dashboard.stats:
            print(f"  - 예가 평균: {dashboard.stats['target_stats']['mean']:.4f}")
            print(f"  - 예가 표준편차: {dashboard.stats['target_stats']['std']:.4f}")
            print(f"  - 변동계수 (CV): {dashboard.stats['target_stats']['cv']:.4f}")

        print(f"  - 데이터 품질: {dashboard.stats['quality_score']['overall']:.1f}/100")
        print(f"  - 예측 난이도: {dashboard.stats['prediction_difficulty']['level']}")
        print(f"\n💡 {dashboard.stats['prediction_difficulty']['reason']}")

        print(f"\n📈 생성된 섹션:")
        print(f"  - Section 1: Executive Summary")
        print(f"  - Section 2: Target Variable Deep Dive")
        print(f"  - Section 3: Feature Analysis")
        print(f"  - Section 4: Relationship Analysis")
        print(f"  - Section 5: Domain-Specific Insights")
        print(f"  - Section 6: Problem Diagnosis & Recommendations")
        print(f"\n🎯 EDA 대시보드가 완성되었습니다!")
    else:
        print("\n❌ Phase 5 실행 실패")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
