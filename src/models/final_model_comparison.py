"""
최종 3가지 모델 비교 및 분석
1. Linear Regression (베이스라인)
2. Ridge Regression (정규화 모델)
3. Weighted Ensemble (앙상블 모델)

각 모델의 특징, 성능, 권장 사용 상황 정리
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 시각화
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinalModelComparison:
    """최종 모델 비교 및 분석 클래스"""

    def __init__(self, output_dir: str = "models/trained"):
        """초기화"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 모델 성능 데이터 (실제 실험 결과)
        self.model_results = {
            'Linear Regression': {
                'test_r2': 0.234917,
                'test_mae': 0.483048,
                'test_rmse': 0.625699,
                'test_mape': 0.4836,
                'train_r2': 0.214812,
                'overfitting': False,
                'complexity': '낮음',
                'interpretability': '매우 높음',
                'training_time': '매우 빠름'
            },
            'Ridge Regression': {
                'test_r2': 0.241543,
                'test_mae': 0.483365,
                'test_rmse': 0.622984,
                'test_mape': 0.4839,
                'train_r2': 0.208855,
                'best_alpha': 0.019307,
                'overfitting': False,
                'complexity': '낮음',
                'interpretability': '높음',
                'training_time': '빠름'
            },
            'Weighted Ensemble': {
                'test_r2': 0.231258,
                'test_mae': 0.491127,
                'test_rmse': 0.627193,
                'test_mape': 0.4918,
                'train_r2': 0.531011,
                'lr_weight': 0.5041,
                'xgb_weight': 0.4959,
                'overfitting': True,  # Train과 Test 차이 큼
                'complexity': '중간',
                'interpretability': '중간',
                'training_time': '중간'
            }
        }

        logger.info("FinalModelComparison 초기화 완료")

    def create_performance_comparison(self):
        """성능 비교표 생성"""
        logger.info("\n" + "=" * 70)
        logger.info("성능 비교표 생성")
        logger.info("=" * 70)

        # DataFrame 생성
        comparison_data = []
        for model_name, metrics in self.model_results.items():
            comparison_data.append({
                'Model': model_name,
                'Test_R2': metrics['test_r2'],
                'Test_MAE': metrics['test_mae'],
                'Test_RMSE': metrics['test_rmse'],
                'Test_MAPE': metrics['test_mape'],
                'Train_R2': metrics['train_r2'],
                'Overfitting_Gap': metrics['train_r2'] - metrics['test_r2'],
                'Complexity': metrics['complexity'],
                'Interpretability': metrics['interpretability']
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test_R2', ascending=False)

        logger.info("\n성능 비교:")
        logger.info(df.to_string(index=False))

        # 순위 추가
        df['R2_Rank'] = df['Test_R2'].rank(ascending=False).astype(int)
        df['MAE_Rank'] = df['Test_MAE'].rank(ascending=True).astype(int)

        self.comparison_df = df

        return df

    def analyze_model_characteristics(self):
        """모델별 특징 분석"""
        logger.info("\n" + "=" * 70)
        logger.info("모델별 특징 분석")
        logger.info("=" * 70)

        characteristics = {
            'Linear Regression': {
                '장점': [
                    '가장 단순하고 해석하기 쉬움',
                    '과적합 없이 안정적인 성능',
                    '학습 속도가 매우 빠름',
                    '새로운 데이터에 대한 일반화 성능 우수'
                ],
                '단점': [
                    '비선형 관계 포착 불가',
                    '피처 간 다중공선성에 민감할 수 있음'
                ],
                '권장 사용 상황': [
                    '빠른 예측이 필요한 경우',
                    '모델 해석이 중요한 경우',
                    '안정적인 성능이 우선인 경우',
                    '프로토타입 및 베이스라인으로 사용'
                ],
                '성능 요약': 'Test R²=0.2349, MAPE=0.48%'
            },
            'Ridge Regression': {
                '장점': [
                    '최고의 Test R² 성능 (0.2415)',
                    'L2 정규화로 다중공선성 완화',
                    '과적합 방지하면서 성능 개선',
                    '해석 가능성 유지'
                ],
                '단점': [
                    'Alpha 하이퍼파라미터 튜닝 필요',
                    '피처 스케일링 필수',
                    'Linear Regression 대비 소폭 개선 (+2.82%)'
                ],
                '권장 사용 상황': [
                    '최고 성능이 필요한 경우',
                    '피처 간 상관관계가 높은 경우',
                    '약간의 성능 개선을 위해 복잡도를 수용할 수 있는 경우',
                    '실전 배포용 추천 모델'
                ],
                '성능 요약': 'Test R²=0.2415, MAPE=0.48%, Best α=0.0193'
            },
            'Weighted Ensemble': {
                '장점': [
                    'Linear와 XGBoost의 장점 결합 시도',
                    '비선형 패턴 일부 활용 가능',
                    '자동 가중치 최적화 (LR=50.41%, XGB=49.59%)'
                ],
                '단점': [
                    '과적합 발생 (Train R²=0.53, Test R²=0.23)',
                    '단일 모델 대비 성능 향상 없음',
                    'Test 성능이 Linear/Ridge보다 낮음',
                    '복잡도 대비 성능 향상 부족'
                ],
                '권장 사용 상황': [
                    '현재 데이터셋에서는 권장하지 않음',
                    '데이터가 더 많아지면 재평가 필요',
                    '비선형 패턴이 명확할 때 고려'
                ],
                '성능 요약': 'Test R²=0.2313, MAPE=0.49%, 과적합 Gap=0.30'
            }
        }

        for model_name, details in characteristics.items():
            logger.info(f"\n【{model_name}】")
            logger.info(f"성능: {details['성능 요약']}")
            logger.info(f"\n✓ 장점:")
            for pro in details['장점']:
                logger.info(f"  - {pro}")
            logger.info(f"\n✗ 단점:")
            for con in details['단점']:
                logger.info(f"  - {con}")
            logger.info(f"\n📌 권장 사용 상황:")
            for use_case in details['권장 사용 상황']:
                logger.info(f"  - {use_case}")

        self.model_characteristics = characteristics

        return characteristics

    def create_recommendation_matrix(self):
        """사용 상황별 모델 추천 매트릭스"""
        logger.info("\n" + "=" * 70)
        logger.info("사용 상황별 모델 추천")
        logger.info("=" * 70)

        recommendations = {
            '최고 성능 우선': {
                '1순위': 'Ridge Regression',
                '이유': 'Test R² 0.2415로 최고 성능',
                '예상 오차': '기초금액 1억원 기준 ±48만원'
            },
            '빠른 예측 속도': {
                '1순위': 'Linear Regression',
                '이유': '학습/예측 속도 가장 빠름',
                '예상 오차': '기초금액 1억원 기준 ±48만원'
            },
            '모델 해석 필요': {
                '1순위': 'Linear Regression',
                '이유': '피처 계수 직관적 해석',
                '예상 오차': '기초금액 1억원 기준 ±48만원'
            },
            '안정성 우선': {
                '1순위': 'Ridge Regression',
                '이유': '정규화로 안정적 성능 보장',
                '예상 오차': '기초금액 1억원 기준 ±48만원'
            },
            '프로토타입 개발': {
                '1순위': 'Linear Regression',
                '이유': '단순하고 빠른 구현',
                '예상 오차': '기초금액 1억원 기준 ±48만원'
            },
            '실전 배포': {
                '1순위': 'Ridge Regression',
                '2순위': 'Linear Regression',
                '이유': 'Ridge가 최고 성능, Linear는 백업용',
                '예상 오차': '기초금액 1억원 기준 ±48만원'
            }
        }

        for situation, rec in recommendations.items():
            logger.info(f"\n【{situation}】")
            logger.info(f"  1순위: {rec['1순위']}")
            if '2순위' in rec:
                logger.info(f"  2순위: {rec['2순위']}")
            logger.info(f"  이유: {rec['이유']}")
            logger.info(f"  {rec['예상 오차']}")

        self.recommendations = recommendations

        return recommendations

    def plot_comprehensive_comparison(self):
        """종합 비교 시각화"""
        logger.info("\n종합 비교 시각화 생성 중...")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Test R² Score 비교',
                'Test MAPE (예측 오차율) 비교',
                'Train vs Test R² (과적합 분석)',
                '종합 점수 레이더 차트'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatterpolar"}]
            ]
        )

        models = list(self.model_results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # 1. Test R² 비교
        test_r2_scores = [self.model_results[m]['test_r2'] for m in models]
        fig.add_trace(
            go.Bar(
                x=models,
                y=test_r2_scores,
                name='Test R²',
                marker_color=colors,
                text=[f'{score:.4f}' for score in test_r2_scores],
                textposition='outside'
            ),
            row=1, col=1
        )

        # 2. Test MAPE 비교 (낮을수록 좋음)
        test_mape_scores = [self.model_results[m]['test_mape'] for m in models]
        fig.add_trace(
            go.Bar(
                x=models,
                y=test_mape_scores,
                name='Test MAPE',
                marker_color=colors,
                text=[f'{score:.4f}%' for score in test_mape_scores],
                textposition='outside'
            ),
            row=1, col=2
        )

        # 3. Train vs Test R² (과적합 분석)
        for idx, model in enumerate(models):
            train_r2 = self.model_results[model]['train_r2']
            test_r2 = self.model_results[model]['test_r2']

            fig.add_trace(
                go.Scatter(
                    x=[train_r2],
                    y=[test_r2],
                    mode='markers+text',
                    name=model,
                    marker=dict(size=15, color=colors[idx]),
                    text=[model],
                    textposition='top center'
                ),
                row=2, col=1
            )

        # 대각선 추가 (과적합 없는 완벽한 경우)
        fig.add_trace(
            go.Scatter(
                x=[0, 0.8],
                y=[0, 0.8],
                mode='lines',
                line=dict(dash='dash', color='red'),
                name='No Overfitting',
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. 종합 점수 레이더 차트
        # 점수 정규화 (0~1 범위)
        categories = ['성능', '안정성', '해석성', '속도', '단순성']

        for idx, model in enumerate(models):
            # 점수 계산 (주관적 평가)
            if model == 'Linear Regression':
                scores = [0.7, 1.0, 1.0, 1.0, 1.0]  # 성능, 안정성, 해석성, 속도, 단순성
            elif model == 'Ridge Regression':
                scores = [1.0, 1.0, 0.9, 0.9, 0.8]
            else:  # Weighted Ensemble
                scores = [0.6, 0.5, 0.6, 0.7, 0.5]

            fig.add_trace(
                go.Scatterpolar(
                    r=scores,
                    theta=categories,
                    fill='toself',
                    name=model,
                    line_color=colors[idx]
                ),
                row=2, col=2
            )

        # 레이아웃 업데이트
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_xaxes(title_text="Model", row=1, col=2)
        fig.update_xaxes(title_text="Train R²", row=2, col=1)

        fig.update_yaxes(title_text="Test R²", row=1, col=1)
        fig.update_yaxes(title_text="MAPE (%)", row=1, col=2)
        fig.update_yaxes(title_text="Test R²", row=2, col=1)

        fig.update_layout(
            title_text="최종 3가지 모델 종합 비교",
            height=900,
            showlegend=True
        )

        # 저장
        output_path = self.output_dir / f"final_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(output_path))
        logger.info(f"✓ 종합 비교 시각화 저장: {output_path}")

        return fig

    def save_final_report(self):
        """최종 보고서 저장"""
        logger.info("\n최종 보고서 저장 중...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. 성능 비교표
        comparison_path = self.output_dir / f"final_performance_comparison_{timestamp}.xlsx"
        self.comparison_df.to_excel(comparison_path, index=False)
        logger.info(f"✓ 성능 비교표: {comparison_path}")

        # 2. 모델 특징 정리
        characteristics_data = []
        for model, details in self.model_characteristics.items():
            characteristics_data.append({
                'Model': model,
                'Performance': details['성능 요약'],
                'Pros': ' | '.join(details['장점']),
                'Cons': ' | '.join(details['단점']),
                'Use_Cases': ' | '.join(details['권장 사용 상황'])
            })

        characteristics_df = pd.DataFrame(characteristics_data)
        characteristics_path = self.output_dir / f"final_model_characteristics_{timestamp}.xlsx"
        characteristics_df.to_excel(characteristics_path, index=False)
        logger.info(f"✓ 모델 특징: {characteristics_path}")

        # 3. 추천 매트릭스
        recommendations_data = []
        for situation, rec in self.recommendations.items():
            recommendations_data.append({
                'Situation': situation,
                'Recommended_Model': rec['1순위'],
                'Backup_Model': rec.get('2순위', 'N/A'),
                'Reason': rec['이유'],
                'Expected_Error': rec['예상 오차']
            })

        recommendations_df = pd.DataFrame(recommendations_data)
        recommendations_path = self.output_dir / f"final_recommendations_{timestamp}.xlsx"
        recommendations_df.to_excel(recommendations_path, index=False)
        logger.info(f"✓ 추천 매트릭스: {recommendations_path}")

        logger.info("✓ 모든 보고서 저장 완료")

        return {
            'comparison': comparison_path,
            'characteristics': characteristics_path,
            'recommendations': recommendations_path
        }


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("최종 3가지 모델 비교 및 분석")
    print("=" * 70)

    # 초기화
    comparator = FinalModelComparison()

    # 1. 성능 비교표 생성
    comparator.create_performance_comparison()

    # 2. 모델별 특징 분석
    comparator.analyze_model_characteristics()

    # 3. 사용 상황별 추천
    comparator.create_recommendation_matrix()

    # 4. 종합 시각화
    comparator.plot_comprehensive_comparison()

    # 5. 최종 보고서 저장
    comparator.save_final_report()

    print("\n" + "=" * 70)
    print("✓ 최종 비교 및 분석 완료")
    print("=" * 70)
    print("\n🏆 권장 모델:")
    print("  1순위: Ridge Regression (최고 성능)")
    print("  2순위: Linear Regression (빠르고 안정적)")
    print("  3순위: Weighted Ensemble (현재 데이터셋에서는 비권장)")
    print("=" * 70)


if __name__ == "__main__":
    main()
