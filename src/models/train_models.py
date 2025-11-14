"""
낙찰하한가 예측 모델 학습 및 비교
- Linear Regression
- XGBoost
- LightGBM
- Random Forest

4가지 모델을 학습하고 성능을 비교하여 최적의 모델을 선정합니다.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 모델 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

# 시각화
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BidPredictionModel:
    """낙찰하한가 예측 모델 학습 및 평가 클래스"""

    def __init__(self, data_path: str = "data/processed/features_engineered.xlsx",
                 output_dir: str = "models/trained"):
        """
        초기화

        Args:
            data_path: 피처 엔지니어링된 데이터 경로
            output_dir: 모델 저장 디렉토리
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_col = '예가/기초(100%)'
        self.models = {}
        self.results = {}
        self.feature_importances = {}

        # 제외할 컬럼 (타겟 관련 및 식별자)
        self.exclude_cols = [
            self.target_col,
            '예가/기초(0%)',
            '1순위사정율(0%)',
            '1순위사정율(100%)',
            '1순위기초대비',
            '입찰공고번호',
            '개찰일',
            '1순위업체',
            '1순위사업자번호',
            '개찰일_dt'  # datetime 컬럼
        ]

        logger.info("BidPredictionModel 초기화 완료")

    def load_and_prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        데이터 로드 및 전처리

        Args:
            test_size: 테스트 데이터 비율
            random_state: 랜덤 시드
        """
        logger.info("=" * 70)
        logger.info("데이터 로드 및 전처리 시작")
        logger.info("=" * 70)

        # 데이터 로드
        df = pd.read_excel(self.data_path)
        logger.info(f"원본 데이터: {df.shape}")

        # 타겟 변수 결측치 제거
        df_clean = df.dropna(subset=[self.target_col])
        logger.info(f"타겟 결측치 제거 후: {df_clean.shape} (제거: {len(df) - len(df_clean)}개)")

        # 피처와 타겟 분리
        feature_cols = [col for col in df_clean.columns if col not in self.exclude_cols]
        X = df_clean[feature_cols].copy()
        y = df_clean[self.target_col].copy()

        logger.info(f"\n초기 피처 수: {len(feature_cols)}")

        # 범주형 변수 식별 및 인코딩
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.info(f"범주형 변수: {len(categorical_cols)}개")
        for col in categorical_cols:
            logger.info(f"  - {col}: {X[col].nunique()}개 고유값")

        # Label Encoding (트리 기반 모델은 이것으로 충분)
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            # 결측치를 'Unknown'으로 채움
            X[col] = X[col].fillna('Unknown')
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        self.label_encoders = label_encoders
        logger.info(f"범주형 인코딩 완료")

        # 수치형 결측치 처리 (중앙값으로 채움)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        missing_before = X[numeric_cols].isnull().sum().sum()
        if missing_before > 0:
            for col in numeric_cols:
                if X[col].isnull().sum() > 0:
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)
                    logger.info(f"  - {col}: {X[col].isnull().sum()}개 결측치를 {median_val:.4f}로 채움")

        missing_after = X.isnull().sum().sum()
        logger.info(f"수치형 결측치 처리: {missing_before}개 → {missing_after}개")

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(f"\n데이터 분할 (80/20):")
        logger.info(f"  - Train: {X_train.shape}")
        logger.info(f"  - Test:  {X_test.shape}")
        logger.info(f"  - 피처 수: {X_train.shape[1]}")

        # 저장
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = X_train.columns.tolist()

        logger.info("=" * 70)
        logger.info("✓ 데이터 준비 완료")
        logger.info("=" * 70)

        return X_train, X_test, y_train, y_test

    def train_linear_regression(self):
        """Linear Regression 모델 학습"""
        logger.info("\n" + "=" * 70)
        logger.info("Linear Regression 학습 시작")
        logger.info("=" * 70)

        model = LinearRegression()
        model.fit(self.X_train, self.y_train)

        # 예측
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # 평가
        train_metrics = self._evaluate_predictions(self.y_train, y_train_pred, "Train")
        test_metrics = self._evaluate_predictions(self.y_test, y_test_pred, "Test")

        # 피처 중요도 (계수의 절대값)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(model.coef_)
        }).sort_values('importance', ascending=False)

        # 저장
        self.models['Linear Regression'] = model
        self.results['Linear Regression'] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            }
        }
        self.feature_importances['Linear Regression'] = feature_importance

        logger.info("✓ Linear Regression 학습 완료")

        return model, test_metrics

    def train_xgboost(self):
        """XGBoost 모델 학습"""
        logger.info("\n" + "=" * 70)
        logger.info("XGBoost 학습 시작")
        logger.info("=" * 70)

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)

        # 예측
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # 평가
        train_metrics = self._evaluate_predictions(self.y_train, y_train_pred, "Train")
        test_metrics = self._evaluate_predictions(self.y_test, y_test_pred, "Test")

        # 피처 중요도
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 저장
        self.models['XGBoost'] = model
        self.results['XGBoost'] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            }
        }
        self.feature_importances['XGBoost'] = feature_importance

        logger.info("✓ XGBoost 학습 완료")

        return model, test_metrics

    def train_lightgbm(self):
        """LightGBM 모델 학습"""
        logger.info("\n" + "=" * 70)
        logger.info("LightGBM 학습 시작")
        logger.info("=" * 70)

        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(self.X_train, self.y_train)

        # 예측
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # 평가
        train_metrics = self._evaluate_predictions(self.y_train, y_train_pred, "Train")
        test_metrics = self._evaluate_predictions(self.y_test, y_test_pred, "Test")

        # 피처 중요도
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 저장
        self.models['LightGBM'] = model
        self.results['LightGBM'] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            }
        }
        self.feature_importances['LightGBM'] = feature_importance

        logger.info("✓ LightGBM 학습 완료")

        return model, test_metrics

    def train_random_forest(self):
        """Random Forest 모델 학습"""
        logger.info("\n" + "=" * 70)
        logger.info("Random Forest 학습 시작")
        logger.info("=" * 70)

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)

        # 예측
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # 평가
        train_metrics = self._evaluate_predictions(self.y_train, y_train_pred, "Train")
        test_metrics = self._evaluate_predictions(self.y_test, y_test_pred, "Test")

        # 피처 중요도
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 저장
        self.models['Random Forest'] = model
        self.results['Random Forest'] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            }
        }
        self.feature_importances['Random Forest'] = feature_importance

        logger.info("✓ Random Forest 학습 완료")

        return model, test_metrics

    def _evaluate_predictions(self, y_true, y_pred, dataset_name: str = ""):
        """
        예측 결과 평가

        Args:
            y_true: 실제 값
            y_pred: 예측 값
            dataset_name: 데이터셋 이름 (Train/Test)

        Returns:
            dict: 평가 지표
        """
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # 예가 예측 오차율 (%)
        # 예가 = 예가/기초(100%) * 기초금액 / 100
        # 실제로는 타겟이 이미 %이므로 직접 계산
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        metrics = {
            'R²': r2,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape  # Mean Absolute Percentage Error
        }

        if dataset_name:
            logger.info(f"\n{dataset_name} 평가 결과:")
            logger.info(f"  - R² Score:  {r2:.6f}")
            logger.info(f"  - MAE:       {mae:.6f}")
            logger.info(f"  - RMSE:      {rmse:.6f}")
            logger.info(f"  - MAPE:      {mape:.4f}%")

        return metrics

    def compare_models(self):
        """모든 모델 성능 비교"""
        logger.info("\n" + "=" * 70)
        logger.info("모델 성능 비교")
        logger.info("=" * 70)

        # 비교 테이블 생성
        comparison = []
        for model_name, results in self.results.items():
            comparison.append({
                'Model': model_name,
                'Train_R2': results['train']['R²'],
                'Test_R2': results['test']['R²'],
                'Train_MAE': results['train']['MAE'],
                'Test_MAE': results['test']['MAE'],
                'Train_RMSE': results['train']['RMSE'],
                'Test_RMSE': results['test']['RMSE'],
                'Test_MAPE': results['test']['MAPE']
            })

        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('Test_R2', ascending=False)

        logger.info("\n성능 비교표:")
        logger.info(df_comparison.to_string(index=False))

        # 최적 모델 선정
        best_model_name = df_comparison.iloc[0]['Model']
        best_r2 = df_comparison.iloc[0]['Test_R2']

        logger.info("\n" + "=" * 70)
        logger.info(f"🏆 최적 모델: {best_model_name}")
        logger.info(f"   Test R² Score: {best_r2:.6f}")
        logger.info("=" * 70)

        self.best_model_name = best_model_name
        self.comparison_df = df_comparison

        return df_comparison, best_model_name

    def plot_comparison(self):
        """모델 비교 시각화"""
        logger.info("\n시각화 생성 중...")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'R² Score 비교',
                'MAE 비교',
                'RMSE 비교',
                '실제 vs 예측 (최적 모델)'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )

        models = list(self.results.keys())

        # 1. R² Score 비교
        train_r2 = [self.results[m]['train']['R²'] for m in models]
        test_r2 = [self.results[m]['test']['R²'] for m in models]

        fig.add_trace(
            go.Bar(name='Train R²', x=models, y=train_r2, marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Test R²', x=models, y=test_r2, marker_color='darkblue'),
            row=1, col=1
        )

        # 2. MAE 비교
        train_mae = [self.results[m]['train']['MAE'] for m in models]
        test_mae = [self.results[m]['test']['MAE'] for m in models]

        fig.add_trace(
            go.Bar(name='Train MAE', x=models, y=train_mae, marker_color='lightcoral',
                   showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='Test MAE', x=models, y=test_mae, marker_color='darkred',
                   showlegend=False),
            row=1, col=2
        )

        # 3. RMSE 비교
        train_rmse = [self.results[m]['train']['RMSE'] for m in models]
        test_rmse = [self.results[m]['test']['RMSE'] for m in models]

        fig.add_trace(
            go.Bar(name='Train RMSE', x=models, y=train_rmse, marker_color='lightgreen',
                   showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name='Test RMSE', x=models, y=test_rmse, marker_color='darkgreen',
                   showlegend=False),
            row=2, col=1
        )

        # 4. 실제 vs 예측 (최적 모델)
        best_model = self.best_model_name
        y_test_pred = self.results[best_model]['predictions']['test']

        fig.add_trace(
            go.Scatter(
                x=self.y_test,
                y=y_test_pred,
                mode='markers',
                marker=dict(size=5, color='purple', opacity=0.6),
                name='Predictions',
                showlegend=False
            ),
            row=2, col=2
        )

        # 대각선 추가 (완벽한 예측)
        min_val = min(self.y_test.min(), y_test_pred.min())
        max_val = max(self.y_test.max(), y_test_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect',
                showlegend=False
            ),
            row=2, col=2
        )

        # 레이아웃 업데이트
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_xaxes(title_text="Model", row=1, col=2)
        fig.update_xaxes(title_text="Model", row=2, col=1)
        fig.update_xaxes(title_text="Actual", row=2, col=2)

        fig.update_yaxes(title_text="R² Score", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=1, col=2)
        fig.update_yaxes(title_text="RMSE", row=2, col=1)
        fig.update_yaxes(title_text="Predicted", row=2, col=2)

        fig.update_layout(
            title_text=f"모델 성능 비교 - 최적 모델: {best_model}",
            height=800,
            showlegend=True
        )

        # 저장
        output_path = self.output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(output_path))
        logger.info(f"✓ 시각화 저장: {output_path}")

        return fig

    def plot_feature_importance(self, top_n: int = 20):
        """피처 중요도 시각화"""
        logger.info(f"\n상위 {top_n}개 피처 중요도 시각화 중...")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"{model} - Top {top_n} Features"
                          for model in self.feature_importances.keys()]
        )

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for idx, (model_name, importance_df) in enumerate(self.feature_importances.items()):
            row, col = positions[idx]
            top_features = importance_df.head(top_n)

            fig.add_trace(
                go.Bar(
                    x=top_features['importance'],
                    y=top_features['feature'],
                    orientation='h',
                    name=model_name,
                    showlegend=False
                ),
                row=row, col=col
            )

            fig.update_xaxes(title_text="Importance", row=row, col=col)
            fig.update_yaxes(title_text="Feature", row=row, col=col)

        fig.update_layout(
            title_text=f"피처 중요도 비교 (Top {top_n})",
            height=1000
        )

        # 저장
        output_path = self.output_dir / f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(output_path))
        logger.info(f"✓ 피처 중요도 시각화 저장: {output_path}")

        return fig

    def save_results(self):
        """결과 저장"""
        logger.info("\n결과 저장 중...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. 모델 비교표
        comparison_path = self.output_dir / f"model_comparison_{timestamp}.xlsx"
        self.comparison_df.to_excel(comparison_path, index=False)
        logger.info(f"✓ 모델 비교표 저장: {comparison_path}")

        # 2. 각 모델의 피처 중요도
        with pd.ExcelWriter(self.output_dir / f"feature_importance_{timestamp}.xlsx") as writer:
            for model_name, importance_df in self.feature_importances.items():
                importance_df.to_excel(writer, sheet_name=model_name, index=False)
        logger.info(f"✓ 피처 중요도 저장: {self.output_dir / f'feature_importance_{timestamp}.xlsx'}")

        # 3. 최적 모델 정보
        best_model_info = {
            'best_model': self.best_model_name,
            'test_r2': self.comparison_df.iloc[0]['Test_R2'],
            'test_mae': self.comparison_df.iloc[0]['Test_MAE'],
            'test_rmse': self.comparison_df.iloc[0]['Test_RMSE'],
            'test_mape': self.comparison_df.iloc[0]['Test_MAPE'],
            'timestamp': timestamp
        }

        info_df = pd.DataFrame([best_model_info])
        info_path = self.output_dir / f"best_model_info_{timestamp}.xlsx"
        info_df.to_excel(info_path, index=False)
        logger.info(f"✓ 최적 모델 정보 저장: {info_path}")

        logger.info("✓ 모든 결과 저장 완료")


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("낙찰하한가 예측 모델 학습")
    print("=" * 70)

    # 모델 초기화
    model_trainer = BidPredictionModel()

    # 1. 데이터 로드 및 전처리 (80/20 분할)
    model_trainer.load_and_prepare_data(test_size=0.2, random_state=42)

    # 2. 모델 학습
    model_trainer.train_linear_regression()
    model_trainer.train_xgboost()
    model_trainer.train_lightgbm()
    model_trainer.train_random_forest()

    # 3. 모델 비교
    comparison_df, best_model = model_trainer.compare_models()

    # 4. 시각화
    model_trainer.plot_comparison()
    model_trainer.plot_feature_importance(top_n=20)

    # 5. 결과 저장
    model_trainer.save_results()

    print("\n" + "=" * 70)
    print("✓ 모든 작업 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
