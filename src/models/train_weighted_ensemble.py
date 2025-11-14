"""
Weighted Ensemble 모델 개발
- Linear Regression + XGBoost 가중 평균
- 최적 가중치 자동 탐색
- 과적합 방지하면서 비선형 패턴 활용
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from scipy.optimize import minimize
import joblib

# 시각화
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeightedEnsemble:
    """Weighted Ensemble 모델 클래스"""

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

        # 제외할 컬럼
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
            '개찰일_dt'
        ]

        logger.info("WeightedEnsemble 초기화 완료")

    def load_and_prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        """데이터 로드 및 전처리"""
        logger.info("=" * 70)
        logger.info("데이터 로드 및 전처리")
        logger.info("=" * 70)

        # 데이터 로드
        df = pd.read_excel(self.data_path)
        logger.info(f"원본 데이터: {df.shape}")

        # 타겟 변수 결측치 제거
        df_clean = df.dropna(subset=[self.target_col])
        logger.info(f"타겟 결측치 제거 후: {df_clean.shape}")

        # 피처와 타겟 분리
        feature_cols = [col for col in df_clean.columns if col not in self.exclude_cols]
        X = df_clean[feature_cols].copy()
        y = df_clean[self.target_col].copy()

        logger.info(f"초기 피처 수: {len(feature_cols)}")

        # 범주형 변수 인코딩
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        logger.info(f"범주형 변수: {len(categorical_cols)}개")

        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = X[col].fillna('Unknown')
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        self.label_encoders = label_encoders

        # 수치형 결측치 처리
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)

        # Train/Test Split (검증 세트 추가)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Train을 다시 Train/Validation으로 분할 (가중치 최적화용)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=random_state
        )

        logger.info(f"데이터 분할:")
        logger.info(f"  - Train:      {X_train.shape}")
        logger.info(f"  - Validation: {X_val.shape}")
        logger.info(f"  - Test:       {X_test.shape}")

        # 저장
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.feature_names = X_train.columns.tolist()

        logger.info("✓ 데이터 준비 완료")
        logger.info("=" * 70)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_base_models(self):
        """베이스 모델 학습 (Linear Regression + XGBoost)"""
        logger.info("\n" + "=" * 70)
        logger.info("베이스 모델 학습")
        logger.info("=" * 70)

        # 1. Linear Regression
        logger.info("\n1. Linear Regression 학습...")
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)

        # 예측
        lr_train_pred = lr_model.predict(self.X_train)
        lr_val_pred = lr_model.predict(self.X_val)
        lr_test_pred = lr_model.predict(self.X_test)

        # 평가
        lr_train_r2 = r2_score(self.y_train, lr_train_pred)
        lr_val_r2 = r2_score(self.y_val, lr_val_pred)
        lr_test_r2 = r2_score(self.y_test, lr_test_pred)

        logger.info(f"   Train R²: {lr_train_r2:.6f}")
        logger.info(f"   Val R²:   {lr_val_r2:.6f}")
        logger.info(f"   Test R²:  {lr_test_r2:.6f}")

        # 2. XGBoost
        logger.info("\n2. XGBoost 학습...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(self.X_train, self.y_train)

        # 예측
        xgb_train_pred = xgb_model.predict(self.X_train)
        xgb_val_pred = xgb_model.predict(self.X_val)
        xgb_test_pred = xgb_model.predict(self.X_test)

        # 평가
        xgb_train_r2 = r2_score(self.y_train, xgb_train_pred)
        xgb_val_r2 = r2_score(self.y_val, xgb_val_pred)
        xgb_test_r2 = r2_score(self.y_test, xgb_test_pred)

        logger.info(f"   Train R²: {xgb_train_r2:.6f}")
        logger.info(f"   Val R²:   {xgb_val_r2:.6f}")
        logger.info(f"   Test R²:  {xgb_test_r2:.6f}")

        # 저장
        self.lr_model = lr_model
        self.xgb_model = xgb_model

        self.lr_preds = {
            'train': lr_train_pred,
            'val': lr_val_pred,
            'test': lr_test_pred
        }

        self.xgb_preds = {
            'train': xgb_train_pred,
            'val': xgb_val_pred,
            'test': xgb_test_pred
        }

        self.base_model_scores = {
            'Linear Regression': {
                'train_r2': lr_train_r2,
                'val_r2': lr_val_r2,
                'test_r2': lr_test_r2
            },
            'XGBoost': {
                'train_r2': xgb_train_r2,
                'val_r2': xgb_val_r2,
                'test_r2': xgb_test_r2
            }
        }

        logger.info("✓ 베이스 모델 학습 완료")

        return lr_model, xgb_model

    def optimize_weights(self):
        """
        최적 가중치 탐색
        - Validation 세트에서 R² 최대화
        - 제약조건: w_lr + w_xgb = 1, 0 <= w <= 1
        """
        logger.info("\n" + "=" * 70)
        logger.info("가중치 최적화")
        logger.info("=" * 70)

        # 목적 함수: Validation R² 최대화 = -R² 최소화
        def objective(weights):
            w_lr, w_xgb = weights
            ensemble_pred = w_lr * self.lr_preds['val'] + w_xgb * self.xgb_preds['val']
            r2 = r2_score(self.y_val, ensemble_pred)
            return -r2  # 최소화 문제로 변환

        # 초기 가중치 (균등)
        initial_weights = [0.5, 0.5]

        # 제약조건
        constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1}  # 합 = 1
        bounds = [(0, 1), (0, 1)]  # 0 <= w <= 1

        logger.info("scipy.optimize.minimize 실행 중...")
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            optimal_weights = result.x
            optimal_lr_weight = optimal_weights[0]
            optimal_xgb_weight = optimal_weights[1]
            optimal_val_r2 = -result.fun

            logger.info(f"\n✓ 최적화 성공")
            logger.info(f"   Linear Regression 가중치: {optimal_lr_weight:.4f}")
            logger.info(f"   XGBoost 가중치:          {optimal_xgb_weight:.4f}")
            logger.info(f"   Validation R²:            {optimal_val_r2:.6f}")

            self.optimal_weights = {
                'lr': optimal_lr_weight,
                'xgb': optimal_xgb_weight
            }
            self.optimal_val_r2 = optimal_val_r2

            return optimal_weights, optimal_val_r2
        else:
            logger.error("✗ 최적화 실패")
            logger.info("균등 가중치 (0.5, 0.5) 사용")
            self.optimal_weights = {'lr': 0.5, 'xgb': 0.5}
            return [0.5, 0.5], None

    def evaluate_ensemble(self):
        """앙상블 모델 평가"""
        logger.info("\n" + "=" * 70)
        logger.info("Weighted Ensemble 평가")
        logger.info("=" * 70)

        w_lr = self.optimal_weights['lr']
        w_xgb = self.optimal_weights['xgb']

        # 앙상블 예측
        ensemble_train_pred = w_lr * self.lr_preds['train'] + w_xgb * self.xgb_preds['train']
        ensemble_val_pred = w_lr * self.lr_preds['val'] + w_xgb * self.xgb_preds['val']
        ensemble_test_pred = w_lr * self.lr_preds['test'] + w_xgb * self.xgb_preds['test']

        # 평가
        train_metrics = self._evaluate_predictions(self.y_train, ensemble_train_pred, "Train")
        val_metrics = self._evaluate_predictions(self.y_val, ensemble_val_pred, "Validation")
        test_metrics = self._evaluate_predictions(self.y_test, ensemble_test_pred, "Test")

        # 저장
        self.ensemble_preds = {
            'train': ensemble_train_pred,
            'val': ensemble_val_pred,
            'test': ensemble_test_pred
        }

        self.ensemble_metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }

        logger.info("✓ Weighted Ensemble 평가 완료")

        return test_metrics

    def _evaluate_predictions(self, y_true, y_pred, dataset_name: str = ""):
        """예측 결과 평가"""
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        metrics = {
            'R²': r2,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

        if dataset_name:
            logger.info(f"\n{dataset_name} 평가 결과:")
            logger.info(f"  - R² Score:  {r2:.6f}")
            logger.info(f"  - MAE:       {mae:.6f}")
            logger.info(f"  - RMSE:      {rmse:.6f}")
            logger.info(f"  - MAPE:      {mape:.4f}%")

        return metrics

    def compare_models(self):
        """모든 모델 비교"""
        logger.info("\n" + "=" * 70)
        logger.info("모델 비교")
        logger.info("=" * 70)

        comparison = []

        # Linear Regression
        comparison.append({
            'Model': 'Linear Regression',
            'Train_R2': self.base_model_scores['Linear Regression']['train_r2'],
            'Val_R2': self.base_model_scores['Linear Regression']['val_r2'],
            'Test_R2': self.base_model_scores['Linear Regression']['test_r2']
        })

        # XGBoost
        comparison.append({
            'Model': 'XGBoost',
            'Train_R2': self.base_model_scores['XGBoost']['train_r2'],
            'Val_R2': self.base_model_scores['XGBoost']['val_r2'],
            'Test_R2': self.base_model_scores['XGBoost']['test_r2']
        })

        # Weighted Ensemble
        comparison.append({
            'Model': 'Weighted Ensemble',
            'Train_R2': self.ensemble_metrics['train']['R²'],
            'Val_R2': self.ensemble_metrics['val']['R²'],
            'Test_R2': self.ensemble_metrics['test']['R²']
        })

        df_comparison = pd.DataFrame(comparison)

        logger.info("\n성능 비교:")
        logger.info(df_comparison.to_string(index=False))

        # 최적 모델
        best_idx = df_comparison['Test_R2'].idxmax()
        best_model = df_comparison.iloc[best_idx]['Model']
        best_r2 = df_comparison.iloc[best_idx]['Test_R2']

        logger.info(f"\n🏆 Test 세트 최고 성능: {best_model} (R²={best_r2:.6f})")

        self.comparison_df = df_comparison

        return df_comparison

    def plot_weight_search(self):
        """가중치 탐색 시각화"""
        logger.info("\n가중치 탐색 시각화 중...")

        # 0~1 범위의 가중치 조합 시도
        lr_weights = np.linspace(0, 1, 101)
        xgb_weights = 1 - lr_weights

        val_r2_scores = []
        test_r2_scores = []

        for w_lr, w_xgb in zip(lr_weights, xgb_weights):
            val_pred = w_lr * self.lr_preds['val'] + w_xgb * self.xgb_preds['val']
            test_pred = w_lr * self.lr_preds['test'] + w_xgb * self.xgb_preds['test']

            val_r2 = r2_score(self.y_val, val_pred)
            test_r2 = r2_score(self.y_test, test_pred)

            val_r2_scores.append(val_r2)
            test_r2_scores.append(test_r2)

        # 시각화
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=lr_weights,
                y=val_r2_scores,
                mode='lines',
                name='Validation R²',
                line=dict(color='blue', width=2)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=lr_weights,
                y=test_r2_scores,
                mode='lines',
                name='Test R²',
                line=dict(color='green', width=2)
            )
        )

        # 최적 가중치 표시
        optimal_lr = self.optimal_weights['lr']
        fig.add_vline(
            x=optimal_lr,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Optimal (LR={optimal_lr:.2f}, XGB={1-optimal_lr:.2f})"
        )

        fig.update_layout(
            title="가중치에 따른 앙상블 성능",
            xaxis_title="Linear Regression 가중치",
            yaxis_title="R² Score",
            height=500,
            hovermode='x unified'
        )

        # 저장
        output_path = self.output_dir / f"ensemble_weight_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(output_path))
        logger.info(f"✓ 가중치 탐색 시각화 저장: {output_path}")

        return fig

    def plot_model_comparison(self):
        """모델 비교 시각화"""
        logger.info("\n모델 비교 시각화 중...")

        fig = go.Figure()

        models = self.comparison_df['Model'].tolist()

        fig.add_trace(
            go.Bar(
                name='Train R²',
                x=models,
                y=self.comparison_df['Train_R2'],
                marker_color='lightblue'
            )
        )

        fig.add_trace(
            go.Bar(
                name='Validation R²',
                x=models,
                y=self.comparison_df['Val_R2'],
                marker_color='orange'
            )
        )

        fig.add_trace(
            go.Bar(
                name='Test R²',
                x=models,
                y=self.comparison_df['Test_R2'],
                marker_color='darkblue'
            )
        )

        fig.update_layout(
            title="모델 성능 비교 (R² Score)",
            xaxis_title="Model",
            yaxis_title="R² Score",
            barmode='group',
            height=500
        )

        # 저장
        output_path = self.output_dir / f"ensemble_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(output_path))
        logger.info(f"✓ 모델 비교 시각화 저장: {output_path}")

        return fig

    def save_results(self):
        """결과 저장"""
        logger.info("\n결과 저장 중...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. 앙상블 모델 정보
        ensemble_info = {
            'model': 'Weighted Ensemble (LR + XGBoost)',
            'lr_weight': self.optimal_weights['lr'],
            'xgb_weight': self.optimal_weights['xgb'],
            'val_r2': self.ensemble_metrics['val']['R²'],
            'test_r2': self.ensemble_metrics['test']['R²'],
            'test_mae': self.ensemble_metrics['test']['MAE'],
            'test_rmse': self.ensemble_metrics['test']['RMSE'],
            'test_mape': self.ensemble_metrics['test']['MAPE'],
            'timestamp': timestamp
        }

        info_df = pd.DataFrame([ensemble_info])
        info_path = self.output_dir / f"ensemble_model_info_{timestamp}.xlsx"
        info_df.to_excel(info_path, index=False)
        logger.info(f"✓ 앙상블 정보 저장: {info_path}")

        # 2. 모델 비교표
        comparison_path = self.output_dir / f"ensemble_comparison_{timestamp}.xlsx"
        self.comparison_df.to_excel(comparison_path, index=False)
        logger.info(f"✓ 모델 비교표 저장: {comparison_path}")

        # 3. 모델 객체 저장
        model_path = self.output_dir / f"ensemble_model_{timestamp}.pkl"
        joblib.dump({
            'lr_model': self.lr_model,
            'xgb_model': self.xgb_model,
            'weights': self.optimal_weights,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, model_path)
        logger.info(f"✓ 앙상블 모델 저장: {model_path}")

        logger.info("✓ 모든 결과 저장 완료")

        return {
            'info': info_path,
            'comparison': comparison_path,
            'model': model_path
        }


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("Weighted Ensemble 모델 개발")
    print("=" * 70)

    # 초기화
    ensemble = WeightedEnsemble()

    # 1. 데이터 준비
    ensemble.load_and_prepare_data(test_size=0.2, random_state=42)

    # 2. 베이스 모델 학습
    ensemble.train_base_models()

    # 3. 가중치 최적화
    ensemble.optimize_weights()

    # 4. 앙상블 평가
    ensemble.evaluate_ensemble()

    # 5. 모델 비교
    ensemble.compare_models()

    # 6. 시각화
    ensemble.plot_weight_search()
    ensemble.plot_model_comparison()

    # 7. 결과 저장
    ensemble.save_results()

    print("\n" + "=" * 70)
    print("✓ Weighted Ensemble 개발 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
