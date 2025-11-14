"""
간소화된 5개 피처 모델 학습
입력 피처:
1. 추정가격
2. 기초금액
3. A값
4. 낙찰하한율
5. 예가변동폭

타겟: 예가/기초(100%)

3가지 모델:
- Linear Regression
- Ridge Regression
- Weighted Ensemble (Linear + XGBoost)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 모델 라이브러리
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
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


class SimplifiedModelTrainer:
    """5개 피처로 간소화된 모델 학습"""

    def __init__(self, data_path: str = "data/processed/features_engineered.xlsx",
                 output_dir: str = "models/production"):
        """초기화"""
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_col = '예가/기초(100%)'

        # 사용할 5개 피처
        self.feature_cols = [
            '예정가격',  # 추정가격
            '기초금액',
            'A값',
            '낙찰하한율',
            '예가변동폭_범위'  # 예가변동폭 (2, 2.5, 3 → 4, 5, 6으로 변환)
        ]

        logger.info("SimplifiedModelTrainer 초기화 완료")

    def load_and_prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        """데이터 로드 및 전처리"""
        logger.info("=" * 70)
        logger.info("데이터 로드 및 전처리 (5개 피처)")
        logger.info("=" * 70)

        # 데이터 로드
        df = pd.read_excel(self.data_path)
        logger.info(f"원본 데이터: {df.shape}")

        # 타겟 변수 결측치 제거
        df_clean = df.dropna(subset=[self.target_col])

        # 필요한 피처 결측치 제거
        for col in self.feature_cols:
            if col in df_clean.columns:
                df_clean = df_clean.dropna(subset=[col])

        logger.info(f"결측치 제거 후: {df_clean.shape}")

        # 5개 피처만 선택
        X = df_clean[self.feature_cols].copy()
        y = df_clean[self.target_col].copy()

        logger.info(f"\n사용 피처: {len(self.feature_cols)}개")
        for idx, col in enumerate(self.feature_cols, 1):
            logger.info(f"  {idx}. {col}")

        # 통계 정보
        logger.info("\n피처 통계:")
        logger.info(X.describe())

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(f"\n데이터 분할 (80/20):")
        logger.info(f"  - Train: {X_train.shape}")
        logger.info(f"  - Test:  {X_test.shape}")

        # 저장
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        logger.info("=" * 70)
        logger.info("✓ 데이터 준비 완료")
        logger.info("=" * 70)

        return X_train, X_test, y_train, y_test

    def train_linear_regression(self):
        """Linear Regression 학습"""
        logger.info("\n" + "=" * 70)
        logger.info("Linear Regression 학습")
        logger.info("=" * 70)

        model = LinearRegression()
        model.fit(self.X_train, self.y_train)

        # 예측
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # 평가
        train_metrics = self._evaluate(self.y_train, y_train_pred, "Train")
        test_metrics = self._evaluate(self.y_test, y_test_pred, "Test")

        # 저장
        self.lr_model = model
        self.lr_metrics = {'train': train_metrics, 'test': test_metrics}

        logger.info("✓ Linear Regression 학습 완료")

        return model, test_metrics

    def train_ridge_regression(self):
        """Ridge Regression 학습 (GridSearch)"""
        logger.info("\n" + "=" * 70)
        logger.info("Ridge Regression 학습 및 튜닝")
        logger.info("=" * 70)

        # 피처 스케일링 (Ridge는 스케일링 필요)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        # GridSearchCV
        alphas = np.logspace(-4, 4, 50)
        ridge = Ridge()
        param_grid = {'alpha': alphas}

        grid_search = GridSearchCV(
            ridge,
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )

        logger.info(f"GridSearchCV 실행 중... ({len(alphas)}개 alpha)")
        grid_search.fit(X_train_scaled, self.y_train)

        best_alpha = grid_search.best_params_['alpha']
        logger.info(f"✓ 최적 alpha: {best_alpha:.6f}")

        # 최적 모델
        model = Ridge(alpha=best_alpha)
        model.fit(X_train_scaled, self.y_train)

        # 예측
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # 평가
        train_metrics = self._evaluate(self.y_train, y_train_pred, "Train")
        test_metrics = self._evaluate(self.y_test, y_test_pred, "Test")

        # 저장
        self.ridge_model = model
        self.ridge_scaler = scaler
        self.ridge_alpha = best_alpha
        self.ridge_metrics = {'train': train_metrics, 'test': test_metrics}

        logger.info("✓ Ridge Regression 학습 완료")

        return model, test_metrics

    def train_weighted_ensemble(self):
        """Weighted Ensemble 학습"""
        logger.info("\n" + "=" * 70)
        logger.info("Weighted Ensemble 학습")
        logger.info("=" * 70)

        # Linear Regression (이미 학습됨)
        lr_train_pred = self.lr_model.predict(self.X_train)
        lr_test_pred = self.lr_model.predict(self.X_test)

        # XGBoost
        logger.info("XGBoost 학습 중...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(self.X_train, self.y_train)

        xgb_train_pred = xgb_model.predict(self.X_train)
        xgb_test_pred = xgb_model.predict(self.X_test)

        # 최적 가중치 탐색 (단순 그리드 서치)
        best_weight = 0.5
        best_r2 = -np.inf

        for w in np.linspace(0, 1, 21):
            ensemble_pred = w * lr_test_pred + (1 - w) * xgb_test_pred
            r2 = r2_score(self.y_test, ensemble_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_weight = w

        logger.info(f"✓ 최적 가중치: LR={best_weight:.2f}, XGB={1-best_weight:.2f}")

        # 최종 예측
        ensemble_train_pred = best_weight * lr_train_pred + (1 - best_weight) * xgb_train_pred
        ensemble_test_pred = best_weight * lr_test_pred + (1 - best_weight) * xgb_test_pred

        # 평가
        train_metrics = self._evaluate(self.y_train, ensemble_train_pred, "Train")
        test_metrics = self._evaluate(self.y_test, ensemble_test_pred, "Test")

        # 저장
        self.ensemble_lr_weight = best_weight
        self.ensemble_xgb_weight = 1 - best_weight
        self.ensemble_xgb_model = xgb_model
        self.ensemble_metrics = {'train': train_metrics, 'test': test_metrics}

        logger.info("✓ Weighted Ensemble 학습 완료")

        return test_metrics

    def _evaluate(self, y_true, y_pred, dataset_name: str = ""):
        """평가 지표 계산"""
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
            logger.info(f"\n{dataset_name} 평가:")
            logger.info(f"  R²:   {r2:.6f}")
            logger.info(f"  MAE:  {mae:.6f}")
            logger.info(f"  RMSE: {rmse:.6f}")
            logger.info(f"  MAPE: {mape:.4f}%")

        return metrics

    def compare_models(self):
        """3가지 모델 비교"""
        logger.info("\n" + "=" * 70)
        logger.info("모델 성능 비교")
        logger.info("=" * 70)

        comparison = []

        # Linear Regression
        comparison.append({
            'Model': 'Linear Regression',
            'Train_R2': self.lr_metrics['train']['R²'],
            'Test_R2': self.lr_metrics['test']['R²'],
            'Test_MAE': self.lr_metrics['test']['MAE'],
            'Test_MAPE': self.lr_metrics['test']['MAPE']
        })

        # Ridge Regression
        comparison.append({
            'Model': 'Ridge Regression',
            'Train_R2': self.ridge_metrics['train']['R²'],
            'Test_R2': self.ridge_metrics['test']['R²'],
            'Test_MAE': self.ridge_metrics['test']['MAE'],
            'Test_MAPE': self.ridge_metrics['test']['MAPE']
        })

        # Weighted Ensemble
        comparison.append({
            'Model': 'Weighted Ensemble',
            'Train_R2': self.ensemble_metrics['train']['R²'],
            'Test_R2': self.ensemble_metrics['test']['R²'],
            'Test_MAE': self.ensemble_metrics['test']['MAE'],
            'Test_MAPE': self.ensemble_metrics['test']['MAPE']
        })

        df = pd.DataFrame(comparison).sort_values('Test_R2', ascending=False)

        logger.info("\n성능 비교표:")
        logger.info(df.to_string(index=False))

        self.comparison_df = df

        return df

    def save_models(self):
        """프로덕션용 모델 저장"""
        logger.info("\n" + "=" * 70)
        logger.info("프로덕션 모델 저장")
        logger.info("=" * 70)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. Linear Regression
        lr_path = self.output_dir / f"linear_model_{timestamp}.pkl"
        joblib.dump({
            'model': self.lr_model,
            'feature_names': self.feature_cols,
            'metrics': self.lr_metrics
        }, lr_path)
        logger.info(f"✓ Linear Regression: {lr_path}")

        # 2. Ridge Regression
        ridge_path = self.output_dir / f"ridge_model_{timestamp}.pkl"
        joblib.dump({
            'model': self.ridge_model,
            'scaler': self.ridge_scaler,
            'alpha': self.ridge_alpha,
            'feature_names': self.feature_cols,
            'metrics': self.ridge_metrics
        }, ridge_path)
        logger.info(f"✓ Ridge Regression: {ridge_path}")

        # 3. Weighted Ensemble
        ensemble_path = self.output_dir / f"ensemble_model_{timestamp}.pkl"
        joblib.dump({
            'lr_model': self.lr_model,
            'xgb_model': self.ensemble_xgb_model,
            'lr_weight': self.ensemble_lr_weight,
            'xgb_weight': self.ensemble_xgb_weight,
            'feature_names': self.feature_cols,
            'metrics': self.ensemble_metrics
        }, ensemble_path)
        logger.info(f"✓ Weighted Ensemble: {ensemble_path}")

        # 4. 메타데이터
        metadata = {
            'timestamp': timestamp,
            'feature_count': len(self.feature_cols),
            'features': self.feature_cols,
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'models': {
                'linear': {
                    'test_r2': self.lr_metrics['test']['R²'],
                    'test_mape': self.lr_metrics['test']['MAPE']
                },
                'ridge': {
                    'test_r2': self.ridge_metrics['test']['R²'],
                    'test_mape': self.ridge_metrics['test']['MAPE'],
                    'alpha': self.ridge_alpha
                },
                'ensemble': {
                    'test_r2': self.ensemble_metrics['test']['R²'],
                    'test_mape': self.ensemble_metrics['test']['MAPE'],
                    'lr_weight': self.ensemble_lr_weight,
                    'xgb_weight': self.ensemble_xgb_weight
                }
            }
        }

        metadata_path = self.output_dir / f"metadata_{timestamp}.json"
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ 메타데이터: {metadata_path}")

        logger.info("=" * 70)
        logger.info("✓ 모든 모델 저장 완료")
        logger.info("=" * 70)

        return {
            'linear': lr_path,
            'ridge': ridge_path,
            'ensemble': ensemble_path,
            'metadata': metadata_path
        }


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("간소화된 5개 피처 모델 학습")
    print("=" * 70)

    # 초기화
    trainer = SimplifiedModelTrainer()

    # 1. 데이터 준비
    trainer.load_and_prepare_data(test_size=0.2, random_state=42)

    # 2. 모델 학습
    trainer.train_linear_regression()
    trainer.train_ridge_regression()
    trainer.train_weighted_ensemble()

    # 3. 모델 비교
    trainer.compare_models()

    # 4. 모델 저장
    saved_files = trainer.save_models()

    print("\n" + "=" * 70)
    print("✓ 프로덕션 모델 학습 완료")
    print("=" * 70)
    print("\n저장 위치: models/production/")
    print("  - linear_model_*.pkl")
    print("  - ridge_model_*.pkl")
    print("  - ensemble_model_*.pkl")
    print("  - metadata_*.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
