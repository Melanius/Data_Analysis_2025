"""
Ridge Regression 하이퍼파라미터 튜닝
- GridSearchCV로 최적 alpha 값 탐색
- 교차 검증으로 안정성 확인
- L2 정규화로 과적합 방지 및 다중공선성 해결
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 모델 라이브러리
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
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


class RidgeRegressionTuner:
    """Ridge Regression 하이퍼파라미터 튜닝 클래스"""

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

        logger.info("RidgeRegressionTuner 초기화 완료")

    def load_and_prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        데이터 로드 및 전처리

        Args:
            test_size: 테스트 데이터 비율
            random_state: 랜덤 시드
        """
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

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Ridge Regression은 피처 스케일링이 중요함
        logger.info("\n피처 스케일링 (StandardScaler) 적용...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # DataFrame으로 변환 (피처 이름 유지)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        logger.info(f"데이터 분할: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")

        # 저장
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler
        self.feature_names = X_train.columns.tolist()

        logger.info("✓ 데이터 준비 완료")
        logger.info("=" * 70)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def tune_hyperparameters(self, cv_folds: int = 5):
        """
        GridSearchCV로 최적 alpha 값 탐색

        Args:
            cv_folds: 교차 검증 폴드 수
        """
        logger.info("\n" + "=" * 70)
        logger.info("Ridge Regression 하이퍼파라미터 튜닝")
        logger.info("=" * 70)

        # Alpha 값 범위 설정 (로그 스케일)
        # 0.0001 ~ 10000 범위를 로그 스케일로 탐색
        alphas = np.logspace(-4, 4, 50)

        logger.info(f"탐색할 alpha 범위: {alphas.min():.4f} ~ {alphas.max():.1f}")
        logger.info(f"탐색할 alpha 개수: {len(alphas)}개")
        logger.info(f"교차 검증 폴드: {cv_folds}")

        # GridSearchCV 설정
        ridge = Ridge()
        param_grid = {'alpha': alphas}

        grid_search = GridSearchCV(
            ridge,
            param_grid,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )

        logger.info("\nGridSearchCV 실행 중...")
        grid_search.fit(self.X_train, self.y_train)

        # 최적 파라미터
        best_alpha = grid_search.best_params_['alpha']
        best_cv_score = grid_search.best_score_

        logger.info(f"\n✓ 최적 alpha 값: {best_alpha:.6f}")
        logger.info(f"✓ 최적 CV R² Score: {best_cv_score:.6f}")

        # 모든 alpha에 대한 결과 저장
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results = cv_results[['param_alpha', 'mean_test_score', 'std_test_score']]
        cv_results.columns = ['alpha', 'mean_r2', 'std_r2']
        cv_results = cv_results.sort_values('mean_r2', ascending=False)

        self.cv_results = cv_results
        self.best_alpha = best_alpha
        self.best_cv_score = best_cv_score

        return best_alpha, best_cv_score, cv_results

    def train_best_model(self):
        """최적 alpha로 모델 학습"""
        logger.info("\n" + "=" * 70)
        logger.info(f"최적 Ridge 모델 학습 (alpha={self.best_alpha:.6f})")
        logger.info("=" * 70)

        # 최적 모델 학습
        model = Ridge(alpha=self.best_alpha)
        model.fit(self.X_train, self.y_train)

        # 예측
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # 평가
        train_metrics = self._evaluate_predictions(self.y_train, y_train_pred, "Train")
        test_metrics = self._evaluate_predictions(self.y_test, y_test_pred, "Test")

        # 교차 검증 점수 계산
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
        logger.info(f"\n교차 검증 R² Scores: {cv_scores}")
        logger.info(f"평균 CV R²: {cv_scores.mean():.6f} (±{cv_scores.std():.6f})")

        # 피처 중요도 (계수의 절대값)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)

        # 저장
        self.model = model
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.cv_scores = cv_scores
        self.feature_importance = feature_importance

        logger.info("✓ Ridge Regression 학습 완료")

        return model, test_metrics

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

    def compare_with_baseline(self, baseline_r2: float = 0.234917):
        """베이스라인 Linear Regression과 비교"""
        logger.info("\n" + "=" * 70)
        logger.info("베이스라인 모델과 비교")
        logger.info("=" * 70)

        improvement = self.test_metrics['R²'] - baseline_r2
        improvement_pct = (improvement / baseline_r2) * 100

        logger.info(f"베이스라인 (Linear Regression) R²: {baseline_r2:.6f}")
        logger.info(f"Ridge Regression R²:                {self.test_metrics['R²']:.6f}")
        logger.info(f"개선도:                              {improvement:+.6f} ({improvement_pct:+.2f}%)")

        if improvement > 0:
            logger.info("✓ Ridge Regression이 더 우수함")
        elif improvement > -0.01:
            logger.info("≈ 거의 동일한 성능")
        else:
            logger.info("✗ 베이스라인이 더 우수함")

        comparison = {
            'baseline_r2': baseline_r2,
            'ridge_r2': self.test_metrics['R²'],
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }

        self.comparison = comparison
        return comparison

    def plot_alpha_tuning(self):
        """Alpha 튜닝 결과 시각화"""
        logger.info("\nAlpha 튜닝 결과 시각화 중...")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Alpha vs R² Score (Log Scale)',
                'Alpha vs R² Score (Linear Scale, Top 20)'
            )
        )

        # 1. 전체 범위 (로그 스케일)
        fig.add_trace(
            go.Scatter(
                x=self.cv_results['alpha'],
                y=self.cv_results['mean_r2'],
                mode='lines+markers',
                name='Mean R²',
                line=dict(color='blue'),
                error_y=dict(
                    type='data',
                    array=self.cv_results['std_r2'],
                    visible=True
                )
            ),
            row=1, col=1
        )

        # 최적 alpha 표시
        fig.add_vline(
            x=self.best_alpha,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Best α={self.best_alpha:.4f}",
            row=1, col=1
        )

        # 2. 상위 20개 (선형 스케일)
        top_20 = self.cv_results.head(20)
        fig.add_trace(
            go.Bar(
                x=top_20['alpha'].astype(str),
                y=top_20['mean_r2'],
                name='R² Score',
                marker_color='lightblue',
                error_y=dict(
                    type='data',
                    array=top_20['std_r2'],
                    visible=True
                ),
                showlegend=False
            ),
            row=1, col=2
        )

        # 레이아웃 업데이트
        fig.update_xaxes(title_text="Alpha (log scale)", type="log", row=1, col=1)
        fig.update_xaxes(title_text="Alpha", row=1, col=2)
        fig.update_yaxes(title_text="R² Score", row=1, col=1)
        fig.update_yaxes(title_text="R² Score", row=1, col=2)

        fig.update_layout(
            title_text=f"Ridge Regression Alpha 튜닝 결과 (최적 α={self.best_alpha:.4f})",
            height=500
        )

        # 저장
        output_path = self.output_dir / f"ridge_alpha_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(output_path))
        logger.info(f"✓ Alpha 튜닝 시각화 저장: {output_path}")

        return fig

    def plot_feature_importance(self, top_n: int = 20):
        """피처 중요도 시각화"""
        logger.info(f"\n상위 {top_n}개 피처 중요도 시각화 중...")

        top_features = self.feature_importance.head(top_n)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=top_features['abs_coefficient'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(
                    color=top_features['abs_coefficient'],
                    colorscale='Blues',
                    showscale=True
                ),
                text=top_features['coefficient'].apply(lambda x: f'{x:.4f}'),
                textposition='outside'
            )
        )

        fig.update_layout(
            title_text=f"Ridge Regression 피처 중요도 (Top {top_n}) - 계수 절대값",
            xaxis_title="Absolute Coefficient",
            yaxis_title="Feature",
            height=600
        )

        # 저장
        output_path = self.output_dir / f"ridge_feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(output_path))
        logger.info(f"✓ 피처 중요도 시각화 저장: {output_path}")

        return fig

    def save_results(self):
        """결과 저장"""
        logger.info("\n결과 저장 중...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. 최적 모델 정보
        model_info = {
            'model': 'Ridge Regression',
            'best_alpha': self.best_alpha,
            'best_cv_r2': self.best_cv_score,
            'cv_mean': self.cv_scores.mean(),
            'cv_std': self.cv_scores.std(),
            'train_r2': self.train_metrics['R²'],
            'test_r2': self.test_metrics['R²'],
            'test_mae': self.test_metrics['MAE'],
            'test_rmse': self.test_metrics['RMSE'],
            'test_mape': self.test_metrics['MAPE'],
            'improvement_vs_baseline': self.comparison['improvement'],
            'improvement_pct': self.comparison['improvement_pct'],
            'timestamp': timestamp
        }

        info_df = pd.DataFrame([model_info])
        info_path = self.output_dir / f"ridge_model_info_{timestamp}.xlsx"
        info_df.to_excel(info_path, index=False)
        logger.info(f"✓ 모델 정보 저장: {info_path}")

        # 2. Alpha 튜닝 결과
        cv_path = self.output_dir / f"ridge_cv_results_{timestamp}.xlsx"
        self.cv_results.to_excel(cv_path, index=False)
        logger.info(f"✓ CV 결과 저장: {cv_path}")

        # 3. 피처 중요도
        importance_path = self.output_dir / f"ridge_feature_importance_{timestamp}.xlsx"
        self.feature_importance.to_excel(importance_path, index=False)
        logger.info(f"✓ 피처 중요도 저장: {importance_path}")

        # 4. 모델 객체 저장 (pickle)
        model_path = self.output_dir / f"ridge_model_{timestamp}.pkl"
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'best_alpha': self.best_alpha
        }, model_path)
        logger.info(f"✓ 모델 객체 저장: {model_path}")

        logger.info("✓ 모든 결과 저장 완료")

        return {
            'info': info_path,
            'cv_results': cv_path,
            'feature_importance': importance_path,
            'model': model_path
        }


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("Ridge Regression 하이퍼파라미터 튜닝")
    print("=" * 70)

    # 초기화
    tuner = RidgeRegressionTuner()

    # 1. 데이터 준비
    tuner.load_and_prepare_data(test_size=0.2, random_state=42)

    # 2. 하이퍼파라미터 튜닝
    best_alpha, best_cv_score, cv_results = tuner.tune_hyperparameters(cv_folds=5)

    # 3. 최적 모델 학습
    model, test_metrics = tuner.train_best_model()

    # 4. 베이스라인과 비교
    tuner.compare_with_baseline(baseline_r2=0.234917)

    # 5. 시각화
    tuner.plot_alpha_tuning()
    tuner.plot_feature_importance(top_n=20)

    # 6. 결과 저장
    saved_files = tuner.save_results()

    print("\n" + "=" * 70)
    print("✓ Ridge Regression 튜닝 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
