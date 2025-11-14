#!/usr/bin/env python3
"""
예가 예측 모델 개발 (타겟: 예가/기초(100%))
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YegaPredictionModel:
    """예가 예측 모델 (List of Successful Bidders 기반 학습)"""
    
    def __init__(self, data_dir="data/processed", model_dir="models", output_dir="docs/analysis"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        
        # 디렉토리 생성
        self.model_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # 모델 및 전처리기 저장용
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.feature_columns = []
        
        # 결과 저장
        self.training_results = {}
    
    def load_training_data(self):
        """훈련용 데이터 로드 (List of Successful Bidders)"""
        logger.info("훈련용 데이터 로딩...")
        
        try:
            # 최신 전처리된 파일 찾기
            successful_files = list(self.data_dir.glob("List_of_Successful_Bidders_processed_*.xlsx"))
            if not successful_files:
                logger.error("전처리된 낙찰자 데이터를 찾을 수 없습니다.")
                return None
            
            latest_file = max(successful_files, key=lambda x: x.stat().st_mtime)
            self.df_train = pd.read_excel(latest_file)
            
            logger.info(f"훈련 데이터 로드: {len(self.df_train)}행, {len(self.df_train.columns)}열")
            logger.info(f"파일: {latest_file.name}")
            
            return self.df_train
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {str(e)}")
            return None
    
    def prepare_features(self):
        """피쳐 준비 및 전처리"""
        logger.info("피쳐 준비 시작...")
        
        if self.df_train is None:
            logger.error("훈련 데이터가 로드되지 않았습니다.")
            return False
        
        # 타겟 변수 확인
        target_col = '예가/기초(100%)'
        if target_col not in self.df_train.columns:
            logger.error(f"타겟 변수를 찾을 수 없습니다: {target_col}")
            return False
        
        # 필요한 피쳐 컬럼들 정의
        categorical_features = ['지역', '업종', '발주기관']
        numerical_features = ['기초금액', 'A값']
        
        # 사용 가능한 피쳐만 선택
        available_categorical = [col for col in categorical_features if col in self.df_train.columns]
        available_numerical = [col for col in numerical_features if col in self.df_train.columns]
        
        logger.info(f"사용 가능한 범주형 피쳐: {available_categorical}")
        logger.info(f"사용 가능한 수치형 피쳐: {available_numerical}")
        
        # 데이터 준비
        df_clean = self.df_train.copy()
        
        # 타겟 변수 정리
        df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
        
        # 수치형 피쳐 정리
        for col in available_numerical:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 결측값 제거
        required_cols = [target_col] + available_categorical + available_numerical
        df_clean = df_clean[required_cols].dropna()
        
        if len(df_clean) == 0:
            logger.error("유효한 데이터가 없습니다.")
            return False
        
        logger.info(f"정리된 데이터: {len(df_clean)}행")
        
        # 범주형 변수 인코딩
        for col in available_categorical:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
            df_clean[f'{col}_encoded'] = self.encoders[col].fit_transform(df_clean[col])
        
        # 파생 변수 생성
        if 'A값' in available_numerical and '기초금액' in available_numerical:
            # A값 비율
            df_clean['A값_비율'] = df_clean['A값'] / df_clean['기초금액']
            
            # 기초금액 로그 변환
            df_clean['기초금액_log'] = np.log10(df_clean['기초금액'] + 1)
            
            # 기초금액 규모 카테고리
            df_clean['기초금액_규모'] = pd.cut(df_clean['기초금액'], 
                                          bins=[0, 1e8, 5e8, 2e9, np.inf],
                                          labels=['소규모', '중규모', '대규모', '특대규모'])
            
            # 규모 카테고리 인코딩
            if '기초금액_규모' not in self.encoders:
                self.encoders['기초금액_규모'] = LabelEncoder()
            df_clean['기초금액_규모_encoded'] = self.encoders['기초금액_규모'].fit_transform(df_clean['기초금액_규모'])
        
        # 최종 피쳐 컬럼 정의
        self.feature_columns = []
        
        # 인코딩된 범주형 피쳐
        for col in available_categorical:
            self.feature_columns.append(f'{col}_encoded')
        
        # 수치형 피쳐
        for col in available_numerical:
            self.feature_columns.append(col)
        
        # 파생 피쳐
        if 'A값_비율' in df_clean.columns:
            self.feature_columns.append('A값_비율')
        if '기초금액_log' in df_clean.columns:
            self.feature_columns.append('기초금액_log')
        if '기초금액_규모_encoded' in df_clean.columns:
            self.feature_columns.append('기초금액_규모_encoded')
        
        # X, y 준비
        self.X = df_clean[self.feature_columns].copy()
        self.y = df_clean[target_col].copy()
        
        # 수치형 피쳐 스케일링
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        logger.info(f"최종 피쳐 개수: {len(self.feature_columns)}")
        logger.info(f"피쳐 목록: {self.feature_columns}")
        logger.info(f"타겟 변수 범위: {self.y.min():.4f} ~ {self.y.max():.4f}")
        
        return True
    
    def train_models(self, test_size=0.2, random_state=42):
        """여러 모델 훈련"""
        logger.info("모델 훈련 시작...")
        
        if self.X_scaled is None or self.y is None:
            logger.error("피쳐 준비가 완료되지 않았습니다.")
            return False
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"훈련 데이터: {len(X_train)}개")
        logger.info(f"테스트 데이터: {len(X_test)}개")
        
        # 모델 정의
        models_to_train = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(alpha=1.0),
            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        }
        
        # 모델별 훈련 및 평가
        for model_name, model in models_to_train.items():
            logger.info(f"\n{model_name} 훈련 중...")
            
            try:
                # 훈련
                model.fit(X_train, y_train)
                
                # 예측
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # 평가 지표 계산
                train_metrics = self._calculate_metrics(y_train, y_train_pred, "Train")
                test_metrics = self._calculate_metrics(y_test, y_test_pred, "Test")
                
                # 결과 저장
                self.models[model_name] = model
                self.training_results[model_name] = {
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'feature_importance': self._get_feature_importance(model)
                }
                
                logger.info(f"{model_name} 완료:")
                logger.info(f"  훈련 MAE: {train_metrics['mae']:.6f}")
                logger.info(f"  테스트 MAE: {test_metrics['mae']:.6f}")
                logger.info(f"  테스트 R²: {test_metrics['r2']:.6f}")
                
            except Exception as e:
                logger.error(f"{model_name} 훈련 실패: {str(e)}")
        
        # 최고 성능 모델 선택
        if self.training_results:
            best_model_name = min(self.training_results.keys(), 
                                 key=lambda k: self.training_results[k]['test_metrics']['mae'])
            
            logger.info(f"\n최고 성능 모델: {best_model_name}")
            logger.info(f"테스트 MAE: {self.training_results[best_model_name]['test_metrics']['mae']:.6f}")
            
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
        
        return True
    
    def _calculate_metrics(self, y_true, y_pred, dataset_name):
        """평가 지표 계산"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE 계산 (0으로 나누기 방지)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape)
        }
    
    def _get_feature_importance(self, model):
        """피쳐 중요도 추출"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(self.feature_columns, importance.tolist()))
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            return dict(zip(self.feature_columns, np.abs(coef).tolist()))
        else:
            return {}
    
    def save_models(self):
        """모델 및 전처리기 저장"""
        logger.info("모델 저장 시작...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 최고 성능 모델 저장
            if hasattr(self, 'best_model'):
                model_path = self.model_dir / f"yega_prediction_model_{timestamp}.joblib"
                joblib.dump(self.best_model, model_path)
                logger.info(f"최고 모델 저장: {model_path}")
            
            # 전처리기 저장
            preprocessing_data = {
                'feature_columns': self.feature_columns,
                'encoders': {name: encoder.classes_.tolist() if hasattr(encoder, 'classes_') else encoder 
                           for name, encoder in self.encoders.items()},
                'scaler_mean': self.scaler.mean_.tolist() if self.scaler else None,
                'scaler_scale': self.scaler.scale_.tolist() if self.scaler else None,
                'best_model_name': getattr(self, 'best_model_name', None)
            }
            
            preprocessing_path = self.model_dir / f"yega_preprocessing_{timestamp}.json"
            with open(preprocessing_path, 'w', encoding='utf-8') as f:
                json.dump(preprocessing_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"전처리기 저장: {preprocessing_path}")
            
            # 훈련 결과 저장
            results_path = self.output_dir / f"yega_training_results_{timestamp}.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"훈련 결과 저장: {results_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"모델 저장 실패: {str(e)}")
            return False
    
    def run_full_training(self):
        """전체 훈련 파이프라인 실행"""
        logger.info("=== 예가 예측 모델 훈련 시작 ===")
        
        # 1. 데이터 로드
        if self.load_training_data() is None:
            return False
        
        # 2. 피쳐 준비
        if not self.prepare_features():
            return False
        
        # 3. 모델 훈련
        if not self.train_models():
            return False
        
        # 4. 모델 저장
        if not self.save_models():
            return False
        
        logger.info("\n=== 예가 예측 모델 훈련 완료 ===")
        
        # 결과 요약 출력
        if self.training_results:
            logger.info("\n📊 모델 성능 요약:")
            for model_name, results in self.training_results.items():
                test_metrics = results['test_metrics']
                logger.info(f"{model_name}:")
                logger.info(f"  MAE: {test_metrics['mae']:.6f}")
                logger.info(f"  MAPE: {test_metrics['mape']:.2f}%")
                logger.info(f"  R²: {test_metrics['r2']:.4f}")
        
        return True

def main():
    """메인 실행 함수"""
    model = YegaPredictionModel()
    success = model.run_full_training()
    
    if success:
        print("\n✅ 예가 예측 모델 훈련이 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 예가 예측 모델 훈련 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main()