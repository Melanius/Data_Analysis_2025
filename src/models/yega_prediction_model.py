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
        df_clean = self.df_train.copy()\n        \n        # 타겟 변수 정리\n        df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')\n        \n        # 수치형 피쳐 정리\n        for col in available_numerical:\n            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')\n        \n        # 결측값 제거\n        required_cols = [target_col] + available_categorical + available_numerical\n        df_clean = df_clean[required_cols].dropna()\n        \n        if len(df_clean) == 0:\n            logger.error(\"유효한 데이터가 없습니다.\")\n            return False\n        \n        logger.info(f\"정리된 데이터: {len(df_clean)}행\")\n        \n        # 범주형 변수 인코딩\n        for col in available_categorical:\n            if col not in self.encoders:\n                self.encoders[col] = LabelEncoder()\n            df_clean[f'{col}_encoded'] = self.encoders[col].fit_transform(df_clean[col])\n        \n        # 파생 변수 생성\n        if 'A값' in available_numerical and '기초금액' in available_numerical:\n            # A값 비율\n            df_clean['A값_비율'] = df_clean['A값'] / df_clean['기초금액']\n            \n            # 기초금액 로그 변환\n            df_clean['기초금액_log'] = np.log10(df_clean['기초금액'] + 1)\n            \n            # 기초금액 규모 카테고리\n            df_clean['기초금액_규모'] = pd.cut(df_clean['기초금액'], \n                                          bins=[0, 1e8, 5e8, 2e9, np.inf],\n                                          labels=['소규모', '중규모', '대규모', '특대규모'])\n            \n            # 규모 카테고리 인코딩\n            if '기초금액_규모' not in self.encoders:\n                self.encoders['기초금액_규모'] = LabelEncoder()\n            df_clean['기초금액_규모_encoded'] = self.encoders['기초금액_규모'].fit_transform(df_clean['기초금액_규모'])\n        \n        # 최종 피쳐 컬럼 정의\n        self.feature_columns = []\n        \n        # 인코딩된 범주형 피쳐\n        for col in available_categorical:\n            self.feature_columns.append(f'{col}_encoded')\n        \n        # 수치형 피쳐\n        for col in available_numerical:\n            self.feature_columns.append(col)\n        \n        # 파생 피쳐\n        if 'A값_비율' in df_clean.columns:\n            self.feature_columns.append('A값_비율')\n        if '기초금액_log' in df_clean.columns:\n            self.feature_columns.append('기초금액_log')\n        if '기초금액_규모_encoded' in df_clean.columns:\n            self.feature_columns.append('기초금액_규모_encoded')\n        \n        # X, y 준비\n        self.X = df_clean[self.feature_columns].copy()\n        self.y = df_clean[target_col].copy()\n        \n        # 수치형 피쳐 스케일링\n        self.scaler = StandardScaler()\n        self.X_scaled = self.scaler.fit_transform(self.X)\n        \n        logger.info(f\"최종 피쳐 개수: {len(self.feature_columns)}\")\n        logger.info(f\"피쳐 목록: {self.feature_columns}\")\n        logger.info(f\"타겟 변수 범위: {self.y.min():.4f} ~ {self.y.max():.4f}\")\n        \n        return True\n    \n    def train_models(self, test_size=0.2, random_state=42):\n        \"\"\"여러 모델 훈련\"\"\"\n        logger.info(\"모델 훈련 시작...\")\n        \n        if self.X_scaled is None or self.y is None:\n            logger.error(\"피쳐 준비가 완료되지 않았습니다.\")\n            return False\n        \n        # 훈련/테스트 분할\n        X_train, X_test, y_train, y_test = train_test_split(\n            self.X_scaled, self.y, test_size=test_size, random_state=random_state\n        )\n        \n        logger.info(f\"훈련 데이터: {len(X_train)}개\")\n        logger.info(f\"테스트 데이터: {len(X_test)}개\")\n        \n        # 모델 정의\n        models_to_train = {\n            'Linear_Regression': LinearRegression(),\n            'Ridge_Regression': Ridge(alpha=1.0),\n            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),\n            'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)\n        }\n        \n        # 모델별 훈련 및 평가\n        for model_name, model in models_to_train.items():\n            logger.info(f\"\\n{model_name} 훈련 중...\")\n            \n            try:\n                # 훈련\n                model.fit(X_train, y_train)\n                \n                # 예측\n                y_train_pred = model.predict(X_train)\n                y_test_pred = model.predict(X_test)\n                \n                # 평가 지표 계산\n                train_metrics = self._calculate_metrics(y_train, y_train_pred, \"Train\")\n                test_metrics = self._calculate_metrics(y_test, y_test_pred, \"Test\")\n                \n                # 결과 저장\n                self.models[model_name] = model\n                self.training_results[model_name] = {\n                    'train_metrics': train_metrics,\n                    'test_metrics': test_metrics,\n                    'feature_importance': self._get_feature_importance(model)\n                }\n                \n                logger.info(f\"{model_name} 완료:\")\n                logger.info(f\"  훈련 MAE: {train_metrics['mae']:.6f}\")\n                logger.info(f\"  테스트 MAE: {test_metrics['mae']:.6f}\")\n                logger.info(f\"  테스트 R²: {test_metrics['r2']:.6f}\")\n                \n            except Exception as e:\n                logger.error(f\"{model_name} 훈련 실패: {str(e)}\")\n        \n        # 최고 성능 모델 선택\n        if self.training_results:\n            best_model_name = min(self.training_results.keys(), \n                                 key=lambda k: self.training_results[k]['test_metrics']['mae'])\n            \n            logger.info(f\"\\n최고 성능 모델: {best_model_name}\")\n            logger.info(f\"테스트 MAE: {self.training_results[best_model_name]['test_metrics']['mae']:.6f}\")\n            \n            self.best_model_name = best_model_name\n            self.best_model = self.models[best_model_name]\n        \n        return True\n    \n    def _calculate_metrics(self, y_true, y_pred, dataset_name):\n        \"\"\"평가 지표 계산\"\"\"\n        mae = mean_absolute_error(y_true, y_pred)\n        mse = mean_squared_error(y_true, y_pred)\n        rmse = np.sqrt(mse)\n        r2 = r2_score(y_true, y_pred)\n        \n        # MAPE 계산 (0으로 나누기 방지)\n        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100\n        \n        return {\n            'mae': float(mae),\n            'mse': float(mse),\n            'rmse': float(rmse),\n            'r2': float(r2),\n            'mape': float(mape)\n        }\n    \n    def _get_feature_importance(self, model):\n        \"\"\"피쳐 중요도 추출\"\"\"\n        if hasattr(model, 'feature_importances_'):\n            importance = model.feature_importances_\n            return dict(zip(self.feature_columns, importance.tolist()))\n        elif hasattr(model, 'coef_'):\n            coef = model.coef_\n            return dict(zip(self.feature_columns, np.abs(coef).tolist()))\n        else:\n            return {}\n    \n    def save_models(self):\n        \"\"\"모델 및 전처리기 저장\"\"\"\n        logger.info(\"모델 저장 시작...\")\n        \n        timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n        \n        try:\n            # 최고 성능 모델 저장\n            if hasattr(self, 'best_model'):\n                model_path = self.model_dir / f\"yega_prediction_model_{timestamp}.joblib\"\n                joblib.dump(self.best_model, model_path)\n                logger.info(f\"최고 모델 저장: {model_path}\")\n            \n            # 전처리기 저장\n            preprocessing_data = {\n                'feature_columns': self.feature_columns,\n                'encoders': {name: encoder.classes_.tolist() if hasattr(encoder, 'classes_') else encoder \n                           for name, encoder in self.encoders.items()},\n                'scaler_mean': self.scaler.mean_.tolist() if self.scaler else None,\n                'scaler_scale': self.scaler.scale_.tolist() if self.scaler else None,\n                'best_model_name': getattr(self, 'best_model_name', None)\n            }\n            \n            preprocessing_path = self.model_dir / f\"yega_preprocessing_{timestamp}.json\"\n            with open(preprocessing_path, 'w', encoding='utf-8') as f:\n                json.dump(preprocessing_data, f, ensure_ascii=False, indent=2)\n            \n            logger.info(f\"전처리기 저장: {preprocessing_path}\")\n            \n            # 훈련 결과 저장\n            results_path = self.output_dir / f\"yega_training_results_{timestamp}.json\"\n            with open(results_path, 'w', encoding='utf-8') as f:\n                json.dump(self.training_results, f, ensure_ascii=False, indent=2)\n            \n            logger.info(f\"훈련 결과 저장: {results_path}\")\n            \n            return True\n            \n        except Exception as e:\n            logger.error(f\"모델 저장 실패: {str(e)}\")\n            return False\n    \n    def predict_yega(self, region, sector, base_amount, a_value, agency=None):\n        \"\"\"예가 예측 (단일 케이스)\"\"\"\n        if not hasattr(self, 'best_model'):\n            return None, \"훈련된 모델이 없습니다.\"\n        \n        try:\n            # 입력 데이터 준비\n            input_data = {\n                '지역': region,\n                '업종': sector,\n                '기초금액': base_amount,\n                'A값': a_value\n            }\n            \n            if agency:\n                input_data['발주기관'] = agency\n            \n            # 피쳐 변환\n            features = self._transform_input_features(input_data)\n            if features is None:\n                return None, \"피쳐 변환 실패\"\n            \n            # 예측\n            prediction = self.best_model.predict([features])[0]\n            \n            return float(prediction), \"예측 성공\"\n            \n        except Exception as e:\n            return None, f\"예측 중 오류: {str(e)}\"\n    \n    def _transform_input_features(self, input_data):\n        \"\"\"입력 데이터를 모델 입력 형태로 변환\"\"\"\n        try:\n            features = []\n            \n            for col in self.feature_columns:\n                if col.endswith('_encoded'):\n                    # 범주형 피쳐 인코딩\n                    original_col = col.replace('_encoded', '')\n                    if original_col in input_data and original_col in self.encoders:\n                        value = input_data[original_col]\n                        encoder = self.encoders[original_col]\n                        if hasattr(encoder, 'classes_'):\n                            if value in encoder.classes_:\n                                encoded_value = encoder.transform([value])[0]\n                            else:\n                                encoded_value = 0  # 미지의 카테고리는 0으로\n                        else:\n                            encoded_value = 0\n                        features.append(encoded_value)\n                    else:\n                        features.append(0)\n                        \n                elif col in input_data:\n                    # 수치형 피쳐\n                    features.append(float(input_data[col]))\n                    \n                elif col == 'A값_비율' and 'A값' in input_data and '기초금액' in input_data:\n                    # A값 비율 계산\n                    ratio = input_data['A값'] / input_data['기초금액']\n                    features.append(float(ratio))\n                    \n                elif col == '기초금액_log' and '기초금액' in input_data:\n                    # 기초금액 로그\n                    log_value = np.log10(input_data['기초금액'] + 1)\n                    features.append(float(log_value))\n                    \n                elif col == '기초금액_규모_encoded' and '기초금액' in input_data:\n                    # 기초금액 규모 카테고리\n                    amount = input_data['기초금액']\n                    if amount < 1e8:\n                        category = '소규모'\n                    elif amount < 5e8:\n                        category = '중규모'\n                    elif amount < 2e9:\n                        category = '대규모'\n                    else:\n                        category = '특대규모'\n                    \n                    encoder = self.encoders.get('기초금액_규모')\n                    if encoder and hasattr(encoder, 'classes_'):\n                        if category in encoder.classes_:\n                            encoded_value = encoder.transform([category])[0]\n                        else:\n                            encoded_value = 0\n                    else:\n                        encoded_value = 0\n                    features.append(encoded_value)\n                else:\n                    features.append(0)\n            \n            # 스케일링\n            if self.scaler:\n                features_scaled = self.scaler.transform([features])[0]\n                return features_scaled\n            else:\n                return features\n                \n        except Exception as e:\n            logger.error(f\"피쳐 변환 실패: {str(e)}\")\n            return None\n    \n    def run_full_training(self):\n        \"\"\"전체 훈련 파이프라인 실행\"\"\"\n        logger.info(\"=== 예가 예측 모델 훈련 시작 ===\")\n        \n        # 1. 데이터 로드\n        if self.load_training_data() is None:\n            return False\n        \n        # 2. 피쳐 준비\n        if not self.prepare_features():\n            return False\n        \n        # 3. 모델 훈련\n        if not self.train_models():\n            return False\n        \n        # 4. 모델 저장\n        if not self.save_models():\n            return False\n        \n        logger.info(\"\\n=== 예가 예측 모델 훈련 완료 ===\")\n        \n        # 결과 요약 출력\n        if self.training_results:\n            logger.info(\"\\n📊 모델 성능 요약:\")\n            for model_name, results in self.training_results.items():\n                test_metrics = results['test_metrics']\n                logger.info(f\"{model_name}:\")\n                logger.info(f\"  MAE: {test_metrics['mae']:.6f}\")\n                logger.info(f\"  MAPE: {test_metrics['mape']:.2f}%\")\n                logger.info(f\"  R²: {test_metrics['r2']:.4f}\")\n        \n        return True\n\ndef main():\n    \"\"\"메인 실행 함수\"\"\"\n    model = YegaPredictionModel()\n    success = model.run_full_training()\n    \n    if success:\n        print(\"\\n✅ 예가 예측 모델 훈련이 성공적으로 완료되었습니다!\")\n        \n        # 간단한 예측 테스트\n        if hasattr(model, 'best_model'):\n            print(\"\\n🧪 예측 테스트:\")\n            prediction, message = model.predict_yega(\n                region=\"서울특별시\",\n                sector=\"토목\", \n                base_amount=1000000000,\n                a_value=150000000\n            )\n            \n            if prediction is not None:\n                print(f\"  입력: 서울특별시, 토목, 기초금액 10억원, A값 1.5억원\")\n                print(f\"  예가 예측: {prediction:.4f}\")\n                print(f\"  예정가격 예측: {prediction * 1000000000:,.0f}원\")\n            else:\n                print(f\"  예측 실패: {message}\")\n    else:\n        print(\"\\n❌ 예가 예측 모델 훈련 중 오류가 발생했습니다.\")\n\nif __name__ == \"__main__\":\n    main()"