#!/usr/bin/env python3
"""
베이스라인 ML 모델 구현 (pandas/sklearn 없이)
"""

import json
import math
from pathlib import Path
import csv

class SimpleLinearRegression:
    """간단한 선형 회귀 모델 (수동 구현)"""
    
    def __init__(self):
        self.weights = []
        self.bias = 0
        self.feature_names = []
        self.trained = False
    
    def fit(self, X, y):
        """
        최소제곱법으로 선형회귀 훈련
        X: 입력 특성 (2D 리스트)
        y: 타겟 값 (1D 리스트)
        """
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        if n_samples == 0 or n_features == 0:
            return False
        
        # 가중치 초기화
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        # 평균 계산
        X_mean = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
        y_mean = sum(y) / n_samples
        
        # 중심화된 데이터
        X_centered = [[X[i][j] - X_mean[j] for j in range(n_features)] for i in range(n_samples)]
        y_centered = [y[i] - y_mean for i in range(n_samples)]
        
        # 정규방정식 해법 (단순화된 버전)
        # 각 특성에 대해 단순 회귀
        for j in range(n_features):
            numerator = sum(X_centered[i][j] * y_centered[i] for i in range(n_samples))
            denominator = sum(X_centered[i][j] ** 2 for i in range(n_samples))
            
            if abs(denominator) > 1e-10:
                self.weights[j] = numerator / denominator
            else:
                self.weights[j] = 0.0
        
        # 편향 계산
        predictions = [self.predict_sample(X[i]) for i in range(n_samples)]
        self.bias = y_mean - sum(predictions) / n_samples
        
        self.trained = True
        return True
    
    def predict_sample(self, x):
        """단일 샘플 예측"""
        if not self.trained or len(x) != len(self.weights):
            return 0.0
        
        return sum(w * x[i] for i, w in enumerate(self.weights)) + self.bias
    
    def predict(self, X):
        """여러 샘플 예측"""
        return [self.predict_sample(x) for x in X]
    
    def score(self, X, y):
        """R² 점수 계산"""
        if not self.trained:
            return 0.0
        
        predictions = self.predict(X)
        
        # R² 계산
        y_mean = sum(y) / len(y)
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((yi - pred) ** 2 for yi, pred in zip(y, predictions))
        
        if abs(ss_tot) < 1e-10:
            return 0.0
        
        return 1 - (ss_res / ss_tot)

class MeanPredictor:
    """평균 기반 예측기 (최단순 베이스라인)"""
    
    def __init__(self):
        self.mean_value = 0.0
        self.trained = False
    
    def fit(self, X, y):
        """훈련 데이터의 평균 계산"""
        if not y:
            return False
        
        self.mean_value = sum(y) / len(y)
        self.trained = True
        return True
    
    def predict_sample(self, x):
        """단일 샘플 예측"""
        if not self.trained:
            return 0.0
        return self.mean_value
    
    def predict(self, X):
        """모든 예측을 평균값으로 반환"""
        if not self.trained:
            return [0.0] * len(X)
        
        return [self.mean_value] * len(X)
    
    def score(self, X, y):
        """평균 절대 오차 기반 점수"""
        predictions = self.predict(X)
        mae = sum(abs(yi - pred) for yi, pred in zip(y, predictions)) / len(y)
        return 1 / (1 + mae)  # 간단한 정규화된 점수

class BidPredictionModel:
    """낙찰하한가 예측 모델 래퍼"""
    
    def __init__(self, model_type="linear"):
        self.model_type = model_type
        
        if model_type == "linear":
            self.model = SimpleLinearRegression()
        elif model_type == "mean":
            self.model = MeanPredictor()
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
        
        self.feature_mappings = {}
        self.trained = False
    
    def load_processed_data(self, data_file="data/processed/sample_processed_data.json"):
        """전처리된 데이터 로드"""
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.feature_mappings = data.get('metadata', {})
            records = data.get('data', [])
            
            return records
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            return []
    
    def prepare_features(self, records):
        """모델 입력을 위한 특성 준비"""
        if not records:
            return [], []
        
        # 입력 특성 정의
        feature_columns = ['지역_encoded', '업종_encoded', '기초금액', 'A값', 'A값_비율']
        target_column = '예가'
        
        X = []
        y = []
        
        for record in records:
            # 특성 추출
            features = []
            valid_record = True
            
            for col in feature_columns:
                if col in record and record[col] is not None:
                    features.append(float(record[col]))
                else:
                    valid_record = False
                    break
            
            # 타겟 값 추출
            if target_column in record and record[target_column] is not None:
                target = float(record[target_column])
            else:
                valid_record = False
            
            if valid_record:
                X.append(features)
                y.append(target)
        
        return X, y
    
    def train(self, data_file="data/processed/sample_processed_data.json"):
        """모델 훈련"""
        print(f"모델 훈련 시작: {self.model_type}")
        
        # 데이터 로드
        records = self.load_processed_data(data_file)
        if not records:
            print("훈련 데이터가 없습니다.")
            return False
        
        print(f"훈련 데이터: {len(records)}개 레코드")
        
        # 특성 준비
        X, y = self.prepare_features(records)
        if not X or not y:
            print("유효한 훈련 데이터가 없습니다.")
            return False
        
        print(f"유효한 훈련 샘플: {len(X)}개")
        print(f"특성 개수: {len(X[0])}개")
        
        # 모델 훈련
        success = self.model.fit(X, y)
        if success:
            self.trained = True
            
            # 훈련 성능 평가
            train_score = self.model.score(X, y)
            predictions = self.model.predict(X)
            
            # MAE 계산
            mae = sum(abs(yi - pred) for yi, pred in zip(y, predictions)) / len(y)
            
            print(f"훈련 완료!")
            print(f"훈련 점수 (R²): {train_score:.4f}")
            print(f"평균 절대 오차 (MAE): {mae:.4f}")
            
            return True
        else:
            print("모델 훈련 실패")
            return False
    
    def predict_yega(self, region, sector, base_amount, a_value):
        """예가 예측"""
        if not self.trained:
            return None, "모델이 훈련되지 않았습니다."
        
        try:
            # 카테고리 인코딩
            region_mappings = self.feature_mappings.get('region_mapping', {})
            sector_mappings = self.feature_mappings.get('sector_mapping', {})
            
            if region not in region_mappings:
                return None, f"알 수 없는 지역: {region}"
            if sector not in sector_mappings:
                return None, f"알 수 없는 업종: {sector}"
            
            region_encoded = region_mappings[region]
            sector_encoded = sector_mappings[sector]
            
            # A값 비율 계산
            a_ratio = a_value / base_amount if base_amount > 0 else 0
            
            # 특성 벡터 구성
            features = [region_encoded, sector_encoded, base_amount, a_value, a_ratio]
            
            # 예측 수행
            prediction = self.model.predict_sample(features)
            
            return prediction, "예측 성공"
            
        except Exception as e:
            return None, f"예측 중 오류: {e}"
    
    def calculate_bid_limit(self, prediction, base_amount, a_value, limit_rate=0.875):
        """낙찰하한가 계산"""
        if prediction is None:
            return None
        
        # 예정가격 계산
        estimated_price = base_amount * prediction
        
        # 낙찰하한가 계산
        bid_limit = (estimated_price - a_value) * limit_rate + a_value
        
        return {
            "예가": prediction,
            "예정가격": estimated_price,
            "낙찰하한가": bid_limit,
            "기초금액": base_amount,
            "A값": a_value,
            "하한율": limit_rate
        }
    
    def save_model(self, filename="trained_model.json"):
        """모델 저장"""
        if not self.trained:
            return False
        
        model_data = {
            "model_type": self.model_type,
            "feature_mappings": self.feature_mappings,
            "trained": self.trained
        }
        
        # 모델 파라미터 저장
        if self.model_type == "linear":
            model_data["weights"] = self.model.weights
            model_data["bias"] = self.model.bias
        elif self.model_type == "mean":
            model_data["mean_value"] = self.model.mean_value
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        save_path = models_dir / filename
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        print(f"모델 저장: {save_path}")
        return True
    
    def load_model(self, filename="trained_model.json"):
        """저장된 모델 로드"""
        load_path = Path("models") / filename
        
        if not load_path.exists():
            return False
        
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            self.model_type = model_data["model_type"]
            self.feature_mappings = model_data["feature_mappings"]
            self.trained = model_data["trained"]
            
            # 모델 파라미터 로드
            if self.model_type == "linear":
                self.model = SimpleLinearRegression()
                self.model.weights = model_data["weights"]
                self.model.bias = model_data["bias"]
                self.model.trained = True
            elif self.model_type == "mean":
                self.model = MeanPredictor()
                self.model.mean_value = model_data["mean_value"]
                self.model.trained = True
            
            print(f"모델 로드 성공: {load_path}")
            return True
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False

def test_models():
    """모델 테스트"""
    print("=== 베이스라인 모델 테스트 ===\n")
    
    # 1. 선형회귀 모델 테스트
    print("1. 선형 회귀 모델")
    linear_model = BidPredictionModel("linear")
    linear_success = linear_model.train()
    
    if linear_success:
        # 예측 테스트
        yega, message = linear_model.predict_yega("서울특별시", "토목", 1000000000, 150000000)
        if yega is not None:
            print(f"예측 결과: {yega:.4f} ({message})")
        else:
            print(f"예측 실패: {message}")
        
        # 낙찰하한가 계산
        bid_result = linear_model.calculate_bid_limit(yega, 1000000000, 150000000)
        if bid_result:
            print(f"낙찰하한가 계산:")
            for key, value in bid_result.items():
                if key in ['예정가격', '낙찰하한가', '기초금액', 'A값']:
                    print(f"  {key}: {value:,.0f}원")
                else:
                    print(f"  {key}: {value}")
        
        # 모델 저장
        linear_model.save_model("linear_baseline.json")
    
    print("\n" + "="*50)
    
    # 2. 평균 모델 테스트
    print("2. 평균 기반 모델")
    mean_model = BidPredictionModel("mean")
    mean_success = mean_model.train()
    
    if mean_success:
        yega, message = mean_model.predict_yega("경기도", "건축", 2000000000, 200000000)
        if yega is not None:
            print(f"예측 결과: {yega:.4f} ({message})")
        else:
            print(f"예측 실패: {message}")
        
        bid_result = mean_model.calculate_bid_limit(yega, 2000000000, 200000000)
        if bid_result:
            print(f"낙찰하한가 계산:")
            for key, value in bid_result.items():
                if key in ['예정가격', '낙찰하한가', '기초금액', 'A값']:
                    print(f"  {key}: {value:,.0f}원")
                else:
                    print(f"  {key}: {value}")
        
        mean_model.save_model("mean_baseline.json")
    
    print("\n모델 테스트 완료!")

def main():
    """메인 실행 함수"""
    test_models()

if __name__ == "__main__":
    main()