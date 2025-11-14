#!/usr/bin/env python3
"""
낙찰하한가 계산 로직 구현
예가 예측 → 낙찰하한가 계산 방식
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BidLimitCalculator:
    """낙찰하한가 계산기 (예가 예측 + 공식 계산)"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        
        # 모델 및 전처리기
        self.yega_model = None
        self.preprocessing_data = None
        
        # 예가변동폭 파싱용 패턴
        self.variation_pattern = re.compile(r'([+-]?\d+(?:\.\d+)?)')
        
    def load_yega_model(self):
        """최신 예가 예측 모델 로드"""
        logger.info("예가 예측 모델 로딩...")
        
        try:
            # 최신 모델 파일 찾기
            model_files = list(self.model_dir.glob("yega_prediction_model_*.joblib"))
            preprocessing_files = list(self.model_dir.glob("yega_preprocessing_*.json"))
            
            if not model_files or not preprocessing_files:
                logger.error("예가 예측 모델 파일을 찾을 수 없습니다.")
                return False
            
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            latest_preprocessing = max(preprocessing_files, key=lambda x: x.stat().st_mtime)
            
            # 모델 로드
            self.yega_model = joblib.load(latest_model)
            
            # 전처리 데이터 로드
            with open(latest_preprocessing, 'r', encoding='utf-8') as f:
                self.preprocessing_data = json.load(f)
            
            logger.info(f"모델 로드 성공: {latest_model.name}")
            logger.info(f"전처리기 로드: {latest_preprocessing.name}")
            logger.info(f"사용 모델: {self.preprocessing_data.get('best_model_name', 'Unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {str(e)}")
            return False
    
    def parse_price_variation(self, variation_str):
        """예가변동폭 파싱"""
        if pd.isna(variation_str) or variation_str == '':
            return 0, 0  # 기본값
        
        try:
            # 문자열에서 숫자 추출
            numbers = self.variation_pattern.findall(str(variation_str))
            
            if len(numbers) >= 2:
                # -x/+y 형태
                min_var = float(numbers[0])
                max_var = float(numbers[1])
                return min_var, max_var
            elif len(numbers) == 1:
                # 단일 값 (예: "3" → -3/+3으로 해석)
                val = float(numbers[0])
                return -abs(val), abs(val)
            else:
                return 0, 0
                
        except Exception as e:
            logger.warning(f"예가변동폭 파싱 실패: {variation_str}, 오류: {e}")
            return 0, 0
    
    def predict_yega_with_variation(self, region, sector, base_amount, a_value, 
                                   agency=None, price_variation=None):
        """예가 예측 (예가변동폭 고려)"""
        if self.yega_model is None or self.preprocessing_data is None:
            return None, "모델이 로드되지 않았습니다."
        
        try:
            # 기본 예가 예측
            base_prediction = self._predict_base_yega(region, sector, base_amount, a_value, agency)
            if base_prediction is None:
                return None, "기본 예가 예측 실패"
            
            # 예가변동폭 적용
            if price_variation:
                min_var, max_var = self.parse_price_variation(price_variation)
                
                # 예가변동폭을 퍼센트로 해석하고 적용
                min_yega = base_prediction * (1 + min_var / 100)
                max_yega = base_prediction * (1 + max_var / 100)
                
                # 중간값 사용 (보수적 접근)
                adjusted_prediction = (min_yega + max_yega) / 2
                
                logger.info(f"예가변동폭 적용: {price_variation}")
                logger.info(f"기본 예가: {base_prediction:.4f}")
                logger.info(f"조정된 예가 범위: {min_yega:.4f} ~ {max_yega:.4f}")
                logger.info(f"최종 예가: {adjusted_prediction:.4f}")
                
                return {
                    'base_prediction': float(base_prediction),
                    'price_variation': price_variation,
                    'min_yega': float(min_yega),
                    'max_yega': float(max_yega),
                    'final_yega': float(adjusted_prediction),
                    'variation_applied': True
                }, "예측 성공"
            else:
                return {
                    'base_prediction': float(base_prediction),
                    'final_yega': float(base_prediction),
                    'variation_applied': False
                }, "예측 성공"
                
        except Exception as e:
            return None, f"예가 예측 중 오류: {str(e)}"
    
    def _predict_base_yega(self, region, sector, base_amount, a_value, agency=None):
        """기본 예가 예측"""
        try:
            # 피쳐 준비
            features = self._prepare_features(region, sector, base_amount, a_value, agency)
            if features is None:
                return None
            
            # 예측
            prediction = self.yega_model.predict([features])[0]
            return float(prediction)
            
        except Exception as e:
            logger.error(f"기본 예가 예측 실패: {str(e)}")
            return None
    
    def _prepare_features(self, region, sector, base_amount, a_value, agency=None):
        """모델 입력을 위한 피쳐 준비"""
        try:
            feature_columns = self.preprocessing_data['feature_columns']
            encoders = self.preprocessing_data['encoders']
            scaler_mean = self.preprocessing_data['scaler_mean']
            scaler_scale = self.preprocessing_data['scaler_scale']
            
            features = []
            
            for col in feature_columns:
                if col == '지역_encoded':
                    if region in encoders.get('지역', []):
                        encoded = encoders['지역'].index(region)
                    else:
                        encoded = 0  # 미지의 지역
                    features.append(encoded)
                    
                elif col == '업종_encoded':
                    if sector in encoders.get('업종', []):
                        encoded = encoders['업종'].index(sector)
                    else:
                        encoded = 0  # 미지의 업종
                    features.append(encoded)
                    
                elif col == '발주기관_encoded':
                    if agency and agency in encoders.get('발주기관', []):
                        encoded = encoders['발주기관'].index(agency)
                    else:
                        encoded = 0  # 미지의 기관 또는 없음
                    features.append(encoded)
                    
                elif col == '기초금액':
                    features.append(float(base_amount))
                    
                elif col == 'A값':
                    # A값이 없으면 0으로 처리
                    a_val = 0 if pd.isna(a_value) else float(a_value)
                    features.append(a_val)
                    
                elif col == 'A값_비율':
                    # A값이 없으면 0으로 처리하여 비율 계산
                    a_val = 0 if pd.isna(a_value) else float(a_value)
                    ratio = a_val / base_amount if base_amount > 0 else 0
                    features.append(float(ratio))
                    
                elif col == '기초금액_log':
                    log_val = np.log10(base_amount + 1)
                    features.append(float(log_val))
                    
                elif col == '기초금액_규모_encoded':
                    if base_amount < 1e8:
                        category = '소규모'
                    elif base_amount < 5e8:
                        category = '중규모'
                    elif base_amount < 2e9:
                        category = '대규모'
                    else:
                        category = '특대규모'
                    
                    if category in encoders.get('기초금액_규모', []):
                        encoded = encoders['기초금액_규모'].index(category)
                    else:
                        encoded = 0
                    features.append(encoded)
                else:
                    features.append(0)
            
            # 스케일링 적용
            if scaler_mean and scaler_scale:
                features = np.array(features)
                scaled_features = (features - np.array(scaler_mean)) / np.array(scaler_scale)
                return scaled_features.tolist()
            else:
                return features
                
        except Exception as e:
            logger.error(f"피쳐 준비 실패: {str(e)}")
            return None
    
    def calculate_bid_limit(self, yega_result, base_amount, a_value, bid_limit_rate=87.5):
        """낙찰하한가 계산"""
        try:
            if yega_result is None:
                return None, "예가 예측 결과가 없습니다."
            
            final_yega = yega_result['final_yega']
            
            # 예정가격 계산
            estimated_price = base_amount * final_yega
            
            # 낙찰하한가 계산: (예정가격 - A값) × 낙찰하한율 + A값
            bid_limit = (estimated_price - a_value) * (bid_limit_rate / 100) + a_value
            
            result = {
                'input_data': {
                    '기초금액': base_amount,
                    'A값': a_value,
                    '낙찰하한율': bid_limit_rate
                },
                'prediction': {
                    '예가': final_yega,
                    '예가변동폭_적용': yega_result.get('variation_applied', False),
                    '예정가격': estimated_price,
                    '낙찰하한가': bid_limit
                },
                'calculation_details': {
                    '공식': '낙찰하한가 = (예정가격 - A값) × 낙찰하한율 + A값',
                    '단계별': {
                        '1_예정가격': f"{base_amount:,.0f} × {final_yega:.4f} = {estimated_price:,.0f}",
                        '2_과세표준': f"{estimated_price:,.0f} - {a_value:,.0f} = {estimated_price-a_value:,.0f}",
                        '3_하한가계산': f"{estimated_price-a_value:,.0f} × {bid_limit_rate}% = {(estimated_price-a_value)*bid_limit_rate/100:,.0f}",
                        '4_최종결과': f"{(estimated_price-a_value)*bid_limit_rate/100:,.0f} + {a_value:,.0f} = {bid_limit:,.0f}"
                    }
                }
            }
            
            if yega_result.get('variation_applied'):
                result['prediction']['예가_범위'] = {
                    '최소': yega_result['min_yega'],
                    '최대': yega_result['max_yega'],
                    '예가변동폭': yega_result.get('price_variation', 'N/A')
                }
            
            return result, "계산 성공"
            
        except Exception as e:
            return None, f"낙찰하한가 계산 중 오류: {str(e)}"
    
    def predict_bid_limit(self, region, sector, base_amount, a_value, 
                         bid_limit_rate=87.5, agency=None, price_variation=None):
        """전체 예측 프로세스 (예가 → 낙찰하한가)"""
        logger.info("낙찰하한가 예측 시작...")
        logger.info(f"입력: 지역={region}, 업종={sector}, 기초금액={base_amount:,.0f}, A값={a_value:,.0f}")
        
        # 1. 예가 예측
        yega_result, yega_message = self.predict_yega_with_variation(
            region, sector, base_amount, a_value, agency, price_variation
        )
        
        if yega_result is None:
            return None, f"예가 예측 실패: {yega_message}"
        
        # 2. 낙찰하한가 계산
        bid_limit_result, calc_message = self.calculate_bid_limit(
            yega_result, base_amount, a_value, bid_limit_rate
        )
        
        if bid_limit_result is None:
            return None, f"낙찰하한가 계산 실패: {calc_message}"
        
        logger.info("낙찰하한가 예측 완료")
        logger.info(f"예가: {bid_limit_result['prediction']['예가']:.4f}")
        logger.info(f"예정가격: {bid_limit_result['prediction']['예정가격']:,.0f}원")
        logger.info(f"낙찰하한가: {bid_limit_result['prediction']['낙찰하한가']:,.0f}원")
        
        return bid_limit_result, "예측 성공"

def main():
    """테스트 실행"""
    calculator = BidLimitCalculator()
    
    # 모델 로드
    if not calculator.load_yega_model():
        print("❌ 모델 로드 실패")
        return
    
    print("✅ 낙찰하한가 계산기 준비 완료")
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "서울 토목공사",
            "region": "서울특별시",
            "sector": "토목",
            "base_amount": 1000000000,
            "a_value": 150000000,
            "bid_limit_rate": 87.5
        },
        {
            "name": "경기 건축공사",
            "region": "경기도",
            "sector": "건축",
            "base_amount": 2000000000,
            "a_value": 200000000,
            "bid_limit_rate": 85.0,
            "price_variation": "-3/+3"
        },
        {
            "name": "부산 전기공사",
            "region": "부산광역시",
            "sector": "전기",
            "base_amount": 500000000,
            "a_value": 80000000,
            "bid_limit_rate": 90.0,
            "price_variation": "-2/+2"
        }
    ]
    
    print(f"\n🧪 {len(test_cases)}개 테스트 케이스 실행:")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"테스트 {i}: {test['name']}")
        print(f"{'='*60}")
        
        result, message = calculator.predict_bid_limit(
            region=test['region'],
            sector=test['sector'],
            base_amount=test['base_amount'],
            a_value=test['a_value'],
            bid_limit_rate=test['bid_limit_rate'],
            price_variation=test.get('price_variation')
        )
        
        if result:
            pred = result['prediction']
            print(f"✅ 예측 성공:")
            print(f"   📊 예가: {pred['예가']:.4f}")
            print(f"   💰 예정가격: {pred['예정가격']:,.0f}원")
            print(f"   🎯 낙찰하한가: {pred['낙찰하한가']:,.0f}원")
            print(f"   📋 낙찰하한율: {test['bid_limit_rate']}%")
            
            if pred.get('예가변동폭_적용'):
                var_info = pred['예가_범위']
                print(f"   📈 예가변동폭: {var_info['예가변동폭']}")
                print(f"   📊 예가범위: {var_info['최소']:.4f} ~ {var_info['최대']:.4f}")
        else:
            print(f"❌ 예측 실패: {message}")

if __name__ == "__main__":
    main()