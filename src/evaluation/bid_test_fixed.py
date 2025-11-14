#!/usr/bin/env python3
"""
Bid list 데이터로 예측 테스트 (수정 버전)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
from datetime import datetime
import json

# 모델 모듈 import를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent / "models"))
from bid_limit_calculator import BidLimitCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BidListPredictionTest:
    """Bid list 데이터를 사용한 예측 테스트"""
    
    def __init__(self, data_dir="data/processed", output_dir="docs/analysis"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.calculator = BidLimitCalculator()
        self.df_bid = None
        self.test_results = []
        
    def load_bid_list_data(self):
        """Bid list 데이터 로드"""
        logger.info("Bid list 데이터 로딩...")
        
        try:
            bid_files = list(self.data_dir.glob("Bid_list_processed_*.xlsx"))
            if not bid_files:
                logger.error("전처리된 Bid list 데이터를 찾을 수 없습니다.")
                return False
            
            latest_file = max(bid_files, key=lambda x: x.stat().st_mtime)
            self.df_bid = pd.read_excel(latest_file)
            
            logger.info(f"Bid list 데이터 로드: {len(self.df_bid)}행, {len(self.df_bid.columns)}열")
            logger.info(f"컬럼: {list(self.df_bid.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {str(e)}")
            return False
    
    def prepare_test_data(self, sample_size=30):
        """테스트용 데이터 준비"""
        logger.info("테스트 데이터 준비...")
        
        if self.df_bid is None:
            return False
        
        # 필요한 컬럼 확인
        required_cols = ['지역', '업종', '기초금액']
        missing_cols = [col for col in required_cols if col not in self.df_bid.columns]
        if missing_cols:
            logger.error(f"필요한 컬럼이 없습니다: {missing_cols}")
            return False
        
        # 유효한 데이터만 선택
        df_valid = self.df_bid.copy()
        df_valid['기초금액'] = pd.to_numeric(df_valid['기초금액'], errors='coerce')
        
        # A값 처리 - 없으면 기본값 사용
        if 'A값' in df_valid.columns:
            df_valid['A값'] = pd.to_numeric(df_valid['A값'], errors='coerce')
        else:
            # A값이 없으면 기초금액의 10%로 추정
            df_valid['A값'] = df_valid['기초금액'] * 0.1
        
        # 결측값 제거
        df_valid = df_valid.dropna(subset=['지역', '업종', '기초금액'])
        df_valid = df_valid[df_valid['기초금액'] > 0]
        
        # A값이 0이거나 음수인 경우 기본값 설정
        df_valid.loc[df_valid['A값'] <= 0, 'A값'] = df_valid['기초금액'] * 0.1
        
        if len(df_valid) == 0:
            logger.error("유효한 테스트 데이터가 없습니다.")
            return False
        
        # 샘플링
        if len(df_valid) > sample_size:
            self.test_data = df_valid.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            self.test_data = df_valid.reset_index(drop=True)
        
        logger.info(f"테스트 데이터 준비 완료: {len(self.test_data)}개")
        self._summarize_test_data()
        
        return True
    
    def _summarize_test_data(self):
        """테스트 데이터 요약"""
        logger.info("테스트 데이터 분포:")
        
        region_counts = self.test_data['지역'].value_counts()
        logger.info(f"  지역 (상위 5개): {dict(region_counts.head())}")
        
        sector_counts = self.test_data['업종'].value_counts()
        logger.info(f"  업종 (상위 5개): {dict(sector_counts.head())}")
        
        base_amount = self.test_data['기초금액']
        logger.info(f"  기초금액 평균: {base_amount.mean():,.0f}원")
        logger.info(f"  기초금액 범위: {base_amount.min():,.0f} ~ {base_amount.max():,.0f}원")
    
    def run_predictions(self):
        """예측 실행"""
        logger.info("예측 실행 시작...")
        
        if not hasattr(self, 'test_data'):
            return False
        
        if not self.calculator.load_yega_model():
            return False
        
        self.test_results = []
        successful_predictions = 0
        
        for idx, row in self.test_data.iterrows():
            try:
                region = row['지역']
                sector = row['업종']
                base_amount = float(row['기초금액'])
                a_value = float(row['A값'])
                
                # 발주기관과 예가변동폭 처리
                agency = row.get('발주기관', None)
                if pd.isna(agency):
                    agency = None
                
                price_variation = row.get('예가변동폭', None)
                if pd.isna(price_variation):
                    price_variation = None
                
                bid_limit_rate = row.get('낙찰하한율', 87.5)
                if pd.isna(bid_limit_rate):
                    bid_limit_rate = 87.5
                else:
                    bid_limit_rate = float(bid_limit_rate)
                
                # 예측 실행
                result, message = self.calculator.predict_bid_limit(
                    region=region,
                    sector=sector,
                    base_amount=base_amount,
                    a_value=a_value,
                    bid_limit_rate=bid_limit_rate,
                    agency=agency,
                    price_variation=price_variation
                )
                
                if result:
                    prediction_result = {
                        'index': idx,
                        'success': True,
                        'input': {
                            '지역': region,
                            '업종': sector,
                            '기초금액': base_amount,
                            'A값': a_value,
                            '낙찰하한율': bid_limit_rate
                        },
                        'output': {
                            '예가': result['prediction']['예가'],
                            '예정가격': result['prediction']['예정가격'],
                            '낙찰하한가': result['prediction']['낙찰하한가']
                        }
                    }
                    successful_predictions += 1
                else:
                    prediction_result = {
                        'index': idx,
                        'success': False,
                        'error': message
                    }
                
                self.test_results.append(prediction_result)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"진행상황: {idx + 1}/{len(self.test_data)} 완료")
                    
            except Exception as e:
                logger.error(f"인덱스 {idx} 예측 실패: {str(e)}")
                self.test_results.append({
                    'index': idx,
                    'success': False,
                    'error': str(e)
                })
        
        success_rate = (successful_predictions / len(self.test_data)) * 100
        logger.info(f"예측 완료: {successful_predictions}/{len(self.test_data)}개 성공 ({success_rate:.1f}%)")
        
        return True
    
    def analyze_results(self):
        """예측 결과 분석"""
        logger.info("예측 결과 분석 시작...")
        
        successful_results = [r for r in self.test_results if r['success']]
        failed_results = [r for r in self.test_results if not r['success']]
        
        analysis = {
            'summary': {
                '총_테스트': len(self.test_results),
                '성공': len(successful_results),
                '실패': len(failed_results),
                '성공률': len(successful_results) / len(self.test_results) * 100
            }
        }
        
        if successful_results:
            yegas = [r['output']['예가'] for r in successful_results]
            bid_limits = [r['output']['낙찰하한가'] for r in successful_results]
            
            analysis['prediction_statistics'] = {
                '예가': {
                    '평균': float(np.mean(yegas)),
                    '표준편차': float(np.std(yegas)),
                    '최소': float(np.min(yegas)),
                    '최대': float(np.max(yegas))
                },
                '낙찰하한가': {
                    '평균': float(np.mean(bid_limits)),
                    '최소': float(np.min(bid_limits)),
                    '최대': float(np.max(bid_limits))
                }
            }
            
            logger.info(f"\\n📊 예측 결과 통계:")
            logger.info(f"성공률: {analysis['summary']['성공률']:.1f}%")
            
            stats = analysis['prediction_statistics']
            logger.info(f"예가 평균: {stats['예가']['평균']:.4f}")
            logger.info(f"예가 범위: {stats['예가']['최소']:.4f} ~ {stats['예가']['최대']:.4f}")
            logger.info(f"낙찰하한가 평균: {stats['낙찰하한가']['평균']:,.0f}원")
        
        self.analysis_results = analysis
        return True
    
    def save_results(self):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            results_path = self.output_dir / f"bid_list_test_results_{timestamp}.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'test_results': self.test_results,
                    'analysis': getattr(self, 'analysis_results', {}),
                    'metadata': {
                        'test_date': timestamp,
                        'total_tests': len(self.test_results)
                    }
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"결과 저장: {results_path}")
            return True
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {str(e)}")
            return False
    
    def run_full_test(self, sample_size=30):
        """전체 테스트 실행"""
        logger.info("=== Bid List 예측 테스트 시작 ===")
        
        steps = [
            ("데이터 로드", self.load_bid_list_data),
            ("테스트 데이터 준비", lambda: self.prepare_test_data(sample_size)),
            ("예측 실행", self.run_predictions),
            ("결과 분석", self.analyze_results),
            ("결과 저장", self.save_results)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\\n{step_name} 실행 중...")
            if not step_func():
                logger.error(f"{step_name} 실패")
                return False
        
        logger.info("\\n=== Bid List 예측 테스트 완료 ===")
        return True

def main():
    tester = BidListPredictionTest()
    success = tester.run_full_test(sample_size=20)
    
    if success:
        print("\\n✅ Bid List 예측 테스트가 성공적으로 완료되었습니다!")
        
        if hasattr(tester, 'analysis_results'):
            summary = tester.analysis_results['summary']
            print(f"\\n📊 테스트 결과:")
            print(f"  총 테스트: {summary['총_테스트']}개")
            print(f"  성공: {summary['성공']}개")
            print(f"  성공률: {summary['성공률']:.1f}%")
    else:
        print("\\n❌ Bid List 예측 테스트 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main()