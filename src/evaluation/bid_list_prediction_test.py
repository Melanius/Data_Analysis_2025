#!/usr/bin/env python3
"""
Bid list 데이터로 예측 테스트
실제 입찰 공고 데이터를 사용한 낙찰하한가 예측
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
            # 최신 전처리된 파일 찾기
            bid_files = list(self.data_dir.glob("Bid_list_processed_*.xlsx"))
            if not bid_files:
                logger.error("전처리된 Bid list 데이터를 찾을 수 없습니다.")
                return False
            
            latest_file = max(bid_files, key=lambda x: x.stat().st_mtime)
            self.df_bid = pd.read_excel(latest_file)
            
            logger.info(f"Bid list 데이터 로드: {len(self.df_bid)}행, {len(self.df_bid.columns)}열")
            logger.info(f"파일: {latest_file.name}")
            
            # 컬럼 정보 출력
            logger.info(f"컬럼: {list(self.df_bid.columns)}")
            
            return True
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {str(e)}")
            return False
    
    def prepare_test_data(self, sample_size=100):
        """테스트용 데이터 준비"""
        logger.info("테스트 데이터 준비...")
        
        if self.df_bid is None:
            return False
        
        # 필요한 컬럼 확인
        required_cols = ['지역', '업종', '기초금액', 'A값', '예가변동폭']
        
        missing_cols = [col for col in required_cols if col not in self.df_bid.columns]
        if missing_cols:
            logger.error(f"필요한 컬럼이 없습니다: {missing_cols}")
            return False
        
        # 유효한 데이터만 선택
        df_valid = self.df_bid.copy()
        
        # 수치형 데이터 정리
        df_valid['기초금액'] = pd.to_numeric(df_valid['기초금액'], errors='coerce')
        df_valid['A값'] = pd.to_numeric(df_valid['A값'], errors='coerce')
        
        # 결측값이 없는 레코드만 선택
        for col in ['지역', '업종', '기초금액']:
            df_valid = df_valid.dropna(subset=[col])
        
        # A값이 있는 레코드만 선택 (예측에 필요)
        df_valid = df_valid.dropna(subset=['A값'])
        
        # 기초금액이 양수인 레코드만
        df_valid = df_valid[df_valid['기초금액'] > 0]
        df_valid = df_valid[df_valid['A값'] > 0]
        
        if len(df_valid) == 0:
            logger.error("유효한 테스트 데이터가 없습니다.")
            return False
        
        # 샘플링
        if len(df_valid) > sample_size:
            self.test_data = df_valid.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            self.test_data = df_valid.reset_index(drop=True)
        
        logger.info(f"테스트 데이터 준비 완료: {len(self.test_data)}개")
        
        # 데이터 분포 요약
        self._summarize_test_data()
        
        return True
    
    def _summarize_test_data(self):
        """테스트 데이터 요약"""
        logger.info("테스트 데이터 분포:")
        
        # 지역 분포
        region_counts = self.test_data['지역'].value_counts()
        logger.info(f"  지역 (상위 5개): {dict(region_counts.head())}")
        
        # 업종 분포  
        sector_counts = self.test_data['업종'].value_counts()
        logger.info(f"  업종 (상위 5개): {dict(sector_counts.head())}")
        
        # 기초금액 통계
        base_amount = self.test_data['기초금액']
        logger.info(f"  기초금액 평균: {base_amount.mean():,.0f}원")
        logger.info(f"  기초금액 범위: {base_amount.min():,.0f} ~ {base_amount.max():,.0f}원")
        
        # 예가변동폭 분포
        if '예가변동폭' in self.test_data.columns:
            variation_counts = self.test_data['예가변동폭'].value_counts()
            logger.info(f"  예가변동폭 (상위 3개): {dict(variation_counts.head(3))}")
    
    def run_predictions(self):
        """예측 실행"""
        logger.info("예측 실행 시작...")
        
        if not hasattr(self, 'test_data'):
            logger.error("테스트 데이터가 준비되지 않았습니다.")
            return False
        
        # 계산기 로드
        if not self.calculator.load_yega_model():
            logger.error("예가 예측 모델 로드 실패")
            return False
        
        self.test_results = []
        successful_predictions = 0
        
        for idx, row in self.test_data.iterrows():
            try:
                # 입력 데이터 준비
                region = row['지역']
                sector = row['업종']
                base_amount = float(row['기초금액'])
                a_value = float(row['A값'])
                
                # 발주기관 정보 (있으면)
                agency = row.get('발주기관', None)
                if pd.isna(agency):
                    agency = None
                
                # 예가변동폭 정보
                price_variation = row.get('예가변동폭', None)
                if pd.isna(price_variation):
                    price_variation = None
                
                # 낙찰하한율 (있으면)
                bid_limit_rate = row.get('낙찰하한율', 87.5)
                if pd.isna(bid_limit_rate):
                    bid_limit_rate = 87.5
                else:\n                    bid_limit_rate = float(bid_limit_rate)\n                \n                # 예측 실행\n                result, message = self.calculator.predict_bid_limit(\n                    region=region,\n                    sector=sector,\n                    base_amount=base_amount,\n                    a_value=a_value,\n                    bid_limit_rate=bid_limit_rate,\n                    agency=agency,\n                    price_variation=price_variation\n                )\n                \n                if result:\n                    prediction_result = {\n                        'index': idx,\n                        'success': True,\n                        'input': {\n                            '지역': region,\n                            '업종': sector,\n                            '기초금액': base_amount,\n                            'A값': a_value,\n                            '낙찰하한율': bid_limit_rate,\n                            '발주기관': agency,\n                            '예가변동폭': price_variation\n                        },\n                        'output': {\n                            '예가': result['prediction']['예가'],\n                            '예정가격': result['prediction']['예정가격'],\n                            '낙찰하한가': result['prediction']['낙찰하한가'],\n                            '예가변동폭_적용': result['prediction'].get('예가변동폭_적용', False)\n                        },\n                        'calculation_details': result.get('calculation_details', {})\n                    }\n                    \n                    successful_predictions += 1\n                else:\n                    prediction_result = {\n                        'index': idx,\n                        'success': False,\n                        'error': message,\n                        'input': {\n                            '지역': region,\n                            '업종': sector,\n                            '기초금액': base_amount,\n                            'A값': a_value\n                        }\n                    }\n                \n                self.test_results.append(prediction_result)\n                \n                # 진행상황 출력\n                if (idx + 1) % 10 == 0:\n                    logger.info(f\"진행상황: {idx + 1}/{len(self.test_data)} 완료\")\n                    \n            except Exception as e:\n                logger.error(f\"인덱스 {idx} 예측 실패: {str(e)}\")\n                error_result = {\n                    'index': idx,\n                    'success': False,\n                    'error': str(e)\n                }\n                self.test_results.append(error_result)\n        \n        success_rate = (successful_predictions / len(self.test_data)) * 100\n        logger.info(f\"예측 완료: {successful_predictions}/{len(self.test_data)}개 성공 ({success_rate:.1f}%)\")\n        \n        return True\n    \n    def analyze_results(self):\n        \"\"\"예측 결과 분석\"\"\"\n        logger.info(\"예측 결과 분석 시작...\")\n        \n        if not self.test_results:\n            logger.error(\"분석할 결과가 없습니다.\")\n            return False\n        \n        successful_results = [r for r in self.test_results if r['success']]\n        failed_results = [r for r in self.test_results if not r['success']]\n        \n        analysis = {\n            'summary': {\n                '총_테스트': len(self.test_results),\n                '성공': len(successful_results),\n                '실패': len(failed_results),\n                '성공률': len(successful_results) / len(self.test_results) * 100\n            },\n            'prediction_statistics': {},\n            'sample_predictions': []\n        }\n        \n        if successful_results:\n            # 예측값 통계\n            yegas = [r['output']['예가'] for r in successful_results]\n            estimated_prices = [r['output']['예정가격'] for r in successful_results]\n            bid_limits = [r['output']['낙찰하한가'] for r in successful_results]\n            \n            analysis['prediction_statistics'] = {\n                '예가': {\n                    '평균': float(np.mean(yegas)),\n                    '표준편차': float(np.std(yegas)),\n                    '최소': float(np.min(yegas)),\n                    '최대': float(np.max(yegas))\n                },\n                '예정가격': {\n                    '평균': float(np.mean(estimated_prices)),\n                    '최소': float(np.min(estimated_prices)),\n                    '최대': float(np.max(estimated_prices))\n                },\n                '낙찰하한가': {\n                    '평균': float(np.mean(bid_limits)),\n                    '최소': float(np.min(bid_limits)),\n                    '최대': float(np.max(bid_limits))\n                }\n            }\n            \n            # 샘플 예측 결과 (상위 5개)\n            analysis['sample_predictions'] = successful_results[:5]\n            \n            # 결과 출력\n            logger.info(\"\\n📊 예측 결과 통계:\")\n            logger.info(f\"성공률: {analysis['summary']['성공률']:.1f}%\")\n            \n            stats = analysis['prediction_statistics']\n            logger.info(f\"예가 평균: {stats['예가']['평균']:.4f}\")\n            logger.info(f\"예가 범위: {stats['예가']['최소']:.4f} ~ {stats['예가']['최대']:.4f}\")\n            \n            logger.info(f\"예정가격 평균: {stats['예정가격']['평균']:,.0f}원\")\n            logger.info(f\"낙찰하한가 평균: {stats['낙찰하한가']['평균']:,.0f}원\")\n        \n        # 실패 원인 분석\n        if failed_results:\n            error_types = {}\n            for result in failed_results:\n                error = result.get('error', 'Unknown')\n                error_types[error] = error_types.get(error, 0) + 1\n            \n            analysis['error_analysis'] = error_types\n            logger.info(f\"\\n❌ 실패 원인 분석:\")\n            for error, count in error_types.items():\n                logger.info(f\"  {error}: {count}건\")\n        \n        self.analysis_results = analysis\n        return True\n    \n    def save_results(self):\n        \"\"\"결과 저장\"\"\"\n        logger.info(\"결과 저장 시작...\")\n        \n        timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n        \n        try:\n            # 상세 결과 저장\n            results_path = self.output_dir / f\"bid_list_prediction_results_{timestamp}.json\"\n            with open(results_path, 'w', encoding='utf-8') as f:\n                json.dump({\n                    'test_results': self.test_results,\n                    'analysis': getattr(self, 'analysis_results', {}),\n                    'metadata': {\n                        'test_date': timestamp,\n                        'total_tests': len(self.test_results),\n                        'model_used': 'Ridge_Regression (Yega Prediction)'\n                    }\n                }, f, ensure_ascii=False, indent=2)\n            \n            logger.info(f\"상세 결과 저장: {results_path}\")\n            \n            # 요약 보고서 저장\n            if hasattr(self, 'analysis_results'):\n                summary_path = self.output_dir / f\"bid_list_prediction_summary_{timestamp}.json\"\n                with open(summary_path, 'w', encoding='utf-8') as f:\n                    json.dump(self.analysis_results, f, ensure_ascii=False, indent=2)\n                \n                logger.info(f\"요약 보고서 저장: {summary_path}\")\n            \n            return True\n            \n        except Exception as e:\n            logger.error(f\"결과 저장 실패: {str(e)}\")\n            return False\n    \n    def run_full_test(self, sample_size=50):\n        \"\"\"전체 테스트 실행\"\"\"\n        logger.info(\"=== Bid List 예측 테스트 시작 ===\")\n        \n        # 1. 데이터 로드\n        if not self.load_bid_list_data():\n            return False\n        \n        # 2. 테스트 데이터 준비\n        if not self.prepare_test_data(sample_size):\n            return False\n        \n        # 3. 예측 실행\n        if not self.run_predictions():\n            return False\n        \n        # 4. 결과 분석\n        if not self.analyze_results():\n            return False\n        \n        # 5. 결과 저장\n        if not self.save_results():\n            return False\n        \n        logger.info(\"\\n=== Bid List 예측 테스트 완료 ===\")\n        return True\n\ndef main():\n    \"\"\"메인 실행 함수\"\"\"\n    tester = BidListPredictionTest()\n    success = tester.run_full_test(sample_size=30)  # 30개 샘플로 테스트\n    \n    if success:\n        print(\"\\n✅ Bid List 예측 테스트가 성공적으로 완료되었습니다!\")\n        \n        if hasattr(tester, 'analysis_results'):\n            summary = tester.analysis_results['summary']\n            print(f\"\\n📊 테스트 결과:\")\n            print(f\"  총 테스트: {summary['총_테스트']}개\")\n            print(f\"  성공: {summary['성공']}개\")\n            print(f\"  성공률: {summary['성공률']:.1f}%\")\n    else:\n        print(\"\\n❌ Bid List 예측 테스트 중 오류가 발생했습니다.\")\n\nif __name__ == \"__main__\":\n    main()"