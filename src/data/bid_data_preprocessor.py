#!/usr/bin/env python3
"""
낙찰하한가 예측 데이터 전처리기
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BidDataPreprocessor:
    """낙찰하한가 예측을 위한 데이터 전처리기"""
    
    def __init__(self, data_dir="data/raw", output_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 처리 결과 저장용
        self.processing_log = {
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "errors": [],
            "summary": {}
        }
    
    def load_data(self):
        """Excel 파일들 로드"""
        try:
            logger.info("데이터 파일 로딩 시작...")
            
            # List of Successful Bidders_merged.xlsx 로드
            successful_file = self.data_dir / "List of Successful Bidders_merged.xlsx"
            self.df_successful = pd.read_excel(successful_file)
            logger.info(f"낙찰자 데이터 로드: {len(self.df_successful)}행, {len(self.df_successful.columns)}열")
            
            # Bid list_merged.xlsx 로드  
            bid_file = self.data_dir / "Bid list_merged.xlsx"
            self.df_bid = pd.read_excel(bid_file)
            logger.info(f"입찰 목록 데이터 로드: {len(self.df_bid)}행, {len(self.df_bid.columns)}열")
            
            # 컬럼 정보 출력
            logger.info(f"낙찰자 데이터 컬럼: {list(self.df_successful.columns)}")
            logger.info(f"입찰 목록 데이터 컬럼: {list(self.df_bid.columns)}")
            
            return True
            
        except Exception as e:
            error_msg = f"데이터 로딩 실패: {str(e)}"
            logger.error(error_msg)
            self.processing_log["errors"].append(error_msg)
            return False
    
    def step1_remove_empty_bid_limit(self):
        """1단계: List of Successful Bidders_merged에서 낙찰하한가 빈 행 삭제"""
        logger.info("1단계: 낙찰하한가 빈 행 삭제 시작...")
        
        original_count = len(self.df_successful)
        
        # 낙찰하한가 컬럼 확인
        bid_limit_column = None
        for col in self.df_successful.columns:
            if '낙찰하한가' in str(col):
                bid_limit_column = col
                break
        
        if bid_limit_column is None:
            error_msg = "낙찰하한가 컬럼을 찾을 수 없습니다"
            logger.error(error_msg)
            self.processing_log["errors"].append(error_msg)
            return False
        
        logger.info(f"낙찰하한가 컬럼 발견: {bid_limit_column}")
        
        # 빈 값 제거
        self.df_successful = self.df_successful.dropna(subset=[bid_limit_column])
        
        removed_count = original_count - len(self.df_successful)
        logger.info(f"삭제된 행: {removed_count}개 (전체 {original_count}개 중)")
        
        self.processing_log["steps"].append({
            "step": 1,
            "description": "낙찰하한가 빈 행 삭제",
            "original_count": original_count,
            "final_count": len(self.df_successful),
            "removed_count": removed_count
        })
        
        return True
    
    def step2_calculate_bid_limit_rate(self):
        """2단계: 낙찰하한율 컬럼 추가"""
        logger.info("2단계: 낙찰하한율 계산 시작...")
        
        # 필요한 컬럼들 찾기
        required_columns = {}
        column_mappings = {
            '낙찰하한가': ['낙찰하한가'],
            'A값': ['A값', 'A'],
            '예정가격': ['예정가격', '예정가']
        }
        
        for key, possible_names in column_mappings.items():
            found = False
            for col in self.df_successful.columns:
                for name in possible_names:
                    if name in str(col):
                        required_columns[key] = col
                        found = True
                        break
                if found:
                    break
            
            if not found:
                error_msg = f"{key} 컬럼을 찾을 수 없습니다"
                logger.error(error_msg)
                self.processing_log["errors"].append(error_msg)
                return False
        
        logger.info(f"사용할 컬럼들: {required_columns}")
        
        try:
            # 낙찰하한율 계산: (낙찰하한가 - A값) / (예정가격 - A값) * 100
            bid_limit = pd.to_numeric(self.df_successful[required_columns['낙찰하한가']], errors='coerce')
            a_value = pd.to_numeric(self.df_successful[required_columns['A값']], errors='coerce')
            estimated_price = pd.to_numeric(self.df_successful[required_columns['예정가격']], errors='coerce')
            
            # 분모가 0이 되는 경우 방지
            denominator = estimated_price - a_value
            
            # 낙찰하한율 계산
            bid_limit_rate = np.where(
                denominator != 0,
                (bid_limit - a_value) / denominator * 100,
                np.nan
            )
            
            self.df_successful['낙찰하한율'] = bid_limit_rate
            
            # 유효 범위 확인 (70% < 낙찰하한율 < 100%)
            valid_mask = (bid_limit_rate > 70) & (bid_limit_rate < 100)
            valid_count = valid_mask.sum()
            invalid_count = len(self.df_successful) - valid_count - pd.isna(bid_limit_rate).sum()
            
            logger.info(f"낙찰하한율 계산 완료")
            logger.info(f"유효 범위(70%-100%) 내 데이터: {valid_count}개")
            logger.info(f"유효 범위 밖 데이터: {invalid_count}개")
            logger.info(f"계산 불가 데이터: {pd.isna(bid_limit_rate).sum()}개")
            
            # 통계 정보
            stats = {
                "mean": float(np.nanmean(bid_limit_rate)),
                "min": float(np.nanmin(bid_limit_rate)),
                "max": float(np.nanmax(bid_limit_rate)),
                "valid_count": int(valid_count),
                "invalid_count": int(invalid_count),
                "nan_count": int(pd.isna(bid_limit_rate).sum())
            }
            
            self.processing_log["steps"].append({
                "step": 2,
                "description": "낙찰하한율 계산",
                "stats": stats
            })
            
            return True
            
        except Exception as e:
            error_msg = f"낙찰하한율 계산 실패: {str(e)}"
            logger.error(error_msg)
            self.processing_log["errors"].append(error_msg)
            return False
    
    def step3_remove_duplicates_bid_list(self):
        """3단계: Bid list_merged에서 공고번호 중복 제거 (A컬럼 번호가 가장 낮은 것만 유지)"""
        logger.info("3단계: Bid list_merged 중복 제거 시작...")
        
        original_count = len(self.df_bid)
        
        # A컬럼 (첫 번째 컬럼)과 C컬럼 (공고번호) 확인
        if len(self.df_bid.columns) < 3:
            error_msg = "Bid list_merged 파일에 필요한 컬럼이 부족합니다"
            logger.error(error_msg)
            self.processing_log["errors"].append(error_msg)
            return False
        
        # A컬럼 (인덱스 0), C컬럼 (인덱스 2) 사용
        a_column = self.df_bid.columns[0]  # A컬럼
        c_column = self.df_bid.columns[2]  # C컬럼 (공고번호)
        
        logger.info(f"A컬럼: {a_column}, C컬럼(공고번호): {c_column}")
        
        try:
            # A컬럼을 숫자로 변환
            self.df_bid[a_column] = pd.to_numeric(self.df_bid[a_column], errors='coerce')
            
            # 공고번호별로 A컬럼 값이 가장 작은 행만 유지
            self.df_bid = self.df_bid.loc[self.df_bid.groupby(c_column)[a_column].idxmin()]
            
            removed_count = original_count - len(self.df_bid)
            logger.info(f"중복 제거 완료: {removed_count}개 행 삭제 (전체 {original_count}개 중)")
            
            self.processing_log["steps"].append({
                "step": 3,
                "description": "Bid list 중복 제거",
                "original_count": original_count,
                "final_count": len(self.df_bid),
                "removed_count": removed_count
            })
            
            return True
            
        except Exception as e:
            error_msg = f"중복 제거 실패: {str(e)}"
            logger.error(error_msg)
            self.processing_log["errors"].append(error_msg)
            return False
    
    def step4_add_price_variation_column(self):
        """4단계: List of Successful Bidders_merged에 예가변동폭 컬럼 추가 (없을 경우만)"""
        logger.info("4단계: 예가변동폭 컬럼 추가 확인...")
        
        # 예가변동폭 컬럼 존재 여부 확인
        price_variation_exists = False
        for col in self.df_successful.columns:
            if '예가변동폭' in str(col):
                price_variation_exists = True
                logger.info(f"예가변동폭 컬럼이 이미 존재합니다: {col}")
                break
        
        if not price_variation_exists:
            self.df_successful['예가변동폭'] = np.nan
            logger.info("예가변동폭 컬럼을 새로 추가했습니다")
        
        self.processing_log["steps"].append({
            "step": 4,
            "description": "예가변동폭 컬럼 추가",
            "column_existed": price_variation_exists
        })
        
        return True
    
    def step5_merge_price_variation_data(self):
        """5단계: 공고번호 기준으로 예가변동폭 데이터 매칭"""
        logger.info("5단계: 예가변동폭 데이터 매칭 시작...")
        
        try:
            # 두 데이터프레임의 공고번호 컬럼 찾기 (C컬럼, 인덱스 2)
            if len(self.df_successful.columns) < 3 or len(self.df_bid.columns) < 3:
                error_msg = "공고번호 컬럼을 찾을 수 없습니다"
                logger.error(error_msg)
                self.processing_log["errors"].append(error_msg)
                return False
            
            successful_notice_col = self.df_successful.columns[2]  # C컬럼
            bid_notice_col = self.df_bid.columns[2]  # C컬럼
            
            logger.info(f"낙찰자 데이터 공고번호 컬럼: {successful_notice_col}")
            logger.info(f"입찰 목록 공고번호 컬럼: {bid_notice_col}")
            
            # Bid list에서 예가변동폭 컬럼 찾기
            price_variation_col_bid = None
            for col in self.df_bid.columns:
                if '예가변동폭' in str(col):
                    price_variation_col_bid = col
                    break
            
            if price_variation_col_bid is None:
                error_msg = "Bid list_merged에서 예가변동폭 컬럼을 찾을 수 없습니다"
                logger.error(error_msg)
                self.processing_log["errors"].append(error_msg)
                return False
            
            logger.info(f"Bid list 예가변동폭 컬럼: {price_variation_col_bid}")
            
            # 매칭 수행
            matched_count = 0
            total_successful_count = len(self.df_successful)
            
            for idx, row in self.df_successful.iterrows():
                notice_number = row[successful_notice_col]
                
                # Bid list에서 같은 공고번호 찾기
                matching_bid = self.df_bid[self.df_bid[bid_notice_col] == notice_number]
                
                if not matching_bid.empty:
                    # 예가변동폭 데이터 복사
                    price_variation_value = matching_bid.iloc[0][price_variation_col_bid]
                    self.df_successful.at[idx, '예가변동폭'] = price_variation_value
                    matched_count += 1
            
            match_rate = (matched_count / total_successful_count) * 100 if total_successful_count > 0 else 0
            
            logger.info(f"매칭 완료: {matched_count}개/{total_successful_count}개 ({match_rate:.1f}%)")
            
            self.processing_log["steps"].append({
                "step": 5,
                "description": "예가변동폭 데이터 매칭",
                "matched_count": matched_count,
                "total_count": total_successful_count,
                "match_rate": match_rate
            })
            
            return True
            
        except Exception as e:
            error_msg = f"데이터 매칭 실패: {str(e)}"
            logger.error(error_msg)
            self.processing_log["errors"].append(error_msg)
            return False
    
    def save_processed_data(self):
        """전처리된 데이터 저장"""
        logger.info("전처리된 데이터 저장 시작...")
        
        try:
            # 파일명에 타임스탬프 추가
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 전처리된 파일 저장
            successful_output = self.output_dir / f"List_of_Successful_Bidders_processed_{timestamp}.xlsx"
            bid_output = self.output_dir / f"Bid_list_processed_{timestamp}.xlsx"
            
            self.df_successful.to_excel(successful_output, index=False)
            self.df_bid.to_excel(bid_output, index=False)
            
            logger.info(f"낙찰자 데이터 저장: {successful_output}")
            logger.info(f"입찰 목록 데이터 저장: {bid_output}")
            
            # 처리 로그 저장
            log_output = self.output_dir / f"preprocessing_log_{timestamp}.json"
            self.processing_log["end_time"] = datetime.now().isoformat()
            self.processing_log["output_files"] = {
                "successful_bidders": str(successful_output),
                "bid_list": str(bid_output)
            }
            
            with open(log_output, 'w', encoding='utf-8') as f:
                json.dump(self.processing_log, f, ensure_ascii=False, indent=2)
            
            logger.info(f"처리 로그 저장: {log_output}")
            
            return True
            
        except Exception as e:
            error_msg = f"데이터 저장 실패: {str(e)}"
            logger.error(error_msg)
            self.processing_log["errors"].append(error_msg)
            return False
    
    def run_preprocessing(self):
        """전체 전처리 파이프라인 실행"""
        logger.info("=== 낙찰하한가 데이터 전처리 시작 ===")
        
        # 단계별 실행
        steps = [
            ("데이터 로딩", self.load_data),
            ("1단계: 낙찰하한가 빈 행 삭제", self.step1_remove_empty_bid_limit),
            ("2단계: 낙찰하한율 계산", self.step2_calculate_bid_limit_rate),
            ("3단계: Bid list 중복 제거", self.step3_remove_duplicates_bid_list),
            ("4단계: 예가변동폭 컬럼 추가", self.step4_add_price_variation_column),
            ("5단계: 예가변동폭 데이터 매칭", self.step5_merge_price_variation_data),
            ("데이터 저장", self.save_processed_data)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"실행 중: {step_name}")
            logger.info(f"{'='*50}")
            
            success = step_func()
            if not success:
                logger.error(f"{step_name} 실패로 전처리 중단")
                return False
        
        # 요약 정보
        self.processing_log["summary"] = {
            "successful_final_count": len(self.df_successful),
            "bid_final_count": len(self.df_bid),
            "total_errors": len(self.processing_log["errors"]),
            "completed_steps": len([step for step in self.processing_log["steps"]])
        }
        
        logger.info(f"\n{'='*50}")
        logger.info("=== 전처리 완료 ===")
        logger.info(f"최종 낙찰자 데이터: {len(self.df_successful)}행")
        logger.info(f"최종 입찰 목록 데이터: {len(self.df_bid)}행")
        logger.info(f"처리된 단계: {len(self.processing_log['steps'])}개")
        logger.info(f"발생한 오류: {len(self.processing_log['errors'])}개")
        logger.info(f"{'='*50}")
        
        return True

def main():
    """메인 실행 함수"""
    preprocessor = BidDataPreprocessor()
    success = preprocessor.run_preprocessing()
    
    if success:
        print("\n✅ 전처리가 성공적으로 완료되었습니다!")
    else:
        print("\n❌ 전처리 중 오류가 발생했습니다.")
        if preprocessor.processing_log["errors"]:
            print("오류 내용:")
            for error in preprocessor.processing_log["errors"]:
                print(f"  - {error}")

if __name__ == "__main__":
    main()