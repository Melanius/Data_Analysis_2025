#!/usr/bin/env python3
"""
전처리된 데이터 EDA 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import logging

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'NanumGothic']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BidDataEDA:
    """낙찰하한가 예측을 위한 EDA 분석기"""
    
    def __init__(self, data_dir="data/processed", output_dir="docs/analysis"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.df_successful = None
        self.df_bid = None
        self.analysis_results = {}
    
    def load_processed_data(self):
        """최신 전처리된 데이터 로드"""
        logger.info("전처리된 데이터 로딩...")
        
        try:
            # 최신 파일 찾기
            successful_files = list(self.data_dir.glob("List_of_Successful_Bidders_processed_*.xlsx"))
            bid_files = list(self.data_dir.glob("Bid_list_processed_*.xlsx"))
            
            if not successful_files or not bid_files:
                logger.error("전처리된 파일을 찾을 수 없습니다.")
                return False
            
            # 가장 최신 파일 사용
            latest_successful = max(successful_files, key=lambda x: x.stat().st_mtime)
            latest_bid = max(bid_files, key=lambda x: x.stat().st_mtime)
            
            self.df_successful = pd.read_excel(latest_successful)
            self.df_bid = pd.read_excel(latest_bid)
            
            logger.info(f"낙찰자 데이터 로드: {len(self.df_successful)}행")
            logger.info(f"입찰 목록 데이터 로드: {len(self.df_bid)}행")
            
            return True
            
        except Exception as e:
            logger.error(f"데이터 로드 실패: {str(e)}")
            return False
    
    def analyze_target_variable(self):
        """타겟 변수 (예가) 분석"""
        logger.info("타겟 변수 분석 시작...")
        
        # 예가/기초(100%) 분석
        yega_col = '예가/기초(100%)'
        if yega_col not in self.df_successful.columns:
            logger.error("예가/기초(100%) 컬럼을 찾을 수 없습니다.")
            return
        
        yega = pd.to_numeric(self.df_successful[yega_col], errors='coerce')
        valid_yega = yega.dropna()
        
        # 기본 통계
        stats = {
            "count": len(valid_yega),
            "mean": float(valid_yega.mean()),
            "std": float(valid_yega.std()),
            "min": float(valid_yega.min()),
            "max": float(valid_yega.max()),
            "q25": float(valid_yega.quantile(0.25)),
            "q50": float(valid_yega.quantile(0.5)),
            "q75": float(valid_yega.quantile(0.75))
        }
        
        # 범위별 분포
        range_analysis = {
            "below_0.8": (valid_yega < 0.8).sum(),
            "0.8_to_0.9": ((valid_yega >= 0.8) & (valid_yega < 0.9)).sum(),
            "0.9_to_1.0": ((valid_yega >= 0.9) & (valid_yega < 1.0)).sum(),
            "1.0_to_1.1": ((valid_yega >= 1.0) & (valid_yega < 1.1)).sum(),
            "above_1.1": (valid_yega >= 1.1).sum()
        }
        
        self.analysis_results['target_analysis'] = {
            "basic_stats": stats,
            "range_distribution": range_analysis
        }
        
        logger.info(f"예가 평균: {stats['mean']:.4f}")
        logger.info(f"예가 표준편차: {stats['std']:.4f}")
        logger.info(f"예가 범위: {stats['min']:.4f} ~ {stats['max']:.4f}")
    
    def analyze_categorical_features(self):
        """범주형 변수 분석"""
        logger.info("범주형 변수 분석 시작...")
        
        categorical_features = ['지역', '업종', '발주기관']
        categorical_analysis = {}
        
        for feature in categorical_features:
            if feature in self.df_successful.columns:
                value_counts = self.df_successful[feature].value_counts()
                
                categorical_analysis[feature] = {
                    "unique_count": len(value_counts),
                    "top_10": dict(value_counts.head(10)),
                    "missing_count": self.df_successful[feature].isnull().sum()
                }
                
                logger.info(f"{feature}: {len(value_counts)}개 고유값")
        
        self.analysis_results['categorical_analysis'] = categorical_analysis
    
    def analyze_numeric_features(self):
        """수치형 변수 분석"""
        logger.info("수치형 변수 분석 시작...")
        
        numeric_features = ['기초금액', 'A값', '예정가격', '낙찰하한가', '낙찰하한율']
        numeric_analysis = {}
        
        for feature in numeric_features:
            if feature in self.df_successful.columns:
                numeric_data = pd.to_numeric(self.df_successful[feature], errors='coerce')
                valid_data = numeric_data.dropna()
                
                if len(valid_data) > 0:
                    stats = {
                        "count": len(valid_data),
                        "mean": float(valid_data.mean()),
                        "std": float(valid_data.std()),
                        "min": float(valid_data.min()),
                        "max": float(valid_data.max()),
                        "q25": float(valid_data.quantile(0.25)),
                        "q50": float(valid_data.quantile(0.5)),
                        "q75": float(valid_data.quantile(0.75)),
                        "missing_count": len(numeric_data) - len(valid_data)
                    }
                    
                    numeric_analysis[feature] = stats
                    logger.info(f"{feature}: 평균 {stats['mean']:,.0f}, 결측값 {stats['missing_count']}개")
        
        self.analysis_results['numeric_analysis'] = numeric_analysis
    
    def analyze_business_logic(self):
        """비즈니스 로직 검증"""
        logger.info("비즈니스 로직 검증 시작...")
        
        # 필요한 컬럼 확인
        required_cols = ['기초금액', '예가/기초(100%)', '예정가격', '낙찰하한가', 'A값', '낙찰하한율']
        missing_cols = [col for col in required_cols if col not in self.df_successful.columns]
        
        if missing_cols:
            logger.error(f"필요한 컬럼이 없습니다: {missing_cols}")
            return
        
        # 데이터 준비
        df_clean = self.df_successful[required_cols].copy()
        for col in required_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 결측값 제거
        df_clean = df_clean.dropna()
        
        if len(df_clean) == 0:
            logger.error("유효한 데이터가 없습니다.")
            return
        
        # 공식 검증 1: 예정가격 = 기초금액 × 예가
        df_clean['예정가격_계산'] = df_clean['기초금액'] * df_clean['예가/기초(100%)']
        price_diff = abs(df_clean['예정가격'] - df_clean['예정가격_계산'])
        price_match_rate = (price_diff < 1000).mean() * 100  # 1000원 이하 차이를 정확한 것으로 간주
        
        # 공식 검증 2: 낙찰하한율 = (낙찰하한가 - A값) / (예정가격 - A값)
        df_clean['낙찰하한율_계산'] = (df_clean['낙찰하한가'] - df_clean['A값']) / (df_clean['예정가격'] - df_clean['A값']) * 100
        rate_diff = abs(df_clean['낙찰하한율'] - df_clean['낙찰하한율_계산'])
        rate_match_rate = (rate_diff < 1).mean() * 100  # 1% 이하 차이를 정확한 것으로 간주
        
        business_logic = {
            "validated_records": len(df_clean),
            "total_records": len(self.df_successful),
            "price_formula_accuracy": float(price_match_rate),
            "rate_formula_accuracy": float(rate_match_rate),
            "price_mean_diff": float(price_diff.mean()),
            "rate_mean_diff": float(rate_diff.mean())
        }
        
        self.analysis_results['business_logic'] = business_logic
        
        logger.info(f"검증된 레코드: {len(df_clean)}개")
        logger.info(f"예정가격 공식 정확도: {price_match_rate:.1f}%")
        logger.info(f"낙찰하한율 공식 정확도: {rate_match_rate:.1f}%")
    
    def analyze_bid_list_features(self):
        """Bid list 데이터 피쳐 분석"""
        logger.info("Bid list 피쳐 분석 시작...")
        
        if self.df_bid is None:
            logger.error("Bid list 데이터가 없습니다.")
            return
        
        # 예가변동폭 분석
        price_variation_col = '예가변동폭'
        if price_variation_col in self.df_bid.columns:
            variation_counts = self.df_bid[price_variation_col].value_counts()
            
            # 수치형 변동폭 추출 시도
            variation_patterns = {}
            for var in variation_counts.index:
                if pd.notna(var):
                    var_str = str(var)
                    variation_patterns[var_str] = int(variation_counts[var])
            
            bid_analysis = {
                "total_records": len(self.df_bid),
                "price_variation_patterns": variation_patterns,
                "missing_variation": self.df_bid[price_variation_col].isnull().sum()
            }
            
            logger.info(f"Bid list 총 레코드: {len(self.df_bid)}개")
            logger.info(f"예가변동폭 패턴: {list(variation_patterns.keys())[:10]}")
        else:
            bid_analysis = {"error": "예가변동폭 컬럼을 찾을 수 없습니다."}
        
        self.analysis_results['bid_list_analysis'] = bid_analysis
    
    def create_visualizations(self):
        """시각화 생성"""
        logger.info("시각화 생성 시작...")
        
        try:
            # 서브플롯 생성
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('낙찰하한가 예측 데이터 EDA', fontsize=16)
            
            # 1. 예가 분포
            yega_col = '예가/기초(100%)'
            if yega_col in self.df_successful.columns:
                yega = pd.to_numeric(self.df_successful[yega_col], errors='coerce').dropna()
                axes[0,0].hist(yega, bins=50, alpha=0.7, color='skyblue')
                axes[0,0].set_title('예가 분포')
                axes[0,0].set_xlabel('예가/기초(100%)')
                axes[0,0].set_ylabel('빈도')
                axes[0,0].axvline(yega.mean(), color='red', linestyle='--', label=f'평균: {yega.mean():.3f}')
                axes[0,0].legend()
            
            # 2. 지역별 분포
            if '지역' in self.df_successful.columns:
                top_regions = self.df_successful['지역'].value_counts().head(15)
                axes[0,1].bar(range(len(top_regions)), top_regions.values, color='lightgreen')
                axes[0,1].set_title('상위 15개 지역별 분포')
                axes[0,1].set_xlabel('지역')
                axes[0,1].set_ylabel('건수')
                axes[0,1].set_xticks(range(len(top_regions)))
                axes[0,1].set_xticklabels(top_regions.index, rotation=45, ha='right')
            
            # 3. 업종별 분포
            if '업종' in self.df_successful.columns:
                top_sectors = self.df_successful['업종'].value_counts().head(10)
                axes[0,2].bar(range(len(top_sectors)), top_sectors.values, color='orange')
                axes[0,2].set_title('상위 10개 업종별 분포')
                axes[0,2].set_xlabel('업종')
                axes[0,2].set_ylabel('건수')
                axes[0,2].set_xticks(range(len(top_sectors)))
                axes[0,2].set_xticklabels(top_sectors.index, rotation=45, ha='right')
            
            # 4. 기초금액 분포 (로그 스케일)
            if '기초금액' in self.df_successful.columns:
                base_amount = pd.to_numeric(self.df_successful['기초금액'], errors='coerce').dropna()
                log_amount = np.log10(base_amount[base_amount > 0])
                axes[1,0].hist(log_amount, bins=30, alpha=0.7, color='purple')
                axes[1,0].set_title('기초금액 분포 (로그 스케일)')
                axes[1,0].set_xlabel('log10(기초금액)')
                axes[1,0].set_ylabel('빈도')
            
            # 5. 낙찰하한율 분포
            if '낙찰하한율' in self.df_successful.columns:
                bid_rate = pd.to_numeric(self.df_successful['낙찰하한율'], errors='coerce').dropna()
                if len(bid_rate) > 0:
                    axes[1,1].hist(bid_rate, bins=30, alpha=0.7, color='coral')
                    axes[1,1].set_title('낙찰하한율 분포')
                    axes[1,1].set_xlabel('낙찰하한율 (%)')
                    axes[1,1].set_ylabel('빈도')
                    axes[1,1].axvline(bid_rate.mean(), color='red', linestyle='--', 
                                     label=f'평균: {bid_rate.mean():.1f}%')
                    axes[1,1].legend()
            
            # 6. 예가 vs 기초금액 산점도
            if yega_col in self.df_successful.columns and '기초금액' in self.df_successful.columns:
                yega = pd.to_numeric(self.df_successful[yega_col], errors='coerce')
                base_amount = pd.to_numeric(self.df_successful['기초금액'], errors='coerce')
                
                # 유효한 데이터만 선택
                valid_mask = yega.notna() & base_amount.notna() & (base_amount > 0)
                yega_valid = yega[valid_mask]
                amount_valid = base_amount[valid_mask]
                
                # 샘플링 (너무 많으면)
                if len(yega_valid) > 2000:
                    sample_idx = np.random.choice(len(yega_valid), 2000, replace=False)
                    yega_valid = yega_valid.iloc[sample_idx]
                    amount_valid = amount_valid.iloc[sample_idx]
                
                axes[1,2].scatter(np.log10(amount_valid), yega_valid, alpha=0.6, s=1)
                axes[1,2].set_title('예가 vs 기초금액 관계')
                axes[1,2].set_xlabel('log10(기초금액)')
                axes[1,2].set_ylabel('예가/기초(100%)')
            
            plt.tight_layout()
            
            # 저장
            viz_path = self.output_dir / "eda_visualizations.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"시각화 저장: {viz_path}")
            
        except Exception as e:
            logger.error(f"시각화 생성 실패: {str(e)}")
    
    def generate_feature_correlation(self):
        """피쳐 간 상관관계 분석"""
        logger.info("상관관계 분석 시작...")
        
        # 수치형 컬럼들 선택
        numeric_cols = ['예가/기초(100%)', '기초금액', 'A값', '예정가격', '낙찰하한가', '낙찰하한율']
        existing_cols = [col for col in numeric_cols if col in self.df_successful.columns]
        
        if len(existing_cols) < 2:
            logger.warning("상관관계 분석을 위한 충분한 수치형 컬럼이 없습니다.")
            return
        
        # 수치형 데이터 준비
        df_numeric = self.df_successful[existing_cols].copy()
        for col in existing_cols:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
        
        # 결측값 제거
        df_numeric = df_numeric.dropna()
        
        if len(df_numeric) > 0:
            # 상관관계 계산
            correlation_matrix = df_numeric.corr()
            
            # 히트맵 생성
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f')
            plt.title('피쳐 간 상관관계')
            
            corr_path = self.output_dir / "feature_correlation.png"
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            # 상관관계 결과 저장
            self.analysis_results['correlation_analysis'] = correlation_matrix.to_dict()
            
            logger.info(f"상관관계 히트맵 저장: {corr_path}")
        else:
            logger.warning("유효한 수치형 데이터가 없어 상관관계 분석을 수행할 수 없습니다.")
    
    def save_analysis_results(self):
        """분석 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self.output_dir / f"eda_analysis_results_{timestamp}.json"
        
        # JSON 직렬화 가능하도록 변환
        serializable_results = {}
        for key, value in self.analysis_results.items():
            if isinstance(value, (dict, list, str, int, float, bool)):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        # 메타데이터 추가
        serializable_results['metadata'] = {
            "analysis_date": timestamp,
            "successful_records": len(self.df_successful) if self.df_successful is not None else 0,
            "bid_records": len(self.df_bid) if self.df_bid is not None else 0
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"분석 결과 저장: {result_path}")
        return result_path
    
    def run_full_eda(self):
        """전체 EDA 분석 실행"""
        logger.info("=== 낙찰하한가 예측 EDA 분석 시작 ===")
        
        # 데이터 로드
        if not self.load_processed_data():
            return False
        
        # 분석 수행
        analysis_steps = [
            ("타겟 변수 분석", self.analyze_target_variable),
            ("범주형 변수 분석", self.analyze_categorical_features),
            ("수치형 변수 분석", self.analyze_numeric_features),
            ("비즈니스 로직 검증", self.analyze_business_logic),
            ("Bid list 피쳐 분석", self.analyze_bid_list_features),
            ("상관관계 분석", self.generate_feature_correlation),
            ("시각화 생성", self.create_visualizations),
            ("결과 저장", self.save_analysis_results)
        ]
        
        for step_name, step_func in analysis_steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"실행 중: {step_name}")
            logger.info(f"{'='*50}")
            
            try:
                step_func()
                logger.info(f"{step_name} 완료")
            except Exception as e:
                logger.error(f"{step_name} 실패: {str(e)}")
        
        logger.info(f"\n{'='*50}")
        logger.info("=== EDA 분석 완료 ===")
        logger.info(f"{'='*50}")
        
        return True

def main():
    """메인 실행 함수"""
    eda = BidDataEDA()
    success = eda.run_full_eda()
    
    if success:
        print("\n✅ EDA 분석이 성공적으로 완료되었습니다!")
        print(f"📊 분석 결과: {eda.output_dir}")
    else:
        print("\n❌ EDA 분석 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main()