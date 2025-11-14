#!/usr/bin/env python3
"""
데이터 분석 및 EDA 모듈
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'NanumGothic']
plt.rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    def __init__(self, data_dir="../data/raw"):
        self.data_dir = Path(data_dir)
        self.df = None
        self.analysis_results = {}
        
    def load_data(self, filename="List of Successful Bidders_merged.xlsx"):
        """기존 Excel 데이터 로드"""
        try:
            file_path = self.data_dir / filename
            self.df = pd.read_excel(file_path)
            print(f"데이터 로드 완료: {len(self.df)}개 레코드")
            print(f"컬럼: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            return False
    
    def basic_info(self):
        """기본 데이터 정보"""
        if self.df is None:
            return "데이터가 로드되지 않았습니다."
        
        info = {
            "총_레코드_수": len(self.df),
            "컬럼_수": len(self.df.columns),
            "데이터_타입": dict(self.df.dtypes),
            "결측값": dict(self.df.isnull().sum()),
            "중복_레코드": self.df.duplicated().sum()
        }
        
        self.analysis_results['basic_info'] = info
        return info
    
    def create_target_variable(self):
        """타겟 변수 생성: 예가 = 예정가격 / 기초금액"""
        if self.df is None:
            return False
        
        try:
            # 숫자형으로 변환
            self.df['예정가격_numeric'] = pd.to_numeric(self.df['예정가격'], errors='coerce')
            self.df['기초금액_numeric'] = pd.to_numeric(self.df['기초금액'], errors='coerce')
            
            # 예가 계산
            self.df['예가'] = self.df['예정가격_numeric'] / self.df['기초금액_numeric']
            
            print(f"예가 변수 생성 완료")
            print(f"유효한 예가 값: {self.df['예가'].notna().sum()}개")
            print(f"예가 범위: {self.df['예가'].min():.4f} ~ {self.df['예가'].max():.4f}")
            
            return True
        except Exception as e:
            print(f"타겟 변수 생성 실패: {e}")
            return False
    
    def analyze_features(self):
        """피쳐 분석"""
        if self.df is None:
            return {}
        
        analysis = {}
        
        # 카테고리 변수 분석
        categorical_features = ['지역', '업종', '발주기관']
        for feature in categorical_features:
            if feature in self.df.columns:
                analysis[feature] = {
                    "고유값_수": self.df[feature].nunique(),
                    "상위_10개": dict(self.df[feature].value_counts().head(10)),
                    "결측값": self.df[feature].isnull().sum()
                }
        
        # 수치 변수 분석
        numeric_features = ['기초금액_numeric', 'A값', '예가']
        for feature in numeric_features:
            if feature in self.df.columns:
                analysis[feature] = {
                    "평균": float(self.df[feature].mean()) if self.df[feature].notna().any() else None,
                    "표준편차": float(self.df[feature].std()) if self.df[feature].notna().any() else None,
                    "최소값": float(self.df[feature].min()) if self.df[feature].notna().any() else None,
                    "최대값": float(self.df[feature].max()) if self.df[feature].notna().any() else None,
                    "25%": float(self.df[feature].quantile(0.25)) if self.df[feature].notna().any() else None,
                    "50%": float(self.df[feature].quantile(0.5)) if self.df[feature].notna().any() else None,
                    "75%": float(self.df[feature].quantile(0.75)) if self.df[feature].notna().any() else None,
                    "결측값": int(self.df[feature].isnull().sum())
                }
        
        self.analysis_results['feature_analysis'] = analysis
        return analysis
    
    def plot_distributions(self, save_path="../docs/data_analysis_plots.png"):
        """분포 시각화"""
        if self.df is None:
            return False
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('데이터 분포 분석', fontsize=16)
            
            # 1. 지역별 분포
            if '지역' in self.df.columns:
                top_regions = self.df['지역'].value_counts().head(15)
                top_regions.plot(kind='bar', ax=axes[0,0])
                axes[0,0].set_title('상위 15개 지역별 입찰 건수')
                axes[0,0].tick_params(axis='x', rotation=45)
            
            # 2. 업종별 분포
            if '업종' in self.df.columns:
                top_sectors = self.df['업종'].value_counts().head(15)
                top_sectors.plot(kind='bar', ax=axes[0,1])
                axes[0,1].set_title('상위 15개 업종별 입찰 건수')
                axes[0,1].tick_params(axis='x', rotation=45)
            
            # 3. 예가 분포
            if '예가' in self.df.columns:
                self.df['예가'].dropna().hist(bins=50, ax=axes[0,2])
                axes[0,2].set_title('예가 분포')
                axes[0,2].set_xlabel('예가 (예정가격/기초금액)')
            
            # 4. 기초금액 분포 (로그 스케일)
            if '기초금액_numeric' in self.df.columns:
                valid_amounts = self.df['기초금액_numeric'].dropna()
                if len(valid_amounts) > 0:
                    log_amounts = np.log10(valid_amounts[valid_amounts > 0])
                    log_amounts.hist(bins=50, ax=axes[1,0])
                    axes[1,0].set_title('기초금액 분포 (로그 스케일)')
                    axes[1,0].set_xlabel('log10(기초금액)')
            
            # 5. A값 분포
            if 'A값' in self.df.columns:
                a_values = pd.to_numeric(self.df['A값'], errors='coerce').dropna()
                if len(a_values) > 0:
                    a_values.hist(bins=30, ax=axes[1,1])
                    axes[1,1].set_title('A값 분포')
            
            # 6. 예가 vs 기초금액 산점도
            if '예가' in self.df.columns and '기초금액_numeric' in self.df.columns:
                valid_data = self.df[['예가', '기초금액_numeric']].dropna()
                if len(valid_data) > 0:
                    # 기초금액이 너무 큰 값들 제외하고 샘플링
                    sample_data = valid_data.sample(min(1000, len(valid_data)))
                    axes[1,2].scatter(sample_data['기초금액_numeric'], sample_data['예가'], alpha=0.6)
                    axes[1,2].set_xlabel('기초금액')
                    axes[1,2].set_ylabel('예가')
                    axes[1,2].set_title('기초금액 vs 예가 관계')
                    axes[1,2].set_xscale('log')
            
            plt.tight_layout()
            
            # 저장
            save_dir = Path(save_path).parent
            save_dir.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"분포 시각화 저장: {save_path}")
            return True
            
        except Exception as e:
            print(f"시각화 실패: {e}")
            return False
    
    def correlation_analysis(self):
        """상관관계 분석"""
        if self.df is None:
            return {}
        
        numeric_cols = ['기초금액_numeric', 'A값', '예가']
        existing_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if len(existing_cols) < 2:
            return {}
        
        # 수치 변환
        numeric_data = self.df[existing_cols].copy()
        for col in existing_cols:
            if col != '예가':
                numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
        
        # 상관관계 계산
        correlation = numeric_data.corr()
        
        self.analysis_results['correlation'] = correlation.to_dict()
        return correlation
    
    def data_quality_report(self):
        """데이터 품질 보고서"""
        if self.df is None:
            return {}
        
        report = {
            "전체_평가": {
                "총_레코드": len(self.df),
                "완전한_레코드": len(self.df.dropna()),
                "완전성_비율": len(self.df.dropna()) / len(self.df) * 100
            },
            "핵심_변수_품질": {},
            "권장사항": []
        }
        
        # 핵심 변수별 품질 평가
        key_features = ['지역', '업종', '기초금액', 'A값', '예가']
        for feature in key_features:
            if feature in self.df.columns or (feature == '예가' and '예가' in self.df.columns):
                col_data = self.df[feature] if feature in self.df.columns else self.df['예가']
                missing_rate = col_data.isnull().sum() / len(col_data) * 100
                
                quality = "우수" if missing_rate < 5 else "양호" if missing_rate < 20 else "개선필요"
                
                report["핵심_변수_품질"][feature] = {
                    "결측률": missing_rate,
                    "품질등급": quality
                }
                
                if missing_rate > 10:
                    report["권장사항"].append(f"{feature} 변수의 결측률이 {missing_rate:.1f}%로 높음. 데이터 보완 필요")
        
        self.analysis_results['quality_report'] = report
        return report
    
    def save_analysis_results(self, save_path="../docs/data_analysis_results.json"):
        """분석 결과 저장"""
        save_dir = Path(save_path).parent
        save_dir.mkdir(exist_ok=True)
        
        # JSON 직렬화 가능하도록 변환
        serializable_results = {}
        for key, value in self.analysis_results.items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = value
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"분석 결과 저장: {save_path}")
        return save_path
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("=== 낙찰하한가 예측 데이터 분석 ===\n")
        
        # 1. 데이터 로드
        if not self.load_data():
            return False
        
        # 2. 기본 정보
        print("1. 기본 데이터 정보")
        basic_info = self.basic_info()
        print(f"레코드 수: {basic_info['총_레코드_수']}")
        print(f"컬럼 수: {basic_info['컬럼_수']}")
        print(f"중복 레코드: {basic_info['중복_레코드']}\n")
        
        # 3. 타겟 변수 생성
        print("2. 타겟 변수 생성")
        if self.create_target_variable():
            print("예가 변수 생성 성공\n")
        else:
            print("예가 변수 생성 실패\n")
        
        # 4. 피쳐 분석
        print("3. 피쳐 분석")
        feature_analysis = self.analyze_features()
        for feature, stats in feature_analysis.items():
            print(f"{feature}: {stats}")
        print()
        
        # 5. 시각화
        print("4. 분포 시각화")
        self.plot_distributions()
        print()
        
        # 6. 상관관계 분석
        print("5. 상관관계 분석")
        correlation = self.correlation_analysis()
        if not correlation.empty:
            print("수치 변수 간 상관관계:")
            print(correlation)
        print()
        
        # 7. 품질 보고서
        print("6. 데이터 품질 평가")
        quality_report = self.data_quality_report()
        print(f"완전성 비율: {quality_report['전체_평가']['완전성_비율']:.1f}%")
        for feature, quality in quality_report['핵심_변수_품질'].items():
            print(f"{feature}: 결측률 {quality['결측률']:.1f}% ({quality['품질등급']})")
        print()
        
        # 8. 결과 저장
        print("7. 분석 결과 저장")
        self.save_analysis_results()
        
        print("분석 완료!")
        return True

def main():
    """메인 실행 함수"""
    analyzer = DataAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()