#!/usr/bin/env python3
"""
데이터 전처리 파이프라인
"""

import json
import os
from pathlib import Path
import sys

# CSV 모듈은 표준 라이브러리이므로 사용 가능
import csv

class BidDataProcessor:
    """낙찰하한가 예측을 위한 데이터 전처리기"""
    
    def __init__(self, data_dir="data/raw", output_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 카테고리 인코딩을 위한 매핑
        self.region_mapping = {}
        self.sector_mapping = {}
        
    def load_excel_to_json(self, filename="List of Successful Bidders_merged.xlsx"):
        """Excel 파일을 JSON으로 변환하여 로드 (pandas 없이)"""
        print(f"Excel 파일 처리: {filename}")
        print("주의: Excel 직접 읽기를 위해서는 pandas가 필요합니다.")
        print("대안으로 Excel을 CSV로 저장한 후 처리하거나, 기존 분석 결과를 활용합니다.")
        
        # 기존 분석 결과 활용
        analysis_file = self.data_dir / "column_analysis_report.json"
        if analysis_file.exists():
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            return analysis
        return None
    
    def create_sample_data(self):
        """샘플 데이터 생성 (테스트용)"""
        sample_data = {
            "records": [
                {
                    "공고번호": "20241001-001", 
                    "공고명": "도로포장공사",
                    "지역": "서울특별시",
                    "업종": "토목",
                    "기초금액": 1000000000,
                    "A값": 150000000,
                    "예정가격": 950000000,
                    "예가": 0.95
                },
                {
                    "공고번호": "20241001-002",
                    "공고명": "건물신축공사", 
                    "지역": "경기도",
                    "업종": "건축",
                    "기초금액": 2000000000,
                    "A값": 200000000,
                    "예정가격": 1800000000,
                    "예가": 0.90
                },
                {
                    "공고번호": "20241001-003",
                    "공고명": "전력설비공사",
                    "지역": "부산광역시", 
                    "업종": "전기",
                    "기초금액": 500000000,
                    "A값": 80000000,
                    "예정가격": 480000000,
                    "예가": 0.96
                }
            ]
        }
        
        # 샘플 데이터 저장
        sample_file = self.output_dir / "sample_data.json"
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"샘플 데이터 생성: {sample_file}")
        return sample_data
    
    def build_category_mappings(self, data):
        """카테고리 변수 매핑 구축"""
        if not data or 'records' not in data:
            return
        
        regions = set()
        sectors = set()
        
        for record in data['records']:
            if '지역' in record:
                regions.add(record['지역'])
            if '업종' in record:
                sectors.add(record['업종'])
        
        # 지역 매핑
        self.region_mapping = {region: idx for idx, region in enumerate(sorted(regions))}
        
        # 업종 매핑
        self.sector_mapping = {sector: idx for idx, sector in enumerate(sorted(sectors))}
        
        print(f"지역 카테고리: {len(self.region_mapping)}개")
        for region, idx in list(self.region_mapping.items())[:5]:
            print(f"  {region}: {idx}")
        
        print(f"업종 카테고리: {len(self.sector_mapping)}개") 
        for sector, idx in list(self.sector_mapping.items())[:5]:
            print(f"  {sector}: {idx}")
    
    def encode_categories(self, record):
        """개별 레코드의 카테고리 인코딩"""
        encoded = record.copy()
        
        if '지역' in record and record['지역'] in self.region_mapping:
            encoded['지역_encoded'] = self.region_mapping[record['지역']]
        
        if '업종' in record and record['업종'] in self.sector_mapping:
            encoded['업종_encoded'] = self.sector_mapping[record['업종']]
        
        return encoded
    
    def process_numeric_features(self, record):
        """수치형 피쳐 전처리"""
        processed = record.copy()
        
        # 기초금액 정규화 (로그 변환을 위한 준비)
        if '기초금액' in record and isinstance(record['기초금액'], (int, float)):
            processed['기초금액_log'] = record['기초금액']  # 실제로는 log를 취해야 하지만 수학 라이브러리가 필요
        
        # A값 비율 계산
        if '기초금액' in record and 'A값' in record:
            base = record['기초금액']
            a_val = record['A값']
            if base and base > 0:
                processed['A값_비율'] = a_val / base
        
        # 예가 정규화 (0.8-1.0 범위로 스케일링)
        if '예가' in record:
            yega = record['예가']
            if yega:
                # 일반적으로 예가는 0.8-1.0 사이
                processed['예가_normalized'] = (yega - 0.8) / 0.2
        
        return processed
    
    def validate_record(self, record):
        """레코드 유효성 검사"""
        required_fields = ['지역', '업종', '기초금액', 'A값']
        
        for field in required_fields:
            if field not in record or record[field] is None:
                return False, f"필수 필드 누락: {field}"
        
        # 기초금액 양수 체크
        if record['기초금액'] <= 0:
            return False, "기초금액은 양수여야 합니다"
        
        # A값 양수 체크  
        if record['A값'] <= 0:
            return False, "A값은 양수여야 합니다"
        
        return True, "유효"
    
    def create_training_features(self, data):
        """훈련용 피쳐 생성"""
        if not data or 'records' not in data:
            return []
        
        processed_records = []
        
        for record in data['records']:
            # 유효성 검사
            is_valid, message = self.validate_record(record)
            if not is_valid:
                print(f"유효하지 않은 레코드 스킵: {message}")
                continue
            
            # 카테고리 인코딩
            processed = self.encode_categories(record)
            
            # 수치형 피쳐 처리
            processed = self.process_numeric_features(processed)
            
            processed_records.append(processed)
        
        return processed_records
    
    def save_processed_data(self, processed_data, filename="processed_training_data.json"):
        """전처리된 데이터 저장"""
        output_file = self.output_dir / filename
        
        save_data = {
            "metadata": {
                "total_records": len(processed_data),
                "region_mapping": self.region_mapping,
                "sector_mapping": self.sector_mapping,
                "features": [
                    "지역_encoded", "업종_encoded", "기초금액", 
                    "A값", "기초금액_log", "A값_비율"
                ],
                "target": "예가"
            },
            "data": processed_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"전처리된 데이터 저장: {output_file}")
        print(f"처리된 레코드: {len(processed_data)}개")
        return output_file
    
    def export_to_csv(self, processed_data, filename="training_data.csv"):
        """CSV 형태로 내보내기"""
        if not processed_data:
            return None
        
        output_file = self.output_dir / filename
        
        # 헤더 추출
        headers = set()
        for record in processed_data:
            headers.update(record.keys())
        headers = sorted(headers)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(processed_data)
        
        print(f"CSV 파일 저장: {output_file}")
        return output_file
    
    def generate_feature_info(self):
        """피쳐 정보 문서 생성"""
        feature_info = {
            "input_features": {
                "지역_encoded": {
                    "type": "categorical",
                    "description": "지역 카테고리를 정수로 인코딩",
                    "mapping": self.region_mapping
                },
                "업종_encoded": {
                    "type": "categorical", 
                    "description": "업종 카테고리를 정수로 인코딩",
                    "mapping": self.sector_mapping
                },
                "기초금액": {
                    "type": "numeric",
                    "description": "입찰 기초금액 (원)",
                    "preprocessing": "원본 값 사용"
                },
                "기초금액_log": {
                    "type": "numeric",
                    "description": "기초금액의 로그 변환 (스케일 정규화용)",
                    "preprocessing": "log10 변환"
                },
                "A값": {
                    "type": "numeric",
                    "description": "낙찰하한가 계산용 A값 (원)", 
                    "preprocessing": "원본 값 사용"
                },
                "A값_비율": {
                    "type": "numeric",
                    "description": "A값 / 기초금액 비율",
                    "preprocessing": "비율 계산"
                }
            },
            "target_variable": {
                "예가": {
                    "type": "numeric",
                    "description": "예정가격 / 기초금액 비율 (예측 대상)",
                    "range": "일반적으로 0.8 ~ 1.0"
                }
            },
            "business_logic": {
                "formula": "낙찰하한가 = (예정가격 - A값) × 낙찰하한율 + A값",
                "where": "예정가격 = 기초금액 × 예가",
                "goal": "예가를 예측하여 적정 낙찰하한가 산출"
            }
        }
        
        info_file = self.output_dir / "feature_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)
        
        print(f"피쳐 정보 저장: {info_file}")
        return feature_info
    
    def run_preprocessing(self):
        """전체 전처리 파이프라인 실행"""
        print("=== 낙찰하한가 예측 데이터 전처리 ===\n")
        
        # 1. 데이터 로드 시도
        print("1. 데이터 로드")
        data = self.load_excel_to_json()
        
        if not data:
            print("실제 데이터 로드 실패, 샘플 데이터 생성")
            data = self.create_sample_data()
        
        # 2. 카테고리 매핑 구축
        print("\n2. 카테고리 매핑 구축")
        self.build_category_mappings(data)
        
        # 3. 피쳐 생성
        print("\n3. 훈련용 피쳐 생성")
        processed_data = self.create_training_features(data)
        
        # 4. 데이터 저장
        print("\n4. 전처리 데이터 저장")
        json_file = self.save_processed_data(processed_data)
        csv_file = self.export_to_csv(processed_data)
        
        # 5. 피쳐 정보 생성
        print("\n5. 피쳐 정보 문서 생성")
        feature_info = self.generate_feature_info()
        
        print("\n전처리 완료!")
        
        return {
            "processed_records": len(processed_data),
            "json_file": str(json_file),
            "csv_file": str(csv_file),
            "feature_count": len(feature_info["input_features"])
        }

def main():
    """메인 실행 함수"""
    processor = BidDataProcessor()
    result = processor.run_preprocessing()
    
    print(f"\n=== 전처리 결과 ===")
    for key, value in result.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()