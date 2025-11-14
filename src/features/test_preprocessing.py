#!/usr/bin/env python3
"""
전처리 파이프라인 테스트
"""

from preprocessing import BidDataProcessor

def test_with_sample_data():
    """샘플 데이터로 전처리 테스트"""
    processor = BidDataProcessor()
    
    # 샘플 데이터 생성
    sample_data = processor.create_sample_data()
    
    # 카테고리 매핑 구축
    processor.build_category_mappings(sample_data)
    
    # 피쳐 생성
    processed_data = processor.create_training_features(sample_data)
    
    # 결과 저장
    json_file = processor.save_processed_data(processed_data, "sample_processed_data.json")
    csv_file = processor.export_to_csv(processed_data, "sample_training_data.csv")
    
    # 피쳐 정보
    feature_info = processor.generate_feature_info()
    
    print("샘플 데이터 전처리 테스트 완료!")
    return processed_data, feature_info

if __name__ == "__main__":
    test_with_sample_data()