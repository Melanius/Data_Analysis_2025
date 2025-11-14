#!/usr/bin/env python3
"""
기본 데이터 확인 스크립트 (minimal dependencies)
"""

import json
from pathlib import Path
import csv
import os

def check_data_files():
    """데이터 파일들 확인"""
    data_dir = Path("data/raw")
    
    print("=== 낙찰하한가 예측 데이터 확인 ===\n")
    print("1. 데이터 파일 목록:")
    
    if not data_dir.exists():
        print(f"데이터 디렉토리가 없습니다: {data_dir}")
        return False
    
    files = list(data_dir.iterdir())
    for file in files:
        size_kb = file.stat().st_size / 1024
        print(f"  - {file.name}: {size_kb:.1f} KB")
    
    return len(files) > 0

def read_column_analysis():
    """컬럼 분석 결과 읽기"""
    analysis_file = Path("data/raw/column_analysis_report.json")
    
    if not analysis_file.exists():
        print("컬럼 분석 파일이 없습니다.")
        return None
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("\n2. 데이터 구조 분석:")
        
        # 기존 Excel 데이터 정보
        excel_info = data.get('data_sources', {}).get('existing_excel', {})
        print(f"  Excel 파일: {excel_info.get('filename', 'Unknown')}")
        print(f"  컬럼 수: {excel_info.get('column_count', 0)}")
        print(f"  주요 컬럼: {', '.join(excel_info.get('columns', [])[:5])}...")
        
        # 모델 요구사항
        model_req = data.get('model_requirements', {})
        print(f"\n3. 모델 입력 변수:")
        for feature in model_req.get('input_features', []):
            print(f"  - {feature}")
        print(f"  타겟 변수: {model_req.get('target', 'Unknown')}")
        
        # 데이터 준비도
        readiness = data.get('data_readiness', {})
        print(f"\n4. 데이터 준비도:")
        print(f"  핵심 준비도: {readiness.get('core_readiness', 0)}%")
        print(f"  전체 준비도: {readiness.get('total_readiness', 0)}%")
        
        return data
        
    except Exception as e:
        print(f"분석 파일 읽기 실패: {e}")
        return None

def check_existing_code():
    """기존 코드 구조 확인"""
    print("\n5. 코드 구조:")
    
    # src 디렉토리 확인
    src_dir = Path("src")
    if src_dir.exists():
        for subdir in src_dir.iterdir():
            if subdir.is_dir():
                files = list(subdir.glob("*.py"))
                print(f"  {subdir.name}/: {len(files)}개 파이썬 파일")
    
    # 노트북 확인
    notebook_dir = Path("notebooks")
    if notebook_dir.exists():
        notebooks = list(notebook_dir.glob("*.ipynb"))
        print(f"  notebooks/: {len(notebooks)}개 노트북")

def development_recommendations():
    """개발 권장사항"""
    print("\n6. 개발 권장사항:")
    print("  ✓ 기존 Excel 데이터를 활용한 베이스라인 모델 개발 가능")
    print("  ✓ 예가 = 예정가격/기초금액 으로 타겟 변수 생성")
    print("  ✓ 지역, 업종 카테고리 인코딩 필요")
    print("  ✓ Streamlit 웹앱으로 사용자 인터페이스 구축")
    print("  ✓ FastAPI로 예측 서비스 API 구축")

def create_next_steps():
    """다음 단계 파일 생성"""
    next_steps = {
        "development_phases": {
            "Phase_1": {
                "name": "데이터 분석 및 전처리",
                "tasks": [
                    "Excel 데이터 로드 및 기본 분석",
                    "예가 타겟 변수 생성",
                    "카테고리 변수 인코딩",
                    "데이터 품질 확인 및 정제"
                ],
                "status": "ready"
            },
            "Phase_2": {
                "name": "베이스라인 모델 개발",
                "tasks": [
                    "LinearRegression 베이스라인 구현",
                    "RandomForest 모델 구현", 
                    "XGBoost 모델 구현",
                    "모델 성능 비교"
                ],
                "status": "pending"
            },
            "Phase_3": {
                "name": "웹 애플리케이션 개발",
                "tasks": [
                    "Streamlit 인터페이스 구축",
                    "입력 폼 및 예측 결과 화면",
                    "예측 히스토리 관리",
                    "시각화 대시보드"
                ],
                "status": "pending"
            },
            "Phase_4": {
                "name": "API 서버 구축",
                "tasks": [
                    "FastAPI 서버 구현",
                    "예측 엔드포인트 개발",
                    "모델 로딩 및 서빙",
                    "API 문서화"
                ],
                "status": "pending"
            }
        },
        "immediate_actions": [
            "Python 패키지 환경 설정",
            "데이터 분석 스크립트 실행",
            "베이스라인 모델 프로토타입 개발"
        ]
    }
    
    with open("development_plan.json", "w", encoding="utf-8") as f:
        json.dump(next_steps, f, ensure_ascii=False, indent=2)
    
    print("\n7. 개발 계획 저장: development_plan.json")

def main():
    """메인 함수"""
    # 1. 데이터 파일 확인
    data_exists = check_data_files()
    
    if not data_exists:
        print("데이터 파일이 없어 분석을 중단합니다.")
        return
    
    # 2. 컬럼 분석 읽기
    analysis = read_column_analysis()
    
    # 3. 코드 구조 확인
    check_existing_code()
    
    # 4. 개발 권장사항
    development_recommendations()
    
    # 5. 다음 단계 계획
    create_next_steps()
    
    print("\n데이터 확인 완료!")

if __name__ == "__main__":
    main()