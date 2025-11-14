#!/usr/bin/env python3
"""
컬럼 구조 분석 및 비교
기존 데이터와 크롤링 예상 데이터 매핑
"""

import json
from datetime import datetime

def get_existing_columns():
    """기존 엑셀 파일에서 추출된 실제 컬럼"""
    return [
        "번호",
        "공고명", 
        "공고번호",
        "발주기관",
        "추정가격",
        "기초금액",
        "A값",
        "순공사원가",
        "예정가격",
        "낙찰하한가",
        "예가/기초(100%)",
        "예가/기초(0%)",
        "1순위업체",
        "1순위사업자번호", 
        "1순위투찰금액",
        "1순위사정율(100%)",
        "1순위사정율(0%)",
        "1순위기초대비",
        "업체수",
        "개찰일",
        "입력일",
        "업종",
        "G2B물품분류",
        "지역"
    ]

def get_model_required_columns():
    """우리 모델이 필요로 하는 핵심 컬럼"""
    return {
        "input_features": [
            "지역",        # 카테고리 변수
            "업종",        # 카테고리 변수  
            "기초금액",     # 수치 변수
            "A값"          # 수치 변수
        ],
        "target": "예가",  # 예정가격/기초금액 비율
        "additional": [
            "공고번호",     # 식별자
            "공고명",       # 참고용
            "발주기관",     # 추가 피쳐 가능성
            "예정가격",     # 타겟 계산용
            "낙찰하한가",   # 최종 목표값
            "개찰일"        # 시계열 분석용
        ]
    }

def analyze_column_mapping():
    """컬럼 매핑 분석"""
    print("=== 컬럼 매핑 분석 ===")
    
    existing = get_existing_columns()
    model_req = get_model_required_columns()
    
    print(f"기존 데이터 컬럼 수: {len(existing)}")
    print(f"모델 필요 컬럼 수: {len(model_req['input_features']) + len(model_req['additional']) + 1}")
    
    # 매핑 확인
    mapping = {}
    available_features = []
    missing_features = []
    
    all_required = model_req['input_features'] + [model_req['target']] + model_req['additional']
    
    for req_col in all_required:
        if req_col in existing:
            mapping[req_col] = req_col
            available_features.append(req_col)
        else:
            # 유사한 이름 찾기
            similar = find_similar_column(req_col, existing)
            if similar:
                mapping[req_col] = similar
                available_features.append(req_col)
            else:
                missing_features.append(req_col)
    
    print(f"\n✅ 사용 가능한 피쳐: {len(available_features)}개")
    for feature in available_features:
        mapped_to = mapping.get(feature, feature)
        if mapped_to != feature:
            print(f"  {feature} → {mapped_to}")
        else:
            print(f"  {feature} ✓")
    
    print(f"\n❌ 누락된 피쳐: {len(missing_features)}개")
    for feature in missing_features:
        print(f"  {feature}")
    
    return mapping, available_features, missing_features

def find_similar_column(target, candidates):
    """유사한 컬럼명 찾기"""
    target_lower = target.lower()
    
    # 직접 매핑
    direct_mappings = {
        "예가": ["예가/기초(100%)", "예가/기초(0%)"],
        "낙찰업체": ["1순위업체"],
        "낙찰사업자번호": ["1순위사업자번호"],
        "낙찰금액": ["1순위투찰금액"]
    }
    
    if target in direct_mappings:
        for candidate in candidates:
            if candidate in direct_mappings[target]:
                return candidate
    
    # 부분 문자열 매칭
    for candidate in candidates:
        if target_lower in candidate.lower() or candidate.lower() in target_lower:
            return candidate
    
    return None

def calculate_data_readiness():
    """데이터 준비도 계산"""
    print("\n=== 데이터 준비도 분석 ===")
    
    mapping, available, missing = analyze_column_mapping()
    model_req = get_model_required_columns()
    
    # 핵심 피쳐 준비도
    core_features_available = sum(1 for f in model_req['input_features'] if f in available)
    core_readiness = core_features_available / len(model_req['input_features']) * 100
    
    # 전체 준비도
    total_required = len(model_req['input_features']) + 1 + len(model_req['additional'])
    total_readiness = len(available) / total_required * 100
    
    print(f"핵심 피쳐 준비도: {core_readiness:.1f}% ({core_features_available}/{len(model_req['input_features'])})")
    print(f"전체 데이터 준비도: {total_readiness:.1f}% ({len(available)}/{total_required})")
    
    # 예가 계산 가능성
    can_calculate_target = "예정가격" in available and "기초금액" in available
    print(f"예가 계산 가능: {'✅' if can_calculate_target else '❌'}")
    
    if can_calculate_target:
        print("  → 예가 = 예정가격 / 기초금액")
    
    return {
        "core_readiness": core_readiness,
        "total_readiness": total_readiness,
        "can_calculate_target": can_calculate_target,
        "mapping": mapping
    }

def simulate_web_scraping_columns():
    """웹 크롤링으로 얻을 수 있는 예상 컬럼"""
    print("\n=== 웹 크롤링 예상 컬럼 ===")
    
    # 일반적인 입찰 정보 사이트에서 제공하는 컬럼
    web_columns = [
        "공고번호",
        "공고명",
        "발주기관", 
        "공고일자",
        "개찰일자",
        "입찰방법",
        "계약방법",
        "업종",
        "지역",
        "추정가격",
        "기초금액",
        "예정가격", 
        "낙찰하한가",
        "A값",
        "하한율",
        "낙찰업체",
        "낙찰가격",
        "낙찰율"
    ]
    
    existing = get_existing_columns()
    
    print(f"웹 크롤링 예상 컬럼: {len(web_columns)}개")
    print(f"기존 데이터 컬럼: {len(existing)}개")
    
    # 교집합과 차집합 분석
    common = set(web_columns) & set(existing)
    web_only = set(web_columns) - set(existing)
    existing_only = set(existing) - set(web_columns)
    
    print(f"\n공통 컬럼: {len(common)}개")
    for col in sorted(common):
        print(f"  ✓ {col}")
    
    print(f"\n웹에서만 얻을 수 있는 컬럼: {len(web_only)}개")
    for col in sorted(web_only):
        print(f"  + {col}")
    
    print(f"\n기존 데이터에만 있는 컬럼: {len(existing_only)}개")
    for col in sorted(existing_only):
        print(f"  - {col}")
    
    return web_columns, common, web_only, existing_only

def generate_final_report():
    """최종 분석 리포트 생성"""
    print("\n=== 최종 분석 리포트 ===")
    
    existing = get_existing_columns()
    model_req = get_model_required_columns()
    mapping, available, missing = analyze_column_mapping()
    readiness = calculate_data_readiness()
    web_cols, common, web_only, existing_only = simulate_web_scraping_columns()
    
    report = {
        "analysis_date": datetime.now().isoformat(),
        "data_sources": {
            "existing_excel": {
                "filename": "List of Successful Bidders_merged.xlsx",
                "columns": existing,
                "column_count": len(existing)
            },
            "web_scraping_expected": {
                "columns": web_cols,
                "column_count": len(web_cols)
            }
        },
        "model_requirements": model_req,
        "column_mapping": mapping,
        "data_readiness": readiness,
        "compatibility_analysis": {
            "common_columns": list(common),
            "web_only_columns": list(web_only),
            "existing_only_columns": list(existing_only),
            "compatibility_score": len(common) / max(len(web_cols), len(existing)) * 100
        },
        "recommendations": {
            "immediate_actions": [
                "기존 엑셀 데이터로 베이스라인 모델 개발 시작",
                "예가 = 예정가격/기초금액 으로 타겟 변수 생성",
                "지역, 업종 카테고리 인코딩 준비"
            ],
            "web_scraping_strategy": [
                "로그인 프로세스 정확한 분석 필요",
                "테이블 구조 파악을 위한 브라우저 개발자 도구 사용",
                "기존 데이터와 일치하는 컬럼 우선 수집",
                "추가 피쳐(공고일자, 입찰방법 등) 점진적 확장"
            ],
            "data_integration": [
                "공고번호를 기준으로 데이터 매칭",
                "시계열 분석을 위한 일자 컬럼 표준화",
                "데이터 품질 검증 프로세스 구축"
            ]
        },
        "next_steps": [
            "1. 기존 데이터로 EDA 및 베이스라인 모델 개발",
            "2. 웹 크롤링 환경 구축 (Windows + 패키지 설치)",
            "3. 브라우저 개발자 도구로 정확한 HTML 구조 분석", 
            "4. 로그인 성공 후 데이터 수집 로직 구현",
            "5. 기존 데이터와 신규 데이터 통합 파이프라인 구축"
        ]
    }
    
    # 리포트 저장
    with open("data/raw/column_analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("✅ 컬럼 분석 리포트 저장: data/raw/column_analysis_report.json")
    
    # 핵심 결과 요약
    print(f"\n📊 핵심 결과:")
    print(f"  기존 데이터 사용 가능: ✅ (모델 개발 즉시 가능)")
    print(f"  핵심 피쳐 준비도: {readiness['core_readiness']:.1f}%")
    print(f"  웹 크롤링 호환성: {report['compatibility_analysis']['compatibility_score']:.1f}%")
    print(f"  예가 계산 가능: {'✅' if readiness['can_calculate_target'] else '❌'}")
    
    return report

def main():
    """메인 분석 실행"""
    print("컬럼 구조 분석 및 매핑 시작")
    print("=" * 60)
    
    # 전체 분석 실행
    final_report = generate_final_report()
    
    print("\n🎯 결론:")
    print("1. 기존 데이터만으로도 모델 개발 가능!")
    print("2. 웹 크롤링은 데이터 확장용으로 점진적 진행")
    print("3. 예가 계산 로직 구현으로 타겟 변수 생성")
    print("4. 다음 단계: EDA 및 베이스라인 모델 개발")

if __name__ == "__main__":
    main()