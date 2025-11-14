#!/usr/bin/env python3
"""
간단한 웹사이트 테스트 (기본 라이브러리만 사용)
"""

import urllib.request
import urllib.parse
import urllib.error
import json
from datetime import datetime
import re

def test_basic_access():
    """기본 웹사이트 접근 테스트"""
    print("=== 기본 웹사이트 접근 테스트 ===")
    
    url = "https://infose.info21c.net/info21c/bids/list?bidtype=con&bid_suc=suc"
    
    # 요청 헤더 설정
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
        'Connection': 'keep-alive',
    }
    
    try:
        # 요청 생성
        req = urllib.request.Request(url, headers=headers)
        
        # 응답 받기
        with urllib.request.urlopen(req, timeout=10) as response:
            status_code = response.getcode()
            content = response.read().decode('utf-8', errors='ignore')
            
            print(f"응답 코드: {status_code}")
            print(f"응답 길이: {len(content)} bytes")
            
            # 기본 HTML 구조 확인
            if '<title>' in content:
                title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                if title_match:
                    print(f"페이지 제목: {title_match.group(1).strip()}")
            
            # 로그인 관련 확인
            login_indicators = ['로그인', 'login', 'Login', 'LOG IN']
            has_login = any(indicator in content for indicator in login_indicators)
            
            if has_login:
                print("⚠️ 로그인이 필요한 페이지로 보입니다.")
            else:
                print("✅ 직접 접근 가능한 페이지입니다.")
            
            # 테이블 구조 대략적 확인
            table_count = content.lower().count('<table')
            tr_count = content.lower().count('<tr')
            
            print(f"테이블 개수: {table_count}")
            print(f"테이블 행 개수: {tr_count}")
            
            # 일부 내용 출력 (첫 500자)
            print(f"\n페이지 내용 미리보기 (첫 500자):")
            print("=" * 50)
            print(content[:500])
            print("=" * 50)
            
            return True, content
            
    except urllib.error.URLError as e:
        print(f"❌ URL 오류: {e}")
        return False, None
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP 오류: {e.code} - {e.reason}")
        return False, None
    except Exception as e:
        print(f"❌ 기타 오류: {e}")
        return False, None

def analyze_existing_data():
    """기존 데이터 파일 간단 분석"""
    print("\n=== 기존 데이터 파일 분석 ===")
    
    filename = "data/raw/List of Successful Bidders_merged.xlsx"
    
    try:
        import openpyxl
        
        # 엑셀 파일 읽기
        wb = openpyxl.load_workbook(filename)
        ws = wb.active
        
        print(f"워크시트명: {ws.title}")
        print(f"최대 행: {ws.max_row}")
        print(f"최대 열: {ws.max_column}")
        
        # 헤더 행 읽기 (첫 번째 행)
        headers = []
        for col in range(1, ws.max_column + 1):
            cell_value = ws.cell(row=1, column=col).value
            if cell_value:
                headers.append(str(cell_value).strip())
        
        print(f"\n컬럼 개수: {len(headers)}")
        print("컬럼 목록:")
        for i, header in enumerate(headers):
            print(f"  {i+1:2d}. {header}")
        
        # 첫 3행 데이터 샘플
        print("\n첫 3행 데이터:")
        for row in range(2, min(5, ws.max_row + 1)):  # 2~4행 (데이터 3행)
            print(f"\n행 {row-1}:")
            for col in range(1, min(len(headers) + 1, ws.max_column + 1)):
                cell_value = ws.cell(row=row, column=col).value
                header = headers[col-1] if col-1 < len(headers) else f"열{col}"
                print(f"  {header}: {cell_value}")
        
        return headers
        
    except ImportError:
        print("❌ openpyxl 라이브러리가 없어 엑셀 파일을 읽을 수 없습니다.")
        print("Windows에서 다음 명령으로 설치하세요: pip install openpyxl")
        return None
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return None

def create_test_report(web_accessible, existing_columns):
    """테스트 결과 리포트 생성"""
    print("\n=== 테스트 결과 리포트 ===")
    
    report = {
        "test_date": datetime.now().isoformat(),
        "web_access_test": {
            "success": web_accessible,
            "requires_login": True  # 대부분의 유료 사이트는 로그인 필요
        },
        "existing_data_columns": existing_columns,
        "next_steps": [
            "1. 웹사이트 로그인 프로세스 분석",
            "2. 실제 테이블 구조 파악",
            "3. 데이터 매핑 정의",
            "4. 크롤링 로직 최적화"
        ]
    }
    
    # 리포트 저장
    with open("data/raw/scraping_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("✅ 테스트 리포트 저장: data/raw/scraping_test_report.json")
    
    print(f"\n📊 결과 요약:")
    print(f"  웹사이트 접근: {'✅ 성공' if web_accessible else '❌ 실패'}")
    print(f"  기존 데이터 컬럼 수: {len(existing_columns) if existing_columns else '분석 실패'}")
    
    if existing_columns:
        print(f"  기존 데이터 주요 컬럼:")
        for col in existing_columns[:5]:
            print(f"    - {col}")
        if len(existing_columns) > 5:
            print(f"    ... 외 {len(existing_columns) - 5}개")

def main():
    """메인 테스트 함수"""
    print("낙찰 데이터 크롤링 기초 테스트")
    print("=" * 50)
    
    # 1. 웹사이트 접근 테스트
    web_accessible, content = test_basic_access()
    
    # 2. 기존 데이터 분석
    existing_columns = analyze_existing_data()
    
    # 3. 테스트 리포트 생성
    create_test_report(web_accessible, existing_columns)
    
    print("\n🎯 다음 단계:")
    print("1. Windows에서 pip install pandas openpyxl requests beautifulsoup4")
    print("2. 실제 로그인 프로세스 분석 (개발자 도구 사용)")
    print("3. 테이블 구조 상세 파악")
    print("4. 컬럼 매핑 정의")

if __name__ == "__main__":
    main()