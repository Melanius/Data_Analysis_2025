#!/usr/bin/env python3
"""
고급 웹 크롤링 테스트
로그인 프로세스와 CSRF 토큰 처리 포함
"""

import urllib.request
import urllib.parse
import urllib.error
import http.cookiejar
import json
import re
from datetime import datetime

class BidScraper:
    def __init__(self):
        # 쿠키 관리를 위한 CookieJar 설정
        self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookie_jar)
        )
        
        # 브라우저처럼 보이는 헤더 설정
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        }
        
        urllib.request.install_opener(self.opener)
    
    def get_page(self, url, data=None):
        """페이지 요청"""
        try:
            if data:
                # POST 요청
                data = urllib.parse.urlencode(data).encode('utf-8')
                req = urllib.request.Request(url, data=data, headers=self.headers)
            else:
                # GET 요청
                req = urllib.request.Request(url, headers=self.headers)
            
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode('utf-8', errors='ignore')
                return response.getcode(), content
                
        except Exception as e:
            print(f"페이지 요청 오류: {e}")
            return None, None
    
    def extract_csrf_token(self, content):
        """CSRF 토큰 추출"""
        # CSRF 토큰 패턴 찾기
        patterns = [
            r'<meta name="csrf-token" content="([^"]+)"',
            r'name="_csrf-frontend" value="([^"]+)"',
            r'_csrf["\']:\s*["\']([^"\']+)["\']',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def analyze_login_page(self):
        """로그인 페이지 분석"""
        print("=== 로그인 페이지 분석 ===")
        
        login_url = "https://infose.info21c.net/info21c/login"
        
        status, content = self.get_page(login_url)
        
        if not content:
            print("❌ 로그인 페이지 접근 실패")
            return None
        
        print(f"로그인 페이지 상태: {status}")
        print(f"페이지 크기: {len(content)} bytes")
        
        # CSRF 토큰 찾기
        csrf_token = self.extract_csrf_token(content)
        if csrf_token:
            print(f"✅ CSRF 토큰 발견: {csrf_token[:20]}...")
        else:
            print("⚠️ CSRF 토큰 없음")
        
        # 로그인 폼 필드 분석
        form_fields = []
        input_patterns = [
            r'<input[^>]+name=["\']([^"\']+)["\'][^>]*>',
            r'name=["\']([^"\']+)["\'][^>]+type=["\']([^"\']+)["\']',
        ]
        
        for pattern in input_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    form_fields.append(match)
                else:
                    form_fields.append(match)
        
        unique_fields = list(set(form_fields))
        print(f"발견된 폼 필드: {unique_fields}")
        
        return csrf_token
    
    def attempt_login(self, username, password):
        """로그인 시도"""
        print("\n=== 로그인 시도 ===")
        
        # 먼저 로그인 페이지에서 토큰 획득
        csrf_token = self.analyze_login_page()
        
        login_url = "https://infose.info21c.net/info21c/login"
        
        # 다양한 필드명 조합으로 시도
        login_attempts = [
            {
                'LoginForm[username]': username,
                'LoginForm[password]': password,
                '_csrf-frontend': csrf_token if csrf_token else '',
            },
            {
                'username': username,
                'password': password,
                '_token': csrf_token if csrf_token else '',
            },
            {
                'userid': username,
                'passwd': password,
                '_csrf': csrf_token if csrf_token else '',
            },
            {
                'user_id': username,
                'user_password': password,
                'csrf_token': csrf_token if csrf_token else '',
            }
        ]
        
        for i, login_data in enumerate(login_attempts):
            print(f"\n로그인 시도 {i+1}: {list(login_data.keys())}")
            
            # 헤더 업데이트 (POST 요청용)
            post_headers = self.headers.copy()
            post_headers.update({
                'Content-Type': 'application/x-www-form-urlencoded',
                'Origin': 'https://infose.info21c.net',
                'Referer': login_url,
            })
            
            try:
                data = urllib.parse.urlencode(login_data).encode('utf-8')
                req = urllib.request.Request(login_url, data=data, headers=post_headers)
                
                with urllib.request.urlopen(req, timeout=15) as response:
                    status = response.getcode()
                    final_url = response.geturl()
                    response_content = response.read().decode('utf-8', errors='ignore')
                    
                    print(f"응답 상태: {status}")
                    print(f"최종 URL: {final_url}")
                    
                    # 로그인 성공 확인
                    success_indicators = ['logout', '로그아웃', 'dashboard', '대시보드']
                    login_success = any(indicator in response_content.lower() for indicator in success_indicators)
                    
                    if login_success or final_url != login_url:
                        print(f"✅ 로그인 성공 가능성 높음 (시도 {i+1})")
                        return True, response_content
                    
                    # 오류 메시지 확인
                    error_indicators = ['error', 'invalid', '오류', '실패', 'fail']
                    has_error = any(indicator in response_content.lower() for indicator in error_indicators)
                    
                    if has_error:
                        print(f"⚠️ 로그인 오류 메시지 감지")
                    
            except Exception as e:
                print(f"로그인 시도 {i+1} 실패: {e}")
                continue
        
        print("❌ 모든 로그인 시도 실패")
        return False, None
    
    def test_data_page_access(self, after_login_content=None):
        """데이터 페이지 접근 테스트"""
        print("\n=== 데이터 페이지 접근 테스트 ===")
        
        data_url = "https://infose.info21c.net/info21c/bids/list?bidtype=con&bid_suc=suc"
        
        status, content = self.get_page(data_url)
        
        if not content:
            print("❌ 데이터 페이지 접근 실패")
            return []
        
        print(f"데이터 페이지 상태: {status}")
        print(f"페이지 크기: {len(content)} bytes")
        
        # 테이블 구조 분석
        table_patterns = [
            r'<table[^>]*>(.*?)</table>',
            r'<thead[^>]*>(.*?)</thead>',
            r'<tbody[^>]*>(.*?)</tbody>',
            r'<tr[^>]*>(.*?)</tr>',
        ]
        
        for pattern_name, pattern in zip(['table', 'thead', 'tbody', 'tr'], table_patterns):
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            print(f"{pattern_name} 태그 발견: {len(matches)}개")
        
        # 테이블 헤더 추출 시도
        headers = self.extract_table_headers(content)
        
        return headers
    
    def extract_table_headers(self, content):
        """테이블 헤더 추출"""
        print("\n=== 테이블 헤더 추출 ===")
        
        # 여러 패턴으로 헤더 찾기
        header_patterns = [
            r'<th[^>]*>(.*?)</th>',
            r'<td[^>]*class=["\'][^"\']*header[^"\']*["\'][^>]*>(.*?)</td>',
            r'<tr[^>]*class=["\'][^"\']*header[^"\']*["\'][^>]*>(.*?)</tr>',
        ]
        
        all_headers = []
        
        for pattern in header_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                # HTML 태그 제거
                clean_header = re.sub(r'<[^>]+>', '', match).strip()
                if clean_header and len(clean_header) < 50:  # 너무 긴 텍스트 제외
                    all_headers.append(clean_header)
        
        # 중복 제거 및 정리
        unique_headers = []
        for header in all_headers:
            if header not in unique_headers and header:
                unique_headers.append(header)
        
        print(f"추출된 헤더 후보: {len(unique_headers)}개")
        
        if unique_headers:
            print("헤더 목록:")
            for i, header in enumerate(unique_headers[:20]):  # 최대 20개만 출력
                print(f"  {i+1:2d}. {header}")
        
        return unique_headers

def simulate_scraped_data():
    """크롤링 데이터 시뮬레이션"""
    print("\n=== 크롤링 데이터 시뮬레이션 ===")
    
    # 일반적인 낙찰 데이터 컬럼 예상
    expected_columns = [
        "공고번호",
        "공고명", 
        "발주기관",
        "공고일자",
        "개찰일자",
        "지역",
        "업종",
        "추정가격",
        "기초금액", 
        "예정가격",
        "낙찰하한가",
        "낙찰가격",
        "낙찰율",
        "하한율",
        "A값",
        "낙찰업체",
        "낙찰업체_대표자",
        "낙찰업체_사업자번호"
    ]
    
    print(f"예상 컬럼 개수: {len(expected_columns)}")
    print("예상 컬럼 목록:")
    for i, col in enumerate(expected_columns):
        print(f"  {i+1:2d}. {col}")
    
    return expected_columns

def create_detailed_report(login_attempted, headers_found, expected_columns):
    """상세 리포트 생성"""
    print("\n=== 상세 테스트 리포트 생성 ===")
    
    report = {
        "test_datetime": datetime.now().isoformat(),
        "website_analysis": {
            "base_url": "https://infose.info21c.net",
            "data_url": "https://infose.info21c.net/info21c/bids/list?bidtype=con&bid_suc=suc",
            "requires_login": True,
            "csrf_protection": True
        },
        "login_test": {
            "attempted": login_attempted,
            "credentials_provided": True,
            "multiple_field_combinations_tested": True
        },
        "data_extraction": {
            "headers_found": headers_found,
            "headers_count": len(headers_found) if headers_found else 0,
            "expected_columns": expected_columns,
            "expected_count": len(expected_columns)
        },
        "comparison": {
            "matching_columns": [],
            "missing_columns": [],
            "extra_columns": []
        },
        "recommendations": [
            "1. 브라우저 개발자 도구로 실제 로그인 프로세스 분석",
            "2. 정확한 폼 필드명과 CSRF 토큰 처리 방식 확인", 
            "3. 로그인 성공 후 쿠키/세션 관리 방법 파악",
            "4. 테이블 구조와 페이지네이션 방식 분석",
            "5. 데이터 추출을 위한 정확한 CSS 선택자 또는 XPath 정의"
        ]
    }
    
    # 컬럼 비교 (시뮬레이션)
    if headers_found and expected_columns:
        headers_lower = [h.lower() for h in headers_found]
        expected_lower = [e.lower() for e in expected_columns]
        
        for expected in expected_columns:
            if expected.lower() in headers_lower:
                report["comparison"]["matching_columns"].append(expected)
            else:
                report["comparison"]["missing_columns"].append(expected)
        
        for header in headers_found:
            if header.lower() not in expected_lower:
                report["comparison"]["extra_columns"].append(header)
    
    # 리포트 저장
    with open("data/raw/detailed_scraping_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("✅ 상세 리포트 저장: data/raw/detailed_scraping_report.json")
    
    # 요약 출력
    print(f"\n📊 테스트 결과 요약:")
    print(f"  로그인 시도: {'✅' if login_attempted else '❌'}")
    print(f"  발견된 헤더: {len(headers_found) if headers_found else 0}개")
    print(f"  예상 컬럼: {len(expected_columns)}개")
    
    if report["comparison"]["matching_columns"]:
        print(f"  매칭되는 컬럼: {len(report['comparison']['matching_columns'])}개")

def main():
    """메인 테스트 실행"""
    print("고급 웹 크롤링 테스트 시작")
    print("=" * 60)
    
    scraper = BidScraper()
    
    # 1. 로그인 시도
    login_success, login_content = scraper.attempt_login("wjoon97", "joon3277^^")
    
    # 2. 데이터 페이지 접근
    headers_found = scraper.test_data_page_access(login_content)
    
    # 3. 예상 컬럼 정의
    expected_columns = simulate_scraped_data()
    
    # 4. 상세 리포트 생성
    create_detailed_report(True, headers_found, expected_columns)
    
    print("\n🎯 다음 단계:")
    print("1. Windows 환경에서 실제 패키지 설치하여 완전한 테스트")
    print("2. 브라우저로 수동 로그인하여 HTML 구조 분석")
    print("3. 성공적인 로그인 후 쿠키/세션 복사하여 테스트")
    print("4. 완전한 크롤링 스크립트 개발")

if __name__ == "__main__":
    main()