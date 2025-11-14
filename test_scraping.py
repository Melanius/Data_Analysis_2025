#!/usr/bin/env python3
"""
웹 크롤링 테스트 스크립트
기존 데이터와 크롤링 데이터 비교
"""

import requests
from bs4 import BeautifulSoup
import csv
import json
from datetime import datetime

def test_website_access():
    """웹사이트 접근 테스트"""
    print("=== 웹사이트 접근 테스트 ===")
    
    url = "https://infose.info21c.net/info21c/bids/list?bidtype=con&bid_suc=suc"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"응답 코드: {response.status_code}")
        print(f"응답 길이: {len(response.text)} bytes")
        
        # HTML 구조 분석
        soup = BeautifulSoup(response.content, 'html.parser')
        
        if soup.title:
            print(f"페이지 제목: {soup.title.get_text().strip()}")
        
        # 로그인 폼 찾기
        forms = soup.find_all('form')
        print(f"폼 개수: {len(forms)}")
        
        # 테이블 구조 확인
        tables = soup.find_all('table')
        print(f"테이블 개수: {len(tables)}")
        
        if tables:
            table = tables[0]
            rows = table.find_all('tr')
            print(f"첫 번째 테이블 행 수: {len(rows)}")
            
            # 헤더 행 확인
            if rows:
                header_row = rows[0]
                headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
                print("테이블 헤더:", headers)
        
        # 로그인이 필요한지 확인
        if "로그인" in response.text or "login" in response.text.lower():
            print("⚠️ 로그인이 필요한 페이지입니다.")
            return False, soup
        else:
            print("✅ 직접 접근 가능한 페이지입니다.")
            return True, soup
            
    except Exception as e:
        print(f"❌ 접근 오류: {e}")
        return False, None

def login_test():
    """로그인 테스트"""
    print("\n=== 로그인 테스트 ===")
    
    session = requests.Session()
    
    # 로그인 페이지 접근
    login_url = "https://infose.info21c.net/info21c/login"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    try:
        # 로그인 페이지 접근
        login_page = session.get(login_url, headers=headers)
        print(f"로그인 페이지 응답: {login_page.status_code}")
        
        soup = BeautifulSoup(login_page.content, 'html.parser')
        
        # 로그인 폼 분석
        forms = soup.find_all('form')
        print(f"로그인 폼 개수: {len(forms)}")
        
        if forms:
            form = forms[0]
            inputs = form.find_all('input')
            print("폼 필드:")
            for inp in inputs:
                name = inp.get('name', '')
                input_type = inp.get('type', '')
                print(f"  - {name}: {input_type}")
        
        # 실제 로그인 시도
        login_data = {
            'userid': 'wjoon97',  # 일반적인 필드명 추정
            'password': 'joon3277^^',
            'username': 'wjoon97',  # 대안 필드명
            'passwd': 'joon3277^^',  # 대안 필드명
        }
        
        login_response = session.post(login_url, data=login_data, headers=headers)
        print(f"로그인 시도 응답: {login_response.status_code}")
        
        # 로그인 성공 확인
        if login_response.status_code == 200:
            # 데이터 페이지에 접근해보기
            data_url = "https://infose.info21c.net/info21c/bids/list?bidtype=con&bid_suc=suc"
            data_response = session.get(data_url, headers=headers)
            
            if "로그아웃" in data_response.text or "logout" in data_response.text.lower():
                print("✅ 로그인 성공!")
                return session, True
            else:
                print("⚠️ 로그인 상태 불명확")
                return session, False
        
        return session, False
        
    except Exception as e:
        print(f"❌ 로그인 오류: {e}")
        return None, False

def extract_table_data(session, max_rows=100):
    """테이블 데이터 추출"""
    print(f"\n=== 데이터 추출 (최대 {max_rows}행) ===")
    
    data_url = "https://infose.info21c.net/info21c/bids/list?bidtype=con&bid_suc=suc"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = session.get(data_url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 테이블 찾기
        tables = soup.find_all('table')
        
        if not tables:
            print("❌ 테이블을 찾을 수 없습니다.")
            return []
        
        table = tables[0]  # 첫 번째 테이블 사용
        rows = table.find_all('tr')
        
        if len(rows) == 0:
            print("❌ 테이블 행이 없습니다.")
            return []
        
        # 헤더 추출
        header_row = rows[0]
        headers = []
        for cell in header_row.find_all(['th', 'td']):
            header_text = cell.get_text().strip()
            headers.append(header_text)
        
        print(f"컬럼 개수: {len(headers)}")
        print("추출된 컬럼:")
        for i, header in enumerate(headers):
            print(f"  {i+1:2d}. {header}")
        
        # 데이터 추출
        extracted_data = []
        data_rows = rows[1:]  # 헤더 제외
        
        for i, row in enumerate(data_rows):
            if i >= max_rows:
                break
                
            cells = row.find_all(['td', 'th'])
            row_data = []
            
            for cell in cells:
                cell_text = cell.get_text().strip()
                row_data.append(cell_text)
            
            if len(row_data) == len(headers):  # 컬럼 수가 일치하는 행만
                extracted_data.append(dict(zip(headers, row_data)))
        
        print(f"✅ {len(extracted_data)}행 추출 완료")
        return extracted_data
        
    except Exception as e:
        print(f"❌ 데이터 추출 오류: {e}")
        return []

def save_test_data(data, filename="web_scraped_test.csv"):
    """추출된 데이터 저장"""
    if not data:
        print("저장할 데이터가 없습니다.")
        return
    
    try:
        filepath = f"data/raw/{filename}"
        
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
            if data:
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
        
        print(f"✅ 데이터 저장 완료: {filepath}")
        
        # JSON 형태로도 저장
        json_filepath = filepath.replace('.csv', '.json')
        with open(json_filepath, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, ensure_ascii=False, indent=2)
        
        print(f"✅ JSON 저장 완료: {json_filepath}")
        
    except Exception as e:
        print(f"❌ 저장 오류: {e}")

def main():
    """메인 테스트 실행"""
    print("낙찰 데이터 웹 크롤링 테스트 시작")
    print("=" * 50)
    
    # 1. 웹사이트 접근 테스트
    can_access, soup = test_website_access()
    
    # 2. 로그인 테스트
    session, login_success = login_test()
    
    if not session:
        print("❌ 세션 생성 실패")
        return
    
    # 3. 데이터 추출
    extracted_data = extract_table_data(session, max_rows=100)
    
    # 4. 데이터 저장
    if extracted_data:
        save_test_data(extracted_data)
        
        # 첫 3개 레코드 출력
        print("\n=== 추출된 데이터 미리보기 ===")
        for i, record in enumerate(extracted_data[:3]):
            print(f"\n레코드 {i+1}:")
            for key, value in record.items():
                print(f"  {key}: {value}")
    
    print("\n테스트 완료!")

if __name__ == "__main__":
    main()