"""
웹 크롤링 기반 데이터 수집 모듈
유료 웹사이트에서 입찰 정보를 수집하는 기능
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BidDataScraper:
    """입찰 데이터 웹 크롤링 클래스"""
    
    def __init__(self, username: str, password: str):
        """
        초기화
        
        Args:
            username: 로그인 아이디
            password: 로그인 비밀번호
        """
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.base_url = "https://infose.info21c.net"
        self.login_url = f"{self.base_url}/info21c/login"
        self.data_url = "https://infose.info21c.net/info21c/bids/list?bidtype=con&bid_suc=suc"
        
        # 헤더 설정 (브라우저처럼 보이게)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
    def login(self) -> bool:
        """
        웹사이트 로그인
        
        Returns:
            bool: 로그인 성공 여부
        """
        try:
            # 로그인 페이지에서 필요한 토큰 등 확인
            login_page = self.session.get(self.login_url)
            soup = BeautifulSoup(login_page.content, 'html.parser')
            
            # 로그인 폼 데이터 준비
            login_data = {
                'username': self.username,
                'password': self.password,
                # 필요한 경우 추가 필드 (CSRF 토큰 등)
            }
            
            # 로그인 시도
            response = self.session.post(self.login_url, data=login_data)
            
            # 로그인 성공 확인 (리다이렉트나 특정 텍스트로 판단)
            if response.status_code == 200:
                logger.info("로그인 성공")
                return True
            else:
                logger.error(f"로그인 실패: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"로그인 중 오류 발생: {e}")
            return False
    
    def get_page_data(self, page: int = 1) -> Optional[List[Dict]]:
        """
        특정 페이지의 데이터 수집
        
        Args:
            page: 페이지 번호
            
        Returns:
            List[Dict]: 수집된 데이터 리스트
        """
        try:
            # 페이지별 URL 구성
            url = f"{self.data_url}&page={page}"
            response = self.session.get(url)
            
            if response.status_code != 200:
                logger.error(f"페이지 {page} 요청 실패: {response.status_code}")
                return None
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 데이터 파싱 (실제 HTML 구조에 맞게 수정 필요)
            data_rows = []
            
            # 테이블 또는 리스트 형태의 데이터 찾기
            # 예시: 테이블 구조인 경우
            table = soup.find('table')
            if table:
                rows = table.find_all('tr')[1:]  # 헤더 제외
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 5:  # 필요한 컬럼 수만큼
                        row_data = {
                            '지역': cols[0].get_text(strip=True) if cols[0] else '',
                            '업종': cols[1].get_text(strip=True) if cols[1] else '',
                            '하한율': cols[2].get_text(strip=True) if cols[2] else '',
                            'A값': cols[3].get_text(strip=True) if cols[3] else '',
                            '기초금액': cols[4].get_text(strip=True) if cols[4] else '',
                            '예가': cols[5].get_text(strip=True) if len(cols) > 5 and cols[5] else '',
                            '수집일시': datetime.now().isoformat()
                        }
                        data_rows.append(row_data)
            
            logger.info(f"페이지 {page}에서 {len(data_rows)}개 데이터 수집")
            return data_rows
            
        except Exception as e:
            logger.error(f"페이지 {page} 데이터 수집 중 오류: {e}")
            return None
    
    def collect_data(self, max_pages: int = 10, delay: float = 1.0) -> pd.DataFrame:
        """
        여러 페이지에서 데이터 수집
        
        Args:
            max_pages: 최대 수집할 페이지 수
            delay: 페이지 간 지연시간 (초)
            
        Returns:
            pd.DataFrame: 수집된 전체 데이터
        """
        if not self.login():
            logger.error("로그인 실패로 인한 데이터 수집 중단")
            return pd.DataFrame()
        
        all_data = []
        
        for page in range(1, max_pages + 1):
            logger.info(f"페이지 {page} 수집 시작...")
            
            page_data = self.get_page_data(page)
            if page_data:
                all_data.extend(page_data)
                
                # 페이지 간 지연
                if page < max_pages:
                    time.sleep(delay)
            else:
                logger.warning(f"페이지 {page}에서 데이터 없음")
                break
        
        df = pd.DataFrame(all_data)
        logger.info(f"총 {len(df)}개 레코드 수집 완료")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        수집된 데이터 저장
        
        Args:
            df: 저장할 데이터프레임
            filename: 파일명 (없으면 자동 생성)
            
        Returns:
            str: 저장된 파일 경로
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bid_data_{timestamp}.csv"
        
        # 데이터 폴더 확인
        data_dir = "../../data/raw"
        os.makedirs(data_dir, exist_ok=True)
        
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        logger.info(f"데이터 저장 완료: {filepath}")
        return filepath

def main():
    """메인 실행 함수"""
    # 크롤러 초기화
    scraper = BidDataScraper(
        username="wjoon97",
        password="joon3277^^"
    )
    
    # 데이터 수집
    df = scraper.collect_data(max_pages=5, delay=2.0)
    
    if not df.empty:
        # 데이터 저장
        saved_file = scraper.save_data(df)
        print(f"데이터 수집 완료: {saved_file}")
        print(f"수집된 레코드 수: {len(df)}")
        print("\n데이터 미리보기:")
        print(df.head())
    else:
        print("데이터 수집 실패")

if __name__ == "__main__":
    main()