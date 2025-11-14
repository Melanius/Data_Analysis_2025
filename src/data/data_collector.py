"""
데이터 수집 관리자 모듈
스케줄링과 데이터 품질 관리를 담당
"""

import pandas as pd
import schedule
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import os
import yaml

from web_scraper import BidDataScraper

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """데이터 수집 관리자 클래스"""
    
    def __init__(self, config_path: str = "../../config/config.yaml"):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self.load_config(config_path)
        self.scraper = None
        
    def load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"설정 파일 없음: {config_path}, 기본값 사용")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """기본 설정값"""
        return {
            'data_collection': {
                'max_pages': 10,
                'delay_seconds': 2.0,
                'retry_attempts': 3,
                'quality_threshold': 0.8
            }
        }
    
    def initialize_scraper(self, username: str, password: str) -> bool:
        """크롤러 초기화"""
        try:
            self.scraper = BidDataScraper(username, password)
            return True
        except Exception as e:
            logger.error(f"크롤러 초기화 실패: {e}")
            return False
    
    def collect_daily_data(self) -> pd.DataFrame:
        """일일 데이터 수집"""
        logger.info("일일 데이터 수집 시작")
        
        if not self.scraper:
            logger.error("크롤러가 초기화되지 않음")
            return pd.DataFrame()
        
        # 데이터 수집
        df = self.scraper.collect_data(
            max_pages=self.config['data_collection']['max_pages'],
            delay=self.config['data_collection']['delay_seconds']
        )
        
        # 데이터 품질 검사
        if self.validate_data_quality(df):
            # 저장
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"daily_bid_data_{timestamp}.csv"
            saved_path = self.scraper.save_data(df, filename)
            
            logger.info(f"일일 데이터 수집 완료: {len(df)}개 레코드")
            return df
        else:
            logger.warning("데이터 품질 기준 미달")
            return pd.DataFrame()
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """데이터 품질 검증"""
        if df.empty:
            logger.warning("수집된 데이터가 없음")
            return False
        
        # 필수 컬럼 확인
        required_columns = ['지역', '업종', '하한율', 'A값', '기초금액']
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            logger.warning(f"필수 컬럼 누락: {missing_columns}")
            return False
        
        # 결측값 비율 확인
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        threshold = self.config['data_collection']['quality_threshold']
        
        if missing_ratio > (1 - threshold):
            logger.warning(f"결측값 비율 과다: {missing_ratio:.2%}")
            return False
        
        logger.info(f"데이터 품질 검증 통과: {len(df)}개 레코드, 결측율 {missing_ratio:.2%}")
        return True
    
    def setup_scheduler(self) -> None:
        """자동 수집 스케줄 설정"""
        # 매일 오전 9시에 데이터 수집
        schedule.every().day.at("09:00").do(self.collect_daily_data)
        
        # 매주 월요일 오전 8시에 주간 요약
        schedule.every().monday.at("08:00").do(self.weekly_summary)
        
        logger.info("스케줄러 설정 완료")
    
    def weekly_summary(self) -> None:
        """주간 수집 데이터 요약"""
        logger.info("주간 데이터 요약 생성")
        
        # 지난 7일간 수집된 파일들 찾기
        data_dir = "../../data/raw"
        files = []
        
        for i in range(7):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            filename = f"daily_bid_data_{date_str}.csv"
            filepath = os.path.join(data_dir, filename)
            
            if os.path.exists(filepath):
                files.append(filepath)
        
        if files:
            # 데이터 통합
            dfs = [pd.read_csv(f) for f in files]
            weekly_df = pd.concat(dfs, ignore_index=True)
            
            # 중복 제거
            weekly_df = weekly_df.drop_duplicates()
            
            # 주간 요약 저장
            week_end = datetime.now().strftime("%Y%m%d")
            summary_filename = f"weekly_summary_{week_end}.csv"
            summary_path = os.path.join(data_dir, summary_filename)
            weekly_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            
            logger.info(f"주간 요약 생성: {len(weekly_df)}개 레코드")
        
    def run_scheduler(self) -> None:
        """스케줄러 실행"""
        logger.info("데이터 수집 스케줄러 시작")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 체크
    
    def manual_collect(self, username: str, password: str, pages: int = 5) -> pd.DataFrame:
        """수동 데이터 수집"""
        if self.initialize_scraper(username, password):
            return self.scraper.collect_data(max_pages=pages, delay=2.0)
        else:
            return pd.DataFrame()

def main():
    """테스트 실행"""
    collector = DataCollector()
    
    # 수동 수집 테스트
    df = collector.manual_collect("wjoon97", "joon3277^^", pages=3)
    
    if not df.empty:
        print(f"수집 완료: {len(df)}개 레코드")
        print("\n컬럼:", df.columns.tolist())
        print("\n데이터 미리보기:")
        print(df.head())
    else:
        print("데이터 수집 실패")

if __name__ == "__main__":
    main()