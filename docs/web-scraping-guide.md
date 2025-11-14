# 웹 크롤링 가이드

## 개요
유료 웹사이트(https://infose.info21c.net)에서 입찰 정보를 수집하는 시스템

## 주요 구성 요소

### 1. BidDataScraper (`src/data/web_scraper.py`)
- **목적**: 웹사이트 로그인 및 데이터 크롤링
- **주요 기능**:
  - 로그인 세션 관리
  - 페이지별 데이터 수집
  - HTML 파싱 및 데이터 추출
  - 수집 속도 제어 (지연시간 설정)

### 2. DataCollector (`src/data/data_collector.py`)
- **목적**: 데이터 수집 작업 관리 및 자동화
- **주요 기능**:
  - 스케줄링 (일일/주간 수집)
  - 데이터 품질 검증
  - 자동 저장 및 백업
  - 수집 통계 관리

## 사용법

### 기본 사용
```python
from src.data import BidDataScraper

# 크롤러 초기화
scraper = BidDataScraper("username", "password")

# 데이터 수집 (5페이지, 2초 지연)
df = scraper.collect_data(max_pages=5, delay=2.0)

# 데이터 저장
scraper.save_data(df, "bid_data.csv")
```

### 자동 수집 설정
```python
from src.data import DataCollector

# 수집 관리자 초기화
collector = DataCollector()

# 크롤러 설정
collector.initialize_scraper("username", "password")

# 스케줄 설정 (매일 9시)
collector.setup_scheduler()

# 스케줄러 실행
collector.run_scheduler()
```

## 설정

### config.yaml에 추가 설정
```yaml
data_collection:
  max_pages: 10          # 최대 수집 페이지
  delay_seconds: 2.0     # 페이지 간 지연시간
  retry_attempts: 3      # 재시도 횟수
  quality_threshold: 0.8 # 데이터 품질 임계값
```

## 중요 고려사항

### 1. 웹사이트 구조 의존성
- HTML 구조 변경 시 파싱 로직 수정 필요
- 로그인 방식 변경에 대한 대응
- 접근 차단 및 CAPTCHA 대응

### 2. 에러 처리
- 네트워크 연결 오류
- 로그인 실패
- 페이지 구조 변경
- 데이터 형식 불일치

### 3. 성능 최적화
- 적절한 지연시간 설정 (서버 부하 방지)
- 세션 재사용으로 로그인 횟수 최소화
- 메모리 효율적인 대용량 데이터 처리

### 4. 보안 및 규정 준수
- 로그인 정보 안전한 저장 (.env 파일 사용)
- 웹사이트 이용약관 준수
- 과도한 요청으로 인한 차단 방지

## 테스트

### Jupyter 노트북으로 테스트
1. `notebooks/01_web_scraping_test.ipynb` 실행
2. 웹사이트 연결 및 로그인 테스트
3. 소량 데이터 수집 테스트
4. 데이터 품질 확인

### 단위 테스트
```bash
python -m pytest tests/test_web_scraper.py
```

## 문제 해결

### 로그인 실패
1. 로그인 페이지 URL 확인
2. 폼 필드명 확인 (개발자 도구 사용)
3. CSRF 토큰 또는 추가 인증 요소 확인

### 데이터 수집 실패
1. HTML 구조 변경 확인
2. 테이블/리스트 선택자 업데이트
3. 데이터 필드 매핑 수정

### 성능 문제
1. 지연시간 증가 (`delay` 파라미터)
2. 배치 크기 감소 (`max_pages` 파라미터)
3. 세션 최적화

## 향후 개선 사항

### 1. 고급 크롤링 기능
- Selenium을 활용한 JavaScript 처리
- 동적 로딩 페이지 대응
- 다중 브라우저 세션 관리

### 2. 데이터 품질 향상
- 데이터 검증 규칙 확장
- 이상값 자동 탐지
- 중복 데이터 정리 알고리즘

### 3. 모니터링 및 알림
- 수집 실패 시 자동 알림
- 데이터 품질 지표 대시보드
- 성능 메트릭 추적

### 4. 확장성
- 다중 웹사이트 지원
- 분산 수집 시스템
- 클라우드 배포 지원