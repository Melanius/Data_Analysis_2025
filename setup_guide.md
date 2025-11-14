# 개발 환경 설정 가이드

## 현재 상태
✅ 프로젝트 구조 생성 완료  
✅ 학습 데이터 복사 완료 (`data/raw/` 폴더)  
✅ 기본 설정 파일 생성 완료  

## 데이터 파일 확인
- `Bid list_merged.xlsx` - 입찰 공고 리스트
- `List of Successful Bidders_merged.xlsx` - 낙찰자 목록

## 다음 단계 (Windows에서 실행)

### 1. Python 가상환경 생성 (Windows)
```cmd
# CMD 또는 PowerShell에서 프로젝트 폴더로 이동
cd C:\Users\star\claude\bid-prediction

# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. Jupyter Lab 실행
```cmd
# 가상환경 활성화 후
jupyter lab
```

### 3. 첫 번째 데이터 탐색
새 노트북 생성 후 다음 코드로 데이터 확인:

```python
import pandas as pd
import numpy as np

# 데이터 로드
bid_list = pd.read_excel('data/raw/Bid list_merged.xlsx')
successful_bidders = pd.read_excel('data/raw/List of Successful Bidders_merged.xlsx')

# 기본 정보 확인
print("입찰공고 데이터:")
print(bid_list.info())
print("\n낙찰자 데이터:")
print(successful_bidders.info())
```

## 프로젝트 구조
```
bid-prediction/
├── data/raw/           ✅ 원시 데이터 저장됨
├── config/            ✅ 설정 파일
├── docs/              ✅ 문서
├── src/               ✅ 소스 코드 폴더
└── requirements.txt   ✅ 의존성 목록
```

## 문제 해결
WSL 환경에서 Python 가상환경 생성에 제약이 있어, Windows 환경에서 직접 작업하는 것을 권장합니다.