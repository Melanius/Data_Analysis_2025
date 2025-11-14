# 낙찰하한가 예측 모델 프로젝트

## 프로젝트 개요
정부 입찰 공고 시점에 알 수 있는 정보를 활용하여 낙찰하한가를 예측하는 머신러닝 모델 개발

### 핵심 목표
- **예측 대상**: 예가 (예정가격 = 기초금액 × 예가)
- **최종 계산**: 낙찰하한가 = (예정가격 - A값) × 낙찰하한율 + A값

## 프로젝트 구조
```
bid-prediction/
├── data/
│   ├── raw/              # 원시 데이터 (입찰공고, 낙찰자 목록)
│   ├── processed/        # 전처리된 훈련 데이터
│   └── external/         # 외부 경제 지표 데이터
├── src/
│   ├── data/             # 데이터 수집/전처리 모듈
│   ├── features/         # 피쳐 엔지니어링
│   ├── models/           # ML 모델 정의
│   ├── evaluation/       # 성능 평가
│   └── api/              # 예측 API
├── notebooks/            # EDA 및 실험 노트북
├── docs/                 # 마크다운 문서
├── models/               # 저장된 모델 파일
└── config/               # 설정 파일
```

## 입력 변수
- **지역** (카테고리): 입찰 지역
- **업종** (카테고리): 사업 분야
- **하한율** (수치): 낙찰하한율
- **A값** (수치): 계산 상수
- **기초금액** (수치): 입찰 시 제공되는 사업 기초 금액

## 설치 및 실행
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# Jupyter 노트북 실행
jupyter lab
```

## 개발 단계
- [x] 프로젝트 구조 생성
- [ ] 데이터 탐색 및 전처리
- [ ] 베이스라인 모델 개발
- [ ] 모델 최적화
- [ ] MLOps 파이프라인 구축

## 문서
- [비즈니스 로직](docs/business-logic.md)
- [데이터 분석 결과](docs/data-analysis.md)
- [모델 개발 과정](docs/model-development.md)