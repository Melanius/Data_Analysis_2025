# 🚀 빠른 시작 가이드

## 1️⃣ Supabase 테이블 생성 (1분)

### Supabase 콘솔에서:
1. https://supabase.com/dashboard/project/kkaltyxehwupjxijpwtv 접속
2. 왼쪽 메뉴 → **SQL Editor** 클릭
3. `supabase_schema.sql` 파일 내용 복사하여 붙여넣기
4. **Run** 클릭
5. 왼쪽 메뉴 → **Table Editor** → `predictions` 테이블 확인 ✅

---

## 2️⃣ 로컬 테스트 (2분)

### 터미널에서:
```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 앱 실행
streamlit run app.py
```

### 브라우저에서:
- http://localhost:8501 자동 오픈
- 예측하기 → 값 입력 → 예측 실행 → 저장 테스트 ✅

---

## 3️⃣ Streamlit Cloud 배포 (5분)

### 1단계: GitHub에 푸시
```bash
git add app.py requirements.txt models/ .streamlit/ supabase_schema.sql
git commit -m "Add bid prediction web app"
git push origin main
```

### 2단계: Streamlit Cloud 배포
1. https://share.streamlit.io/ 접속
2. **New app** 클릭
3. 저장소 선택: `your-repo/bid-prediction`
4. Branch: `main`
5. Main file: `app.py`
6. **Advanced settings** → **Secrets** 추가:
   ```toml
   SUPABASE_URL = "https://kkaltyxehwupjxijpwtv.supabase.co"
   SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtrYWx0eXhlaHd1cGp4aWpwd3R2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE0MTY0NTMsImV4cCI6MjA0Njk5MjQ1M30.Pf-GaTUzM3-fRzSOwR3SRU5Jt6dSbzWqz1LWqZwKh_Q"
   ```
7. **Deploy!** 클릭
8. 배포 완료 대기 (2-3분)
9. 앱 URL로 접속 ✅

---

## ✅ 완성!

이제 다음 기능을 사용할 수 있습니다:

### 🎯 예측하기
- 5개 입력 필드 (추정가격, 기초금액, A값, 낙찰하한율, 예가변동폭)
- 3가지 모델 예측 (Ridge, Linear, Ensemble)
- 모델별 예측 결과 비교
- 원하는 예측 선택 및 저장

### 📊 예측 이력
- 저장된 모든 예측 조회
- 실제 낙찰하한가 입력
- 예측 vs 실제 비교

### 📈 정확도 분석
- 모델별 성능 통계 (MAPE, 표준편차)
- 시각화 차트 (모델 비교, 예측 vs 실제)
- 최근 예측 상세 내역

---

## 🔧 트러블슈팅

### 모델 파일 오류
```bash
# Git LFS 설치 및 모델 파일 추적
git lfs install
git lfs track "*.pkl"
git add .gitattributes models/production/*.pkl
git commit -m "Add models with LFS"
git push
```

### Supabase 연결 오류
- API Key 확인 (Anon key 사용)
- RLS 정책 확인 (모든 사용자 읽기/쓰기 허용)
- 테이블 이름 확인 (`predictions`)

### 의존성 오류
```bash
# 가상환경 사용 권장
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📞 도움이 필요하신가요?

자세한 배포 가이드는 `DEPLOYMENT.md`를 참고하세요.
