# 낙찰하한가 예측 시스템 배포 가이드

## 📋 배포 전 체크리스트

### 1. Supabase 데이터베이스 설정

#### 1.1 Supabase 콘솔 접속
- URL: https://supabase.com/dashboard
- 프로젝트: https://kkaltyxehwupjxijpwtv.supabase.co

#### 1.2 테이블 생성
1. Supabase 대시보드 → SQL Editor
2. `supabase_schema.sql` 파일의 내용을 복사하여 실행
3. 테이블 생성 확인: Table Editor에서 `predictions` 테이블 확인

```sql
-- 테이블 확인 쿼리
SELECT * FROM predictions LIMIT 1;
```

### 2. 환경 변수 설정

#### 2.1 로컬 개발용
```bash
# .env 파일 생성
cp .env.example .env
```

`.env` 파일 내용:
```
SUPABASE_URL=https://kkaltyxehwupjxijpwtv.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtrYWx0eXhlaHd1cGp4aWpwd3R2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE0MTY0NTMsImV4cCI6MjA0Njk5MjQ1M30.Pf-GaTUzM3-fRzSOwR3SRU5Jt6dSbzWqz1LWqZwKh_Q
```

#### 2.2 Streamlit Cloud용
- Streamlit Cloud 배포 시 Secrets에 추가

### 3. 로컬 테스트

#### 3.1 의존성 설치
```bash
pip install -r requirements.txt
```

#### 3.2 로컬 실행
```bash
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속하여 확인

#### 3.3 테스트 체크리스트
- [ ] 예측하기 페이지 로드
- [ ] 5개 입력 필드 정상 작동
- [ ] 3가지 모델 예측 실행
- [ ] 예측 결과 표시
- [ ] 예측 저장 기능
- [ ] 예측 이력 조회
- [ ] 실제값 입력
- [ ] 정확도 분석 표시

---

## 🚀 Streamlit Cloud 배포

### 1. GitHub 저장소 준비

#### 1.1 .gitignore 설정
```bash
# .gitignore에 추가
.env
*.pkl
__pycache__/
*.pyc
.DS_Store
```

#### 1.2 필수 파일 확인
- [x] `app.py` - 메인 애플리케이션
- [x] `requirements.txt` - 의존성
- [x] `models/production/*.pkl` - 학습된 모델 (Git에 포함 필요)
- [x] `.streamlit/config.toml` (선택사항)

#### 1.3 Git 저장소에 푸시
```bash
git add app.py requirements.txt models/production/
git commit -m "Add Streamlit web app for bid prediction"
git push origin main
```

### 2. Streamlit Cloud 배포

#### 2.1 Streamlit Cloud 접속
- URL: https://share.streamlit.io/
- GitHub 계정으로 로그인

#### 2.2 새 앱 배포
1. "New app" 클릭
2. 저장소 선택: `your-username/bid-prediction`
3. Branch: `main`
4. Main file path: `app.py`
5. "Deploy!" 클릭

#### 2.3 환경 변수 설정
1. 배포된 앱 → Settings → Secrets
2. 아래 내용 추가:

```toml
SUPABASE_URL = "https://kkaltyxehwupjxijpwtv.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtrYWx0eXhlaHd1cGp4aWpwd3R2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE0MTY0NTMsImV4cCI6MjA0Njk5MjQ1M30.Pf-GaTUzM3-fRzSOwR3SRU5Jt6dSbzWqz1LWqZwKh_Q"
```

3. "Save" 클릭
4. 앱 자동 재시작

### 3. 배포 확인
- [ ] 앱 URL 접속 확인
- [ ] 모든 페이지 정상 로드
- [ ] 데이터베이스 연결 확인
- [ ] 예측 및 저장 기능 테스트

---

## 🔧 트러블슈팅

### 모델 파일을 찾을 수 없음
**문제**: `models/production/*.pkl` 파일을 찾을 수 없다는 오류

**해결**:
1. Git LFS 사용 (파일이 큰 경우)
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add models/production/*.pkl
git commit -m "Add models with Git LFS"
git push
```

2. 또는 모델 파일을 Git에 직접 포함
```bash
git add -f models/production/*.pkl
git commit -m "Add trained models"
git push
```

### Supabase 연결 오류
**문제**: Supabase 테이블에 접근할 수 없음

**해결**:
1. Supabase 콘솔에서 RLS 정책 확인
2. API Key 확인 (Anon key 사용 중인지)
3. 테이블이 정상 생성되었는지 확인

### 의존성 설치 오류
**문제**: Streamlit Cloud에서 패키지 설치 실패

**해결**:
1. `requirements.txt`에서 버전 충돌 확인
2. Python 버전 확인 (3.8-3.11 권장)
3. 불필요한 패키지 제거

---

## 📊 모델 업데이트

새로운 데이터로 모델을 재학습한 경우:

### 1. 모델 재학습
```bash
python src/models/train_simplified_models.py
```

### 2. 새 모델 파일 확인
```bash
ls -lh models/production/
# linear_model_*.pkl
# ridge_model_*.pkl
# ensemble_model_*.pkl
# metadata_*.json
```

### 3. Git에 커밋 및 푸시
```bash
git add models/production/*_20251112_*.pkl
git commit -m "Update models with new training data"
git push
```

### 4. Streamlit Cloud 자동 재배포
- Git 푸시 후 자동으로 재배포됨
- 또는 수동으로 "Reboot app" 클릭

---

## 🎯 성능 최적화

### 캐싱 설정
`app.py`의 `@st.cache_resource` 데코레이터가 이미 적용되어 있음:
- 모델 로드: 최초 1회만 실행
- Supabase 클라이언트: 세션당 1회만 생성

### 메모리 최적화
큰 모델 파일의 경우:
1. 모델 압축 (joblib compress 옵션)
2. 불필요한 데이터 제거
3. Streamlit Cloud 리소스 업그레이드

---

## 📈 모니터링

### Streamlit Cloud 모니터링
- Logs: 실시간 로그 확인
- Analytics: 사용자 접속 통계
- Performance: 앱 성능 지표

### Supabase 모니터링
- Database → Table Editor: 저장된 데이터 확인
- SQL Editor: 쿼리 실행 및 분석
- Logs: API 요청 로그 확인

---

## 🔒 보안 고려사항

### API 키 관리
- ✅ Supabase Anon Key 사용 (공개 가능)
- ✅ RLS 정책으로 접근 제어
- ❌ Service Role Key는 절대 노출 금지

### 데이터 보호
- RLS (Row Level Security) 활성화
- 민감 정보는 저장하지 않음
- HTTPS 연결 사용

---

## 📞 지원

### 문제 발생 시
1. Streamlit Cloud Logs 확인
2. Supabase Logs 확인
3. GitHub Issues 등록
4. 로컬 환경에서 재현 테스트

### 유용한 링크
- Streamlit 문서: https://docs.streamlit.io/
- Supabase 문서: https://supabase.com/docs
- Python Supabase 클라이언트: https://github.com/supabase-community/supabase-py
