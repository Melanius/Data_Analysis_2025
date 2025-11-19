-- =====================================================
-- 회원 인증 시스템 DB 마이그레이션
-- 실행 위치: Supabase SQL Editor
-- =====================================================

-- 1. users 테이블 생성
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. users 테이블 코멘트
COMMENT ON TABLE users IS '사용자 계정 정보';
COMMENT ON COLUMN users.id IS '사용자 고유 ID (UUID)';
COMMENT ON COLUMN users.username IS '로그인 아이디';
COMMENT ON COLUMN users.password_hash IS 'bcrypt 해시된 비밀번호';
COMMENT ON COLUMN users.email IS '이메일 (비밀번호 찾기용)';
COMMENT ON COLUMN users.is_admin IS '관리자 권한 여부';

-- 3. predictions 테이블에 user_id 컬럼 추가
ALTER TABLE predictions
ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id) ON DELETE SET NULL;

-- 4. predictions 테이블 코멘트
COMMENT ON COLUMN predictions.user_id IS '예측을 생성한 사용자 ID (NULL = 공유 데이터)';

-- 5. 인덱스 생성 (성능 최적화)
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- 6. updated_at 자동 업데이트 트리거 함수 (있으면 재사용)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 7. users 테이블에 updated_at 트리거 추가
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- 마이그레이션 완료
-- 다음 단계: create_initial_users.py 실행하여 초기 계정 생성
-- =====================================================
