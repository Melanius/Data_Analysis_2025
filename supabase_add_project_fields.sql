-- 공사명 및 낙찰 발표일 컬럼 추가
-- Supabase SQL Editor에서 실행

-- 1. 컬럼 추가
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS project_name TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS bid_announcement_date DATE;

-- 2. 인덱스 생성 (검색 성능 향상)
CREATE INDEX IF NOT EXISTS idx_predictions_project_name ON predictions(project_name);
CREATE INDEX IF NOT EXISTS idx_predictions_bid_date ON predictions(bid_announcement_date);

-- 3. 컬럼 코멘트
COMMENT ON COLUMN predictions.project_name IS '공사명 (프로젝트 식별용)';
COMMENT ON COLUMN predictions.bid_announcement_date IS '낙찰하한가 발표일 (선택사항)';
