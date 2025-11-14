-- predictions 테이블에 bid_number 컬럼 추가
-- Supabase SQL Editor에서 실행

-- 1. 컬럼 추가
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS bid_number VARCHAR(100);

-- 2. 인덱스 생성 (검색 성능 향상)
CREATE INDEX IF NOT EXISTS idx_predictions_bid_number ON predictions(bid_number);

-- 3. 컬럼 코멘트
COMMENT ON COLUMN predictions.bid_number IS '공고번호 (선택사항)';
