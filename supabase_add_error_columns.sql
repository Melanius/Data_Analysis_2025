-- predictions 테이블에 오차 관련 컬럼 추가
-- Supabase SQL Editor에서 실행

-- 1. 오차 컬럼 추가
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS error_amount NUMERIC;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS error_rate NUMERIC;

-- 2. 컬럼 코멘트
COMMENT ON COLUMN predictions.error_amount IS '예측 오차 (원) = 실제값 - 예측값';
COMMENT ON COLUMN predictions.error_rate IS '예측 오차율 (%) = (실제값 - 예측값) / 실제값 * 100';

-- 3. 인덱스 생성 (정렬 및 필터링 성능 향상)
CREATE INDEX IF NOT EXISTS idx_predictions_error_rate ON predictions(error_rate);
