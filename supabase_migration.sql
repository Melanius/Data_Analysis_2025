-- 기존 테이블에 prediction_group_id 컬럼 추가 (마이그레이션)
-- Supabase 콘솔 SQL Editor에서 실행

-- 1. 기존 테이블이 있는 경우 컬럼 추가
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS prediction_group_id UUID;

-- 2. 기존 데이터가 있는 경우, 각 행에 고유한 그룹 ID 부여
UPDATE predictions
SET prediction_group_id = gen_random_uuid()
WHERE prediction_group_id IS NULL;

-- 3. NOT NULL 제약 조건 추가
ALTER TABLE predictions ALTER COLUMN prediction_group_id SET NOT NULL;

-- 4. 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_predictions_group_id ON predictions(prediction_group_id);

-- 5. 코멘트 추가
COMMENT ON COLUMN predictions.prediction_group_id IS '예측 그룹 ID (같은 입력에 대한 3개 모델 결과 그룹화)';
