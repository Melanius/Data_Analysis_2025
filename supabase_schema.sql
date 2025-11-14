-- 낙찰하한가 예측 테이블
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,

    -- 예측 그룹 ID (같은 입력에 대한 3개 모델을 그룹화)
    prediction_group_id UUID NOT NULL,

    -- 입력 데이터
    chujeong_price NUMERIC NOT NULL,
    gichogeum NUMERIC NOT NULL,
    a_value NUMERIC NOT NULL DEFAULT 0,
    nakchalhahan_rate NUMERIC NOT NULL,
    yega_range NUMERIC NOT NULL,

    -- 모델 정보
    model_type VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,

    -- 예측 결과
    predicted_yega NUMERIC NOT NULL,
    predicted_yejeong_price NUMERIC NOT NULL,
    predicted_nakchalhahan_price NUMERIC NOT NULL,

    -- 실제 결과 (나중에 입력)
    actual_nakchalhahan_price NUMERIC,

    -- 타임스탬프
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model_type ON predictions(model_type);
CREATE INDEX IF NOT EXISTS idx_predictions_group_id ON predictions(prediction_group_id);
CREATE INDEX IF NOT EXISTS idx_predictions_actual_not_null ON predictions(actual_nakchalhahan_price) WHERE actual_nakchalhahan_price IS NOT NULL;

-- RLS (Row Level Security) 활성화
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- 모든 사용자가 읽기/쓰기 가능하도록 정책 설정
CREATE POLICY "Enable read access for all users" ON predictions
    FOR SELECT USING (true);

CREATE POLICY "Enable insert access for all users" ON predictions
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Enable update access for all users" ON predictions
    FOR UPDATE USING (true);

-- 테이블 코멘트
COMMENT ON TABLE predictions IS '낙찰하한가 예측 이력 및 정확도 추적';
COMMENT ON COLUMN predictions.prediction_group_id IS '예측 그룹 ID (같은 입력에 대한 3개 모델 결과 그룹화)';
COMMENT ON COLUMN predictions.chujeong_price IS '추정가격 (입찰 공고 명시)';
COMMENT ON COLUMN predictions.gichogeum IS '기초금액 (예가 산정 기준)';
COMMENT ON COLUMN predictions.a_value IS 'A값 (낙찰하한율 계산용, 없으면 0)';
COMMENT ON COLUMN predictions.nakchalhahan_rate IS '낙찰하한율 (%)';
COMMENT ON COLUMN predictions.yega_range IS '예가변동폭 (2, 2.5, 3)';
COMMENT ON COLUMN predictions.model_type IS '모델 타입 (linear, ridge, ensemble)';
COMMENT ON COLUMN predictions.model_name IS '모델 이름 (표시용)';
COMMENT ON COLUMN predictions.predicted_yega IS '예측된 예가 비율 (%)';
COMMENT ON COLUMN predictions.predicted_yejeong_price IS '예측된 예정가격';
COMMENT ON COLUMN predictions.predicted_nakchalhahan_price IS '예측된 낙찰하한가';
COMMENT ON COLUMN predictions.actual_nakchalhahan_price IS '실제 낙찰하한가 (낙찰 후 입력)';
