-- =====================================================
-- 특정 predictions 데이터에 user_id 할당
-- id 4,5,6,7,8,9 → wjoon97 사용자에게 할당
-- =====================================================

-- 1. wjoon97의 user_id 확인 (참고용)
SELECT id, username FROM users WHERE username = 'wjoon97';

-- 2. 해당 데이터에 wjoon97의 user_id 할당
UPDATE predictions
SET user_id = (SELECT id FROM users WHERE username = 'wjoon97')
WHERE id IN (4, 5, 6, 7, 8, 9);

-- 3. 업데이트 결과 확인
SELECT id, user_id, bid_number, project_name, created_at
FROM predictions
WHERE id IN (4, 5, 6, 7, 8, 9);

-- 4. 전체 user_id 분포 확인
SELECT
    CASE
        WHEN user_id IS NULL THEN 'NULL (공유 데이터)'
        ELSE u.username
    END as owner,
    COUNT(*) as count
FROM predictions p
LEFT JOIN users u ON p.user_id = u.id
GROUP BY user_id, u.username
ORDER BY count DESC;
