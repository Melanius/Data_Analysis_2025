"""
초기 사용자 계정 생성 스크립트
- admin 계정 (관리자)
- wjoon97 계정 (일반 사용자)

실행 방법:
python3 create_initial_users.py
"""

import os
import sys
from pathlib import Path

# bcrypt 설치 확인 및 자동 설치
try:
    import bcrypt
except ImportError:
    print("bcrypt가 설치되어 있지 않습니다. 설치 중...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "bcrypt", "--break-system-packages"])
    import bcrypt

from supabase import create_client

# 환경 변수 로드
def load_env_file(env_path='.env'):
    """간단한 .env 파일 파서"""
    env_vars = {}
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return env_vars

# Supabase 설정
env_vars = load_env_file('/mnt/c/Users/star/claude/bid-prediction/.env')
SUPABASE_URL = env_vars.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = env_vars.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ 오류: SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다.")
    print("   .env 파일을 확인해주세요.")
    sys.exit(1)

# Supabase 클라이언트 생성
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def hash_password(password: str) -> str:
    """비밀번호를 bcrypt로 해싱"""
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    password_hash = bcrypt.hashpw(password_bytes, salt)
    return password_hash.decode('utf-8')


def create_user(username: str, password: str, email: str, is_admin: bool = False):
    """사용자 생성"""
    try:
        # 비밀번호 해싱
        password_hash = hash_password(password)

        # 사용자 데이터
        user_data = {
            'username': username,
            'password_hash': password_hash,
            'email': email,
            'is_admin': is_admin
        }

        # Supabase에 삽입
        response = supabase.table('users').insert(user_data).execute()

        if response.data:
            role = "관리자" if is_admin else "일반 사용자"
            print(f"✅ {username} 계정 생성 완료 ({role})")
            print(f"   이메일: {email}")
            return True
        else:
            print(f"❌ {username} 계정 생성 실패")
            return False

    except Exception as e:
        if "duplicate key" in str(e).lower() or "unique" in str(e).lower():
            print(f"⚠️  {username} 계정이 이미 존재합니다.")
        else:
            print(f"❌ {username} 계정 생성 중 오류: {e}")
        return False


def main():
    """메인 함수"""
    print("=" * 70)
    print("초기 사용자 계정 생성")
    print("=" * 70)
    print()

    # 계정 정보
    accounts = [
        {
            'username': 'admin',
            'password': 'gnswjd88',
            'email': 'melanius88@naver.com',
            'is_admin': True
        },
        {
            'username': 'wjoon97',
            'password': 'joon3277^^',
            'email': 'axis19@naver.com',
            'is_admin': False
        }
    ]

    # 계정 생성
    success_count = 0
    for account in accounts:
        if create_user(**account):
            success_count += 1
        print()

    # 결과 출력
    print("=" * 70)
    print(f"완료: {success_count}/{len(accounts)}개 계정 생성")
    print("=" * 70)
    print()
    print("✅ 다음 단계:")
    print("   1. 로그인 테스트")
    print("   2. app.py 인증 기능 개발")
    print()

    # 생성된 계정 확인
    try:
        response = supabase.table('users').select('username, email, is_admin, created_at').execute()
        if response.data:
            print("📋 생성된 계정 목록:")
            for user in response.data:
                role = "🔑 관리자" if user['is_admin'] else "👤 일반"
                print(f"   {role} - {user['username']} ({user['email']})")
    except Exception as e:
        print(f"⚠️  계정 목록 조회 실패: {e}")


if __name__ == "__main__":
    main()
