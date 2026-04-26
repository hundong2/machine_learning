#!/usr/bin/env bash
# Isaac Sim 초보자 가이드 - 환경 설정 스크립트
#
# 사용법:
#   bash scripts/setup.sh
#
# 이 스크립트는:
#   1. Python 3.10 venv 생성
#   2. Isaac Sim 5.0 설치
#   3. 추가 의존성 설치

set -e

echo "=========================================="
echo "Isaac Sim 초보자 가이드 - 환경 설정"
echo "=========================================="

# Python 버전 확인
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "[1/4] Python 버전: $PYTHON_VERSION"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python 3.10 이상이 필요합니다."
    exit 1
fi

# 가상환경 생성
if [ ! -d "venv" ]; then
    echo "[2/4] 가상환경 생성 중..."
    python3 -m venv venv
else
    echo "[2/4] 기존 가상환경 사용"
fi

# shellcheck disable=SC1091
source venv/bin/activate

# pip 업그레이드
echo "[3/4] pip 업그레이드..."
pip install --upgrade pip setuptools wheel

# Isaac Sim 설치
echo "[4/4] Isaac Sim 5.0 및 의존성 설치..."
pip install isaacsim==5.0.0 --extra-index-url https://pypi.nvidia.com
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ 설치 완료!"
echo "=========================================="
echo ""
echo "다음 명령어로 환경을 활성화하세요:"
echo "    source venv/bin/activate"
echo ""
echo "첫 예제를 실행해보세요:"
echo "    python examples/01_basics/01_hello_world.py"
echo ""
