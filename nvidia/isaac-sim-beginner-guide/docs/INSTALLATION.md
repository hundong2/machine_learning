# 📦 Isaac Sim 설치 가이드 (2026년 4월 기준)

Isaac Sim 5.0 이상을 기준으로 작성되었습니다.

## 🎯 설치 방법 선택

### 방법 A: pip 설치 (가장 쉬움, 권장)

Python 가상환경에서 직접 설치할 수 있습니다. **초보자 추천**.

```bash
# 1. conda 가상환경 생성 (Python 3.10)
conda create -n isaac-sim python=3.10 -y
conda activate isaac-sim

# 2. PyTorch 설치 (CUDA 11.8 예시)
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. Isaac Sim 설치
pip install isaacsim==5.0.0 --extra-index-url https://pypi.nvidia.com

# 4. 확장 패키지 (선택)
pip install isaacsim-extscache-physics==5.0.0 \
            isaacsim-extscache-kit==5.0.0 \
            isaacsim-extscache-kit-sdk==5.0.0 \
            --extra-index-url https://pypi.nvidia.com

# 5. 첫 실행 (EULA 동의)
isaacsim
```

### 방법 B: Binary 설치 (Omniverse Launcher)

1. [Omniverse Launcher 다운로드](https://www.nvidia.com/en-us/omniverse/download/)
2. Launcher에서 "Exchange" → "Isaac Sim" 설치
3. 기본 경로: `~/.local/share/ov/pkg/isaac-sim-5.0.0/`

### 방법 C: Docker / NGC Container

```bash
docker pull nvcr.io/nvidia/isaac-sim:5.0.0
docker run --name isaac-sim --entrypoint bash -it --gpus all \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    --network=host \
    --runtime=nvidia \
    nvcr.io/nvidia/isaac-sim:5.0.0
```

### 방법 D: GitHub에서 빌드 (최신/개발자용)

```bash
git clone https://github.com/isaac-sim/IsaacSim.git
cd IsaacSim
./build.sh
```

## ✅ 설치 확인

```bash
# 1. Python import 확인
python -c "from isaacsim import SimulationApp; print('✅ Isaac Sim 설치 완료')"

# 2. GPU 인식 확인
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 3. 간단한 예제 실행
cd isaac-sim-beginner-guide
python examples/01_basics/01_hello_world.py
```

## 🔧 시스템 요구사항

### 최소
- **GPU**: RTX 3070 (8GB VRAM)
- **RAM**: 32GB
- **OS**: Ubuntu 22.04 또는 Windows 10/11
- **Driver**: NVIDIA 535+

### 권장
- **GPU**: RTX 4080/4090 (16GB+ VRAM)
- **RAM**: 64GB
- **Storage**: NVMe SSD 100GB+
- **Driver**: 최신 Studio 또는 Production Branch

### ❌ 지원 안 됨
- GTX 시리즈 (RT Core 없음)
- AMD / Intel GPU
- macOS

## ⚠️ 자주 발생하는 문제

### 1. "Error: Nucleus is not available"
```bash
# Nucleus Cache 실행 (방법 A 사용자는 무시 가능)
# Enterprise Nucleus Server 설치 권장 (2025년 10월부터)
```

### 2. Python 버전 문제
```bash
# Isaac Sim 5.0은 Python 3.10만 지원
python --version  # Python 3.10.x여야 함
```

### 3. "Cannot load USD file"
```bash
# 에셋 서버 접근 확인
# 또는 로컬 에셋 팩 다운로드:
# https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html
```

### 4. "ImportError: libGL.so.1"
```bash
# Linux에서 OpenGL 라이브러리 설치
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 \
    libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```

## 📚 공식 문서

- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- [Installation Guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/index.html)
- [Isaac Sim GitHub](https://github.com/isaac-sim/IsaacSim)
- [NVIDIA Developer Forum](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-sim/69)
