# Git Push HTTP 400 해결과 Git LFS 도입 가이드

## 이 저장소에서 실제로 발생한 일

2026-05-23 기준 `git push origin main` 실행 시 아래와 같은 에러가 발생했다.

```bash
error: RPC failed; HTTP 400 curl 22 The requested URL returned error: 400
send-pack: unexpected disconnect while reading sideband packet
fatal: the remote end hung up unexpectedly
```

처음에는 이미지가 포함된 `.ipynb` 파일이 너무 커져서 발생한 문제로 의심했지만, 실제로 push 대상 객체 크기를 확인해보니 **그 원인은 아니었다**.

문제가 있던 커밋의 주요 파일 크기:

- `dacon/6.1.4.Exercise.ipynb`: 약 564 KB
- `dacon/7.HyperParameterTunning.ipynb`: 약 3.3 MB
- 원격으로 전송되는 전체 pack 크기: 약 3.0 MB

정리하면:

- 이번 push는 GitHub 일반 Git 파일 크기 제한 때문에 실패한 것이 아니다.
- 실제 문제는 HTTPS push 과정에서 `git-receive-pack` 단계 연결이 끊긴 것이었다.
- 실무적으로는 `HTTP/1.1` 강제와 `http.postBuffer` 증가로 해결됐다.

## 바로 적용할 해결 방법

### 1회성 push

```bash
git -c http.version=HTTP/1.1 -c http.postBuffer=10485760 push origin main
```

### 이 저장소에 로컬 설정으로 고정

```bash
git config --local http.version HTTP/1.1
git config --local http.postBuffer 10485760
```

### 확인

```bash
git config --show-origin --get-all http.version
git config --show-origin --get-all http.postBuffer
git push origin main
```

## Git LFS가 실제로 필요한 경우

Git LFS는 저장소에 일반 Git으로 관리하기 불편하거나, GitHub 제한에 걸릴 가능성이 있는 대용량 파일이 들어오기 시작할 때 도입하는 것이 맞다.

대표적인 대상:

- 모델 가중치: `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`
- 직렬화 산출물: `*.pkl`, `*.joblib`
- 대용량 데이터셋: `*.parquet`, `*.zip`, `*.tar.gz`, `*.csv`
- 미디어 자산: `*.png`, `*.jpg`, `*.jpeg`, `*.gif`, `*.mp4`

노트북은 조금 다르게 보는 것이 좋다.

- 모든 `*.ipynb`를 기본적으로 LFS에 넣는 것은 권장하지 않는다.
- `.ipynb`는 JSON 텍스트라서 일반 Git diff가 유용한 경우가 많다.
- 용량 증가 원인이 플롯, 이미지, 표 출력이라면 우선 출력 제거를 먼저 고려하는 편이 낫다.

GitHub 공식 문서 기준으로 참고할 점:

- 일반 Git은 50 MiB를 넘는 파일에 대해 경고를 준다.
- 일반 Git은 100 MiB를 넘는 파일을 차단한다.
- Git LFS는 실제 대용량 콘텐츠를 일반 Git 객체 밖에 저장하고, 저장소에는 포인터 파일을 둔다.

## Git LFS 도입 방법

### 1. Git LFS 설치

macOS + Homebrew 기준:

```bash
brew install git-lfs
git lfs install
```

### 2. 정말 필요한 파일 타입만 추적

예시:

```bash
git lfs track "*.parquet"
git lfs track "*.pkl"
git lfs track "*.pt"
git lfs track "*.png"
```

이 명령은 `.gitattributes`를 수정한다. 따라서 `.gitattributes`도 같이 커밋해야 한다.

```bash
git add .gitattributes
git commit -m "Configure Git LFS tracking"
```

### 3. 이후에는 평소처럼 add, commit, push

```bash
git add large_model.pt data/train.parquet
git commit -m "Add model and dataset via Git LFS"
git push origin main
```

## 이미 큰 파일을 커밋한 뒤라면

상황은 보통 두 가지다.

### 경우 1. 문제 커밋이 로컬에만 있고 아직 push 전인 경우

먼저 파일 타입을 LFS로 추적하게 만든 뒤, 해당 파일을 다시 add해서 LFS 방식으로 저장되게 한다.

```bash
git lfs track "*.pt"
git rm --cached model.pt
git add .gitattributes model.pt
git commit --amend
```

### 경우 2. 큰 파일이 이미 커밋 히스토리에 들어가 있고 push가 막히는 경우

이 경우에는 히스토리 재작성 작업이 필요할 수 있다.

```bash
git lfs migrate import --include="*.pt,*.pkl,*.parquet"
git push --force-with-lease origin main
```

주의:

- `git lfs migrate import`는 커밋 해시를 바꾼다.
- 여러 사람이 함께 쓰는 브랜치라면 먼저 조율해야 한다.
- 단일 파일만 옮길 때는 더 좁은 include 패턴이나 경로를 쓰는 것이 안전하다.

## 이 저장소에 권장하는 운영 방식

현 시점에서 합리적인 기본값은 아래와 같다.

1. 일반적인 노트북은 계속 일반 Git으로 관리한다.
2. 렌더링 이미지가 큰 노트북은 커밋 전에 출력 제거를 먼저 검토한다.
3. 실제로 큰 바이너리 산출물만 Git LFS로 옮긴다.
4. `*.ipynb` 전체를 LFS로 묶는 것은, 반복적으로 용량 문제가 생길 때만 검토한다.

ML 저장소에서 먼저 LFS 후보로 보기 좋은 확장자:

- `*.pt`
- `*.pth`
- `*.ckpt`
- `*.onnx`
- `*.pkl`
- `*.joblib`
- `*.parquet`
- `*.zip`

## 참고 문서

- GitHub Docs: About large files on GitHub
  - https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- GitHub Docs: About Git Large File Storage
  - https://docs.github.com/repositories/working-with-files/managing-large-files/about-git-large-file-storage
- GitHub Docs: Installing Git Large File Storage
  - https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage
- GitHub Docs: Moving a file in your repository to Git Large File Storage
  - https://docs.github.com/en/repositories/working-with-files/managing-large-files/moving-a-file-in-your-repository-to-git-large-file-storage
