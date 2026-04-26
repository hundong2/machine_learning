# 🎨 USD (Universal Scene Description) 이해하기

Isaac Sim의 모든 씬은 **USD** 형식으로 저장됩니다. USD를 이해하면 Isaac Sim이 훨씬 쉬워집니다.

## 1. USD란?

**Pixar가 만든 3D 씬 기술 형식**. 현재 3D 파이프라인의 사실상 표준.

### 특징
- **계층적 구조**: 파일 시스템처럼 트리 구조
- **Non-destructive**: 원본을 수정하지 않고 레이어로 수정
- **컴포지션**: 여러 파일을 조합 가능 (variants, sublayers, references)

## 2. USD의 기본 개념

### Prim (프림)
씬의 최소 단위. 큐브, 구, 로봇, 카메라 등 모든 것이 prim.

```
/World           ← Xform prim (그룹)
├── /GroundPlane  ← Mesh prim
├── /Cube         ← Mesh prim
│   └── /Material ← Material prim
└── /Robot        ← Xform prim
    ├── /Base     ← Mesh prim
    └── /Arm      ← Mesh prim
```

### Prim Path
파일 시스템 경로와 똑같이 생겼습니다.

```python
# 절대 경로 (/로 시작)
"/World/Robot/Arm"

# 각 prim은 고유한 path를 가짐
```

### Property (속성)
Prim이 가진 속성들. 위치, 색상, 메쉬 데이터 등.

```python
# Isaac Sim에서 속성 설정 예시
cube.set_world_pose(
    position=[1, 0, 0.5],      # position 속성
    orientation=[1, 0, 0, 0]    # orientation 속성
)
```

### Schema (스키마)
Prim의 "타입"을 정의. 예: Cube는 `UsdGeomCube` 스키마.

**자주 쓰는 스키마**:
- `UsdGeomMesh`: 메쉬 (임의의 3D 모델)
- `UsdGeomCube`, `UsdGeomSphere`: 기본 도형
- `UsdGeomXform`: 변환 (그룹)
- `UsdPhysicsRigidBodyAPI`: 강체 물리
- `UsdPhysicsRevoluteJoint`: 회전 조인트
- `UsdLuxDistantLight`: 태양광

### API Schema
Prim에 **기능을 추가**하는 스키마.

```python
# 예: 큐브에 물리 추가
# UsdGeomCube + UsdPhysicsRigidBodyAPI + UsdPhysicsCollisionAPI
# → 중력 받고 충돌도 가능한 큐브
```

## 3. USD 파일 형식

### .usd (바이너리 or 텍스트)
```
#usda 1.0

def Xform "World"
{
    def Cube "MyCube"
    {
        double3 xformOp:translate = (0, 0, 1)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
```

### .usda (텍스트, 읽기 좋음)
위와 동일한 형식이지만 명시적으로 텍스트.

### .usdc (바이너리, 빠름)
대용량 씬에 적합.

### .usdz (압축 패키지)
에셋 공유용 (텍스처, 메쉬 다 포함).

## 4. Python에서 USD 다루기 (pxr)

Isaac Sim 내부에서 USD를 직접 조작할 수 있습니다.

```python
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

# 현재 stage 가져오기
stage = omni.usd.get_context().get_stage()

# Prim 생성
cube_prim = UsdGeom.Cube.Define(stage, Sdf.Path("/World/MyCube"))

# 크기 설정
cube_prim.CreateSizeAttr(1.0)

# 위치 설정
UsdGeom.XformCommonAPI(cube_prim).SetTranslate(Gf.Vec3d(0, 0, 1.0))

# 물리 추가
UsdPhysics.RigidBodyAPI.Apply(cube_prim.GetPrim())
UsdPhysics.CollisionAPI.Apply(cube_prim.GetPrim())
```

## 5. Isaac Sim Core API vs USD API

Isaac Sim은 두 가지 레벨의 API를 제공합니다:

### High-level (Core API) - **권장**
```python
from isaacsim.core.api.objects import DynamicCuboid

cube = DynamicCuboid(
    prim_path="/World/Cube",
    position=np.array([0, 0, 1]),
    scale=np.array([0.5, 0.5, 0.5]),
    color=np.array([1, 0, 0]),
    mass=1.0,
)
```
✅ 간단, 초보자에게 좋음
❌ 세밀한 제어 어려움

### Low-level (USD API)
```python
from pxr import UsdGeom, UsdPhysics

cube_prim = UsdGeom.Cube.Define(stage, "/World/Cube")
UsdPhysics.RigidBodyAPI.Apply(cube_prim.GetPrim())
# ... 많은 설정 필요
```
✅ 모든 USD 기능 접근 가능
❌ 코드가 복잡

## 6. Reference & Payload

### Reference (참조)
다른 USD 파일을 현재 씬에 불러옴.

```python
from isaacsim.core.utils.stage import add_reference_to_stage

add_reference_to_stage(
    usd_path="/path/to/robot.usd",
    prim_path="/World/MyRobot"
)
```

### 왜 중요한가?
- **에셋 재사용**: 로봇 모델을 한 번 만들고 여러 씬에서 사용
- **업데이트 편의**: 원본 파일 수정 시 모든 참조에 반영
- **모듈화**: 큰 씬을 작은 파일로 분할

## 7. Variants (변형)

하나의 prim이 여러 "버전"을 가질 수 있음.

예시: 같은 로봇이 색깔별 variants
```
robot
├── variant: red
├── variant: blue
└── variant: green
```

## 8. 실습: USD 파일 직접 보기

```bash
# 텍스트 에디터로 열어보기 (작은 파일)
cat /path/to/simple_scene.usda

# 또는 usdview (Pixar 공식 뷰어)
usdview scene.usd
```

## ✅ 체크리스트

- [ ] Prim과 Property의 차이를 안다
- [ ] Prim Path 구조를 이해한다
- [ ] Core API와 USD API를 구분할 수 있다
- [ ] Reference가 무엇인지 설명할 수 있다

## 📚 더 공부하기

- [OpenUSD 공식 문서](https://openusd.org/)
- [NVIDIA USD Documentation](https://docs.omniverse.nvidia.com/usd/latest/index.html)
- [Pixar USD Tutorials](https://openusd.org/release/tut_usd_tutorials.html)
