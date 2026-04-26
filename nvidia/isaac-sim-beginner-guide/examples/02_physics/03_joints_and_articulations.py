"""
03_joints_and_articulations.py
==============================
조인트를 이용한 간단한 로봇 만들기 (USD 직접 작성)

학습 목표:
    1. USD로 간단한 2-링크 로봇 구성
    2. Revolute Joint (회전 조인트) 이해
    3. Articulation API로 로봇 제어
    4. Joint position/velocity 읽기 및 쓰기

📝 참고:
    이 예제는 복잡할 수 있어서 다음 03_robotics에서
    미리 만들어진 로봇(Franka, Jetbot 등)을 쓰는 것을 권장합니다.
    이 예제는 "조인트가 어떻게 동작하는지" 이해하기 위한 것입니다.
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import Articulation
from pxr import UsdPhysics, UsdGeom, Gf, Sdf, Usd, PhysxSchema
import numpy as np


def create_revolute_joint(stage, joint_path, body0_path, body1_path,
                           local_pos_0, local_pos_1, axis="Z"):
    """
    두 바디 사이에 Revolute Joint(회전 조인트) 생성
    
    Args:
        stage: USD stage
        joint_path: 조인트 prim path
        body0_path: 부모 바디 path
        body1_path: 자식 바디 path
        local_pos_0: body0의 로컬 좌표에서 조인트 위치
        local_pos_1: body1의 로컬 좌표에서 조인트 위치
        axis: 회전축 ("X", "Y", "Z")
    """
    joint = UsdPhysics.RevoluteJoint.Define(stage, Sdf.Path(joint_path))
    
    # 연결할 두 바디 지정
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    
    # 로컬 앵커 위치
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos_0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos_1))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    
    # 회전축 지정
    joint.CreateAxisAttr(axis)
    
    # 각도 제한 (degree)
    joint.CreateLowerLimitAttr(-90.0)
    joint.CreateUpperLimitAttr(90.0)
    
    # PhysX drive (관절 제어기) 추가
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateMaxForceAttr(1000.0)
    drive.CreateTargetPositionAttr(0.0)  # 목표 각도 (deg)
    drive.CreateDampingAttr(100.0)       # D gain
    drive.CreateStiffnessAttr(1000.0)    # P gain (스프링처럼 0으로 돌아가려는 힘)
    
    return joint


def main():
    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
    world.scene.add_default_ground_plane()
    
    stage = world.stage
    
    # ============================================================
    # 2-DOF 로봇 팔 만들기
    # ============================================================
    # 구조:
    #   base (고정) ── joint1 ── link1 ── joint2 ── link2 (엔드이펙터)
    
    # Root prim 생성 (articulation의 최상위)
    robot_root_path = "/World/SimpleRobot"
    robot_xform = UsdGeom.Xform.Define(stage, Sdf.Path(robot_root_path))
    UsdPhysics.ArticulationRootAPI.Apply(robot_xform.GetPrim())
    
    # Base (지면에 고정)
    base = DynamicCuboid(
        prim_path=f"{robot_root_path}/Base",
        position=np.array([0, 0, 0.1]),
        scale=np.array([0.3, 0.3, 0.2]),
        color=np.array([0.2, 0.2, 0.2]),
        mass=10.0,
    )
    # Base를 월드에 고정 (FixedJoint)
    fixed_joint = UsdPhysics.FixedJoint.Define(
        stage, Sdf.Path(f"{robot_root_path}/BaseFixedJoint")
    )
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{robot_root_path}/Base")])
    
    # Link 1 (상완)
    link1 = DynamicCuboid(
        prim_path=f"{robot_root_path}/Link1",
        position=np.array([0, 0, 0.55]),
        scale=np.array([0.1, 0.1, 0.6]),
        color=np.array([0.8, 0.2, 0.2]),
        mass=2.0,
    )
    
    # Link 2 (하완)
    link2 = DynamicCuboid(
        prim_path=f"{robot_root_path}/Link2",
        position=np.array([0, 0, 1.15]),
        scale=np.array([0.08, 0.08, 0.5]),
        color=np.array([0.2, 0.8, 0.2]),
        mass=1.0,
    )
    
    # Joint 1: base와 link1 연결 (Y축 기준 회전 - 어깨)
    create_revolute_joint(
        stage,
        joint_path=f"{robot_root_path}/Joint1",
        body0_path=f"{robot_root_path}/Base",
        body1_path=f"{robot_root_path}/Link1",
        local_pos_0=(0, 0, 0.1),   # base 위쪽
        local_pos_1=(0, 0, -0.3),  # link1 아래쪽
        axis="Y",
    )
    
    # Joint 2: link1과 link2 연결 (Y축 기준 회전 - 팔꿈치)
    create_revolute_joint(
        stage,
        joint_path=f"{robot_root_path}/Joint2",
        body0_path=f"{robot_root_path}/Link1",
        body1_path=f"{robot_root_path}/Link2",
        local_pos_0=(0, 0, 0.3),    # link1 위쪽
        local_pos_1=(0, 0, -0.25),  # link2 아래쪽
        axis="Y",
    )
    
    world.reset()
    
    # ============================================================
    # Articulation으로 로봇 제어
    # ============================================================
    robot = Articulation(prim_paths_expr=robot_root_path, name="simple_robot")
    robot.initialize()
    
    print("✅ 2-DOF 로봇 생성 완료")
    print(f"   - DOF 수: {robot.num_dof}")
    print(f"   - Joint 이름: {robot.dof_names}")
    
    # ============================================================
    # 사인파 궤적으로 관절 움직이기
    # ============================================================
    dt = world.get_physics_dt()
    
    for step in range(1200):  # 20초
        sim_time = step * dt
        
        # 사인파 형태의 목표 각도 (라디안)
        target_angle_1 = 0.5 * np.sin(sim_time * 2.0)       # 어깨
        target_angle_2 = 0.5 * np.sin(sim_time * 3.0 + 0.5) # 팔꿈치 (다른 주파수)
        
        # 관절 위치 목표 설정
        target_positions = np.array([[target_angle_1, target_angle_2]])
        robot.set_joint_position_targets(target_positions)
        
        world.step(render=True)
        
        # 0.5초마다 현재 상태 출력
        if step % 30 == 0:
            current_pos = robot.get_joint_positions()
            print(f"[{sim_time:.2f}s] 목표: [{target_angle_1:.3f}, {target_angle_2:.3f}] "
                  f"/ 현재: {current_pos[0].round(3)}")
    
    simulation_app.close()


if __name__ == "__main__":
    main()
