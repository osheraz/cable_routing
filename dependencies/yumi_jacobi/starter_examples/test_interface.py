from yumi_jacobi.interface import Interface
from autolab_core import RigidTransform, Point


def run():
    interface = Interface(speed=0.26)
    print(interface.get_FK("left"))
    print(interface.get_FK("right"))
    # interface.home()
    interface.calibrate_grippers()
    interface.open_grippers()
    print(interface.driver_left.get_gripper_pos())
    print(interface.driver_right.get_gripper_pos())
    interface.close_grippers()
    print(interface.driver_left.get_gripper_pos())
    print(interface.driver_right.get_gripper_pos())

    wp1_l = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]], translation=[0.4, 0.3, 0.2]
    )
    wp2_l = RigidTransform(
        rotation=[
            [-1.0000000, -0.0000000, 0.0000000],
            [-0.0000000, 0.9659258, -0.2588190],
            [-0.0000000, -0.2588190, -0.9659258],
        ],
        translation=[0.4, 0.4, 0.1],
    )
    wp3_l = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]], translation=[0.3, 0.3, 0.15]
    )
    wp1_r = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]], translation=[0.3, -0.2, 0.15]
    )
    wp2_r = RigidTransform(
        rotation=[[-1, 0, 0], [0, 1, 0], [0, 0, -1]], translation=[0.35, -0.05, 0.2]
    )
    wp3_r = RigidTransform(
        rotation=[
            [-0.7071068, 0.7071068, 0.0000000],
            [0.5000000, 0.5000000, 0.7071068],
            [0.5000000, 0.5000000, -0.7071068],
        ],
        translation=[0.45, -0.08, 0.15],
    )
    interface.go_cartesian_waypoints(
        l_targets=[wp1_l, wp2_l, wp3_l], r_targets=[wp1_r, wp2_r, wp3_r]
    )
    interface.go_linear_single(l_target=wp1_l, r_target=wp1_r)
    interface.go_linear_single(l_target=wp2_l, r_target=wp2_r)
    interface.go_linear_single(l_target=wp3_l, r_target=wp3_r)
    interface.home()

    interface.go_delta([0, 0, -0.1])
    interface.go_delta([0, -0.1, 0], [0, 0.1, 0])
    interface.go_delta([0.1, -0.1, 0], [0.1, 0.1, 0])
    interface.home()


if __name__ == "__main__":
    run()
