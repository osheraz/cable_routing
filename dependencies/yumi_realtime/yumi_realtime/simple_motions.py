import rospy
import time
import jax.numpy as jnp
import onp
from yumi_realtime.base import YuMiBaseInterface
from loguru import logger
from sensor_msgs.msg import Float64MultiArray

class YuMiMotionTester(YuMiBaseInterface):
    """A test interface to test motion commands."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        rospy.init_node('yumi_motion_tester', anonymous=True)
        self.joint_pub = rospy.Publisher(
            "yumi/egm/joint_group_position_controller/command", 
            Float64MultiArray, 
            queue_size=10
        )
        
        logger.info("YuMi Motion Tester initialized.")
    
    def test_motion(self, l_xyz, l_wxyz, l_gripper_cmd, r_xyz, r_wxyz, r_gripper_cmd, rate_hz=150):
        """Test motion by sending direct pose updates."""
        rate = rospy.Rate(rate_hz)
        
        super().update_target_pose(
            side='left',
            position=l_xyz,
            wxyz=l_wxyz,
            gripper_state=bool(l_gripper_cmd > self.gripper_thres),
            enable=True
        )
        
        super().update_target_pose(
            side='right',
            position=r_xyz,
            wxyz=r_wxyz,
            gripper_state=bool(r_gripper_cmd > self.gripper_thres),
            enable=False
        )
        
        self.solve_ik()
        self.update_visualization()
        self.publish_joint_commands()
        
        rate.sleep()
    
    def publish_joint_commands(self):
        """Publish joint commands to the robot."""
        joint_desired = onp.array([
            self.joints[7], self.joints[8], self.joints[9],
            self.joints[10], self.joints[11], self.joints[12], self.joints[13],
            self.joints[0], self.joints[1], self.joints[2],
            self.joints[3], self.joints[4], self.joints[5], self.joints[6]
        ], dtype=onp.float32)
        msg = Float64MultiArray(data=joint_desired)
        self.joint_pub.publish(msg)
        
    def move_to_home(self):
        """Move the robot to its home position."""
        self.joints = self.rest_pose.copy()
        self.solve_ik()
        self.update_visualization()
        self.publish_joint_commands()
        logger.info("Moved to home position.")
        time.sleep(2)
        
if __name__ == "__main__":
    tester = YuMiMotionTester()
    tester.move_to_home()
    
    # Example motion test
    tester.test_motion(
        l_xyz=jnp.array([0.2, 0.3, 0.4]),
        l_wxyz=jnp.array([1, 0, 0, 0]),
        l_gripper_cmd=1,
        r_xyz=jnp.array([0.3, -0.2, 0.5]),
        r_wxyz=jnp.array([1, 0, 0, 0]),
        r_gripper_cmd=0
    )