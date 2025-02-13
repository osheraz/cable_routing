import numpy as np
import rospy
import tf
from yumi_jacobi.interface import Interface
from autolab_core import RigidTransform


class YuMiTFPublisher:
    def __init__(self):
        rospy.init_node("yumi_tf_publisher", anonymous=True)
        self.rate = rospy.Rate(100)
        self.br = tf.TransformBroadcaster()
        self.interface = Interface()

    def matrix_to_quaternion(self, rotation_matrix):
        """Converts a 3x3 rotation matrix to a quaternion."""
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        return tf.transformations.quaternion_from_matrix(transformation_matrix)

    def publish_transforms(self):
        while not rospy.is_shutdown():
            # Get the end-effector poses
            left_ee_pose, right_ee_pose = self.interface.get_FK(
                "left"
            ), self.interface.get_FK("right")
            base_link_pose = RigidTransform(rotation=np.eye(3), translation=[0, 0, 0])

            self.br.sendTransform(
                base_link_pose.translation,
                self.matrix_to_quaternion(base_link_pose.rotation),
                rospy.Time.now(),
                "yumi_base_link",
                "world",
            )

            self.br.sendTransform(
                left_ee_pose.translation,
                self.matrix_to_quaternion(left_ee_pose.rotation),
                rospy.Time.now(),
                "left_ee",
                "yumi_base_link",
            )

            # Publish right end-effector transformation
            self.br.sendTransform(
                right_ee_pose.translation,
                self.matrix_to_quaternion(right_ee_pose.rotation),
                rospy.Time.now(),
                "right_ee",
                "yumi_base_link",
            )

            self.rate.sleep()


if __name__ == "__main__":
    try:
        yumi_tf_publisher = YuMiTFPublisher()
        yumi_tf_publisher.publish_transforms()
    except rospy.ROSInterruptException:
        pass
