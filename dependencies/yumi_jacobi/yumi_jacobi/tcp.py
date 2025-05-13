from autolab_core import RigidTransform

#to use these you have to give it the right frames: like BLACK_GRIPPER.as_frames("l_tcp","gripper_l_base")

#gripper (fat, black fingertips)
BLACK_GRIPPER = RigidTransform(translation=[0,0,.175])
METAL_GRIPPER = RigidTransform(translation=[0,0,.165])
#suction
SUCTION = RigidTransform(translation=[0,0,.156])

TWEEZERS = RigidTransform(translation=[0,-.01,.26])

ABB_WHITE = RigidTransform(translation=[0,0,.1325])