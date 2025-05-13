from jacobi.drivers import ABBDriver
from autolab_core import RigidTransform # To support later
import time
from enum import Enum
import sys
from io import StringIO

class GripperState(Enum):
            NONE = 0
            INITIALIZE = 1
            CALIBRATE = 2
            MOVE_TO = 3
            GRIP_IN = 4
            GRIP_OUT = 5
            BLOW_ON_1 = 6
            BLOW_ON_2 = 7
            BLOW_OFF_1 = 8
            BLOW_OFF_2 = 9
            VACUUM_ON_1 = 10
            VACUUM_ON_2 = 11
            VACUUM_OFF_1 = 12
            VACUUM_OFF_2 = 13
            
class YuMiArm(ABBDriver):
    """
    Jacobi arm driver base class ABBDriver + SmartGripper interface for YuMi robot. 
    SmartGripper controlled by I/O signals and interrupts that were manually exposed in RAPID modules.
    """

    def __init__(self, planner, robot, host, port, module, version, on_fail=None):
        super().__init__(planner, robot, host, port, module, version)
        self._on_fail = on_fail 
        
        if self.port == 6511:
            self._side = 'left'
            self._gripper_comm = ['192.168.125.40', 'Hand_L']
            self._gripper_state_IO = 'cmd_GripperState_L'
            self._gripper_pos_IO = 'cmd_GripperPos_L'
            self._gripper_pos_read_IO = 'hand_ActualPosition_L'
        elif self.port == 6512:
            self._side = 'right'
            self._gripper_comm = ['192.168.125.30', 'Hand_R']
            self._gripper_state_IO = 'cmd_GripperState_R'
            self._gripper_pos_IO = 'cmd_GripperPos_R'
            self._gripper_pos_read_IO = 'hand_ActualPosition_R'
        self._set_signal(self._gripper_state_IO, 0)
        self._set_signal(self._gripper_pos_IO, 0)

    
    async def run_async(self, trajectory, *args, **kwargs):
        result = await super().run_async(trajectory, *args, **kwargs)
        print("Result repr:", repr(result))

        if "[-" in repr(result):
            self.last_error = result
            if self._on_fail:
                # print(f"[{self._side}] failed motion triggered")
                self._on_fail()
        else:
            self.last_error = None  # Clear it on success?
            
        return result


    def _get_signal(self, signal):
        return self.rws.get_signal(signal = signal, network = self._gripper_comm[0], device = self._gripper_comm[1])

    def _set_signal(self, signal, value):
        self.rws.set_signal(signal = signal, value = str(value), network = self._gripper_comm[0], device = self._gripper_comm[1])
        return int(self._get_signal(signal).lvalue) == value
    
    def get_gripper_state(self):
        return GripperState(int(self._get_signal(self._gripper_state_IO).lvalue))
    
    def get_gripper_pos(self):
        """Returns the current position of the gripper in meters"""
        return int(self._get_signal(self._gripper_pos_read_IO).lvalue)/10000

    async def initialize_gripper(self, sync=False):
        # Unecessary?
        self._set_signal(self._gripper_state_IO, 1)
        if not sync:
            self._set_signal('RUN_SG_ROUTINE', 1)
            time.sleep(0.05)
            self._set_signal('RUN_SG_ROUTINE', 0)
            self._set_signal(self._gripper_state_IO, 0)
        return True
        
    async def calibrate_gripper(self, sync=False):
        self._set_signal(self._gripper_state_IO, 2)
        if not sync:
            self._set_signal('RUN_SG_ROUTINE', 1)
            time.sleep(0.05)
            self._set_signal('RUN_SG_ROUTINE', 0)
            self._set_signal(self._gripper_state_IO, 0)
        print(f"Calibrated {self._side} gripper")
        return True
            
    async def open_gripper(self, sync=False):
        self._set_signal(self._gripper_state_IO, 5)
        if not sync:
            self._set_signal('RUN_SG_ROUTINE', 1)
            time.sleep(0.05)
            self._set_signal('RUN_SG_ROUTINE', 0)
            self._set_signal(self._gripper_state_IO, 0)
        return True
        
    async def close_gripper(self, sync=False):
        self._set_signal(self._gripper_state_IO, 4)
        if not sync:
            self._set_signal('RUN_SG_ROUTINE', 1)
            time.sleep(0.05)
            self._set_signal('RUN_SG_ROUTINE', 0)
            self._set_signal(self._gripper_state_IO, 0)
        return True
          
    async def move_gripper(self, value, sync=False):
        "Value is valid between 0 to 25 (Physical gripper travel range in [mm]), though RAPID modules have extra check and will saturate values"
        self._set_signal(self._gripper_pos_IO, value)
        self._set_signal(self._gripper_state_IO, 3)
        if not sync:
            self._set_signal('RUN_SG_ROUTINE', 1)
            time.sleep(0.05)
            self._set_signal('RUN_SG_ROUTINE', 0)
            self._set_signal(self._gripper_state_IO, 0)
        return True
