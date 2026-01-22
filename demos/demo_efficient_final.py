"""
EfficientTransportAction Demo
"""

from pycram.datastructures.dataclasses import Color
from pycram.process_module import simulated_robot
from pycram.ros_utils.object_state_updater import RobotStateUpdater
from pycram.worlds.bullet_world import BulletWorld
from pycram.robot_plans.actions import *
from pycram.designators.object_designator import *
from pycram.datastructures.enums import ObjectType, Arms, Grasp, WorldMode
from pycram.object_descriptors.urdf import ObjectDescription
from pycram.language import SequentialPlan, ParallelPlan
import time

from pycrap.ontologies import Robot

# Object starting positions
box1Start = PoseStamped.from_list([0.25, 1.25, 0.95])
box2Start = PoseStamped.from_list([0.25, 2.5, 0.95])
box3Start = PoseStamped.from_list([0.25, 1.75, 0.95])

# Target positions
box1Target = PoseStamped.from_list([0.26, 1.3, 1.00])
box2Target = PoseStamped.from_list([0.26, 1.3, 1.15])
box3Target = PoseStamped.from_list([0.26, 1.3, 1.30])


world = BulletWorld(WorldMode.GUI)

tracy = Object("tracy", Robot, "tracy.urdf", 
               pose=PoseStamped.from_list([-0.5, 1.5, 0], [0, 0, 0, 1]),
               ignore_cached_files=True)

tracy_description = ObjectDesignatorDescription(names=["tracy"]).resolve()

# Load objects
box1 = Object("box1", PhysicalObject, "block_blue.urdf", 
              pose=box1Start, color=Color(1, 0, 0, 1))
box2 = Object("box2", PhysicalObject, "block_green.urdf", 
              pose=box2Start, color=Color(0, 1, 0, 1))
box3 = Object("box3", PhysicalObject, "block_red.urdf", 
              pose=box3Start, color=Color(0, 0, 1, 1))


with simulated_robot:
    sp = SequentialPlan(
        # EfficientTransport automatically chooses the best arm!
        EfficientTransportActionDescription(box1, box1Target),
        EfficientTransportActionDescription(box2, box2Target),
        EfficientTransportActionDescription(box3, box3Target),
    )
    sp.perform()


time.sleep(10)
world.exit()
