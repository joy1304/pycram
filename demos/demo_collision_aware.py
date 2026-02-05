"""
CollisionAwareTransportAction Demo
"""

from pycram.datastructures.dataclasses import Color
from pycram.process_module import simulated_robot
from pycram.robot_plans.actions.composite.collision_aware_transport import CollisionAwareTransportActionDescription
from pycram.worlds.bullet_world import BulletWorld
from pycram.robot_plans.actions import *
from pycram.designators.object_designator import *
from pycram.datastructures.enums import ObjectType, Arms, Grasp, WorldMode
from pycram.language import SequentialPlan
import time

from pycrap.ontologies import Robot

# Object starting positions
box1Start = PoseStamped.from_list([0.25, 1.25, 0.95])
box2Start = PoseStamped.from_list([0.25, 1.5, 0.95])
box3Start = PoseStamped.from_list([0.25, 1.75, 0.95])

# All boxes go to the SAME target position
# CollisionAwareTransport will automatically offset when occupied
sharedTarget = PoseStamped.from_list([0.26, 1.3, 1.00])


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
        # CollisionAwareTransport automatically adjusts position if occupied!
        CollisionAwareTransportActionDescription(box1, sharedTarget),
        CollisionAwareTransportActionDescription(box2, sharedTarget),
        CollisionAwareTransportActionDescription(box3, sharedTarget),
    )
    sp.perform()


time.sleep(10)
world.exit()
