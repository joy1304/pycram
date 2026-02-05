"""
CollisionAwareTransportAction with GraspClassifier Demo
"""

from pycram.datastructures.dataclasses import Color
from pycram.process_module import simulated_robot
from pycram.worlds.bullet_world import BulletWorld
from pycram.robot_plans.actions import *
from pycram.designators.object_designator import *
from pycram.datastructures.enums import ObjectType, Arms, Grasp, WorldMode
from pycram.language import SequentialPlan
import yaml
import time

from pycrap.ontologies import Robot

# Import collision-aware transport with grasp classifier
from pycram.robot_plans.actions.composite.collision_aware_transport_graspcls import (
    CollisionAwareTransportActionDescription,
    GraspClassifier,
    load_grasp_data
)

# Object starting positions
box1Start = PoseStamped.from_list([0.25, 1.25, 0.95])
box2Start = PoseStamped.from_list([0.25, 1.5, 0.95])
box3Start = PoseStamped.from_list([0.25, 1.75, 0.95])
box4Start = PoseStamped.from_list([0.25, 2.25, 0.95])
box5Start = PoseStamped.from_list([0.25, 2.5, 0.95])
box6Start = PoseStamped.from_list([0.25, 2.75, 0.95])
box7Start = PoseStamped.from_list([0.25, 0.25, 0.95])
box8Start = PoseStamped.from_list([0.25, 0.5, 0.95])
box9Start = PoseStamped.from_list([0.25, 0.75, 0.95])

# All boxes go to the SAME target position
# CollisionAwareTransport will automatically offset when occupied
sharedTarget = PoseStamped.from_list([0.26, 1.3, 1.00])

# Load grasp data from YAML
grasp_data = load_grasp_data("Cube_Pad_grasps.yaml")
grasp_classifier = GraspClassifier(grasp_data)

# Print grasp statistics
print("Grasp Statistics:")
print(grasp_classifier.get_grasp_count_by_direction())


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
box4 = Object("box4", PhysicalObject, "block_blue.urdf",
              pose=box4Start, color=Color(1, 0, 0, 1))
box5 = Object("box5", PhysicalObject, "block_green.urdf",
              pose=box5Start, color=Color(0, 1, 0, 1))
box6 = Object("box6", PhysicalObject, "block_red.urdf",
              pose=box6Start, color=Color(0, 0, 1, 1))
box7 = Object("box7", PhysicalObject, "block_blue.urdf",
              pose=box7Start, color=Color(1, 0, 0, 1))
box8 = Object("box8", PhysicalObject, "block_green.urdf",
              pose=box8Start, color=Color(0, 1, 0, 1))
box9 = Object("box9", PhysicalObject, "block_red.urdf",
              pose=box9Start, color=Color(0, 0, 1, 1))


with simulated_robot:
    sp = SequentialPlan(
        # CollisionAwareTransport with GraspClassifier!
        CollisionAwareTransportActionDescription(box1, sharedTarget, grasp_classifier=grasp_classifier, distance_weight=0.6, orientation_weight=0.4),
        CollisionAwareTransportActionDescription(box2, sharedTarget, grasp_classifier=grasp_classifier, distance_weight=0.6, orientation_weight=0.4),
        CollisionAwareTransportActionDescription(box3, sharedTarget, grasp_classifier=grasp_classifier, distance_weight=0.6, orientation_weight=0.4),
        CollisionAwareTransportActionDescription(box4, sharedTarget, grasp_classifier=grasp_classifier, distance_weight=0.6, orientation_weight=0.4),
        CollisionAwareTransportActionDescription(box5, sharedTarget, grasp_classifier=grasp_classifier, distance_weight=0.6, orientation_weight=0.4),
        CollisionAwareTransportActionDescription(box6, sharedTarget, grasp_classifier=grasp_classifier, distance_weight=0.6, orientation_weight=0.4),
        CollisionAwareTransportActionDescription(box7, sharedTarget, grasp_classifier=grasp_classifier,
                                                 distance_weight=0.6, orientation_weight=0.4),
        CollisionAwareTransportActionDescription(box8, sharedTarget, grasp_classifier=grasp_classifier,
                                                 distance_weight=0.6, orientation_weight=0.4),
        CollisionAwareTransportActionDescription(box9, sharedTarget, grasp_classifier=grasp_classifier,
                                                 distance_weight=0.6, orientation_weight=0.4),
    )
    sp.perform()


time.sleep(10)
world.exit()
