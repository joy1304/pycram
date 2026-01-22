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

box1Start= PoseStamped.from_list([0.25,1.25,0.7])
box2Start = PoseStamped.from_list([1.5,1.5,0.7])
box3Start = PoseStamped.from_list([0.25,1.75,0.7])

box1Target = PoseStamped.from_list([0.25, 1.5, 0.75])
box2Target = PoseStamped.from_list([2.25, 1.5, 0.7])
box3Target = PoseStamped.from_list([0.25, 1.5, 0.8])

world = BulletWorld(WorldMode.GUI)

#extension = ObjectDescription.get_file_extension()
#robot = Object("pr2", Robot, f"pr2{extension}", pose=PoseStamped.from_list([-0.5, 1.5, 0], [0,0,0,1]))

tracy = Object("tracy", Robot, "tracy.urdf", pose=PoseStamped.from_list([-0.5, 1.5, 0.7], [0, 0, 0, 1]), ignore_cached_files=True)
tracy_description = ObjectDesignatorDescription(names=["tracy"]).resolve()

#table = Object("table", PhysicalObject, "table.urdf", pose=PoseStamped.from_list([4,1,0]))
box1 = Object("box1", PhysicalObject, "horizontal_stabilizer.urdf", pose=box1Start, color=Color(1, 0, 0, 1))
box2 = Object("box2", PhysicalObject, "horizontal_stabilizer.urdf", pose=box2Start, color=Color(0, 1, 0, 1))
box3 = Object("box3", PhysicalObject, "horizontal_stabilizer.urdf", pose=box3Start, color=Color(0, 0, 1, 1))

grasp_description = GraspDescription(ApproachDirection.FRONT, VerticalAlignment.TOP)
with simulated_robot:
    sp = SequentialPlan(
        PickAndPlaceActionDescription(box1, box1Target, Arms.RIGHT, grasp_description),
        PickAndPlaceActionDescription(box2, box2Target, Arms.LEFT, grasp_description),
        #PickAndPlaceActionDescription(box3, box1Start, Arms.RIGHT, grasp_description),
        #PickAndPlaceActionDescription(box1, PoseStamped.from_list([0.25,1.25,0.75]), Arms.RIGHT, grasp_description),
        #PickAndPlaceActionDescription(box2, PoseStamped.from_list([0.25,1.25,0.8]), Arms.RIGHT, grasp_description),
        #PickAndPlaceActionDescription(box2, box3Start, Arms.LEFT, grasp_description),
        #PickAndPlaceActionDescription(box1, PoseStamped.from_list([0.25, 1.75, 0.75]), Arms.LEFT, grasp_description),
        #PickAndPlaceActionDescription(box3, PoseStamped.from_list([0.25, 1.75, 0.8]), Arms.LEFT, grasp_description),
        #PickAndPlaceActionDescription(box3, box2Start, Arms.LEFT, grasp_description),
        #PickAndPlaceActionDescription(box1, box1Start, Arms.LEFT, grasp_description)
    )
    sp.perform()

time.sleep(10)
world.exit()