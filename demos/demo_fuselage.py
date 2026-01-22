import os
import time
import pybullet as p

# Core CRAM imports
from pycram.worlds.bullet_world import BulletWorld
from pycram.process_module import simulated_robot
from pycram.language import SequentialPlan, ParallelPlan  # Import ParallelPlan
from pycram.designators.object_designator import *
from pycram.robot_plans.actions import *
from pycram.robot_plans.motions import MoveJointsMotion
from pycram.datastructures.enums import Arms, WorldMode
from pycram.datastructures.pose import PoseStamped
from pycram.datastructures.dataclasses import Color

# Additional necessary imports
from pycrap.ontologies import Robot, PhysicalObject

# --- WORLD SETUP ---
world = BulletWorld(WorldMode.GUI)

# --- ROBOT AND OBJECT SETUP ---
tracy = Object("tracy", Robot, "tracy.urdf", pose=PoseStamped.from_list([-0.5, 1.5, 0.7], [0, 0, 0, 1]),
               ignore_cached_files=True)
tracy_description = ObjectDesignatorDescription(names=["tracy"]).resolve()

box1Start = PoseStamped.from_list([0.25, 1.25, 0.7])
box2Start = PoseStamped.from_list([0.25, 1.5, 0.7])
box3Start = PoseStamped.from_list([0.25, 1.75, 0.7])

box1Target = PoseStamped.from_list([0.25, 1.5, 0.75])
box2Target = PoseStamped.from_list([0.25, 1.5, 0.7])
box3Target = PoseStamped.from_list([0.25, 1.5, 0.8])

box1 = Object("box1", PhysicalObject, "fuselage_basic.urdf", pose=box1Start, color=Color(1, 0, 0, 1))
box2 = Object("box2", PhysicalObject, "fuselage_basic.urdf", pose=box2Start, color=Color(0, 1, 0, 1))
box3 = Object("box3", PhysicalObject, "fuselage_basic.urdf", pose=box3Start, color=Color(0, 0, 1, 1))

grasp_description = GraspDescription(ApproachDirection.FRONT, VerticalAlignment.TOP)

# --- EXECUTION WITH GRACEFUL ERROR HANDLING ---
try:
    with simulated_robot:
        sp = SequentialPlan(
            PickAndPlaceActionDescription(box1, box1Target, Arms.RIGHT, grasp_description),
            PickAndPlaceActionDescription(box3, box3Target, Arms.LEFT, grasp_description),
            PickAndPlaceActionDescription(box3, box1Start, Arms.RIGHT, grasp_description),
            PickAndPlaceActionDescription(box1, PoseStamped.from_list([0.25, 1.25, 0.75]), Arms.RIGHT,
                                          grasp_description),
            PickAndPlaceActionDescription(box2, PoseStamped.from_list([0.25, 1.25, 0.8]), Arms.RIGHT,
                                          grasp_description),
            PickAndPlaceActionDescription(box2, box3Start, Arms.LEFT, grasp_description),
            PickAndPlaceActionDescription(box1, PoseStamped.from_list([0.25, 1.75, 0.75]), Arms.LEFT,
                                          grasp_description),
            PickAndPlaceActionDescription(box3, PoseStamped.from_list([0.25, 1.75, 0.8]), Arms.LEFT, grasp_description),
            PickAndPlaceActionDescription(box3, box2Start, Arms.LEFT, grasp_description),
            PickAndPlaceActionDescription(box1, box1Start, Arms.LEFT, grasp_description)
        )
        print("Executing main plan...")
        sp.perform()
    print("Plan completed successfully!")
    time.sleep(10)

except NotImplementedError:
    print("\n" + "=" * 70)
    print("Caught a fatal NotImplementedError!")
    print("=" * 70)
    print("This happens because high-level actions (like PickAndPlace) are not")
    print("fully implemented for complex, multi-link objects in BulletWorld.")
    print("\nExecuting a recovery plan to reset the robot's arms and exit gracefully.")

    # Define a safe "home" configuration for Tracy's arms
    home_config_right = {'right_shoulder_pan_joint': -1.5, 'right_shoulder_lift_joint': -1.0, 'right_elbow_joint': 1.5,
                         'right_wrist_1_joint': -1.5, 'right_wrist_2_joint': -1.5, 'right_wrist_3_joint': 0.0}
    home_config_left = {'left_shoulder_pan_joint': 1.5, 'left_shoulder_lift_joint': -1.0, 'left_elbow_joint': -1.5,
                        'left_wrist_1_joint': 1.5, 'left_wrist_2_joint': 1.5, 'left_wrist_3_joint': 0.0}

    try:
        with simulated_robot:
            # CORRECTED: MoveJointsMotion requires both 'names' and 'positions' keyword arguments.
            recovery_plan = ParallelPlan(
                MoveJointsMotion(names=list(home_config_right.keys()), positions=list(home_config_right.values())),
                MoveJointsMotion(names=list(home_config_left.keys()), positions=list(home_config_left.values()))
            )
            recovery_plan.perform()
        print("Robot arms have been reset.")
    except Exception as recovery_e:
        print(f"An error occurred during recovery: {recovery_e}")

    time.sleep(5)  # Give user time to read the message

finally:
    # This block will run whether an error occurred or not, ensuring the simulation closes.
    print("Exiting simulation.")
    world.exit()

