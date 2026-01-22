"""
A test script to validate the functionality of the intelligent PlanGuardian
error handler, specifically for PR2 (a fixed-base manipulator).
"""

import time

from pycram.designator import ObjectDesignatorDescription
# --- Core PyCRAM Imports ---
from pycram.process_module import simulated_robot
from pycram.designators.object_designator import BelieveObject
from pycram.datastructures.pose import PoseStamped
from pycram.datastructures.enums import Arms, WorldMode, Enum
from pycram.worlds.bullet_world import BulletWorld
from pycram.world_concepts.world_object import Object
from pycram.datastructures.dataclasses import Color
from pycrap.ontologies import Robot, PhysicalObject
from pycram.ros import loginfo, logwarn
from pycram.robot_description import RobotDescription

# --- Actions and Recovery ---
from pycram.robot_plans.actions.core import PickUpAction
from pycram.recovery.error_recovery import with_error_handling, GraspUnfeasibleFailure
from pycram.failures import PlanFailure, LowLevelFailure, GripperGoalNotReached


class Grasp(Enum):
    TOP = "top"
    SIDE = "side"
    FRONT = "front"

# --- Mock Grasp Data ---
# FIX: Changed 'orientation' from a list to a dictionary to match GraspClassifier's expectation.
mock_grasp_data = {
    'grasps': [
        {'id': 'front', 'position': {'x': 0, 'y': -0.05, 'z': 0}, 'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1}},
        {'id': 'top', 'position': {'x': 0, 'y': 0, 'z': 0.05}, 'orientation': {'x': 0, 'y': 0.707, 'z': 0, 'w': 0.707}},
        {'id': 'left', 'position': {'x': -0.05, 'y': 0, 'z': 0}, 'orientation': {'x': 0, 'y': 0, 'z': -0.707, 'w': 0.707}},
        {'id': 'right', 'position': {'x': 0.05, 'y': 0, 'z': 0}, 'orientation': {'x': 0, 'y': 0, 'z': 0.707, 'w': 0.707}},
    ]
}

# --- Main Plan ---
@with_error_handling
def guarded_grasp_plan(grasp_data=None, grasp_classifier=None, guardian=None, **_):
    """
    The plan creates a grasp action that is likely to fail,
    then lets PlanGuardian attempt recovery using alternative grasp approaches.
    """
    # Designators for objects
    reachable_box_desig = BelieveObject(names=["reachable_box"])
    unreachable_box_desig = BelieveObject(names=["unreachable_box"])
    resolved_reachable_box = reachable_box_desig.resolve()

    # --- SCENARIO 1: Force a likely failure, let guardian recover ---
    loginfo("--- SCENARIO 1: Testing recovery from a failed grasp ---")
    failed_grasp_action = PickUpAction(
        object_designator=resolved_reachable_box,
        arm=Arms.RIGHT,
        grasp_description=Grasp.TOP.value
    )
    context = {
        'object_designator': resolved_reachable_box,
        'arm': Arms.RIGHT,
    }

    try:
        failed_grasp_action.perform()
        loginfo("--- SCENARIO 1: Initial grasp unexpectedly succeeded ---")
    except (GripperGoalNotReached, LowLevelFailure) as e:
        loginfo(f"Initial grasp failed as expected. Handing over to PlanGuardian.")
        guardian.handle_error(e, failed_grasp_action, context)
        loginfo("--- SCENARIO 1: Recovery successful ---")

    # --- SCENARIO 2: Demonstrate unreachable object handling ---
    loginfo("\n--- SCENARIO 2: Testing handling of an unreachable object ---")
    if grasp_classifier is None:
        raise GraspUnfeasibleFailure("No GraspClassifier available.")

    reachable_grasps = grasp_classifier.get_n_best_reachable_grasps(
        n=1, arm=Arms.RIGHT, target_object=unreachable_box_desig.resolve()
    )
    if not reachable_grasps:
        try:
            raise GraspUnfeasibleFailure("No reachable grasps for the target object.")
        except GraspUnfeasibleFailure as e:
             guardian.handle_error(e, None, {'object_designator': unreachable_box_desig.resolve()})
    else:
        loginfo("--- SCENARIO 2: Unexpectedly found a reachable grasp for the unreachable object ---")


# --- Execution ---
if __name__ == '__main__':
    world = BulletWorld(WorldMode.GUI)
    world.set_gravity([0, 0, -9.81])

    robot_object = Object("tracy", Robot, "tracy.urdf", pose=PoseStamped.from_list([-0.5, 1.5, 0.7], [0, 0, 0, 1]))


    robot_description = RobotDescription.current_robot_description
    
    right_gripper = robot_description.get_arm_chain(Arms.RIGHT).end_effector
    # Fallback to directly modifying the dictionary
    for grasp in mock_grasp_data['grasps']:
        # This part requires orientation to be a list for the KeyError fix.
        # Let's adapt it to handle the new dictionary format.
        orientation_dict = grasp['orientation']
        orientation_list = [orientation_dict['x'], orientation_dict['y'], orientation_dict['z'], orientation_dict['w']]
        right_gripper.grasps[grasp['id']] = orientation_list

    loginfo("Dynamically added grasps to the robot's right gripper.")

    # Spawn objects
    reachable_box = Object("reachable_box", PhysicalObject, "block_blue.urdf", color=Color(0, 1, 0, 1))
    reachable_box.set_pose(PoseStamped.from_list([0.25,1.25,0.7]))
    unreachable_box = Object("unreachable_box", PhysicalObject, "block_red.urdf", color=Color(1, 0, 0, 1))
    unreachable_box.set_pose(PoseStamped.from_list([0.25,1.5,0.7]))

    world.step()

    with simulated_robot:
        try:
            guarded_grasp_plan(grasp_data=mock_grasp_data)
        except PlanFailure as e:
            logwarn(f"\n--- TEST SCENARIO COMPLETED ---")
            logwarn(f"Plan failed as expected. Final error: {e}")

    loginfo("\nSimulation finished. Closing in 5 seconds.")
    time.sleep(5)
    world.exit()
