"""
It shows how PlanGuardian recovers from ManipulationGoalNotReached by parking arms
"""

import time
from pycram.process_module import simulated_robot
from pycram.designators.object_designator import BelieveObject
from pycram.datastructures.pose import PoseStamped
from pycram.datastructures.enums import Arms, WorldMode
from pycram.worlds.bullet_world import BulletWorld
from pycram.world_concepts.world_object import Object
from pycram.datastructures.dataclasses import Color
from pycrap.ontologies import Robot, PhysicalObject
from pycram.ros import loginfo, logwarn
from pycram.robot_description import RobotDescription
from pycram.recovery.error_recovery import with_error_handling
from pycram.failures import PlanFailure, ManipulationGoalNotReached, LowLevelFailure

mock_grasp_data = {
    'grasps': [
        {'id': 'top', 'position': {'x': 0, 'y': 0, 'z': 0.05},
         'orientation': {'x': 0, 'y': 0.707, 'z': 0, 'w': 0.707}},
    ]
}

@with_error_handling
def manipulation_recovery_demo(grasp_data=None, grasp_classifier=None, guardian=None, **_):
    loginfo("Scenario: Robot configuration unreachable, guardian resets and retries")

    # with a robot-specific action that might fail.
    logwarn("Simulating a 'ManipulationGoalNotReached' failure...")

    simulated_arms = [Arms.RIGHT] #dummy value
    simulated_body = []          #dummy value
    raise ManipulationGoalNotReached(
        "This is a simulated failure to test the recovery handler.",
        arms=simulated_arms,
        body=simulated_body
    )

if __name__ == '__main__':
    world = BulletWorld(WorldMode.GUI)

    # Spawn Tracy robot
    robot = Object("tracy", Robot, "tracy.urdf",
                   pose=PoseStamped.from_list([-0.5, 1.5, 0.7], [0, 0, 0, 1]))

    # Configure grasps
    robot_description = RobotDescription.current_robot_description
    right_gripper = robot_description.get_arm_chain(Arms.RIGHT).end_effector

    for grasp in mock_grasp_data['grasps']:
        orientation_dict = grasp['orientation']
        orientation_list = [orientation_dict['x'], orientation_dict['y'],
                          orientation_dict['z'], orientation_dict['w']]
        right_gripper.grasps[grasp['id']] = orientation_list

    loginfo("Robot configured")

    # Add some objects to make scene more realistic
    box = Object("box", PhysicalObject, "block_blue.urdf",
                   color=Color(0.6, 0.4, 0.2, 1))
    box.set_pose(PoseStamped.from_list([0.5, 0.5, 0.5]))

    world.step()

    with simulated_robot:
        try:
            manipulation_recovery_demo(grasp_data=mock_grasp_data)
        except PlanFailure as e:
            logwarn(f"Plan failed after recovery: {e}")
        except LowLevelFailure as e:
            logwarn(f"Demo failed with an unhandled error: {e}")

    loginfo("Demo finished. Closing world.")
    time.sleep(7)
    world.exit()