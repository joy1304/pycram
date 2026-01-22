"""
Complete Grasp Recovery with Error Logging
"""

import time
import yaml
import os
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
from pycram.robot_plans.actions.core import PickUpAction, PlaceAction
from pycram.recovery.error_recovery import with_error_handling
from pycram.failures import PlanFailure, LowLevelFailure

# Comprehensive grasp data with multiple approaches
'''mock_grasp_data = {
    'grasps': [
        {'id': 'front', 'position': {'x': 0, 'y': -0.05, 'z': 0},
         'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1}},
        {'id': 'top', 'position': {'x': 0, 'y': 0, 'z': 0.05},
         'orientation': {'x': 0, 'y': 0.707, 'z': 0, 'w': 0.707}},
'       {'id': 'left', 'position': {'x': -0.05, 'y': 0, 'z': 0},
         'orientation': {'x': 0, 'y': 0, 'z': -0.707, 'w': 0.707}},
        {'id': 'right', 'position': {'x': 0.05, 'y': 0, 'z': 0},
         'orientation': {'x': 0, 'y': 0, 'z': 0.707, 'w': 0.707}},
    ]
}'''

def load_grasps_from_file(filename: str):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist!")

    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
        loginfo(f"Successfully loaded {len(data['grasps'])} grasps from {filename}")
        return data


@with_error_handling
def complete_recovery_demo(grasp_data=None, grasp_classifier=None, guardian=None, **_):
    """
    Complete demonstration of PlanGuardian's capabilities:
    - Attempts to pick up multiple objects
    - Shows detailed error messages when grasps fail
    - Places successfully grasped objects at new locations
    """
    loginfo("=" * 70)
    loginfo("RECOVERY DEMONSTRATION")
    loginfo("=" * 70)

    # Objects to manipulate
    object_configs = [
        {"name": "red_box", "color": "Red", "mesh": "block_red.urdf"},
        {"name": "blue_box", "color": "Blue", "mesh": "block_blue.urdf"},
        {"name": "green_box", "color": "Green", "mesh": "block_green.urdf"},
    ]

    successful_operations = 0
    total_operations = len(object_configs)

    for i, obj_config in enumerate(object_configs, 1):
        obj_name = obj_config["name"]
        obj_color = obj_config["color"]

        loginfo(f"{'=' * 70}")
        loginfo(f" OBJECT {i}/{total_operations}: {obj_color} Box ({obj_name})")
        loginfo(f"{'=' * 70}")

        # Get object designator
        obj_desig = BelieveObject(names=[obj_name])
        obj = obj_desig.resolve()

        loginfo(f"Target object: {obj_name}")
        loginfo(f"Position: x={obj.pose.position.x:.3f}, y={obj.pose.position.y:.3f}, z={obj.pose.position.z:.3f}")

        # Create pickup action with initial grasp
        initial_grasp = 'top'
        loginfo(f"[ATTEMPT] Trying initial grasp strategy: '{initial_grasp}'")

        pickup_action = PickUpAction(
            object_designator=obj,
            arm=Arms.RIGHT,
            grasp_description=initial_grasp
        )

        context = {
            'object_designator': obj,
            'arm': Arms.RIGHT,
            'object_name': obj_name,
            'object_number': i,
        }

        try:
            # Attempt pickup
            pickup_action.perform()
            loginfo(f"[SUCCESS] Initial grasp '{initial_grasp}' succeeded for {obj_name}!")
            successful_operations += 1

            # Place object at new location
            place_x = 0.0
            place_y = 1.0 + (i * 0.2)
            place_z = 0.8
            place_pose = PoseStamped.from_list([place_x, place_y, place_z])

            loginfo(f"[ACTION] Placing {obj_name} at new location...")
            loginfo(f"         Target: x={place_x:.3f}, y={place_y:.3f}, z={place_z:.3f}")

            place_action = PlaceAction(obj, arm=Arms.RIGHT, target_location=place_pose)
            place_action.perform()

            loginfo(f"[SUCCESS] {obj_name} placed successfully!")

        except LowLevelFailure as e:
            # Detailed error information
            error_type = type(e).__name__
            error_msg = str(e)

            logwarn(f"[FAILURE] Initial grasp '{initial_grasp}' failed for {obj_name}")
            logwarn(f"          Error Type: {error_type}")

            # Extract key information from error message
            if "not reachable" in error_msg.lower():
                logwarn(f"          Reason: Pose not reachable with current configuration")
            else:
                logwarn(f"          Reason: {error_msg[:100]}...")

            loginfo(f"[RECOVERY] Activating PlanGuardian for {obj_name}...")
            loginfo(f"           Searching for alternative grasp strategies...")

            try:
                # Let guardian handle the error and try alternatives
                guardian.handle_error(e, pickup_action, context)

                loginfo(f"[SUCCESS] PlanGuardian found working grasp for {obj_name}!")
                loginfo(f"          Recovery strategy successful!")
                successful_operations += 1

                # Place the object after successful recovery
                place_x = 0.0
                place_y = 1.0 + (i * 0.2)
                place_z = 0.8
                place_pose = PoseStamped.from_list([place_x, place_y, place_z])

                loginfo(f"[ACTION] Placing {obj_name} at new location...")
                place_action = PlaceAction(obj, arm=Arms.RIGHT, target_location=place_pose)
                place_action.perform()
                loginfo(f"[SUCCESS] {obj_name} placed successfully after recovery!")

            except PlanFailure as pf:
                logwarn(f"[FAILURE] PlanGuardian could not recover for {obj_name}")
                logwarn(f"          All grasp strategies exhausted")
                logwarn(f"          Final error: {str(pf)[:100]}")

    # Summary Section
    loginfo(f"{'=' * 70}")
    loginfo(" DEMONSTRATION SUMMARY")
    loginfo(f"{'=' * 70}")
    loginfo(f"Total Objects Processed: {total_operations}")
    loginfo(f"Successful Operations: {successful_operations}")
    loginfo(f"Success Rate: {(successful_operations / total_operations) * 100:.1f}%")

    # Error Log Analysis
    loginfo(f"{'=' * 70}")
    loginfo(" ERROR LOG ANALYSIS")
    loginfo(f"{'=' * 70}")

    if not guardian.error_log:
        loginfo("No errors were logged - all operations succeeded on first attempt!")
    else:
        loginfo(f"Total Errors Handled: {len(guardian.error_log)}")

        # Error type statistics
        error_types = {}
        for entry in guardian.error_log:
            error_type = entry['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1

        loginfo("Error Type Breakdown:")
        for error_type, count in error_types.items():
            loginfo(f"  • {error_type}: {count} occurrence(s)")

        # Detailed error log
        loginfo(f"{'=' * 70}")
        loginfo(" DETAILED ERROR LOG")
        loginfo(f"{'=' * 70}")

        for idx, log_entry in enumerate(guardian.error_log, 1):
            loginfo(f"  Error #{idx}:")
            loginfo(f"  Timestamp: {log_entry['timestamp']:.3f}s")
            loginfo(f"  Error Type: {log_entry['error_type']}")
            loginfo(f"  Failed Action: {log_entry['failed_action']}")

            # Show context information
            context_info = log_entry.get('context', {})
            if 'object_name' in context_info:
                loginfo(f"  Object: {context_info['object_name']}")
            if 'arm' in context_info:
                loginfo(f"  Arm: {context_info['arm']}")

    loginfo(f"{'=' * 70}")
    loginfo(" DEMONSTRATION COMPLETED")
    loginfo(f"{'=' * 70}")


if __name__ == '__main__':
    # Initialize simulation world
    world = BulletWorld(WorldMode.GUI)
    world.set_gravity([0, 0, -9.81])

    loginfo("Initializing simulation environment...")

    # Spawn Tracy robot at optimal position
    robot = Object("tracy", Robot, "tracy.urdf",
                   pose=PoseStamped.from_list([-0.5, 1.5, 0.7], [0, 0, 0, 1]))
    loginfo("Tracy robot spawned")

    # Configure grasp approaches
    robot_description = RobotDescription.current_robot_description
    right_gripper = robot_description.get_arm_chain(Arms.RIGHT).end_effector
    left_gripper = robot_description.get_arm_chain(Arms.LEFT).end_effector

    '''for grasp in mock_grasp_data['grasps']:
        orientation_dict = grasp['orientation']
        orientation_list = [orientation_dict['x'], orientation_dict['y'],
                            orientation_dict['z'], orientation_dict['w']]
        right_gripper.grasps[grasp['id']] = orientation_list

    loginfo(f"Configured {len(mock_grasp_data['grasps'])} grasp strategies")'''

    try:
        # We assume the script is run from the project root, so the path is 'config/...'
        GRASP_FILE_PATH = "Cube_Pad_grasps.yaml"
        grasp_data = load_grasps_from_file(GRASP_FILE_PATH)
    except Exception as e:
        logwarn(f"Failed to load YAML file from {GRASP_FILE_PATH}: {e}")
        raise
        #logwarn("Falling back to mock data...")
        # Note: You would need to define 'mock_grasp_data' here if the file is missing
        #grasp_data = mock_grasp_data

        # Load the grasps into the robot's grippers
    for grasp in grasp_data['grasps']:
        orientation_dict = grasp['orientation']
        orientation_list = [
            orientation_dict['x'],
            orientation_dict['y'],
            orientation_dict['z'],
            orientation_dict['w']
        ]

        # The IDs are strings in the YAML, ensure they are compatible keys
        grasp_id = str(grasp['id'])
        right_gripper.grasps[grasp_id] = orientation_list
        left_gripper.grasps[grasp_id] = orientation_list

    if '3' in right_gripper.grasps:
        right_gripper.grasps['top'] = right_gripper.grasps['3']  # <- FIX 1
        left_gripper.grasps['top'] = left_gripper.grasps['3']  # <- FIX 1
        loginfo("Aliased numerical grasp ID '0' to named grasp 'top' for demo compatibility.")

    #loginfo(f"Configured {len(mock_grasp_data['grasps'])} grasp strategies")

    else:
        logwarn("Grasp ID '0' not found; continuing with numerical IDs.")

    loginfo("Robot configured with grasps from YAML file")



    # Spawn objects at strategic positions for varied reachability
    # Position objects to create different challenge levels
    object_positions = [
        ([0.12, 1.38, 0.75], (1, 0, 0, 1), "block_red.urdf", "red_box"),
        ([0.08, 1.35, 0.75], (0, 0, 1, 1), "block_blue.urdf", "blue_box"),
        ([0.50, 1.38, 0.75], (0, 1, 0, 1), "block_green.urdf", "green_box"),
    ]

    for pos, color, mesh, name in object_positions:
        obj = Object(name, PhysicalObject, mesh, color=Color(*color))
        obj.set_pose(PoseStamped.from_list(pos))
        loginfo(f"Spawned {name} at position {pos}")

    world.step()
    loginfo("Simulation setup complete. Starting demonstration...")
    time.sleep(1)  # Brief pause before starting

    # Run the demonstration
    with simulated_robot:
        try:
            complete_recovery_demo(grasp_data)
            loginfo("[FINAL STATUS] DEMONSTRATION SUCCESSFUL")
        except PlanFailure as e:
            logwarn(f"[FINAL STATUS] DEMONSTRATION FAILED")
            logwarn(f"Critical Error: {e}")
        except Exception as e:
            logwarn(f"[FINAL STATUS] UNEXPECTED ERROR")
            logwarn(f"Exception: {e}")

    loginfo("Simulation will close in 5 seconds...")
    time.sleep(5)
    world.exit()