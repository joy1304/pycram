"""
Complete Grasp Recovery with Error Logging and Comprehensive Testing
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
from pycrap.ontologies import Robot, PhysicalObject
from pycram.ros import loginfo, logwarn
from pycram.robot_description import RobotDescription
from pycram.robot_plans.actions.core import PickUpAction, PlaceAction, ParkArmsAction
#from pycram.recovery.error_recovery import with_error_handling
from pycram.failures import PlanFailure, LowLevelFailure
from pycram.datastructures.dataclasses import Color # Added for object coloring

# Global tracking for statistics
grasp_results = {
    'successful_grasps': {},  # {grasp_id: count, ...}
    'failed_grasps': {},      # {grasp_id: count, ...}
    'total_attempts': 0
}

def load_grasps_from_file(filename: str):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist!")

    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
        # Use loginfo to confirm file loading
        loginfo(f"Successfully loaded {len(data['grasps'])} grasps from {filename}")
        return data


def _log_final_summary(guardian):
    """Logs the final grasp statistics, implementing a successful ranking."""
    global grasp_results

    total_successful = sum(grasp_results['successful_grasps'].values())
    total_failed = sum(grasp_results['failed_grasps'].values())

    # Calculate the Score (currently success ratio, as all successful are tied)
    grasp_scores = {}
    for grasp_id, success_count in grasp_results['successful_grasps'].items():
        # Score = Total successful attempts for this ID
        grasp_scores[grasp_id] = success_count

    # 3. Grasp Ranking (Ranked by raw success count)
    ranked_grasps = sorted(
        grasp_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    loginfo(f"\n\n{'=' * 70}")
    loginfo(" DEMONSTRATION SUMMARY & GRASP ANALYSIS")
    loginfo(f"{'=' * 70}")
    loginfo(f"Total Unique Grasps Tested: {len(grasp_results['successful_grasps']) + len(grasp_results['failed_grasps'])}")
    loginfo(f"Total Pickup Attempts: {grasp_results['total_attempts']}")
    loginfo(f"Total Successful Pickups (over all grasps): {total_successful}")
    loginfo(f"Total Final Failures (PlanFailure): {total_failed}")

    if total_successful + total_failed > 0:
        success_rate = (total_successful / (total_successful + total_failed)) * 100
        loginfo(f"Overall Success Rate: {success_rate:.1f}%")

    # --- RANKING OUTPUT ---
    loginfo(f"\nTop 5 Grasp IDs Ranked by Success Count:")
    for rank, (grasp_id, count) in enumerate(ranked_grasps[:5]):
        loginfo(f"  {rank+1}. ID '{grasp_id}' - {count} successful pickups")

    loginfo(f"\nTop 5 Most Failed Grasp IDs:")
    final_fails = sorted(
        grasp_results['failed_grasps'].items(),
        key=lambda item: item[1],
        reverse=True
    )
    for rank, (grasp_id, count) in enumerate(final_fails[:5]):
        loginfo(f"  {rank+1}. ID '{grasp_id}' - {count} times led to PlanFailure")


    # Error Log Analysis
    loginfo(f"\n{'=' * 70}")
    loginfo(" PLAN GUARDIAN LOG ANALYSIS")
    loginfo(f"{'=' * 70}")

    if not guardian.error_log:
        loginfo("No errors were logged, meaning all pickups succeeded on the first attempt!")
    else:
        loginfo(f"Total Errors Logged by Guardian: {len(guardian.error_log)}")

        error_types = {}
        for entry in guardian.error_log:
            error_type = entry['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1

        loginfo("Error Type Breakdown:")
        for error_type, count in error_types.items():
            loginfo(f"  • {error_type}: {count} occurrence(s) (These were attempts the guardian tried to fix)")

    loginfo(f"{'=' * 70}")
    loginfo(" DEMONSTRATION COMPLETED")
    loginfo(f"{'=' * 70}")


#@with_error_handling
def complete_recovery_demo(grasp_data=None, grasp_classifier=None, guardian=None, **_):
    """
    Comprehensive demonstration: Iterates through ALL grasps for ALL objects
    and uses both arms, tracking the success rate of each grasp ID.
    """
    global grasp_results
    loginfo("=" * 70)
    loginfo("COMPREHENSIVE GRASP TEST DEMONSTRATION")
    loginfo("=" * 70)

    # Objects to manipulate
    object_configs = [
        {"name": "red_box", "arm": Arms.RIGHT, "place_offset": 0.2},
        {"name": "blue_box", "arm": Arms.LEFT, "place_offset": 0.4}, # Demonstrates Left Arm usage
        {"name": "green_box", "arm": Arms.RIGHT, "place_offset": 0.6},
    ]

    # Extract all grasp IDs from the loaded YAML data (as strings)
    all_grasp_ids = [str(g['id']) for g in grasp_data['grasps']]

    total_grasp_attempts = 0

    # 1. Iterate through each object
    for i, obj_config in enumerate(object_configs, 1):
        obj_name = obj_config["name"]
        target_arm = obj_config["arm"]
        place_y_offset = obj_config["place_offset"]

        obj_desig = BelieveObject(names=[obj_name])
        obj = obj_desig.resolve()

        loginfo(f"\n{'=' * 30}")
        loginfo(f"TESTING OBJECT {i}: {obj_name} with {target_arm.name} Arm")
        loginfo(f"Total Grasps to Test: {len(all_grasp_ids)}")
        loginfo(f"{'=' * 30}")

        # 2. Iterate through all grasp IDs for the current object
        for grasp_id in all_grasp_ids:

            # Reset the object's pose before each pickup attempt for a clean start
            # Must resolve the object again to get the latest pose update
            obj = obj_desig.resolve()
            initial_pose = obj.get_pose()

            total_grasp_attempts += 1
            grasp_results['total_attempts'] += 1

            # Ensure the object is not currently held by the robot
            ParkArmsAction(Arms.BOTH).perform()

            loginfo(f"Attempt #{total_grasp_attempts}: Testing grasp ID '{grasp_id}'...")

            pickup_action = PickUpAction(
                object_designator=obj,
                arm=target_arm,
                grasp_description=grasp_id
            )

            # Context should be detailed for the PlanGuardian log
            context = {
                'object_designator': obj,
                'arm': target_arm,
                'object_name': obj_name,
                'grasp_id': grasp_id,
                'initial_pose': initial_pose,
            }

            try:
                # Attempt pickup (PlanGuardian handles all retries/recoveries internally)
                pickup_action.perform()

                # Success Logic
                loginfo(f"[SUCCESS] Grasp '{grasp_id}' succeeded on {obj_name}.")
                grasp_results['successful_grasps'][grasp_id] = \
                    grasp_results['successful_grasps'].get(grasp_id, 0) + 1

                # Place object at new location to free up space
                place_pose = PoseStamped.from_list([0.0, 1.0 + place_y_offset, 0.8])
                place_action = PlaceAction(obj, arm=target_arm, target_location=place_pose)
                place_action.perform()
                loginfo(f"[ACTION] Placed successfully. Parking arm...")

                # Park the arm for the next attempt
                ParkArmsAction(target_arm).perform()

            except PlanFailure as pf:
                # Failure Logic (PlanGuardian raised PlanFailure because all recovery attempts failed)
                logwarn(f"[FAILURE] Grasp '{grasp_id}' failed all recovery attempts.")
                grasp_results['failed_grasps'][grasp_id] = \
                    grasp_results['failed_grasps'].get(grasp_id, 0) + 1

                # Reset the object to its starting position for the next grasp attempt
                obj.set_pose(initial_pose)
                ParkArmsAction(target_arm).perform()

            except Exception as e:
                # Catch any unexpected runtime errors
                logwarn(f"[CRITICAL ERROR] Unexpected exception during action: {e}")
                # Reset the object and continue
                obj.set_pose(initial_pose)
                ParkArmsAction(target_arm).perform()

            # Time delay to make the simulation visible
            time.sleep(0)

    # Final Summary Section
    _log_final_summary(guardian)


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

    try:
        # Assuming Cube_Pad_grasps.yaml is in the execution path
        GRASP_FILE_PATH = "Cube_Pad_grasps.yaml"
        grasp_data = load_grasps_from_file(GRASP_FILE_PATH)
    except Exception as e:
        logwarn(f"Failed to load YAML file from {GRASP_FILE_PATH}: {e}")
        raise

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

    # Aliasing a numeric ID to 'top' for compatibility with the general action pattern
    if '0' in right_gripper.grasps:
        right_gripper.grasps['top'] = right_gripper.grasps['0']
        left_gripper.grasps['top'] = left_gripper.grasps['0']
        loginfo("Aliased numerical grasp ID '3' to named grasp 'top' for demo compatibility.")
    else:
        logwarn("Grasp ID '3' not found; continuing with numerical IDs.")

    loginfo("Robot configured with grasps from YAML file")

    # Spawn objects at strategic positions for varied reachability
    object_positions = [
        ([0.12, 1.38, 0.75], (1, 0, 0, 1), "block_red.urdf", "red_box"),
        ([0.08, 1.35, 0.75], (0, 0, 1, 1), "block_blue.urdf", "blue_box"),
        ([0.20, 1.38, 0.75], (0, 1, 0, 1), "block_green.urdf", "green_box"),
    ]

    for pos, color, mesh, name in object_positions:
        obj = Object(name, PhysicalObject, mesh, color=Color(*color))
        obj.set_pose(PoseStamped.from_list(pos))
        loginfo(f"Spawned {name} at position {pos}")

    world.step()
    loginfo("Simulation setup complete. Starting demonstration...")
    time.sleep(0)

    # Run the demonstration
    with simulated_robot:
        try:
            complete_recovery_demo(grasp_data=grasp_data)
            loginfo("[FINAL STATUS] ALL DEMO RUNS COMPLETE")
        except PlanFailure as e:
            logwarn(f"[FINAL STATUS] DEMONSTRATION FAILED CRITICALLY")
            logwarn(f"Critical Error: {e}")
        except Exception as e:
            logwarn(f"[FINAL STATUS] UNEXPECTED ERROR")
            logwarn(f"Exception: {e}")

    loginfo("Simulation will close in 5 seconds...")
    time.sleep(0)
    world.exit()