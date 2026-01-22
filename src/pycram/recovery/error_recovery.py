"""
Enhanced Error handling and recovery module for Tracy
Includes: retry counters, perception recovery, place recovery, and jiggle recovery
"""

from __future__ import annotations
import time
import random
from typing import Dict, Any, Callable, Type, Optional

# FIX: Correct imports for Gripper Motion
from pycram.robot_plans.motions.gripper import MoveGripperMotion
from pycram.robot_plans import MoveTCPMotion, PlaceAction
from pycram.robot_plans.actions.core import ParkArmsAction, DetectAction
from pycram.designators.object_designator import BelieveObject
from pycram.datastructures.enums import Arms, GripperState
from pycram.datastructures.pose import Pose, PoseStamped, Vector3
from pycram.failures import (
    PlanFailure,
    ManipulationGoalNotReached,
    ManipulationPoseUnreachable,
    GripperGoalNotReached,
    LowLevelFailure,
)
from pycram.world_concepts.world_object import Object
from pycram.robot_description import RobotDescription
from pycram.ros import logwarn, loginfo
from pycram.designators.grasp_classifier import GraspClassifier


class GraspUnfeasibleFailure(LowLevelFailure):
    """Raised when the GraspClassifier finds no reachable grasps."""
    pass


class PlacementFailedError(LowLevelFailure):
    """Raised when PlaceAction fails to place object at target location."""
    pass


class PlanGuardian:
    """
    An enhanced error handler that diagnoses failures and attempts recovery.
    """

    def __init__(self, world, robot: Object, grasp_classifier: GraspClassifier | None = None,
                 max_retries: int = 2, enable_perception_update: bool = True):
        self.world = world
        self.robot = robot
        self.grasp_classifier = grasp_classifier
        self.max_retries = max_retries
        self.enable_perception_update = enable_perception_update

        self.retry_counts: Dict[str, int] = {}
        self.error_log: list[dict] = []

        self.recovery_strategies: Dict[Type[LowLevelFailure], Callable] = {
            # If grasp fails (slip) OR is unreachable (IK fail), try next grasp
            GripperGoalNotReached: self._recover_from_object_not_grasped,
            ManipulationPoseUnreachable: self._recover_from_object_not_grasped,
            GraspUnfeasibleFailure: self._recover_from_unfeasible_grasp,

            # Motion/Arm errors
            ManipulationGoalNotReached: self._recover_from_configuration_not_reached,

            # Place errors
            PlacementFailedError: self._recover_from_failed_place,

            # Missing features
            NotImplementedError: self._recover_from_not_implemented,
        }

    def handle_error(self, error: LowLevelFailure, failed_action: Any, context: dict) -> None:
        act_name = type(failed_action).__name__ if failed_action else "<unknown>"
        action_id = id(failed_action) if failed_action else "no_action"

        logwarn(f"PlanGuardian caught: {type(error).__name__} during action: {act_name}")
        self._log_error(error, failed_action, context)

        if self._should_retry(action_id, error):
            try:
                self._simple_retry(failed_action, action_id)
                loginfo("PlanGuardian: Simple retry succeeded!")
                self._reset_retry_count(action_id)
                return
            except Exception as retry_error:
                logwarn(f"Retry attempt failed: {retry_error}")

        strategy = self.recovery_strategies.get(type(error), self._generic_recovery)

        try:
            strategy(error, failed_action, context)
            loginfo("PlanGuardian successfully recovered from the error.")
            self._reset_retry_count(action_id)
        except Exception as e:
            logwarn(f"Recovery attempt failed: {e}")
            self._reset_retry_count(action_id)
            raise PlanFailure(f"Could not recover from {type(error).__name__}") from e

    def _should_retry(self, action_id: str, error: LowLevelFailure) -> bool:
        current_retries = self.retry_counts.get(action_id, 0)
        retryable_errors = (ManipulationGoalNotReached,)
        if isinstance(error, retryable_errors) and current_retries < self.max_retries:
            return True
        return False

    def _simple_retry(self, action: Any, action_id: str) -> None:
        if not action or not hasattr(action, "perform"):
            raise ValueError("Action cannot be retried - no perform method")
        self.retry_counts[action_id] = self.retry_counts.get(action_id, 0) + 1
        retry_num = self.retry_counts[action_id]
        loginfo(f"Attempting simple retry #{retry_num}/{self.max_retries}...")
        time.sleep(0.5)
        action.perform()

    def _reset_retry_count(self, action_id: str) -> None:
        if action_id in self.retry_counts:
            del self.retry_counts[action_id]

    def _log_error(self, error: LowLevelFailure, action: Any, context: dict) -> None:
        log_entry = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'failed_action': type(action).__name__ if action else "<unknown>",
            'context': {k: str(v) for k, v in context.items()}
        }
        self.error_log.append(log_entry)

    def _update_object_perception(self, target_object: Object) -> None:
        if not self.enable_perception_update:
            return
        loginfo(f"Running perception update for object: {target_object.name}")
        try:
            detect_action = DetectAction(target_object.name)
            detected = detect_action.perform()
            if detected:
                loginfo(f"Object pose updated: {target_object.get_pose()}")
            else:
                logwarn(f"Could not re-detect object {target_object.name}")
        except Exception as e:
            logwarn(f"Perception update failed: {e}")

    def _recover_from_object_not_grasped(self, error: LowLevelFailure, action: Any, context: dict) -> None:
        loginfo(f"Executing recovery for {type(error).__name__} (Trying alternative grasps)...")

        if not self.grasp_classifier:
            raise ValueError("GraspClassifier is required for this recovery strategy.")

        target_object = context.get('object_designator')
        if not target_object and action and hasattr(action, 'object_designator'):
            target_object = action.object_designator

        if not target_object:
            raise ValueError("Target object not found in context or action.")

        arm = context.get('arm')
        if not arm and action and hasattr(action, 'arm'):
            arm = action.arm

        self._update_object_perception(target_object)
        action_type = type(action)

        all_grasps = self.grasp_classifier.get_n_best_reachable_grasps(
            n=10, arm=arm, target_object=target_object
        )
        if not all_grasps:
             raise PlanFailure("Grasp classifier found no reachable grasps for recovery.")

        for grasp_info in all_grasps:
            grasp_id = str(grasp_info['id'])

            # Skip the failed one
            failed_id = context.get('grasp_description')
            if not failed_id and action and hasattr(action, 'grasp_description'):
                failed_id = action.grasp_description

            if failed_id == grasp_id:
                continue

            loginfo(f"Trying new grasp strategy: {grasp_id}")
            try:
                new_action = action_type(
                    object_designator=target_object,
                    arm=arm,
                    grasp_description=grasp_id
                )
                new_action.perform()
                loginfo(f"Successfully performed action with grasp '{grasp_id}'.")
                return
            except LowLevelFailure as e:
                logwarn(f"Grasp attempt with '{grasp_id}' failed: {type(e).__name__}")

        raise PlanFailure("Exhausted all grasp recovery strategies without success.")

    def _recover_from_configuration_not_reached(self, error: ManipulationGoalNotReached, action: Any, context: dict) -> None:
        loginfo("Executing recovery for ManipulationGoalNotReached...")
        try:
            self._perform_jiggle_motion(context.get('arm', Arms.BOTH))
        except Exception as e:
            logwarn(f"Jiggle motion failed: {e}")

        loginfo("Resetting arm posture by parking.")
        ParkArmsAction(Arms.BOTH).perform()
        time.sleep(1)

        if action and hasattr(action, "perform"):
            loginfo("Re-attempting the original action.")
            action.perform()
        else:
            raise PlanFailure("Original action not provided; cannot re-attempt.")

    def _perform_jiggle_motion(self, arm: Arms) -> None:
        try:
            tcp_link = self.robot.robot_description.get_arm_chain(arm).end_effector.tool_frame
            current_pose = self.robot.get_link_pose(tcp_link)
            jiggle_offset = Pose(
                position=[random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03), random.uniform(0.02, 0.05)],
                orientation=[0, 0, 0, 1]
            )
            jiggled_pose = current_pose * jiggle_offset
            MoveTCPMotion(target=jiggled_pose, arm=arm).perform()
            time.sleep(0.3)
        except Exception as e:
            logwarn(f"Could not perform jiggle motion: {e}")

    def _recover_from_failed_place(self, error: PlacementFailedError, action: Any, context: dict) -> None:
        loginfo("Executing recovery for PlacementFailedError...")
        target_location = context.get('target_location')
        held_object = context.get('object_designator')
        arm = context.get('arm')

        if not target_location or not held_object:
            if action and hasattr(action, 'object_designator'):
                held_object = action.object_designator
            if action and hasattr(action, 'target_location'):
                target_location = action.target_location

            if not target_location or not held_object:
                raise ValueError("Cannot recover from place failure: missing context data")

        nearby_offsets = [(0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05), (0.05, 0.05)]
        action_type = type(action) if action else None

        for i, (dx, dy) in enumerate(nearby_offsets):
            loginfo(f"Trying alternative placement location #{i+1} (Offset: {dx:.2f}, {dy:.2f})")
            try:
                alternative_pose = target_location.copy()
                alternative_pose.position.x += dx
                alternative_pose.position.y += dy

                if action_type:
                    new_action = action_type(
                        object_designator=held_object,
                        target_location=alternative_pose,
                        arm=arm
                    )
                    new_action.perform()
                    loginfo(f"✓ Successfully placed at alternative location #{i+1}")
                    return
            except LowLevelFailure as e:
                logwarn(f"Alternative location #{i+1} failed: {e}")

        loginfo("All nearby locations failed. Moving to discard zone...")

        frame_id = "map"
        if hasattr(target_location, 'header'):
             frame_id = target_location.header.frame_id
        elif hasattr(target_location, 'frame_id'):
             frame_id = target_location.frame_id

        self._place_in_discard_zone(held_object, arm, frame_id)

    def _place_in_discard_zone(self, held_object: Object, arm: Arms, frame_id: str = "map") -> None:
        discard_pos = [-0.5, 1.5, 0.7]
        discard_orn = [0, 0, 0, 1]
        discard_pose = PoseStamped.from_list(discard_pos, discard_orn, frame_id)

        loginfo(f"Placing {held_object.name} in discard zone at {discard_pos}")

        try:
            MoveTCPMotion(target=discard_pose, arm=arm).perform()
            time.sleep(0.5)
            # FIX: Correct Gripper Motion
            MoveGripperMotion(motion=GripperState.OPEN, gripper=arm).perform()
            loginfo("✓ Object placed in discard zone. Gripper freed.")
        except Exception as e:
            raise PlanFailure(f"Failed to place in discard zone: {e}") from e


    def _recover_from_unfeasible_grasp(self, error: LowLevelFailure, action: Any, context: dict) -> None:
        logwarn(f"Object is considered unreachable. No recovery possible.")
        raise PlanFailure("Object is not reachable from the current robot position.")

    def _recover_from_not_implemented(self, error: NotImplementedError, action: Any, context: dict) -> None:
        if "get_object_rotated_bounding_box" in str(error):
            logwarn("Attempting to use axis-aligned bounding box as a fallback.")
            try:
                raise PlanFailure("Action needs to be modified to use the fallback method.")
            except Exception as e:
                raise PlanFailure("Could not recover from NotImplementedError.") from e
        else:
            raise PlanFailure("No specific fallback for this NotImplementedError.") from error

    def _generic_recovery(self, error: LowLevelFailure, action: Any, context: dict) -> None:
        logwarn(f"No specific strategy for {type(error).__name__}. Triggering generic recovery.")
        ParkArmsAction(Arms.BOTH).perform()
        raise PlanFailure("Generic recovery triggered for an unhandled error.")

    def get_error_statistics(self) -> Dict[str, Any]:
        if not self.error_log:
            return {"total_errors": 0}
        error_types = {}
        for entry in self.error_log:
            error_type = entry['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        return {
            "total_errors": len(self.error_log),
            "error_types": error_types,
            "first_error": self.error_log[0]['timestamp'],
            "last_error": self.error_log[-1]['timestamp'],
        }


def with_error_handling(plan_function: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        robot_name = RobotDescription.current_robot_description.name
        robot = BelieveObject(names=[robot_name]).resolve()
        world = robot.world
        grasp_classifier = None
        if 'grasp_data' in kwargs:
            grasp_classifier = GraspClassifier(kwargs['grasp_data'], robot)
        guardian = PlanGuardian(
            world,
            robot,
            grasp_classifier,
            max_retries=kwargs.get('max_retries', 2),
            enable_perception_update=kwargs.get('enable_perception_update', True)
        )
        kwargs['grasp_classifier'] = grasp_classifier
        kwargs['guardian'] = guardian
        try:
            return plan_function(*args, **kwargs)
        except LowLevelFailure as e:
            context = {'args': args, 'kwargs': kwargs}
            guardian.handle_error(e, None, context)
    return wrapper