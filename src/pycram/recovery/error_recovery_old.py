# -*- coding: utf-8 -*-

"""
A robust, intelligent error handling and recovery module for PyCRAM plans,
specifically adapted for fixed-base manipulators like PR2.
"""

from __future__ import annotations
import time
from typing import Dict, Any, Callable, Type

# --- Core PyCRAM Imports ---
from pycram.robot_plans.actions.core import ParkArmsAction
from pycram.designators.object_designator import BelieveObject
from pycram.datastructures.enums import Arms
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


class PlanGuardian:
    """
    An intelligent error handler that diagnoses failures and attempts recovery
    through arm manipulation and grasp re-selection.
    """

    def __init__(self, world, robot: Object, grasp_classifier: GraspClassifier | None = None):
        self.world = world
        self.robot = robot
        self.grasp_classifier = grasp_classifier
        self.error_log: list[dict] = []
        self.recovery_strategies: Dict[Type[LowLevelFailure], Callable] = {
            GripperGoalNotReached: self._recover_from_object_not_grasped,
            ManipulationGoalNotReached: self._recover_from_configuration_not_reached,
            ManipulationPoseUnreachable: self._recover_from_unfeasible_grasp,
            GraspUnfeasibleFailure: self._recover_from_unfeasible_grasp,
        }

    def handle_error(self, error: LowLevelFailure, failed_action: Any, context: dict) -> None:
        """Entry point for handling a detected error."""
        act_name = type(failed_action).__name__ if failed_action else "<unknown>"
        logwarn(f"PlanGuardian caught: {type(error).__name__} during action: {act_name}")
        self._log_error(error, failed_action, context)

        strategy = self.recovery_strategies.get(type(error), self._generic_recovery)

        try:
            strategy(error, failed_action, context)
            loginfo("PlanGuardian successfully recovered from the error.")
        except Exception as e:
            logwarn(f"Recovery attempt failed: {e}")
            raise PlanFailure(f"Could not recover from {type(error).__name__}") from e

    def _log_error(self, error: LowLevelFailure, action: Any, context: dict) -> None:
        log_entry = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'failed_action': type(action).__name__ if action else "<unknown>",
            'context': {k: str(v) for k, v in context.items()}
        }
        self.error_log.append(log_entry)

    def _recover_from_object_not_grasped(self, error: GripperGoalNotReached, action: Any, context: dict) -> None:
        """
        Try alternative grasps using the GraspClassifier.
        This is a generic recovery function that is NOT dependent on PickUpAction.
        """
        loginfo("Executing recovery for GripperGoalNotReached...")

        if not self.grasp_classifier:
            raise ValueError("GraspClassifier is required for this recovery strategy.")
        if not action:
            raise ValueError("The failed action must be provided for recovery.")

        target_object = context.get('object_designator')
        arm = context.get('arm')

        # Get the class of the failed action (e.g., PickUpAction)
        action_type = type(action)

        all_grasps = self.grasp_classifier.get_n_best_reachable_grasps(
            n=10, arm=arm, target_object=target_object
        )
        if not all_grasps:
             raise PlanFailure("Grasp classifier found no reachable grasps for recovery.")

        for grasp_info in all_grasps:
            grasp_id = grasp_info['id']
            loginfo(f"Trying new grasp strategy: {grasp_id}")
            try:
                # Re-create the failed action with a new grasp
                new_action = action_type(
                    object_designator=target_object,
                    arm=arm,
                    grasp_description=grasp_id
                )
                new_action.perform()
                loginfo(f"Successfully performed action with grasp '{grasp_id}'.")
                return
            except LowLevelFailure as e:
                logwarn(f"Grasp attempt with '{grasp_id}' failed: {e}")

        raise PlanFailure("Exhausted all grasp recovery strategies without success.")

    def _recover_from_configuration_not_reached(self, error: ManipulationGoalNotReached, action: Any, context: dict) -> None:
        loginfo("Executing recovery for ManipulationGoalNotReached...")
        loginfo("Resetting arm posture by parking.")
        ParkArmsAction(Arms.BOTH).perform()
        time.sleep(1)
        if action and hasattr(action, "perform"):
            loginfo("Re-attempting the original action.")
            action.perform()
        else:
            raise PlanFailure("Original action not provided; cannot re-attempt.")

    def _recover_from_unfeasible_grasp(self, error: LowLevelFailure, action: Any, context: dict) -> None:
        obj = context.get('object_designator')
        logwarn(f"Object '{obj}' is considered unreachable. No recovery possible.")
        raise PlanFailure("Object is not reachable from the current robot position.")

    def _generic_recovery(self, error: LowLevelFailure, action: Any, context: dict) -> None:
        logwarn(f"No specific strategy for {type(error).__name__}. Triggering generic recovery.")
        ParkArmsAction(Arms.BOTH).perform()
        raise PlanFailure("Generic recovery triggered for an unhandled error.")


def with_error_handling(plan_function: Callable) -> Callable:
    """
    Decorator that injects a PlanGuardian and optionally a GraspClassifier.
    """
    def wrapper(*args, **kwargs):
        # FIX: Access the class attribute directly, without parentheses
        robot_name = RobotDescription.current_robot_description.name
        robot = BelieveObject(names=[robot_name]).resolve()
        world = robot.world

        grasp_classifier = None
        if 'grasp_data' in kwargs:
            grasp_classifier = GraspClassifier(kwargs['grasp_data'], robot)

        guardian = PlanGuardian(world, robot, grasp_classifier)

        kwargs['grasp_classifier'] = grasp_classifier
        kwargs['guardian'] = guardian

        return plan_function(*args, **kwargs)

    return wrapper
