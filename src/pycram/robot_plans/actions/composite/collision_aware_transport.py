"""
Collision-Aware Transport Action for PyCRAM

This module provides a transport action that automatically detects if the target
position is occupied by another object and adjusts the placement position accordingly.
"""

from __future__ import annotations
from scipy.spatial.transform import Rotation as R

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from typing_extensions import Union, Optional, Type, Any, Iterable, List

from pycram.datastructures.dataclasses import FrozenObject
from pycram.datastructures.enums import Arms, VerticalAlignment, ApproachDirection, AxisIdentifier
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.datastructures.pose import PoseStamped, Vector3
from pycram.designators.object_designator import BelieveObject
from pycram.failures import ConfigurationNotReached
from pycram.has_parameters import has_parameters
from pycram.plan import with_plan
from pycram.robot_description import RobotDescription
from pycram.robot_plans.actions.base import ActionDescription, record_object_pre_perform
from pycram.robot_plans.actions.core import ParkArmsActionDescription, PickUpActionDescription, PlaceActionDescription
from pycram.ros import loginfo, logwarn
from pycram.world_concepts.world_object import Object
from pycram.datastructures.world import World


def get_all_world_objects(world: World, exclude_names: List[str] = None) -> List[Object]:
    """
    Get all objects in the world, excluding specified names.
    
    Args:
        world: The BulletWorld instance
        exclude_names: List of object names to exclude (e.g., robot name, floor)
        
    Returns:
        List of world objects
    """
    if exclude_names is None:
        exclude_names = []
    
    objects = []
    for obj in world.objects:
        if obj.name not in exclude_names and obj.obj_type != "environment":
            objects.append(obj)
    return objects


def check_position_occupied(target_position: PoseStamped, 
                           world: World,
                           exclude_objects: List[Object] = None,
                           tolerance: float = 0.08) -> Optional[Object]:
    """
    Check if a target position is occupied by any object in the world.
    
    Args:
        target_position: The target position to check
        world: The BulletWorld instance
        exclude_objects: Objects to exclude from collision check (e.g., the object being moved)
        tolerance: Distance threshold to consider as "occupied" (meters)
        
    Returns:
        The occupying object if found, None otherwise
    """
    if exclude_objects is None:
        exclude_objects = []
    
    exclude_names = [obj.name for obj in exclude_objects]
    
    # Get robot name to exclude
    try:
        robot_name = RobotDescription.current_robot_description.name
        exclude_names.append(robot_name)
    except:
        pass
    
    # Also exclude common environment objects
    exclude_names.extend(["floor", "ground", "plane", "environment"])
    
    target_pos = np.array([
        target_position.position.x,
        target_position.position.y,
        target_position.position.z
    ])
    
    for obj in world.objects:
        if obj.name in exclude_names:
            continue
        
        # Skip environment type objects
        if hasattr(obj, 'obj_type') and obj.obj_type == "environment":
            continue
            
        obj_pos = np.array([
            obj.pose.position.x,
            obj.pose.position.y,
            obj.pose.position.z
        ])
        
        distance = np.linalg.norm(target_pos - obj_pos)
        
        if distance < tolerance:
            return obj
    
    return None


def find_free_position(original_target: PoseStamped,
                       world: World,
                       exclude_objects: List[Object] = None,
                       offset: float = 0.05,
                       max_attempts: int = 8,
                       tolerance: float = 0.08) -> PoseStamped:
    """
    Find a free position near the original target by trying offsets in different directions.
    
    Args:
        original_target: The original target position
        world: The BulletWorld instance
        exclude_objects: Objects to exclude from collision check
        offset: Distance to offset from original position (meters), default 5cm
        max_attempts: Maximum number of directions to try
        tolerance: Distance threshold for collision detection
        
    Returns:
        A free position (either original or adjusted)
    """
    # First check if original position is free
    occupying_obj = check_position_occupied(original_target, world, exclude_objects, tolerance)
    
    if occupying_obj is None:
        loginfo(f"Target position is free, using original target.")
        return original_target
    
    logwarn(f"Target position occupied by '{occupying_obj.name}'. Finding alternative position...")
    
    # Define offset directions to try (in XY plane)
    # Try: +X, -X, +Y, -Y, and diagonals
    directions = [
        (1, 0, 0),    # +X (right)
        (-1, 0, 0),   # -X (left)
        (0, 1, 0),    # +Y (forward)
        (0, -1, 0),   # -Y (backward)
        (1, 1, 0),    # diagonal +X+Y
        (-1, 1, 0),   # diagonal -X+Y
        (1, -1, 0),   # diagonal +X-Y
        (-1, -1, 0),  # diagonal -X-Y
    ]
    
    # Normalize diagonal directions
    directions = [(np.array(d) / np.linalg.norm(d)).tolist() for d in directions]
    
    for attempt, direction in enumerate(directions[:max_attempts]):
        # Calculate new position
        new_x = original_target.position.x + direction[0] * offset
        new_y = original_target.position.y + direction[1] * offset
        new_z = original_target.position.z + direction[2] * offset
        
        new_target = PoseStamped.from_list(
            [new_x, new_y, new_z],
            [
                original_target.orientation.x,
                original_target.orientation.y,
                original_target.orientation.z,
                original_target.orientation.w
            ]
        )

        # Check if new position is free
        if check_position_occupied(new_target, world, exclude_objects, tolerance) is None:
            loginfo(f"Found free position at offset ({direction[0]*offset:.2f}, {direction[1]*offset:.2f}, {direction[2]*offset:.2f})")
            return new_target

    # If all attempts failed, try increasing offset
    logwarn(f"Could not find free position with {offset}m offset. Trying larger offset...")

    for multiplier in [2, 3, 4]:
        larger_offset = offset * multiplier
        for direction in directions[:4]:  # Try cardinal directions with larger offset
            new_x = original_target.position.x + direction[0] * larger_offset
            new_y = original_target.position.y + direction[1] * larger_offset
            new_z = original_target.position.z + direction[2] * larger_offset

            new_target = PoseStamped.from_list(
                [new_x, new_y, new_z],
                [
                    original_target.orientation.x,
                    original_target.orientation.y,
                    original_target.orientation.z,
                    original_target.orientation.w
                ]
            )

            if check_position_occupied(new_target, world, exclude_objects, tolerance) is None:
                loginfo(f"Found free position at larger offset ({direction[0]*larger_offset:.2f}, {direction[1]*larger_offset:.2f})")
                return new_target

    # Last resort: return original and let placement handle collision
    logwarn("Could not find completely free position. Using original target.")
    return original_target


@has_parameters
@dataclass
class CollisionAwareTransportAction(ActionDescription):
    """
    Transport an object by automatically:
    1. Checking if target position is occupied
    2. Finding a free position if occupied (offset by 5cm)
    3. Choosing the closest arm (Left vs Right)
    4. Choosing the best grasp approach and vertical alignment
    """
    object_designator: Object
    target_location: PoseStamped

    grasp_description: Optional[GraspDescription] = None

    placement_offset: float = 0.05
    """Offset distance (in meters) to use when target is occupied. Default: 5cm"""

    collision_tolerance: float = 0.08
    """Distance threshold to consider positions as 'occupied'. Default: 8cm"""

    actual_placement_location: Optional[PoseStamped] = field(init=False, repr=False, default=None)
    """The actual location where the object was placed (may differ from target if adjusted)"""

    object_at_execution: Optional[FrozenObject] = field(init=False, repr=False, default=None)

    _pre_perform_callbacks = []

    def __post_init__(self):
        super().__post_init__()
        self.pre_perform(record_object_pre_perform)

    def _choose_best_arm(self, robot: Object, obj: Object) -> Arms:
        """Intelligently choose the closest available arm."""
        rd = RobotDescription.current_robot_description
        try:
            try:
                left_tool = rd.get_arm_chain(Arms.LEFT).get_tool_frame()
                right_tool = rd.get_arm_chain(Arms.RIGHT).get_tool_frame()
            except:
                left_tool = "l_gripper_tool_frame"
                right_tool = "r_gripper_tool_frame"

            left_tip = robot.get_link_position(left_tool).to_numpy()
            right_tip = robot.get_link_position(right_tool).to_numpy()

        except Exception as e:
            loginfo(f"Warning: Could not get arm positions, defaulting to RIGHT: {e}")
            return Arms.RIGHT

        obj_pos = np.array([obj.pose.position.x, obj.pose.position.y, obj.pose.position.z])

        left_dist = np.linalg.norm(left_tip - obj_pos)
        right_dist = np.linalg.norm(right_tip - obj_pos)

        attached = robot._attached_objects.values() if hasattr(robot, '_attached_objects') else []
        left_free = left_tool not in attached
        right_free = right_tool not in attached

        if left_free and (not right_free or left_dist <= right_dist):
            return Arms.LEFT
        elif right_free:
            return Arms.RIGHT
        else:
            raise ConfigurationNotReached("No free arm available.")

    def _calculate_closest_faces(self, vec_obj_frame: Vector3) -> tuple:
        """Helper method to calculate faces locally."""
        all_axes = [AxisIdentifier.X, AxisIdentifier.Y, AxisIdentifier.Z]
        vec_list = vec_obj_frame.to_list()

        sorted_axes = sorted(all_axes, key=lambda axis: abs(vec_list[axis.value.index(1)]), reverse=True)

        primary_axis = sorted_axes[0]
        primary_sign = int(np.sign(vec_list[primary_axis.value.index(1)]))

        primary_class = VerticalAlignment if primary_axis == AxisIdentifier.Z else ApproachDirection
        primary_face = primary_class.from_axis_direction(primary_axis, primary_sign)

        if len(sorted_axes) > 1:
            secondary_axis = sorted_axes[1]
            secondary_sign = int(np.sign(vec_list[secondary_axis.value.index(1)]))
        else:
            secondary_axis = primary_axis
            secondary_sign = -primary_sign

        secondary_class = VerticalAlignment if secondary_axis == AxisIdentifier.Z else ApproachDirection
        secondary_face = secondary_class.from_axis_direction(secondary_axis, secondary_sign)

        return primary_face, secondary_face

    def _choose_best_grasp(self, robot: Object, obj: Object) -> GraspDescription:
        """Automatically calculate BOTH Approach and Vertical Alignment."""
        robot_pos = np.array(robot.pose.position.to_list())
        obj_pos = np.array(obj.pose.position.to_list())

        vec_world = robot_pos - obj_pos

        obj_orientation = obj.pose.orientation.to_list()
        rotation = R.from_quat(obj_orientation)
        vec_local = rotation.inv().apply(vec_world)
        vec_local_obj = Vector3.from_list(vec_local.tolist())

        primary, secondary = self._calculate_closest_faces(vec_local_obj)

        final_approach = ApproachDirection.FRONT
        final_vertical = VerticalAlignment.TOP

        if isinstance(primary, VerticalAlignment):
            final_vertical = primary
            if isinstance(secondary, ApproachDirection):
                final_approach = secondary
        elif isinstance(primary, ApproachDirection):
            final_approach = primary
            if isinstance(secondary, VerticalAlignment):
                final_vertical = secondary

        loginfo(f"Auto-Grasp: Approach={final_approach.name}, Vertical={final_vertical.name}")

        return GraspDescription(
            approach_direction=final_approach,
            vertical_alignment=final_vertical,
            rotate_gripper=False
        )

    def plan(self) -> None:
        robot = BelieveObject(names=[RobotDescription.current_robot_description.name]).resolve()
        obj = self.object_designator
        world = robot.world

        if not obj or not obj.pose:
            raise ConfigurationNotReached(f"Cannot resolve object pose: {self.object_designator}")

        # === COLLISION DETECTION AND POSITION ADJUSTMENT ===
        loginfo(f"Checking if target position is occupied...")

        # Find free position (will be original if not occupied)
        self.actual_placement_location = find_free_position(
            original_target=self.target_location,
            world=world,
            exclude_objects=[obj],  # Exclude the object we're moving
            offset=self.placement_offset,
            tolerance=self.collision_tolerance
        )

        if self.actual_placement_location != self.target_location:
            logwarn(f"Adjusted placement from ({self.target_location.position.x:.3f}, "
                   f"{self.target_location.position.y:.3f}, {self.target_location.position.z:.3f}) "
                   f"to ({self.actual_placement_location.position.x:.3f}, "
                   f"{self.actual_placement_location.position.y:.3f}, "
                   f"{self.actual_placement_location.position.z:.3f})")

        # === ARM AND GRASP SELECTION ===
        chosen_arm = self._choose_best_arm(robot, obj)

        if self.grasp_description is None:
            self.grasp_description = self._choose_best_grasp(robot, obj)

        loginfo(f"Action: Transporting '{obj.name}' with {chosen_arm.name}")

        # === EXECUTE TRANSPORT ===
        ParkArmsActionDescription(Arms.BOTH).perform()

        PickUpActionDescription(
            object_designator=self.object_designator,
            arm=chosen_arm,
            grasp_description=self.grasp_description
        ).perform()

        ParkArmsActionDescription(Arms.BOTH).perform()

        PlaceActionDescription(
            object_designator=self.object_designator,
            target_location=self.actual_placement_location,
            arm=chosen_arm
        ).perform()

        ParkArmsActionDescription(Arms.BOTH).perform()

    def validate(self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None):
        pass

    @classmethod
    @with_plan
    def description(cls,
                    object_designator: Union[Iterable[Object], Object],
                    target_location: Union[Iterable[PoseStamped], PoseStamped],
                    grasp_description: Optional[GraspDescription] = None,
                    placement_offset: float = 0.05,
                    collision_tolerance: float = 0.08) -> \
            PartialDesignator[Type['CollisionAwareTransportAction']]:
        return PartialDesignator(cls,
                                 object_designator=object_designator,
                                 target_location=target_location,
                                 grasp_description=grasp_description,
                                 placement_offset=placement_offset,
                                 collision_tolerance=collision_tolerance)


# Convenience alias
CollisionAwareTransportActionDescription = CollisionAwareTransportAction.description