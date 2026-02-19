"""
Collision-Aware Transport Action with Grasp Classifier Integration for PyCRAM

This module provides a transport action that:
1. Automatically detects if target position is occupied and adjusts placement
2. Uses GraspClassifier to select the best grasp from YAML grasp data
3. Scores grasps by distance to robot and orientation simplicity
"""

from __future__ import annotations
from scipy.spatial.transform import Rotation as R

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import yaml
from typing_extensions import Union, Optional, Type, Any, Iterable, List, Dict

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


class GraspClassifier:

    def __init__(self, grasp_data: Dict, robot=None):
        self.grasp_data = grasp_data
        self.robot = robot
        self.classified_grasps = self._classify_grasps()

    def _classify_grasps(self) -> Dict[ApproachDirection, List[Dict]]:
        """Classify grasps by approach direction with initial scores"""
        classified = {direction: [] for direction in ApproachDirection}

        for grasp in self.grasp_data['grasps']:
            pose = self._create_pose(grasp)
            approach_dir = self._determine_approach_direction(pose)
            vertical_align = self._determine_vertical_alignment(pose)

            grasp_info = {
                'id': grasp['id'],
                'pose': pose,
                'approach_direction': approach_dir,
                'vertical_alignment': vertical_align,
                'score': 1.0  # Default score, will be updated by scoring functions
            }

            classified[approach_dir].append(grasp_info)

        return classified

    def _create_pose(self, grasp_data: Dict) -> PoseStamped:
        position = [
            grasp_data['position']['x'],
            grasp_data['position']['y'],
            grasp_data['position']['z']
        ]
        orientation = [
            grasp_data['orientation']['x'],
            grasp_data['orientation']['y'],
            grasp_data['orientation']['z'],
            grasp_data['orientation']['w']
        ]
        return PoseStamped.from_list(position, orientation)

    def _determine_approach_direction(self, pose: PoseStamped) -> ApproachDirection:
        """Determine approach direction from pose orientation"""
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        r = R.from_quat(quat)
        approach_vector = r.apply([0, 0, -1])
        abs_approach_xy = np.abs(approach_vector[:2])

        if abs_approach_xy[0] > abs_approach_xy[1]:
            return ApproachDirection.LEFT if approach_vector[0] < 0 else ApproachDirection.RIGHT
        else:
            return ApproachDirection.BACK if approach_vector[1] < 0 else ApproachDirection.FRONT

    def _determine_vertical_alignment(self, pose: PoseStamped) -> Optional[VerticalAlignment]:
        """Determine vertical alignment from pose"""
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        r = R.from_quat(quat)
        up_vector = r.apply([0, 0, 1])

        if up_vector[2] > 0.5:
            return VerticalAlignment.TOP
        elif up_vector[2] <= -0.5:
            return VerticalAlignment.BOTTOM
        return None


    def _calculate_distance_score(self, grasp: Dict, robot_pos: np.ndarray, obj_pos: np.ndarray) -> float:
        """
        Calculate score based on distance from robot to grasp position.

        Closer grasps get HIGHER scores.

        Returns:
            Score between 0 and 1 (higher = closer = better)
        """
        grasp_pose = grasp['pose']

        # Grasp position is relative to object, so add object position
        grasp_world_pos = np.array([
            obj_pos[0] + grasp_pose.position.x,
            obj_pos[1] + grasp_pose.position.y,
            obj_pos[2] + grasp_pose.position.z
        ])

        # Calculate distance from robot to grasp point
        distance = np.linalg.norm(robot_pos - grasp_world_pos)

        distance_score = 1.0 / (1.0 + distance)

        return distance_score

    def _calculate_orientation_score(self, grasp: Dict) -> float:
        """
        Calculate score based on orientation simplicity.

        Simpler orientations (less rotation) get HIGHER scores.
        Prefers grasps that are more "upright" and easier to execute.
        """
        grasp_pose = grasp['pose']
        quat = [
            grasp_pose.orientation.x,
            grasp_pose.orientation.y,
            grasp_pose.orientation.z,
            grasp_pose.orientation.w
        ]

        # Simple orientation = quaternion close to identity [0, 0, 0, 1]
        # or close to simple rotations around Z axis

        tilt_penalty = abs(quat[0]) + abs(quat[1])

        # tilt_penalty ranges from 0 (no tilt) to ~2 (max tilt)
        orientation_score = 1.0 / (1.0 + tilt_penalty)

        return orientation_score

    def _calculate_combined_score(self, grasp: Dict, robot_pos: np.ndarray, obj_pos: np.ndarray,
                                   distance_weight: float = 0.6,
                                   orientation_weight: float = 0.4) -> float:
        """
        Calculate combined score from distance and orientation.

        Args:
            grasp: Grasp dictionary
            robot_pos: Robot position [x, y, z]
            obj_pos: Object position [x, y, z]
            distance_weight: Weight for distance score (default 0.6)
            orientation_weight: Weight for orientation score (default 0.4)

        Returns:
            Combined weighted score
        """
        distance_score = self._calculate_distance_score(grasp, robot_pos, obj_pos)
        orientation_score = self._calculate_orientation_score(grasp)

        combined_score = (distance_weight * distance_score) + (orientation_weight * orientation_score)

        return combined_score

    def update_scores(self, robot_pos: np.ndarray, obj_pos: np.ndarray,
                      distance_weight: float = 0.6,
                      orientation_weight: float = 0.4) -> None:
        """
        Update all grasp scores based on robot and object positions.
        """
        robot_pos = np.array(robot_pos)
        obj_pos = np.array(obj_pos)

        loginfo(f"Updating grasp scores: robot_pos={robot_pos}, obj_pos={obj_pos}")
        loginfo(f"Weights: distance={distance_weight}, orientation={orientation_weight}")

        for direction, grasps in self.classified_grasps.items():
            for grasp in grasps:
                grasp['score'] = self._calculate_combined_score(
                    grasp, robot_pos, obj_pos,
                    distance_weight, orientation_weight
                )


    def get_best_grasp_for_direction(self,
                                      approach_direction: ApproachDirection,
                                      vertical_alignment: Optional[VerticalAlignment] = None) -> Optional[Dict]:
        """
        Get the best grasp for a given approach direction.
        """
        grasps = self.classified_grasps.get(approach_direction, [])

        if vertical_alignment:
            grasps = [g for g in grasps if g['vertical_alignment'] == vertical_alignment]

        if not grasps:
            return None

        # Return highest scored grasp
        grasps_sorted = sorted(grasps, key=lambda x: x['score'], reverse=True)

        best_grasp = grasps_sorted[0]

        return best_grasp

    def get_top_n_grasps_for_direction(self,
                                        approach_direction: ApproachDirection,
                                        n: int = 5,
                                        vertical_alignment: Optional[VerticalAlignment] = None) -> List[Dict]:
        """
        Get top N grasps for a given approach direction, sorted by score.
        """
        grasps = self.classified_grasps.get(approach_direction, [])

        if vertical_alignment:
            grasps = [g for g in grasps if g['vertical_alignment'] == vertical_alignment]

        if not grasps:
            return []

        # Sort by score and return top N
        grasps_sorted = sorted(grasps, key=lambda x: x['score'], reverse=True)

        return grasps_sorted[:n]

    def get_all_grasps_sorted(self) -> List[Dict]:
        """Get all grasps sorted by score (highest first)"""
        all_grasps = []
        for direction, grasps in self.classified_grasps.items():
            all_grasps.extend(grasps)
        all_grasps.sort(key=lambda x: x['score'], reverse=True)
        return all_grasps

    def get_grasp_by_id(self, grasp_id: int) -> Optional[Dict]:
        """Get a specific grasp by its ID"""
        for direction, grasps in self.classified_grasps.items():
            for grasp in grasps:
                if grasp['id'] == grasp_id:
                    return grasp
        return None

    def get_grasp_count_by_direction(self) -> Dict[str, int]:
        """Get count of grasps per direction (useful for debugging)"""
        return {dir.name: len(grasps) for dir, grasps in self.classified_grasps.items()}

    def print_score_summary(self) -> None:
        """Print a summary of grasp scores by direction"""
        print("\n" + "="*60)
        print("GRASP SCORE SUMMARY")
        print("="*60)

        for direction in ApproachDirection:
            grasps = self.classified_grasps.get(direction, [])
            if grasps:
                scores = [g['score'] for g in grasps]
                print(f"\n{direction.name}:")
                print(f"  Count: {len(grasps)}")
                print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
                print(f"  Best grasp ID: {max(grasps, key=lambda x: x['score'])['id']}")

        print("\n" + "="*60)

#HELPER FUNCTION

def load_grasp_data(yaml_path: str) -> Dict:
    """Load grasp data from YAML file"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def check_position_occupied(target_position: PoseStamped,
                           world: World,
                           exclude_objects: List[Object] = None,
                           tolerance: float = 0.08) -> Optional[Object]:
    """
    Check if a target position is occupied by any object in the world.
    """
    if exclude_objects is None:
        exclude_objects = []

    exclude_names = [obj.name for obj in exclude_objects]

    try:
        robot_name = RobotDescription.current_robot_description.name
        exclude_names.append(robot_name)
    except:
        pass

    exclude_names.extend(["floor", "ground", "plane", "environment"])

    target_pos = np.array([
        target_position.position.x,
        target_position.position.y,
        target_position.position.z
    ])

    for obj in world.objects:
        if obj.name in exclude_names:
            continue

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
    """
    occupying_obj = check_position_occupied(original_target, world, exclude_objects, tolerance)

    if occupying_obj is None:
        loginfo(f"Target position is free, using original target.")
        return original_target

    logwarn(f"Target position occupied by '{occupying_obj.name}'. Finding alternative position...")

    directions = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (1, 1, 0),
        (-1, 1, 0),
        (1, -1, 0),
        (-1, -1, 0),
    ]

    directions = [(np.array(d) / np.linalg.norm(d)).tolist() for d in directions]

    for attempt, direction in enumerate(directions[:max_attempts]):
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

        if check_position_occupied(new_target, world, exclude_objects, tolerance) is None:
            loginfo(f"Found free position at offset ({direction[0]*offset:.2f}, {direction[1]*offset:.2f}, {direction[2]*offset:.2f})")
            return new_target

    logwarn(f"Could not find free position with {offset}m offset. Trying larger offset...")

    for multiplier in [2, 3, 4]:
        larger_offset = offset * multiplier
        for direction in directions[:4]:
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

    logwarn("Could not find completely free position. Using original target.")
    return original_target


# MAIN ACTION CLASS

@has_parameters
@dataclass
class CollisionAwareTransportAction(ActionDescription):
    """
    Transport an object by automatically:
    1. Checking if target position is occupied
    2. Finding a free position if occupied (offset by 5cm)
    3. Choosing the closest arm (Left vs Right)
    4. Using GraspClassifier with distance & orientation scoring to select the best grasp
    """
    object_designator: Object
    target_location: PoseStamped

    grasp_classifier: Optional[GraspClassifier] = None
    """Optional GraspClassifier instance for selecting grasps from YAML data"""

    grasp_description: Optional[GraspDescription] = None
    """Manual grasp description (used if grasp_classifier is not provided)"""

    placement_offset: float = 0.05
    """Offset distance (in meters) to use when target is occupied. Default: 5cm"""

    collision_tolerance: float = 0.08
    """Distance threshold to consider positions as 'occupied'. Default: 8cm"""

    distance_weight: float = 0.6
    """Weight for distance scoring (0-1). Default: 0.6"""

    orientation_weight: float = 0.4
    """Weight for orientation scoring (0-1). Default: 0.4"""

    actual_placement_location: Optional[PoseStamped] = field(init=False, repr=False, default=None)
    """The actual location where the object was placed (may differ from target if adjusted)"""

    selected_grasp_id: Optional[int] = field(init=False, repr=False, default=None)
    """The ID of the grasp selected from the classifier (for debugging)"""

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

    def _choose_best_grasp_auto(self, robot: Object, obj: Object) -> GraspDescription:
        """Automatically calculate grasp based on robot-object geometry."""
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

        loginfo(f"Auto-Grasp (geometry): Approach={final_approach.name}, Vertical={final_vertical.name}")

        return GraspDescription(
            approach_direction=final_approach,
            vertical_alignment=final_vertical,
            rotate_gripper=False
        )

    def _choose_grasp_from_classifier(self, robot: Object, obj: Object) -> GraspDescription:
        """
        Use GraspClassifier to select the best grasp based on:
        1. Robot-object geometry (approach direction)
        2. Distance scoring (closer = better)
        3. Orientation simplicity scoring (simpler = better)
        """
        # Get positions for scoring
        robot_pos = np.array(robot.pose.position.to_list())
        obj_pos = np.array(obj.pose.position.to_list())

        # === UPDATE GRASP SCORES BASED ON DISTANCE AND ORIENTATION ===
        self.grasp_classifier.update_scores(
            robot_pos=robot_pos,
            obj_pos=obj_pos,
            distance_weight=self.distance_weight,
            orientation_weight=self.orientation_weight
        )

        # Calculate ideal approach direction based on geometry
        vec_world = robot_pos - obj_pos
        obj_orientation = obj.pose.orientation.to_list()
        rotation = R.from_quat(obj_orientation)
        vec_local = rotation.inv().apply(vec_world)
        vec_local_obj = Vector3.from_list(vec_local.tolist())

        primary, secondary = self._calculate_closest_faces(vec_local_obj)

        # Determine ideal approach direction
        if isinstance(primary, ApproachDirection):
            ideal_approach = primary
        elif isinstance(secondary, ApproachDirection):
            ideal_approach = secondary
        else:
            ideal_approach = ApproachDirection.FRONT

        # Determine ideal vertical alignment
        if isinstance(primary, VerticalAlignment):
            ideal_vertical = primary
        elif isinstance(secondary, VerticalAlignment):
            ideal_vertical = secondary
        else:
            ideal_vertical = VerticalAlignment.TOP

        loginfo(f"Ideal grasp direction: Approach={ideal_approach.name}, Vertical={ideal_vertical.name if ideal_vertical else 'ANY'}")

        # LOG TOP 5 GRASPS FOR THE IDEAL DIRECTION
        top_5_for_direction = self.grasp_classifier.get_top_n_grasps_for_direction(ideal_approach, n=5)
        if top_5_for_direction:
            loginfo(f"Top 5 grasps for {ideal_approach.name} direction:")
            for i, g in enumerate(top_5_for_direction):
                v_align = g['vertical_alignment'].name if g['vertical_alignment'] else 'NONE'
                loginfo(f"  {i+1}. ID={g['id']}, Score={g['score']:.4f}, Vertical={v_align}")
        else:
            loginfo(f"No grasps found for {ideal_approach.name} direction")

        # Try to get a grasp from classifier matching the ideal direction
        grasp_info = self.grasp_classifier.get_best_grasp_for_direction(ideal_approach, ideal_vertical)

        if grasp_info is None:
            # Try without vertical alignment constraint
            grasp_info = self.grasp_classifier.get_best_grasp_for_direction(ideal_approach)

        if grasp_info is None:
            # Fallback: get best grasp from any direction (highest score)
            all_grasps = self.grasp_classifier.get_all_grasps_sorted()
            if all_grasps:
                grasp_info = all_grasps[0]

        if grasp_info:
            self.selected_grasp_id = grasp_info['id']
            approach = grasp_info['approach_direction']
            vertical = grasp_info['vertical_alignment'] or VerticalAlignment.TOP

            loginfo(f"GraspClassifier selected grasp ID={grasp_info['id']}: "
                   f"Approach={approach.name}, Vertical={vertical.name if vertical else 'NONE'}, "
                   f"Score={grasp_info['score']:.4f}")

            return GraspDescription(
                approach_direction=approach,
                vertical_alignment=vertical,
                rotate_gripper=False
            )
        else:
            logwarn("GraspClassifier found no grasps, falling back to auto-grasp")
            return self._choose_best_grasp_auto(robot, obj)

    def plan(self) -> None:
        robot = BelieveObject(names=[RobotDescription.current_robot_description.name]).resolve()
        obj = self.object_designator
        world = robot.world

        if not obj or not obj.pose:
            raise ConfigurationNotReached(f"Cannot resolve object pose: {self.object_designator}")

        # COLLISION DETECTION AND POSITION ADJUSTMENT
        loginfo(f"Checking if target position is occupied...")

        self.actual_placement_location = find_free_position(
            original_target=self.target_location,
            world=world,
            exclude_objects=[obj],
            offset=self.placement_offset,
            tolerance=self.collision_tolerance
        )

        if self.actual_placement_location != self.target_location:
            logwarn(f"Adjusted placement from ({self.target_location.position.x:.3f}, "
                   f"{self.target_location.position.y:.3f}, {self.target_location.position.z:.3f}) "
                   f"to ({self.actual_placement_location.position.x:.3f}, "
                   f"{self.actual_placement_location.position.y:.3f}, "
                   f"{self.actual_placement_location.position.z:.3f})")

        # ARM SELECTION
        chosen_arm = self._choose_best_arm(robot, obj)

        # GRASP SELECTION
        if self.grasp_description is not None:
            # Use manually provided grasp
            loginfo(f"Using manually specified grasp: {self.grasp_description}")
        elif self.grasp_classifier is not None:
            # Use GraspClassifier with scoring to select grasp
            self.grasp_description = self._choose_grasp_from_classifier(robot, obj)
        else:
            # Fallback to automatic geometry-based grasp
            self.grasp_description = self._choose_best_grasp_auto(robot, obj)

        loginfo(f"Action: Transporting '{obj.name}' with {chosen_arm.name}")

        # EXECUTE TRANSPORT
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
                    grasp_classifier: Optional[GraspClassifier] = None,
                    grasp_description: Optional[GraspDescription] = None,
                    placement_offset: float = 0.05,
                    collision_tolerance: float = 0.08,
                    distance_weight: float = 0.6,
                    orientation_weight: float = 0.4) -> \
            PartialDesignator[Type['CollisionAwareTransportAction']]:
        return PartialDesignator(cls,
                                 object_designator=object_designator,
                                 target_location=target_location,
                                 grasp_classifier=grasp_classifier,
                                 grasp_description=grasp_description,
                                 placement_offset=placement_offset,
                                 collision_tolerance=collision_tolerance,
                                 distance_weight=distance_weight,
                                 orientation_weight=orientation_weight)


CollisionAwareTransportActionDescription = CollisionAwareTransportAction.description