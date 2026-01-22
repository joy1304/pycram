"""
Tracy Reachability Validator
"""

import numpy as np
from typing import Optional, Tuple, List
from pycram.datastructures.pose import Pose
from pycram.datastructures.enums import Arms
from pycram.world_concepts.world_object import Object
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TracyReachability")


class TracyReachabilityValidator:
    def __init__(self, safety_margin: float = 0.10):
        """
        Initialize reachability validator with Tracy measurements.
        """
        self.safety_margin = safety_margin

        self.arm_mount_x = 0.135  # Forward from table center
        self.arm_mount_z = 0.173  # Height above table
        self.arm_mount_y_offset = 0.050925  # Left/right offset

        self.shoulder_height = 0.1807  # Base link to shoulder
        self.upper_arm_length = 0.6127  # Shoulder to elbow
        self.forearm_length = 0.57155  # Elbow to wrist
        self.wrist_length = 0.2364  # Wrist joints combined
        self.gripper_length = 0.144  # Gripper tool frame

        # Calculate total theoretical reach
        total_theoretical = (self.upper_arm_length +
                           self.forearm_length +
                           self.wrist_length +
                           self.gripper_length)

        # Practical reach
        practical_reach = total_theoretical * 0.85

        # Tracy's workspace parameters
        self.workspace = {
            'right': {
                # Horizontal reach from arm mount base
                'max_reach': 1.20,
                'min_reach': 0.20,

                # Vertical limits (Z-axis, world coordinates)
                'height_min': 0.05,     # Just above table surface
                'height_max': 1.40,     # Maximum upward reach
                'optimal_height': 0.75, # Best manipulation height

                # Lateral reach (Y-axis for right arm)
                'lateral_reach': 0.85,  # Side-to-side capability

                # Arm mount position (for accurate distance calculation)
                'mount_x': self.arm_mount_x,
                'mount_y': -self.arm_mount_y_offset,  # Right arm (negative Y)
                'mount_z': self.arm_mount_z,
            },
            'left': {
                'max_reach': 1.20,
                'min_reach': 0.20,
                'height_min': 0.05,
                'height_max': 1.40,
                'optimal_height': 0.75,
                'lateral_reach': 0.85,
                'mount_x': self.arm_mount_x,
                'mount_y': self.arm_mount_y_offset,  # Left arm (positive Y)
                'mount_z': self.arm_mount_z,
            }
        }

        logger.info("Tracy Reachability Validator")
        logger.info(f"Total theoretical reach: {total_theoretical:.3f}m")
        logger.info(f"Practical reach (85%): {practical_reach:.3f}m")
        logger.info(f"Safety margin: {safety_margin:.3f}m")
        logger.info(f"Effective max reach: {1.20 - safety_margin:.3f}m")
        logger.info(f"Arm mount height: {self.arm_mount_z}m")

    def update_workspace_parameters(
        self,
        arm: str,
        max_reach: Optional[float] = None,
        min_reach: Optional[float] = None,
        height_min: Optional[float] = None,
        height_max: Optional[float] = None,
        lateral_reach: Optional[float] = None
    ):

        if arm not in ['right', 'left']:
            raise ValueError("arm must be 'right' or 'left'")

        if max_reach is not None:
            self.workspace[arm]['max_reach'] = max_reach
        if min_reach is not None:
            self.workspace[arm]['min_reach'] = min_reach
        if height_min is not None:
            self.workspace[arm]['height_min'] = height_min
        if height_max is not None:
            self.workspace[arm]['height_max'] = height_max
        if lateral_reach is not None:
            self.workspace[arm]['lateral_reach'] = lateral_reach

        logger.info(f"Updated {arm} arm workspace parameters")

    def get_robot_base_pose(self, robot: Object) -> Pose:
        """
        Get Tracy's base pose (table frame) in world coordinates.
        """
        return robot.get_pose()

    def get_arm_mount_position(self, robot: Object, arm: str) -> Tuple[float, float, float]:
        """
        Calculate the actual arm mount position in world coordinates.
        """
        base_pose = self.get_robot_base_pose(robot)
        workspace = self.workspace[arm]

        # Arm mount is at fixed offset from table base
        mount_x = base_pose.position.x + workspace['mount_x']
        mount_y = base_pose.position.y + workspace['mount_y']
        mount_z = base_pose.position.z + workspace['mount_z']

        return (mount_x, mount_y, mount_z)

    def is_object_reachable(
        self,
        object_pose: Pose,
        robot: Object,
        arm: str = "right",
        verbose: bool = True
    ) -> Tuple[bool, str]:
        """
        Check if an object at given pose is reachable by Tracy's arm.
        """
        # Handle Arms enum
        if isinstance(arm, Arms):
            arm = "right" if arm == Arms.RIGHT else "left"

        # Get arm mount position (not base position!)
        mount_x, mount_y, mount_z = self.get_arm_mount_position(robot, arm)
        workspace = self.workspace[arm]

        # Calculate distance from ARM MOUNT to object (3D)
        dx = object_pose.position.x - mount_x
        dy = object_pose.position.y - mount_y
        dz = object_pose.position.z - mount_z

        # Horizontal distance (XY plane)
        horizontal_distance = np.sqrt(dx**2 + dy**2)

        # 3D distance
        distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)

        # Object height in world coordinates
        object_height = object_pose.position.z

        # Apply safety margin
        max_reach_safe = workspace['max_reach'] - self.safety_margin
        min_reach_safe = workspace['min_reach'] + self.safety_margin

        # Check 1: Too far horizontally?
        if horizontal_distance > max_reach_safe:
            reason = (f"Object too far: {horizontal_distance:.3f}m from arm mount "
                     f"(max safe: {max_reach_safe:.3f}m)")
            if verbose:
                logger.warning(f" {reason}")
            return False, reason

        # Check 2: Too close? (causes singularities)
        if horizontal_distance < min_reach_safe:
            reason = (f"Object too close: {horizontal_distance:.3f}m "
                     f"(min safe: {min_reach_safe:.3f}m, singularity zone)")
            if verbose:
                logger.warning(f" {reason}")
            return False, reason

        # Check 3: Below table surface?
        if object_height < workspace['height_min']:
            reason = (f"Object too low: {object_height:.3f}m "
                     f"(min: {workspace['height_min']:.3f}m, below table)")
            if verbose:
                logger.warning(f" {reason}")
            return False, reason

        # Check 4: Too high?
        if object_height > workspace['height_max']:
            reason = (f"Object too high: {object_height:.3f}m "
                     f"(max: {workspace['height_max']:.3f}m, above reach)")
            if verbose:
                logger.warning(f" {reason}")
            return False, reason

        # Check 5: Lateral reach (Y-axis offset from arm centerline)
        # For right arm (mount_y negative), check absolute Y deviation
        # For left arm (mount_y positive), check absolute Y deviation
        lateral_offset = abs(dy)
        if lateral_offset > workspace['lateral_reach']:
            reason = (f"Object too far laterally: {lateral_offset:.3f}m "
                     f"(max: {workspace['lateral_reach']:.3f}m)")
            if verbose:
                logger.warning(f" {reason}")
            return False, reason

        # Check 6: 3D distance check (overall reachability sphere)
        # Adjust max reach based on vertical deviation from optimal height
        height_deviation = abs(object_height - workspace['optimal_height'])
        height_penalty = height_deviation / workspace['height_max']  # 0 to 1
        effective_max_reach = workspace['max_reach'] * (1.0 - height_penalty * 0.3)

        if distance_3d > effective_max_reach - self.safety_margin:
            reason = (f"Object outside 3D reach envelope: {distance_3d:.3f}m "
                     f"(effective max: {effective_max_reach - self.safety_margin:.3f}m)")
            if verbose:
                logger.warning(f" {reason}")
            return False, reason

        # All checks passed!
        if verbose:
            logger.info(f" Object reachable by {arm} arm:")
            logger.info(f" Horizontal distance: {horizontal_distance:.3f}m")
            logger.info(f" 3D distance: {distance_3d:.3f}m")
            logger.info(f" Height: {object_height:.3f}m")
            logger.info(f" Lateral offset: {lateral_offset:.3f}m")

        return True, f"Object is reachable (dist: {horizontal_distance:.3f}m, height: {object_height:.3f}m)"

    def filter_reachable_objects(
        self,
        object_list: List[Object],
        robot: Object,
        arm: str = "right"
    ) -> List[Object]:
        """
        List of reachable objects
        """
        reachable = []
        logger.info(f" Filtering {len(object_list)} objects for {arm} arm reachability...")

        for obj in object_list:
            try:
                obj_pose = obj.get_pose()
                is_reachable, reason = self.is_object_reachable(
                    obj_pose, robot, arm, verbose=False
                )
                if is_reachable:
                    reachable.append(obj)
                    logger.info(f"  Reachable {obj.name:15} - {reason}")
                else:
                    logger.info(f"  Not reachable {obj.name:15} - {reason}")
            except Exception as e:
                logger.error(f"  Error checking {obj.name}: {e}")
                continue

        logger.info(f" Found {len(reachable)}/{len(object_list)} reachable objects")
        return reachable

    def get_reachability_score(
        self,
        object_pose: Pose,
        robot: Object,
        arm: str = "right"
    ) -> float:
        """
        Calculate a reachability score (0-1) for an object.
        """
        # Handle Arms enum
        if isinstance(arm, Arms):
            arm = "right" if arm == Arms.RIGHT else "left"

        mount_x, mount_y, mount_z = self.get_arm_mount_position(robot, arm)
        workspace = self.workspace[arm]

        # Calculate distances
        dx = object_pose.position.x - mount_x
        dy = object_pose.position.y - mount_y
        horizontal_distance = np.sqrt(dx**2 + dy**2)

        # Optimal distance is mid-range (not too close, not too far)
        optimal_distance = (workspace['min_reach'] + workspace['max_reach']) / 2
        distance_score = 1.0 - abs(horizontal_distance - optimal_distance) / workspace['max_reach']
        distance_score = max(0, min(1, distance_score))

        # Height score (optimal at comfortable manipulation height)
        height = object_pose.position.z
        optimal_height = workspace['optimal_height']
        height_range = workspace['height_max'] - workspace['height_min']
        height_score = 1.0 - abs(height - optimal_height) / height_range
        height_score = max(0, min(1, height_score))

        # Lateral score (prefer objects close to arm centerline)
        lateral_offset = abs(dy)
        lateral_score = 1.0 - (lateral_offset / workspace['lateral_reach'])
        lateral_score = max(0, min(1, lateral_score))

        # Combined score (weighted)
        score = (distance_score * 0.5 +  # Distance most important
                height_score * 0.3 +      # Height fairly important
                lateral_score * 0.2)      # Lateral less critical

        return score

    def visualize_workspace(self, robot: Object) -> str:
        """
        Generate a text visualization of Tracy's workspace with exact measurements.

        Args:
            robot: Robot Object (Tracy)

        Returns:
            Formatted string showing workspace boundaries
        """
        base_pose = self.get_robot_base_pose(robot)

        viz = "\n" + "="*70
        viz += "\n TRACY WORKSPACE "
        viz += "\n" + "="*70
        viz += f"\n Table (base) position: ({base_pose.position.x:.3f}, {base_pose.position.y:.3f}, {base_pose.position.z:.3f})"
        viz += f"\n Robot: UR10e dual-arm with Robotiq 85 grippers"
        viz += f"\n Upper arm: {self.upper_arm_length}m | Forearm: {self.forearm_length}m"
        viz += f"\n Safety margin: {self.safety_margin}m"

        for arm_name, workspace in self.workspace.items():
            mount_x, mount_y, mount_z = (
                workspace['mount_x'],
                workspace['mount_y'],
                workspace['mount_z']
            )

            viz += f"\n\n{arm_name.upper()} ARM:"
            viz += f"\n  Arm mount: ({mount_x:.3f}, {mount_y:.3f}, {mount_z:.3f})"
            viz += f"\n  Horizontal reach: {workspace['min_reach']:.2f}m - {workspace['max_reach']:.2f}m"
            viz += f"\n  Height range: {workspace['height_min']:.2f}m - {workspace['height_max']:.2f}m"
            viz += f"\n  Optimal height: {workspace['optimal_height']:.2f}m"
            viz += f"\n  Lateral reach: ±{workspace['lateral_reach']:.2f}m"
            viz += f"\n  Safe reach zone: {workspace['min_reach'] + self.safety_margin:.2f}m - {workspace['max_reach'] - self.safety_margin:.2f}m"

        viz += "\n" + "="*70
        viz += "\n TIP: Place objects at height ~0.75m for best reachability!"
        viz += "\n  Table is integrated in URDF - DO NOT load separate table!"
        viz += "\n" + "="*70 + "\n"
        return viz


# Global validator instance
_validator_instance = None


def get_validator(safety_margin: float = 0.10) -> TracyReachabilityValidator:
    """
    Get or create global validator instance.
    Returns:TracyReachabilityValidator instance
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = TracyReachabilityValidator(safety_margin=safety_margin)
    return _validator_instance


def check_object_reachable(
    obj: Object,
    robot: Object,
    arm: str = "right"
) -> bool:
    """
    Check if object is reachable.
    """
    validator = get_validator()
    obj_pose = obj.get_pose()
    is_reachable, _ = validator.is_object_reachable(obj_pose, robot, arm, verbose=False)
    return is_reachable


def safe_manipulation_wrapper(
    manipulation_function,
    obj: Object,
    robot: Object,
    arm: str = "right"
):
    """
    Wrapper that checks reachability before executing manipulation.
    """
    validator = get_validator()
    obj_pose = obj.get_pose()

    is_reachable, reason = validator.is_object_reachable(obj_pose, robot, arm)

    if not is_reachable:
        logger.error(f" Cannot perform manipulation: {reason}")
        raise ValueError(f"Object unreachable: {reason}")

    # Object is reachable, perform action
    logger.info(" Object is reachable, proceeding with manipulation")
    return manipulation_function()