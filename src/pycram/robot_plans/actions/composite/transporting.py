from __future__ import annotations
from scipy.spatial.transform import Rotation as R

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from typing_extensions import Union, Optional, Type, Any, Iterable

from ....config.action_conf import ActionConfig
from .facing import FaceAtActionDescription
from ..core import ParkArmsActionDescription, NavigateActionDescription, PickUpActionDescription, PlaceActionDescription, \
    PlaceAction
from ....datastructures.dataclasses import FrozenObject
from ....datastructures.enums import Arms, Grasp, VerticalAlignment, ApproachDirection, AxisIdentifier
from ....datastructures.grasp import GraspDescription
from ....datastructures.partial_designator import PartialDesignator
from ....datastructures.pose import PoseStamped, Vector3
from ....designators.location_designator import ProbabilisticCostmapLocation, CostmapLocation
from ....designators.object_designator import BelieveObject
from ....failures import ObjectUnfetchable, ReachabilityFailure, ConfigurationNotReached
from ....has_parameters import has_parameters
from ....plan import with_plan
from ....robot_description import RobotDescription
from ....robot_plans.actions.base import ActionDescription, record_object_pre_perform
from ....robot_plans.motions.gripper import MoveTCPMotion
from ....ros import loginfo
from ....world_concepts.world_object import Object
from ....datastructures.world import World


@has_parameters
@dataclass
class TransportAction(ActionDescription):
    """
    Transports an object to a position using an arm
    """

    object_designator: Object = field(repr=False)
    """
    Object designator_description describing the object that should be transported.
    """
    target_location: PoseStamped
    """
    Target Location to which the object should be transported
    """
    arm: Optional[Arms]
    """
    Arm that should be used
    """
    place_rotation_agnostic: Optional[bool] = False
    """
    If True, the robot will place the object in the same orientation as it is itself, no matter how the object was grasped.
    """

    object_at_execution: Optional[FrozenObject] = field(init=False, repr=False, default=None)
    """
    The object at the time this Action got created. It is used to be a static, information holding entity. It is
    not updated when the BulletWorld object is changed.
    """

    _pre_perform_callbacks = []
    """
    List to save the callbacks which should be called before performing the action.
    """

    def __post_init__(self):
        super().__post_init__()

        # Store the object's data copy at execution
        self.pre_perform(record_object_pre_perform)

    def plan(self) -> None:
        robot_desig_resolved = BelieveObject(names=[RobotDescription.current_robot_description.name]).resolve()
        ParkArmsActionDescription(Arms.BOTH).perform()
        pickup_loc = ProbabilisticCostmapLocation(target=self.object_designator,
                                                  reachable_for=robot_desig_resolved,
                                                  reachable_arm=self.arm)
        # Tries to find a pick-up position for the robot that uses the given arm
        pickup_pose = pickup_loc.resolve()
        if not pickup_pose:
            raise ObjectUnfetchable(
                f"Found no pose for the robot to grasp the object: {self.object_designator} with arm: {self.arm}")

        NavigateActionDescription(pickup_pose, True).perform()
        PickUpActionDescription(self.object_designator, pickup_pose.arm,
                                grasp_description=pickup_pose.grasp_description).perform()
        ParkArmsActionDescription(Arms.BOTH).perform()
        try:
            place_loc = ProbabilisticCostmapLocation(
                target=self.target_location,
                reachable_for=robot_desig_resolved,
                reachable_arm=pickup_pose.arm,
                grasp_descriptions=[pickup_pose.grasp_description],
                object_in_hand=self.object_designator,
                rotation_agnostic=self.place_rotation_agnostic,
            ).resolve()
        except StopIteration:
            raise ReachabilityFailure(
                self.object_designator, robot_desig_resolved, pickup_pose.arm, pickup_pose.grasp_description)
        NavigateActionDescription(place_loc, True).perform()

        if self.place_rotation_agnostic:
            # Placing rotation agnostic currently means that the robot will position its gripper in the same orientation
            # as it is itself, no matter how the object was grasped
            robot_rotation = robot_desig_resolved.get_pose().orientation
            self.target_location.orientation = robot_rotation
            approach_direction = GraspDescription(pickup_pose.grasp_description.approach_direction, VerticalAlignment.NoAlignment, False)
            side_grasp = np.array(
                robot_desig_resolved.robot_description.get_arm_chain(pickup_pose.arm).end_effector.grasps[
                    approach_direction])
            # Inverting the quaternion for the used grasp to cancel it out during placing, since placing considers the
            # object orientation relative to the gripper )
            side_grasp *= np.array([-1, -1, -1, 1])
            self.target_location.rotate_by_quaternion(side_grasp.tolist())

        PlaceActionDescription(self.object_designator, self.target_location, pickup_pose.arm).perform()
        ParkArmsActionDescription(Arms.BOTH).perform()

    def validate(self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None):
        # The validation of each core action is done in the action itself, so no more validation needed here.
        pass

    @classmethod
    @with_plan
    def description(cls, object_designator: Union[Iterable[Object], Object],
                    target_location: Union[Iterable[PoseStamped], PoseStamped],
                    arm: Union[Iterable[Arms], Arms] = None, place_rotation_agnostic: Optional[bool] = False) -> \
    PartialDesignator[Type[TransportAction]]:
        return PartialDesignator(TransportAction, object_designator=object_designator,
                                 target_location=target_location,
                                 arm=arm, place_rotation_agnostic=place_rotation_agnostic)


@has_parameters
@dataclass
class PickAndPlaceAction(ActionDescription):
    """
    Transports an object to a position using an arm without moving the base of the robot
    """

    object_designator: Object
    """
    Object designator_description describing the object that should be transported.
    """
    target_location: PoseStamped
    """
    Target Location to which the object should be transported
    """
    arm: Arms
    """
    Arm that should be used
    """
    grasp_description: GraspDescription
    """
    Description of the grasp to pick up the target
    """
    _pre_perform_callbacks = []
    """
    List to save the callbacks which should be called before performing the action.
    """

    def __post_init__(self):
        super().__post_init__()

        # Store the object's data copy at execution
        self.pre_perform(record_object_pre_perform)

    def plan(self) -> None:
        ParkArmsActionDescription(Arms.BOTH).perform()
        PickUpActionDescription(self.object_designator, self.arm,
                     grasp_description=self.grasp_description).perform()
        PlaceActionDescription(self.object_designator, self.target_location, self.arm).perform()
        ParkArmsActionDescription(Arms.BOTH).perform()

    def validate(self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None):
        if self.object_designator.pose.__eq__(self.target_location):
            pass
        else:
            raise ValueError("Object not moved to the target location")

    @classmethod
    @with_plan
    def description(cls, object_designator: Union[Iterable[Object], Object],
                    target_location: Union[Iterable[PoseStamped], PoseStamped],
                    arm: Union[Iterable[Arms], Arms] = None,
                    grasp_description = GraspDescription) -> PartialDesignator[Type[PickAndPlaceAction]]:
        return PartialDesignator(PickAndPlaceAction, object_designator=object_designator,
                                 target_location=target_location,
                                 arm=arm,
                                 grasp_description=grasp_description)

@has_parameters
@dataclass
class MoveAndPlaceAction(ActionDescription):
    """
    Navigate to `standing_position`, then turn towards the object and pick it up.
    """

    standing_position: PoseStamped
    """
    The pose to stand before trying to pick up the object
    """

    object_designator: Object
    """
    The object to pick up
    """

    target_location: PoseStamped
    """
    The location to place the object.
    """

    arm: Arms
    """
    The arm to use
    """

    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    def plan(self):
        NavigateActionDescription(self.standing_position, self.keep_joint_states).perform()
        FaceAtActionDescription(self.target_location, self.keep_joint_states).perform()
        PlaceAction(self.object_designator, self.target_location, self.arm).perform()

    def validate(self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None):
        # The validation will be done in each of the core action perform methods so no need to validate here.
        pass

    @classmethod
    @with_plan
    def description(cls, standing_position: Union[Iterable[PoseStamped], PoseStamped],
                    object_designator: Union[Iterable[Object], Object],
                    target_location: Union[Iterable[PoseStamped], PoseStamped],
                    arm: Union[Iterable[Arms], Arms] = None,
                    keep_joint_states: Union[Iterable[bool], bool] = ActionConfig.navigate_keep_joint_states, ) -> \
            PartialDesignator[Type[MoveAndPlaceAction]]:
        return PartialDesignator(MoveAndPlaceAction,
                                 standing_position=standing_position,
                                 object_designator=object_designator,
                                 target_location=target_location,
                                 arm=arm)



@has_parameters
@dataclass
class MoveAndPickUpAction(ActionDescription):
    """
    Navigate to `standing_position`, then turn towards the object and pick it up.
    """

    standing_position: PoseStamped
    """
    The pose to stand before trying to pick up the object
    """

    object_designator: Object
    """
    The object to pick up
    """

    arm: Arms
    """
    The arm to use
    """

    grasp_description: GraspDescription
    """
    The grasp to use
    """

    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    object_at_execution: Optional[FrozenObject] = field(init=False, repr=False, default=None)
    """
    The object at the time this Action got created. It is used to be a static, information holding entity. It is
    not updated when the BulletWorld object is changed.
    """

    _pre_perform_callbacks = []
    """
    List to save the callbacks which should be called before performing the action.
    """

    def __post_init__(self):
        super().__post_init__()

        # Store the object's data copy at execution
        self.pre_perform(record_object_pre_perform)

    def plan(self):
        NavigateActionDescription(self.standing_position, self.keep_joint_states).perform()
        FaceAtActionDescription(self.object_designator.pose, self.keep_joint_states).perform()
        PickUpActionDescription(self.object_designator, self.arm, self.grasp_description).perform()

    def validate(self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None):
        # The validation will be done in each of the core action perform methods so no need to validate here.
        pass

    @classmethod
    @with_plan
    def description(cls, standing_position: Union[Iterable[PoseStamped], PoseStamped],
                    object_designator: Union[Iterable[PoseStamped], PoseStamped],
                    arm: Union[Iterable[Arms], Arms] = None,
                    grasp_description: Union[Iterable[Grasp], Grasp] = None,
                    keep_joint_states: Union[Iterable[bool], bool] = ActionConfig.navigate_keep_joint_states) -> \
            PartialDesignator[Type[MoveAndPickUpAction]]:
        return PartialDesignator(MoveAndPickUpAction,
                                 standing_position=standing_position,
                                 object_designator=object_designator,
                                 arm=arm,
                                 grasp_description=grasp_description,
                                 keep_joint_states=keep_joint_states)



'''
@has_parameters
@dataclass
class EfficientTransportAction(ActionDescription):
    """
    To transport an object to a target location by choosing the closest
    available arm using simple Euclidean distance.
    """
    object_designator: Object
    target_location: PoseStamped

    def _choose_best_arm(self, robot: Object, obj: Object) -> Arms:
        """
        Function to find the closest available arm.
        """
        rd = RobotDescription.current_robot_description
        try:
            left_tool_frame = rd.get_arm_chain(Arms.LEFT).get_tool_frame()
            right_tool_frame = rd.get_arm_chain(Arms.RIGHT).get_tool_frame()
            left_tip = robot.get_link_position(left_tool_frame)
            right_tip = robot.get_link_position(right_tool_frame)
        except Exception as e:
            raise ConfigurationNotReached(f"Could not get tool frames or link positions for arms: {e}")

        # Calculating the distance from gripper to the object
        object_pos_vec = np.array([obj.pose.position.x, obj.pose.position.y, obj.pose.position.z])
        left_dist = np.linalg.norm(np.array(left_tip) - object_pos_vec)
        right_dist = np.linalg.norm(np.array(right_tip) - object_pos_vec)

        # If the arms are free or not
        attached_links = robot._attached_objects.values() if hasattr(robot, '_attached_objects') else []
        left_free = left_tool_frame not in attached_links
        right_free = right_tool_frame not in attached_links

        # Decide which arm to use based on proximity and availability
        if left_free and (not right_free or left_dist <= right_dist):
            return Arms.LEFT
        elif right_free:
            return Arms.RIGHT
        else:
            raise ConfigurationNotReached("No free arm available to grasp the object.")

    def plan(self) -> None:
        """
        The main plan for the transport action, optimized for a stationary robot.
        """
        robot = BelieveObject(names=[RobotDescription.current_robot_description.name]).resolve()
        obj = self.object_designator

        if not obj or not obj.pose:
            raise ConfigurationNotReached(f"Couldn't resolve the pose for the object: {self.object_designator}")

        # Intelligently choose the best arm
        chosen_arm = self._choose_best_arm(robot, obj)
        loginfo(f"Chosen arm for transport: {chosen_arm.name}")

        ParkArmsActionDescription(Arms.BOTH).perform()

        PickUpActionDescription(
            object_designator=self.object_designator,
            arm=chosen_arm
        ).perform()

        ParkArmsActionDescription(Arms.BOTH).perform()

        # Attempting the placement.
        PlaceActionDescription(
            object_designator=self.object_designator,
            target_location=self.target_location,
            arm=chosen_arm
        ).perform()


        ParkArmsActionDescription(Arms.BOTH).perform()

    @classmethod
    @with_plan
    def description(cls, object_designator: Union[Iterable[Object], Object],
                    target_location: Union[Iterable[PoseStamped], PoseStamped]) -> PartialDesignator[Type['EfficientTransportAction']]:
        return PartialDesignator(cls,
                                 object_designator=object_designator,
                                 target_location=target_location)
'''


@has_parameters
@dataclass
class EfficientTransportAction(ActionDescription):
    """
    Transport an object by automatically choosing:
    1. The closest arm (Left vs Right)
    2. The best grasp approach (Front, Side, etc.) AND Vertical Alignment (Top, Bottom)
    """
    object_designator: Object
    target_location: PoseStamped

    grasp_description: Optional[GraspDescription] = None

    def _choose_best_arm(self, robot: Object, obj: Object) -> Arms:
        """
        Intelligently choose the closest available arm.
        """
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
        """
        Helper method to calculate faces locally, avoiding dependency on updated grasp.py.
        """
        all_axes = [AxisIdentifier.X, AxisIdentifier.Y, AxisIdentifier.Z]
        # Convert vector to list for indexing
        vec_list = vec_obj_frame.to_list()

        # Sort axes by magnitude (largest absolute value first)
        sorted_axes = sorted(all_axes, key=lambda axis: abs(vec_list[axis.value.index(1)]), reverse=True)

        primary_axis = sorted_axes[0]
        # Get sign (+1 or -1)
        primary_sign = int(np.sign(vec_list[primary_axis.value.index(1)]))

        # Determine Primary Face (Vertical vs Approach)
        primary_class = VerticalAlignment if primary_axis == AxisIdentifier.Z else ApproachDirection
        primary_face = primary_class.from_axis_direction(primary_axis, primary_sign)

        # Determine Secondary Face
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
        """
        Automatically calculate BOTH Approach and Vertical Alignment.
        """
        # 1. Vector Math Setup
        robot_pos = np.array(robot.pose.position.to_list())
        obj_pos = np.array(obj.pose.position.to_list())

        vec_world = robot_pos - obj_pos

        # Handle object rotation
        obj_orientation = obj.pose.orientation.to_list()
        rotation = R.from_quat(obj_orientation)
        vec_local = rotation.inv().apply(vec_world)
        vec_local_obj = Vector3.from_list(vec_local.tolist())

        # 2. Get the Two Best Faces (Using internal helper now!)
        primary, secondary = self._calculate_closest_faces(vec_local_obj)

        # 3. Smart Logic to Assign Roles
        final_approach = ApproachDirection.FRONT
        final_vertical = VerticalAlignment.TOP

        # CASE A: The robot is mostly above/below the object (Primary is Vertical)
        if isinstance(primary, VerticalAlignment):
            final_vertical = primary
            if isinstance(secondary, ApproachDirection):
                final_approach = secondary

        # CASE B: The robot is mostly to the side (Primary is Approach)
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

        if not obj or not obj.pose:
            raise ConfigurationNotReached(f"Cannot resolve object pose: {self.object_designator}")

        # 1. Auto-Choose Arm
        chosen_arm = self._choose_best_arm(robot, obj)

        # 2. Auto-Choose Grasp (if not provided)
        if self.grasp_description is None:
            self.grasp_description = self._choose_best_grasp(robot, obj)

        loginfo(f"Action: Transporting '{obj.name}' with {chosen_arm.name}")

        ParkArmsActionDescription(Arms.BOTH).perform()

        PickUpActionDescription(
            object_designator=self.object_designator,
            arm=chosen_arm,
            grasp_description=self.grasp_description
        ).perform()

        ParkArmsActionDescription(Arms.BOTH).perform()

        PlaceActionDescription(
            object_designator=self.object_designator,
            target_location=self.target_location,
            arm=chosen_arm
        ).perform()

        ParkArmsActionDescription(Arms.BOTH).perform()

    @classmethod
    @with_plan
    def description(cls,
                    object_designator: Union[Iterable[Object], Object],
                    target_location: Union[Iterable[PoseStamped], PoseStamped],
                    grasp_description: Optional[GraspDescription] = None) -> \
            PartialDesignator[Type['EfficientTransportAction']]:
        return PartialDesignator(cls,
                                 object_designator=object_designator,
                                 target_location=target_location,
                                 grasp_description=grasp_description)


@has_parameters
@dataclass
class HoldAction(ActionDescription):
    """
    Transports an object to a position using an arm without moving the base of the robot
    """

    object_designator: Object
    """
    Object designator_description describing the object that should be transported.
    """
    target_location: PoseStamped
    """
    Target Location to which the object should be transported
    """
    arm: Arms
    """
    Arm that should be used
    """
    grasp_description: GraspDescription
    """
    Description of the grasp to pick up the target
    """
    _pre_perform_callbacks = []
    """
    List to save the callbacks which should be called before performing the action.
    """

    def __post_init__(self):
        super().__post_init__()

        # Store the object's data copy at execution
        self.pre_perform(record_object_pre_perform)

    def plan(self) -> None:
        ParkArmsActionDescription(Arms.BOTH).perform()
        PickUpActionDescription(self.object_designator, self.arm,
                     grasp_description=self.grasp_description).perform()
        MoveTCPMotion(self.target_location, self.arm).perform()

    def validate(self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None):
        if self.object_designator.pose.__eq__(self.target_location):
            pass
        else:
            raise ValueError("Object not moved to the target location")

    @classmethod
    @with_plan
    def description(cls, object_designator: Union[Iterable[Object], Object],
                    target_location: Union[Iterable[PoseStamped], PoseStamped],
                    arm: Union[Iterable[Arms], Arms] = None,
                    grasp_description = GraspDescription) -> PartialDesignator[Type[PickAndPlaceAction]]:
        return PartialDesignator(HoldAction, object_designator=object_designator,
                                 target_location=target_location,
                                 arm=arm,
                                 grasp_description=grasp_description)

TransportActionDescription = TransportAction.description
PickAndPlaceActionDescription = PickAndPlaceAction.description
MoveAndPlaceActionDescription = MoveAndPlaceAction.description
MoveAndPickUpActionDescription = MoveAndPickUpAction.description
EfficientTransportActionDescription = EfficientTransportAction.description
HoldActionDescription = HoldAction.description
