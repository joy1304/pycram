import time
from pycram.datastructures.grasp import GraspDescription
from pycram.worlds.bullet_world import BulletWorld
from pycram.process_module import simulated_robot
from pycram.language import SequentialPlan
from pycram.designators.object_designator import *
from pycram.robot_plans.actions import PickAndPlaceActionDescription
from pycram.datastructures.enums import Arms, WorldMode, ApproachDirection, VerticalAlignment
from pycram.datastructures.pose import PoseStamped
from pycram.datastructures.dataclasses import Color
from pycrap.ontologies import Robot, PhysicalObject



world = BulletWorld(WorldMode.GUI)

# Tracy set-up
tracy = Object("tracy", Robot, "tracy.urdf", pose=PoseStamped.from_list([-0.5, 1.5, 0.5], [0, 0, 0, 1]), ignore_cached_files=True)

# Defining Object Pose
fuselage_start_pose = PoseStamped.from_list([0.25, 1.5, 0.7, 0, 0, 0, 0.1])
wing_start_pose = PoseStamped.from_list([0.25, 1.3, 0.7])
propeller_start_pose = PoseStamped.from_list([0.25, 1.7, 0.7])
tail_fin_start_pose = PoseStamped.from_list([0.4, 1.4, 0.7])
horizontal_stabilizer_start_pose = PoseStamped.from_list([0.4, 1.6, 0.7])
left_wheel_start_pose = PoseStamped.from_list([0.1, 1.4, 0.7])
right_wheel_start_pose = PoseStamped.from_list([0.1, 1.6, 0.7])
back_wheel_start_pose = PoseStamped.from_list([0.1, 1.5, 0.7])

# Target Poses and Orientation
assembly_x, assembly_y, assembly_z = 0.6, 1.5, 0.7
fuselage_target_pose = PoseStamped.from_list([assembly_x, assembly_y, assembly_z, 0, 0, 0, 1])
wing_target_pose = PoseStamped.from_list([assembly_x, assembly_y, assembly_z, 0, 0, 0, 1])
propeller_target_pose = PoseStamped.from_list([assembly_x + 0.25, assembly_y, assembly_z, 0, 0.707, 0, 0.707])
tail_fin_target_pose = PoseStamped.from_list([assembly_x - 0.22, assembly_y, assembly_z + 0.05, 0, 0, 0, 1])
horizontal_stabilizer_target_pose = PoseStamped.from_list([assembly_x - 0.22, assembly_y, assembly_z, 0, 0, 0, 1])
left_wheel_target_pose = PoseStamped.from_list([assembly_x + 0.1, assembly_y - 0.05, assembly_z - 0.05, 0, 0, 0, 1])
right_wheel_target_pose = PoseStamped.from_list([assembly_x + 0.1, assembly_y + 0.05, assembly_z - 0.05, 0, 0, 0, 1])
back_wheel_target_pose = PoseStamped.from_list([assembly_x - 0.15, assembly_y, assembly_z - 0.05, 0, 0, 0, 1])


# Loading Airplane Parts
fuselage = Object("fuselage", PhysicalObject, "fuselage_basic.urdf", pose=fuselage_start_pose, color=Color(0, 0, 1, 1))
wing = Object("wing", PhysicalObject, "wing.urdf", pose=wing_start_pose, color=Color(1, 1, 1, 1))
propeller = Object("propeller", PhysicalObject, "propeller.urdf", pose=propeller_start_pose, color=Color(1, 0, 0, 1))
tail_fin = Object("tail_fin", PhysicalObject, "tail_fin.urdf", pose=tail_fin_start_pose, color=Color(1, 0, 0, 1))
horizontal_stabilizer = Object("horizontal_stabilizer", PhysicalObject, "horizontal_stabilizer.urdf", pose=horizontal_stabilizer_start_pose, color=Color(1, 1, 1, 1))
left_wheel = Object("left_wheel", PhysicalObject, "wheel.urdf", pose=left_wheel_start_pose, color=Color(0.1, 0.1, 0.1, 1))
right_wheel = Object("right_wheel", PhysicalObject, "wheel.urdf", pose=right_wheel_start_pose, color=Color(0.1, 0.1, 0.1, 1))
back_wheel = Object("back_wheel", PhysicalObject, "wheel.urdf", pose=back_wheel_start_pose, color=Color(0.1, 0.1, 0.1, 1))


grasp_description = GraspDescription(ApproachDirection.FRONT, VerticalAlignment.TOP)
main_plan = SequentialPlan(
    PickAndPlaceActionDescription(fuselage, fuselage_target_pose, Arms.RIGHT, grasp_description),
    PickAndPlaceActionDescription(wing, wing_target_pose, Arms.LEFT, grasp_description),
    PickAndPlaceActionDescription(propeller, propeller_target_pose, Arms.RIGHT, grasp_description),
    PickAndPlaceActionDescription(horizontal_stabilizer, horizontal_stabilizer_target_pose, Arms.LEFT, grasp_description),
    PickAndPlaceActionDescription(tail_fin, tail_fin_target_pose, Arms.RIGHT, grasp_description),
    PickAndPlaceActionDescription(left_wheel, left_wheel_target_pose, Arms.LEFT, grasp_description),
    PickAndPlaceActionDescription(right_wheel, right_wheel_target_pose, Arms.RIGHT, grasp_description),
    PickAndPlaceActionDescription(back_wheel, back_wheel_target_pose, Arms.LEFT, grasp_description)
)


try:
    with simulated_robot:
        print("Executing main assembly plan...")
        main_plan.perform()
    print("Plan completed successfully!")
    time.sleep(10)

except Exception as e:
    # This block is now for catching any *unexpected* errors.
    print(f"An unexpected error occurred during the plan: {e}")
    time.sleep(5)

finally:
    print("Exiting simulation.")
    world.exit()