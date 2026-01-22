"""
Simple Tracy Demo with Reachability Validation
"""

from pycram.datastructures.dataclasses import Color
from pycram.process_module import simulated_robot
from pycram.reachability_validator.tracy_reachability_validator import TracyReachabilityValidator, \
    check_object_reachable
from pycram.worlds.bullet_world import BulletWorld
from pycram.robot_plans.actions import *
from pycram.designators.object_designator import *
from pycram.datastructures.enums import ObjectType, Arms, WorldMode
import time

from pycrap.ontologies import Robot


box1Start = PoseStamped.from_list([0.25, 1.25, 0.7])  # Close to tracy
box2Start = PoseStamped.from_list([1.5, 1.5, 0.7])   # Out of reach
box3Start = PoseStamped.from_list([0.25, 1.75, 0.7])  # Close to tracy


world = BulletWorld(WorldMode.GUI)

tracy = Object("tracy", Robot, "tracy.urdf", 
               pose=PoseStamped.from_list([-0.5, 1.5, 0.7], [0, 0, 0, 1]),  # Changed to origin!
               ignore_cached_files=True)

# Load objects
box1 = Object("box1", PhysicalObject, "block_blue.urdf", 
              pose=box1Start, color=Color(1, 0, 0, 1))
box2 = Object("box2", PhysicalObject, "block_green.urdf", 
              pose=box2Start, color=Color(0, 1, 0, 1))
box3 = Object("box3", PhysicalObject, "block_red.urdf", 
              pose=box3Start, color=Color(0, 0, 1, 1))

time.sleep(1)  # Let objects settle

# Create validator
validator = TracyReachabilityValidator(safety_margin=0.10)

# Show Tracy's workspace
print(validator.visualize_workspace(tracy))

# Check reachability of all objects
all_objects = [box1, box2, box3]
reachable_objects = []

for obj in all_objects:
    obj_pose = obj.get_pose()
    
    # Get position
    pos = obj_pose.pose.position
    
    # Check both arms
    is_reachable_right, reason_right = validator.is_object_reachable(
        obj_pose, tracy, "right", verbose=False
    )
    is_reachable_left, reason_left = validator.is_object_reachable(
        obj_pose, tracy, "left", verbose=False
    )
    
    # Get scores
    score_right = validator.get_reachability_score(obj_pose, tracy, "right")
    score_left = validator.get_reachability_score(obj_pose, tracy, "left")
    
    print(f" {obj.name:10} at ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
    print(f"   Right arm: {'Okay' if is_reachable_right else '.':2} (score: {score_right:.3f})")
    if not is_reachable_right:
        print(f"              {reason_right}")
    print(f"   Left arm:  {'Okay' if is_reachable_left else '.':2} (score: {score_left:.3f})")
    if not is_reachable_left:
        print(f"              {reason_left}")
    
    if is_reachable_right or is_reachable_left:
        reachable_objects.append(obj)
        print(f"Can be manipulated!")
    else:
        print(f"NOT REACHABLE by either arm!")
    print()

print(f" Summary: {len(reachable_objects)}/{len(all_objects)} objects are reachable")

# If no objects are reachable
if len(reachable_objects) == 0:
    print(" NO OBJECTS ARE REACHABLE!")
    time.sleep(5)
    world.exit()
    exit(0)

# Demonstrate manipulation with reachability checks
grasp_description = GraspDescription(ApproachDirection.FRONT, VerticalAlignment.TOP)
with simulated_robot:
    
    # Try to pick and place reachable objects
    actions_executed = 0
    
    for obj in reachable_objects:
        # Determine which arm to use
        is_right = check_object_reachable(obj, tracy, "right")
        arm = Arms.RIGHT if is_right else Arms.LEFT
        arm_name = "right" if is_right else "left"
        
        print(f" Attempting to manipulate {obj.name} with {arm_name} arm...")
        
        try:
            # Simple pick and place to nearby location
            target = PoseStamped.from_list([0.25, 1.5, 0.75])
            
            if check_object_reachable(obj, tracy, arm_name):
                print(f"{obj.name} is reachable, executing pick and place...")
                PickAndPlaceActionDescription(obj, target, arm, grasp_description).perform()
                actions_executed += 1
                print(f"Success!")
            else:
                print(f"{obj.name} became unreachable, skipping...")
            
            time.sleep(1)  # Small delay between actions
            
        except Exception as e:
            print(f"Error manipulating {obj.name}: {e}")
            # Continue with next object even if one fails
            continue
    
    print(f"Executed {actions_executed} actions successfully!")

print("DEMO COMPLETED!")

time.sleep(10)
world.exit()
