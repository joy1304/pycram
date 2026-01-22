import time
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
from pycram.robot_plans.actions.core import PickUpAction
from pycram.recovery.error_recovery import with_error_handling
from pycram.failures import PlanFailure, LowLevelFailure


mock_grasp_data = {
    'grasps': [
        {'id': 'front', 'position': {'x': 0, 'y': -0.05, 'z': 0}, 
         'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1}},
        {'id': 'top', 'position': {'x': 0, 'y': 0, 'z': 0.05}, 
         'orientation': {'x': 0, 'y': 0.707, 'z': 0, 'w': 0.707}},
        {'id': 'left', 'position': {'x': -0.05, 'y': 0, 'z': 0}, 
         'orientation': {'x': 0, 'y': 0, 'z': -0.707, 'w': 0.707}},
    ]
}


@with_error_handling
def error_logging_demo(grasp_data=None, grasp_classifier=None, guardian=None, **_):
    loginfo("Scenario: Multiple failures to demonstrate error tracking")
    
    object_names = ["obj_1", "obj_2"]
    
    for i, obj_name in enumerate(object_names, 1):
        loginfo(f"\n--- Attempt {i}: {obj_name} ---")
        
        obj_desig = BelieveObject(names=[obj_name])
        obj = obj_desig.resolve()
        
        pickup_action = PickUpAction(
            object_designator=obj,
            arm=Arms.RIGHT,
            grasp_description='top'
        )
        
        context = {
            'object_designator': obj,
            'arm': Arms.RIGHT,
            'attempt_number': i,
        }
        
        try:
            pickup_action.perform()
            loginfo(f"✓ Pickup succeeded for {obj_name}")
        except LowLevelFailure as e:
            loginfo(f"✗ Pickup failed for {obj_name}: {type(e).__name__}")
            try:
                guardian.handle_error(e, pickup_action, context)
                loginfo(f"✓ Recovery completed for {obj_name}")
            except PlanFailure as pf:
                logwarn(f"Recovery failed for {obj_name}: {pf}")

    loginfo("ERROR LOG SUMMARY")
    
    if not guardian.error_log:
        loginfo("No errors were logged")
    else:
        loginfo(f"Total errors handled: {len(guardian.error_log)}")
        
        for idx, log_entry in enumerate(guardian.error_log, 1):
            loginfo(f"  Type: {log_entry['error_type']}")
            loginfo(f"  Action: {log_entry['failed_action']}")
            loginfo(f"  Message: {log_entry['error_message'][:80]}...")  # Truncate long messages
            loginfo(f"  Timestamp: {log_entry['timestamp']:.2f}")

    
    # Statistics
    error_types = {}
    for entry in guardian.error_log:
        error_type = entry['error_type']
        error_types[error_type] = error_types.get(error_type, 0) + 1
    
    if error_types:
        loginfo("Error Statistics:")
        for error_type, count in error_types.items():
            loginfo(f"  {error_type}: {count} occurrence(s)")

if __name__ == '__main__':
    world = BulletWorld(WorldMode.GUI)
    
    # Spawn Tracy robot
    robot = Object("tracy", Robot, "tracy.urdf", 
                   pose=PoseStamped.from_list([-0.5, 1.5, 0.7], [0, 0, 0, 1]))
    
    # Configure grasps
    robot_description = RobotDescription.current_robot_description
    right_gripper = robot_description.get_arm_chain(Arms.RIGHT).end_effector
    
    for grasp in mock_grasp_data['grasps']:
        orientation_dict = grasp['orientation']
        orientation_list = [orientation_dict['x'], orientation_dict['y'], 
                          orientation_dict['z'], orientation_dict['w']]
        right_gripper.grasps[grasp['id']] = orientation_list
    
    loginfo("Grasps configured")
    
    # Spawn objects at reachable positions
    for i in range(2):
        obj = Object(f"obj_{i+1}", PhysicalObject, "block_blue.urdf", 
                    color=Color(0.2*i, 0.5, 1-0.2*i, 1))
        obj.set_pose(PoseStamped.from_list([0.12 + i*0.1, 1.38, 0.75]))
    
    world.step()
    
    with simulated_robot:
        try:
            error_logging_demo(grasp_data=mock_grasp_data)
            loginfo("DEMO COMPLETED SUCCESSFULLY")
            loginfo("Error logging system demonstrated")
        except PlanFailure as e:
            logwarn(f"DEMO FAILED: {e}")
    
    loginfo("Closing simulation in 5 seconds...")
    time.sleep(5)
    world.exit()
