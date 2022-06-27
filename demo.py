from Particle_Filter import *
from ekf import EKF_Localization_oneiter, VehicleModel
from pybullet_tools.utils import wait_for_user

# set parameters
num_of_particles = 100

def main(screenshot=False):

     # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))

    def collision(x, y, yaw):
        return collision_fn((x, y, yaw))

    wait_for_user()
    print("==============================================")
    print("The whole process should take approximately 2 minutes.")
    print("The simulation runs EKF and PF together.")
    print("Black balls indicates ground truth poses.")
    print("Red balls indicates PF estimation without collision.")
    print("Blue balls indicates PF estimation with collision.")
    print("Green balls indicates EKF estimation without collision.")
    print("White ball indicates EKF estimation with collision.")
    print("==============================================")
    wait_for_user()


    initial_pose = tuple(get_joint_positions(robots['pr2'], base_joints))
    lines = open("path.txt").read().splitlines()
    v = eval(lines[0])
    w = eval(lines[1])
    ITERATIONS = len(v)

    ## Prevent EKF division by zero
    for i in range(len(v)):
        if w[i]==0.0:
            w[i] = 0.00000001

    
    pf = PF(initial_pose, num_of_particles)
    it = 0
    pf_numbers_of_collision = 0
    error = 0
    odom_previous_pose = initial_pose
    true_previous_pose = initial_pose

    ## Initial setting for EKF ##
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    cur_pose = np.asarray(start_config).reshape(3,1)
    cur_cov = np.zeros(shape=(3,3))
    true_prev_pose = cur_pose


    while(it < ITERATIONS):
        # get input from pre-defined path
        input = [v[it], w[it]]

        print("=============== Iteration", it, "===============")
        # get odom
        odom_cur_pose = Odometry(input, odom_previous_pose)

        # get sensor
        sensor_mean, ground_truth = SensorModel(input, robots['pr2'], base_joints, true_previous_pose)
        print("Ground_truth pose: ", ground_truth)

        # apply action
        pf.action_model(odom_cur_pose)

        # update weight
        pf.update_weight(sensor_mean)

        # resample particles
        pf.low_variance_sampling()

        # estimate pose
        pf.estimate_pose()
        print("PF estimated pose", pf.pose)
        
        state_errors = ground_truth - pf.pose
        error += np.linalg.norm(state_errors)

        if(collision_fn(pf.pose)):
            pf_numbers_of_collision += 1
        
        set_joint_positions(robots['pr2'], base_joints, ground_truth)

        
        # EKF Localization :
        
        control = input
        StochasticGain = [5, 5, 5, 5]
        Q = np.array([[0.01, 0.00, 0.00],[0.00, 0.01, 0.00], [0.00, 0.00, 0.01]])
        k = 0.05
        dt = 0.1 # 0.5 second for each control input
        pos_pose, pos_cov, true_pose, collision_time, error = EKF_Localization_oneiter(control, StochasticGain, Q,  dt, robots, base_joints, cur_pose, cur_cov, true_prev_pose , collision,  k = 0.05)
        # Update cur_pose, cur_cov
        cur_pose = pos_pose 
        cur_cov = pos_cov
        true_prev_pose = true_pose

        print("EKF estimated pose: ",pos_pose.reshape(1,-1))

        # PF Collision: Draw blue
        if(collision_fn(pf.pose)):
            draw_sphere_marker((pf.pose[0], pf.pose[1], 1), 0.05, (0, 0, 1, 1))

        # EKF Collision: Draw white
        if(collision(pos_pose[0], pos_pose[1], pos_pose[2])):
            draw_sphere_marker((pos_pose[0], pos_pose[1], 1), 0.05, (1, 1, 1, 1))


        if(it % 10 == 0):

            draw_sphere_marker((ground_truth[0], ground_truth[1], 1), 0.05, (0, 0, 0, 1))

            # Draw PF ball
            if(not collision_fn(pf.pose)):
                draw_sphere_marker((pf.pose[0], pf.pose[1], 1), 0.05, (1, 0, 0, 1))

            # Draw EKF ball
            if(not collision(pos_pose[0], pos_pose[1], pos_pose[2])):
                draw_sphere_marker((pos_pose[0], pos_pose[1], 1), 0.05, (0, 1, 0, 1))


        odom_previous_pose = odom_cur_pose
        true_previous_pose = ground_truth
        it += 1

        print("\n")

    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()