import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, find, get_closest_edge_point, get_joint_positions, is_point_in_polygon, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time


### YOUR IMPORTS HERE ###
from utils import load_env, get_collision_fn_PR2, execute_trajectory, draw_sphere_marker, draw_line
from pybullet_tools.utils import connect, disconnect, wait_if_gui, wait_for_user, joint_from_name, get_joint_info, get_link_pose, link_from_name
from queue import PriorityQueue
import math

from numpy.random import multivariate_normal as mvnrnd
from math import sin, cos


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

"""

TODO:

1. If collide: plot a green ball or sth
3. Add another robot for visualization

"""

PLOT_ITER = 10

#this plots the covariance matrix as an ellipsoid at 2*sigma
def plot_cov(mean,cov, color, plot_axes):
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    # print("arrcos ",np.arccos(v[0, 0]))

    if abs(v[0,0]) > 1:
        # print("v[0,0] ", v[0,0])
        return 

    if np.iscomplex(np.arccos(v[0,0])):
        # print("is complex! don't plot eclipse")
        return

    try:
        ell = Ellipse(xy=mean,
                width=lambda_[0]*2, height=lambda_[1]*2,
                angle=np.rad2deg(np.arccos(v[0, 0])))
        #ell.set_facecolor('none')
        ell.set_facecolor((1.0, 1.0, 1.0, 0))
        ell.set_edgecolor((0, 0, 0, 1))
        plot_axes.add_artist(ell)
        plt.scatter(mean[0,0],mean[1,0],color=color,s=5)
    except:
        # print("rad2deg error. don't plot eclipse")
        pass


#########################


def VehicleModel(c1, c2, c3, c4, dt, controls, prev_pose):
    v, w = controls
    # Disturb the actual input
    r = np.array([ [ (c1*abs(v)+c2*abs(w))**2 ,         0  ], 
                   [        0,                  ( c3*abs(v)+c4*abs(w))**2  ] ])
    ran = mvnrnd(np.array([v, w]), r)
    # v_actual = ran[0]
    # w_actual = ran[1]
    v_actual = v
    w_actual = w

    # print("v, w actual: ", v_actual, w_actual)
    # print("prev_pose[2] " , prev_pose[2])

    # should be 3x1 vector
    change_pose = np.array([
        [ v_actual*dt* cos(prev_pose[2] + w_actual*dt) ],
        [ v_actual*dt* sin(prev_pose[2] + w_actual*dt)],
        [ w_actual * dt]
    ]
    )

    new_pose = prev_pose + change_pose
    return new_pose


def SensorModel(robots, base_joints):
    # 3x1 : [x y theta]'
    true_pose = get_joint_positions(robots['pr2'], base_joints)
    # return true location by get_joint_position(base)
    return np.asarray(true_pose).reshape(3,1) # return a 3x1 vector 



def EKF_Localization_oneiter(control, StochasticGain, Q,  dt, robots, base_joints, cur_pose, cur_cov,  true_prev_pose, collision,  k = 0.05):

    """
        @ arguments:
         - controls: control inputs (lists of [v w])
         - StochasticGain : [c1, c2, c3, c4]
         - Q : covariance matrix of the sensor model
         - dt : sampling time (default = 0.1)
         - robots: pr2
         - base_joints: [0, 1, 2] => x, y, theta
         - cur_pose: current pose
         - cur_cov: current cov matrix P
         - ITER : number of iteration
         - collision: a function to check whether pose (x, y, yaw) would collide with obstacles
         - k : the diagonal term of the covariance matrix for noisy measurement

    """

    c1, c2, c3, c4 = StochasticGain  
    prev_pose = cur_pose
    prev_cov = cur_cov

    collision_times = 0

    # print("Initial pose: ", init_pose)

    ###YOUR CODE HERE###
 
    # This is our current ground_truth pose!
    v,w = control
    true_cur_pose = VehicleModel(c1, c2, c3, c4, dt, control, true_prev_pose)
    # Move PR2 Actually
    waypoint = [true_cur_pose[0].item(0), true_cur_pose[1].item(0), true_cur_pose[2].item(0)]
    # print("True Pose:  ", waypoint)
    set_joint_positions(robots['pr2'], base_joints, waypoint)

    ########## EKF Algorithm here #################

    # Motion Model : Gt, Vt and Mt
    theta = prev_pose[2]

    Gt = np.array([[1, 0, (-v/w)*cos(theta) + (v/w)*cos(theta+w*dt)],
                [0, 1, (-v/w)*sin(theta) + (v/w)*sin(theta+w*dt)],
                [0, 0, 1]])
    Vt = np.array([[ (-sin(theta)+sin(theta+w*dt))/w,  v*(sin(theta)-sin(theta+w*dt))/(w**2) + (v*cos(theta+w*dt)*dt)/w ],
                [ (cos(theta)-cos(theta+w*dt))/w,  -v*(cos(theta)-cos(theta+w*dt))/(w**2) + (v*sin(theta+w*dt)*dt)/w ],
                [0, dt]])

    Mt = np.array([ [c1*v**2 + c2*w**2,              0            ],
                [  0,                       c3*v**2 + c4*w**2 ] ])

    Matrix = np.array([[(-v/w)*sin(theta)+(v/w)*sin(theta+w*dt)],
                        [(v/w)*cos(theta)-(v/w)*cos(theta+w*dt)],
                        [w*dt]])

    prior_pose =  prev_pose +  Matrix
    prior_cov = Gt @ prev_cov @ Gt.T + Vt @ Mt @ Vt.T

    # print("Prior pose: ", prior_pose.reshape(1,3))

    # Kamlan Gain
    C = np.eye(3)
    H = C        
    K = prior_cov * H.T * np.linalg.inv( (H @ prior_cov @ H.T) + Q)

    # Correction using true_pose
    sensor_mean = np.array([0, 0, 0])


    sensor_cov = np.array([[k, 0, 0],[0, k, 0],[0, 0, k]]) 
    sensor_noise = mvnrnd(sensor_mean, sensor_cov).reshape(3,-1) # 3x1 vector

    # return true location by get_joint_position(base)
    true_pose = np.asarray(get_joint_positions(robots['pr2'], base_joints)).reshape(3,1) + sensor_noise

    pos_pose = prior_pose + K @ (true_pose - (C @ prior_pose))
    pos_cov = (np.eye(3) - K@H) @ prior_cov


    ###############################################

    # print("pos_pose ", pos_pose.reshape(1,3))


    # ### Collision check for estimated pose ###
    x = pos_pose[0]
    y = pos_pose[1]
    yaw = pos_pose[2]
    if collision(x, y, yaw):
        collision_times +=1
        print("Collision!")
        collision_color = (1, 1, 1, 1)
        sphere_radius = 0.05
        # draw_sphere_marker((x, y, 1), sphere_radius, collision_color)

    error = np.linalg.norm(true_cur_pose - pos_pose)


    return pos_pose, pos_cov, true_cur_pose, collision_times, error # return final pos





def EKF_Localization(controls, StochasticGain, Q,  dt, robots, base_joints, init_pose,init_cov, ITER, collision,  k = 0.05):

    """
        @ arguments:
         - controls: control inputs (lists of [v w])
         - StochasticGain : [c1, c2, c3, c4]
         - Q : covariance matrix of the sensor model
         - dt : sampling time (default = 0.1)
         - robots: pr2
         - base_joints: [0, 1, 2] => x, y, theta
         - init_pose
         - init_cov
         - ITER : number of iteration
         - collision: a function to check whether pose (x, y, yaw) would collide with obstacles
         - k : the diagonal term of the covariance matrix for noisy measurement

    """

    c1, c2, c3, c4 = StochasticGain  
    prev_pose = init_pose
    prev_cov = init_cov
    true_prev_pose = init_pose

    pos_poses = []
    pos_covs = []
    true_poses =[]
    collision_times = 0

    # print("Initial pose: ", init_pose)


    ###YOUR CODE HERE###
    #go through each measurement and action...
    #and estimate the state using the Kalman filter
    
    plt.ion()
    plot_axes = plt.subplot(111, aspect='equal')       
    true_xy =  np.zeros( (2, ITER))
    estimated_states = np.zeros((2,ITER)) # [x y]' * number of iteration
    noisy_measurements = np.zeros((2,ITER))

    errors = 0

    for it in range(ITER):
        print('\n\n-------------- Iteration ', it, ' -------------------\n')
        # This is our current ground_truth pose!
        control = controls[it]
        v,w = control
        true_cur_pose = VehicleModel(c1, c2, c3, c4, dt, control, true_prev_pose)
        # Move PR2 Actually
        waypoint = [true_cur_pose[0].item(0), true_cur_pose[1].item(0), true_cur_pose[2].item(0)]
        print("True Pose:  ", waypoint)
        set_joint_positions(robots['pr2'], base_joints, waypoint)
        true_poses.append(true_cur_pose)
        true_prev_pose = true_cur_pose
        ########## EKF Algorithm here #################

        # Motion Model : Gt, Vt and Mt
        theta = prev_pose[2]

        Gt = np.array([[1, 0, (-v/w)*cos(theta) + (v/w)*cos(theta+w*dt)],
                    [0, 1, (-v/w)*sin(theta) + (v/w)*sin(theta+w*dt)],
                    [0, 0, 1]])
        Vt = np.array([[ (-sin(theta)+sin(theta+w*dt))/w,  v*(sin(theta)-sin(theta+w*dt))/(w**2) + (v*cos(theta+w*dt)*dt)/w ],
                    [ (cos(theta)-cos(theta+w*dt))/w,  -v*(cos(theta)-cos(theta+w*dt))/(w**2) + (v*sin(theta+w*dt)*dt)/w ],
                    [0, dt]])

        Mt = np.array([ [c1*v**2 + c2*w**2,              0            ],
                    [  0,                       c3*v**2 + c4*w**2 ] ])

        Matrix = np.array([[(-v/w)*sin(theta)+(v/w)*sin(theta+w*dt)],
                         [(v/w)*cos(theta)-(v/w)*cos(theta+w*dt)],
                         [w*dt]])
    
        prior_pose =  prev_pose +  Matrix
        prior_cov = Gt @ prev_cov @ Gt.T + Vt @ Mt @ Vt.T

        print("Prior pose: ", prior_pose.reshape(1,3))

        # Kamlan Gain
        C = np.eye(3)
        H = C        
        K = prior_cov * H.T * np.linalg.inv( (H @ prior_cov @ H.T) + Q)

        # Correction using true_pose
        sensor_mean = np.array([0, 0, 0])


        sensor_cov = np.array([[k, 0, 0],[0, k, 0],[0, 0, k]]) 
        sensor_noise = mvnrnd(sensor_mean, sensor_cov).reshape(3,-1) # 3x1 vector
        true_pose = SensorModel(robots, base_joints) + sensor_noise
        true_xy[:,it] = np.squeeze(true_cur_pose[0:2])
        noisy_measurements[:,it] = np.squeeze(true_pose[0:2])
        
        pos_pose = prior_pose + K @ (true_pose - (C @ prior_pose))
        pos_cov = (np.eye(3) - K@H) @ prior_cov

        # pos_pose = prior_pose 
        # pos_cov = prior_cov

        ###############################################

        print("pos_pose ", pos_pose.reshape(1,3))

        # update prev_pose, prev_cov
        prev_pose = pos_pose
        prev_cov = pos_cov

        pos_poses.append(pos_pose)
        pos_covs.append(pos_cov)


        # ### Collision check for estimated pose ###
        x = pos_pose[0]
        y = pos_pose[1]
        yaw = pos_pose[2]
        if collision(x, y, yaw):
            collision_times +=1
            print("Collision! at iteration "+str(it))
            collision_color = (1, 1, 1, 1)
            sphere_radius = 0.05
            # draw_sphere_marker((x, y, 1), sphere_radius, collision_color)


        ### Draw estimated(red) and true(blue) ball
        # if it% PLOT_ITER ==0:
        #     sphere_radius = 0.05
        #     sphere_color_r = (1, 0, 0, 1) # R, G, B, A 

        #     if not collision(pos_pose[0], pos_pose[1], pos_pose[2]):
        #         draw_sphere_marker((pos_pose[0], pos_pose[1], 1), sphere_radius, sphere_color_r)
    
        #     sphere_color_g = (0, 0, 1, 1) # R, G, B, A 
        #     draw_sphere_marker((true_pose[0], true_pose[1], 1), sphere_radius, sphere_color_g)  
        #     sphere_color_g = (0, 0, 0, 1) # R, G, B, A 
        #     draw_sphere_marker((true_cur_pose[0], true_cur_pose[1], 1), sphere_radius, sphere_color_g)              


        errors += np.linalg.norm(true_cur_pose - pos_pose)

        #store the result
        estimated_states[:,it] = np.squeeze(pos_pose[0:2])  
        #draw covariance every 3 steps (drawing every step is too cluttered)
        if it%10==0:
            plot_cov(pos_pose, pos_cov, 'black', plot_axes)

    true_xy = np.asarray(true_xy).reshape(2, -1)


    ############### End N Iteration, generate plot! #################
    #compute the error between your estimate and ground trutㄥ
    state_errors = estimated_states[:,0:ITER] - true_xy[:,0:ITER]
    total_error=np.sum(np.linalg.norm(state_errors, axis=0))
    # print("Total Error: %f"%total_error)

    #draw the data and result
    plt.scatter(true_xy[0,0:ITER], true_xy[1,0:ITER], c='b',s=2, label='ground truth')
    plt.scatter(noisy_measurements[0,0:ITER], noisy_measurements[1,0:ITER],c='g',s=2, label='measurements')
    plt.scatter(estimated_states[0,0:ITER], estimated_states[1,0:ITER],c='r', s=2, label='pose estimation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    # plt.title('EKF Pose Estimation vs True Pose')
    plt.title('EKF Pose Estimation vs True Pose (No Correction)')

    plt.pause(.001)
    plt.ioff()
    plt.show()


    return pos_poses, pos_covs, true_poses, collision_times, errors # return final pos


def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway-ekf.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']] # [0,1,2]
    # Example use of setting body poses
    # set_pose(obstacles['ikeatable6'], ((0, 0, 0), (1, 0, 0, 0)))
    
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, -1.3, -np.pi/2) # x, y, theta
    # goal_config = (2.6, 1, -np.pi/2) # x, y, theta
    path = []
    start_time = time.time()
    ### YOUR CODE HERE ###
 
    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))

    def collision(x, y, yaw):
        return collision_fn((x, y, yaw))



    ######### Parameters ############
    # StochasticGain = [0.05, 0.05, 0.05, 0.05]
    # Q = np.array([[0.02, 0.00, 0.00],[0.00, 0.02, 0.00], [0.00, 0.00, 0.02]])
    # StochasticGain = [0.5, 0.5, 0.5, 0.5]
    StochasticGain = [5, 5, 5, 5]
    Q = np.array([[0.01, 0.00, 0.00],[0.00, 0.01, 0.00], [0.00, 0.00, 0.01]])
    k = 0.05
    dt = 0.1 # 0.5 second for each control input
    
    init_pose = np.asarray(start_config).reshape(3,1) # 3x1 .# tuple?
    init_cov =  np.zeros(shape=(3,3))
    
    ######## Control Inputs (Pre defined path) ##########
    # v = [0.1 for i in range(ITERATION) ]
    # w = [0.01 for i in range(ITERATION)]
    lines = open("path.txt").read().splitlines()
    v = eval(lines[0])
    w = eval(lines[1])
    # print("type v ", type(v))
    # print("len (v) ", len(v))
    controls = []
    for i in range(len(v)):
        if w[i]==0.0:
            w[i] = 0.00000001
        controls.append([v[i],w[i]])

    ITERATION = len(v)
    # ITERATION = 100


    sum_exec_time = 0
    sum_col_times=0
    sum_errors=0


    AVERAGE_NUMBER = 1

    for _ in range(AVERAGE_NUMBER):
        start_time = time.time()
        pos_poses, pos_covs, true_poses, collision_times, errors = EKF_Localization(controls, StochasticGain, Q,  dt, robots, base_joints, init_pose,init_cov, ITERATION, collision, k)
        end_time = time.time()       

        sum_exec_time += end_time - start_time
        sum_col_times += collision_times
        sum_errors += errors

    
    print("Average exec time: ", sum_exec_time/AVERAGE_NUMBER)
    print("Average number of collision: ", sum_col_times/AVERAGE_NUMBER)
    print("Average error " , sum_errors/AVERAGE_NUMBER)


    print("\n =============== End EKF Localization ================= \n")
    print("Final True Pose: ", true_poses[-1].reshape(1,3))
    print("Final Estimated Pose: ", pos_poses[-1].reshape(1,3))
    print("Final Covariance: ", pos_covs[-1])
    # print("Collision times: ", collision_times)
    # print("execution time: ", end_time - start_time)


    # ######################
    # Execute planned path
    path = []
    for i in range(100):
        if i*10 < len(true_poses):
            path.append(true_poses[(i+1)*10])
        else:
            path.append(true_poses[-1])
            break
    # path = true_poses
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()






if __name__ == '__main__':
    main()


"""
    # IT1 = 120
    # IT2 = 55 # Forward
    # IT3 = 120 # Turn Right
    # IT4 = 90 # Forward
    # IT5 = 110 # Turn Right
    # IT6 = 60  # Forward
    # IT7 = 120 # Turn Left
    # IT8 = 40  # Forward
    # IT9 = 120 # Turn Left
    # IT10= 30 # Forward
    # # IT11 = 120 # Turn Left
    # # IT12 = 40 # Forward


    # ITERATION = IT1 + IT2 + IT3 + IT4 + IT5 + IT6 + IT7 + IT8
    # v1 = [0.001 for i in range(IT1) ]
    # w1 = [0.03 for i in range(IT1)]
    # v2 = [0.1 for i in range(IT2)]
    # w2 = [0.000001 for i in range(IT2)]
    # v3 = [0.0 for i in range(IT3)]
    # w3 = [-0.03 for i in range(IT3)]
    # v4 = [0.1 for i in range(IT4)]
    # w4 = [0.000001 for i in range(IT4)]

    # v5 = [0.001 for i in range(IT5) ]
    # w5 = [-0.03 for i in range(IT5)]
    # v6 = [0.1 for i in range(IT6)]
    # w6 = [0.000001 for i in range(IT6)]
    # v7 = [0.0 for i in range(IT7)]
    # w7 = [0.03 for i in range(IT7)]
    # v8 =  [0.1 for i in range(IT8)]
    # w8 = [0.000001 for i in range(IT8)]

    # v9 =  [0.0 for i in range(IT9)]
    # w9 =  [0.03 for i in range(IT9)]
    # v10 = [0.1 for i in range(IT10)]
    # w10 = [0.000001 for i in range(IT10)]


    # v = v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10
    # w = w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + w10
"""