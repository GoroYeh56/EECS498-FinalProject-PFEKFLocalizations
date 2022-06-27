import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
import random
import math
from scipy.stats import multivariate_normal
import copy
import matplotlib.pyplot as plt


d_t = 0.1
# INTERATIONS = 1000
draw_particles = False
num_of_particles = 100
sensor_cov_k = 0.05
repeat_times = 1

class PF():
    def __init__(self, initial_pose, num_of_particles):
        self.k = 0.1
        self.sensor_cov = np.eye(3) * self.k
        self.draw_spacing = 5
        self.particle_number = num_of_particles
        # self.particles = [self.particle] * self.particle_number
        self.particles = []

        for i in range(self.particle_number):
            p = particle()
            self.particles.append(p)

        self.pervious_odom_pose = initial_pose
        self.k1 = 0.5
        self.k2 = 0.5
        self.pose = [0, 0, 0]
        self.pose[0] = initial_pose[0]
        self.pose[1] = initial_pose[1]
        self.pose[2] = initial_pose[2]

        # initialize particles
        for p in self.particles:
            p.x = initial_pose[0]
            p.y = initial_pose[1]
            p.theta = initial_pose[2]
            p.weight = 1.0/self.particle_number

        # for p in self.particles:
        #     print(p.weight)


    # def set_partilce_number(self, K):
    #     self.particle_number = K
    #     self.particles = []

    #     for i in range(self.particle_number):
    #         p = particle()
    #         self.particles.append(p)

    def set_showing_particles(self, n):
        self.draw_spacing = self.particle_number / n

    def print_weight_sum(self):
        sum = 0
        for p in self.particles:
            # print(p.x, p.y, p.theta, p.weight)
            sum += p.weight
        print("sum of weight: ", sum)

    def print_particles(self):
        # print("all", len(self.particles))
        sum = 0
        for p in self.particles:
            print(p)
            # print(p.x, p.y, p.theta, p.weight)
            sum += p.weight
        print("sum of weight: ", sum)

    def action_model(self, odom_pose):
        delta_x = odom_pose[0] - self.pervious_odom_pose[0]
        delta_y = odom_pose[1] - self.pervious_odom_pose[1]
        delta_theta = odom_pose[2] - self.pervious_odom_pose[2]

        rot1 = self.angle_diff(np.arctan2(delta_y, delta_x), self.pervious_odom_pose[2])
        direction = 1.0
        if(abs(rot1) > np.pi/2):
            rot1 = self.angle_diff(np.pi, rot1)
            direction = -1.0

        trans = math.sqrt(delta_x**2 + delta_y**2)
        trans = trans * direction
        rot2 = self.angle_diff(delta_theta, rot1)

        rot1_std = math.sqrt(self.k1 * abs(rot1))
        trans_std = math.sqrt(self.k2 * abs(trans))
        rot2_std = math.sqrt(self.k1 * abs(rot2))


        # update particles
        for p in self.particles:
            sampled_rot1 = np.random.normal(rot1, rot1_std)
            sampled_trans = np.random.normal(trans, trans_std)
            sampled_rot2 = np.random.normal(rot2, rot2_std)
            p.x += sampled_trans * np.cos(p.theta + sampled_rot1)
            p.y += sampled_trans * np.sin(p.theta + sampled_rot1)
            p.theta = self.wrap_to_pi(p.theta + sampled_rot1 + sampled_rot2)
        
        self.pervious_odom_pose = odom_pose



    def update_weight(self, sensor_mean):
        weight_sum = 0
        # print(sensor_mean)
        rv = multivariate_normal(sensor_mean, self.sensor_cov)
        for p in self.particles:
            p.weight = rv.pdf([p.x, p.y, p.theta])
            weight_sum += p.weight
        # for p in self.particles:
        #     rv = multivariate_normal([p.x, p.y, p.theta], self.sensor_cov)
        #     p.weight = rv.pdf(sensor_mean)
        #     weight_sum += p.weight
            # print(weight)

        # normalize weight
        # for p in self.particles:
        #     weight_sum += p.weight
        for p in self.particles:
            p.weight /= weight_sum

        
    
    def estimate_pose(self):
        # initialize var
        x_mean = 0
        y_mean = 0
        cos_theta_mean = 0
        sin_theta_mean = 0

        # calculate pose
        for p in self.particles:
            x_mean += p.weight * p.x
            y_mean += p.weight * p.y
            cos_theta_mean += p.weight * np.cos(p.theta)
            sin_theta_mean += p.weight * np.sin(p.theta)

        self.pose[0] = x_mean
        self.pose[1] = y_mean
        self.pose[2] = np.arctan2(sin_theta_mean, cos_theta_mean)



    def low_variance_sampling(self):
        weight_sum = 0
        new_particles = copy.deepcopy(self.particles)
        r = random.uniform(0, 1.0 / self.particle_number) 
        # print("r", r)       
        c = self.particles[0].weight
        # print("c", c)
        i = 0
        for m in range(self.particle_number):
            U = r + m * (1 / self.particle_number)
            # print("U", U)
            # while(1):
            #     pass
            while(U > c and i < self.particle_number - 1):
            # while(U > c):
                i += 1
                # i = min(i, self.particle_number - 1)
                c += self.particles[i].weight
            new_particles[m] = copy.deepcopy(self.particles[i])
            weight_sum += self.particles[i].weight
        self.particles = new_particles

        for p in self.particles:
            p.weight /= weight_sum
        
        if(draw_particles):
            self.draw_particles()


    def wrap_to_pi(self, angle):
        if(angle < -np.pi):
            while(angle < -np.pi):
                angle += 2.0*np.pi
        elif(angle > np.pi):
            while(angle > np.pi):
                angle -= 2.0*np.pi
        return angle

    def angle_diff(self, theta1, theta2):
        diff = theta1 - theta2
        if(diff > np.pi):
            diff = diff - 2*np.pi
        elif(diff < -np.pi):
            diff = diff + 2*np.pi
        return diff

    def draw_particles(self):
        # self.draw_spacing = 5
        for i in range(self.particle_number):
            if(i % self.draw_spacing == 0):
                x = self.particles[i].x
                y = self.particles[i].y
                draw_sphere_marker((x, y, 1), 0.01, (1, 0, 0, 1))


class particle():
    def __init__(self):
        self.x = 0
        self.y = 0 
        self.theta = 0
        self.weight = 0

# odometry
def Odometry(input, previous_pose):
    c1 = 0.5
    c2 = 0.5
    c3 = 0.5
    c4 = 0.5
    delta_t = d_t
    v = input[0]
    w = input[1]
    R = np.array([[(c1* abs(v)+c2*abs(w))**2, 0], [0, (c3* abs(v)+c4*abs(w))**2]])
    rand = np.random.multivariate_normal(input, R)
    v_actual = rand[0]
    w_actual = rand[1]
    # print("v_actual", v_actual)
    # print("w_actual", w_actual)
    delta_x = v_actual*delta_t*np.cos(previous_pose[2] + w_actual*delta_t)
    delta_y = v_actual*delta_t*np.sin(previous_pose[2] + w_actual*delta_t)
    delta_theta = w_actual * delta_t
    delta_pose = np.array([delta_x, delta_y, delta_theta])
    new_pose = previous_pose + delta_pose
    return new_pose
    
def SensorModel(input, robots, base_joints, previous_pose):
    # 1x3 : [x y theta], 3x3 : cov
    k = sensor_cov_k
    v = input[0]
    w = input[1]
    delta_t = d_t
    delta_x = v*delta_t*np.cos(previous_pose[2] + w*delta_t)
    delta_y = v*delta_t*np.sin(previous_pose[2] + w*delta_t)
    delta_theta = w * delta_t
    delta_pose = np.array([delta_x, delta_y, delta_theta])
    ground_truth = previous_pose + delta_pose

    # sensor_mean = get_joint_positions(robots, base_joints)
    sensor_cov = np.array([[k, 0, 0], [0, k, 0], [0, 0, k]])
    new_pose = np.random.multivariate_normal(ground_truth, sensor_cov)
    return new_pose, ground_truth

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, -1.3, -np.pi/2)))

    # Example use of setting body poses
    # set_pose(obstacles['ikeatable6'], ((0, 0, 0), (1, 0, 0, 0)))

    # Example of draw 
    # draw_sphere_marker((0, 0, 1), 0.1, (1, 0, 0, 1))

    initial_pose = tuple(get_joint_positions(robots['pr2'], base_joints))
    lines = open("path.txt").read().splitlines()
    v = eval(lines[0])
    w = eval(lines[1])
    ITERATIONS = len(v)
    
    plt.ion()
    # plot_axes = plt.subplot(111, aspect='equal')  
    true_xy = np.zeros((2, ITERATIONS))
    estimate_xy = np.zeros((2, ITERATIONS))
    noisy_measurement = np.zeros((2, ITERATIONS))
    odom_xy = np.zeros((2, ITERATIONS))
    particle_xlist = []
    particle_ylist = []

    time_list = []
    error_list = []
    collision_list = []

    for i in range(repeat_times):
        pf = PF(initial_pose, num_of_particles)
        # pf.set_showing_particles(20)


        it = 0
        numbers_of_collision = 0
        error = 0
        odom_previous_pose = initial_pose
        true_previous_pose = initial_pose

        # print(INTERATIONS)
        start_time = time.time()
        while(it < ITERATIONS):
            
            # get input from pre-defined path
            input = [v[it], w[it]]

            # print("=============== Iteration", it, "===============")
            # get odom
            odom_cur_pose = Odometry(input, odom_previous_pose)
            # print("odometry pose", odom_cur_pose)
            odom_xy[:, it] = np.squeeze(odom_cur_pose[0:2])

            # get sensor
            sensor_mean, ground_truth = SensorModel(input, robots['pr2'], base_joints, true_previous_pose)
            # print("Sensor meansurement: ", sensor_mean)
            # print("Ground_truth: ", ground_truth)

            true_xy[:, it] = np.squeeze(ground_truth[0:2])
            noisy_measurement[:, it] = np.squeeze(sensor_mean[0:2])

            # apply action
            pf.action_model(odom_cur_pose)

            # update weight
            pf.update_weight(sensor_mean)

            # resample particles
            pf.low_variance_sampling()

            if(it % 100 == 0):
                for p in pf.particles:
                    particle_xlist.append(p.x)
                    particle_ylist.append(p.y)
            # pf.print_weight_sum()

            # estimate pose
            pf.estimate_pose()
            estimate_xy[:, it] = np.squeeze(pf.pose[0:2])

            # print("pose estimation", pf.pose)
            
            state_errors = ground_truth - pf.pose
            error += np.linalg.norm(state_errors)

            if(collision_fn(pf.pose)):
                # draw_sphere_marker((pf.pose[0], pf.pose[1], 1), 0.05, (1, 1, 1, 1))
                numbers_of_collision += 1
                # print("Collision!!!!!!!!!!!!!!!!!!!!!!!!!")
            
            set_joint_positions(robots['pr2'], base_joints, ground_truth)
            
            # if(it % 10 == 0):

            #     draw_sphere_marker((ground_truth[0], ground_truth[1], 1), 0.05, (0, 0, 0, 1))
                
            #     draw_sphere_marker((sensor_mean[0], sensor_mean[1], 1), 0.05, (0, 0, 1, 1))

            #     draw_sphere_marker((odom_cur_pose[0], odom_cur_pose[1], 1), 0.05, (0, 1, 0, 1))

            #     if(not collision_fn(pf.pose)):
            #         draw_sphere_marker((pf.pose[0], pf.pose[1], 1), 0.05, (1, 0, 0, 1))


            odom_previous_pose = odom_cur_pose
            true_previous_pose = ground_truth
            it += 1

            # print("\n")

        # print("Goal: ", pf.pose)
        execution_time = time.time() - start_time
        print("Time: ", execution_time)
        print("Numbers of collision points: ", numbers_of_collision)
        print("Sum of squared error: ", error)
        time_list.append(execution_time)
        error_list.append(error)
        collision_list.append(numbers_of_collision)

        plt.figure(1)
        plt.scatter(true_xy[0,0:ITERATIONS], true_xy[1,0:ITERATIONS], c='b',s=5, label='ground truth')
        plt.scatter(noisy_measurement[0,0:ITERATIONS], noisy_measurement[1,0:ITERATIONS],c='g',s=5, label='sensor measurement')
        plt.scatter(estimate_xy[0,0:ITERATIONS], estimate_xy[1,0:ITERATIONS],c='r', s=5, label='pose estimation')
        # plt.scatter(particle_xlist, particle_ylist, c='black', marker='X', s=10)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("PF True Pose & Pose Estimation")


        plt.figure(2)
        plt.scatter(true_xy[0,0:ITERATIONS], true_xy[1,0:ITERATIONS], c='b',s=5, label='ground truth')
        # plt.scatter(noisy_measurement[0,0:ITERATIONS], noisy_measurement[1,0:ITERATIONS],c='g',s=5)
        plt.scatter(estimate_xy[0,0:ITERATIONS], estimate_xy[1,0:ITERATIONS],c='r', s=5, label='pose estimation')
        plt.scatter(particle_xlist, particle_ylist, c='black', marker='X', s=10, label='particles')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("PF Particles & Pose Estimation")
        plt.legend()

        plt.figure(3)
        plt.scatter(true_xy[0,0:ITERATIONS], true_xy[1,0:ITERATIONS], c='b',s=5, label='ground truth')
        # plt.scatter(noisy_measurement[0,0:ITERATIONS], noisy_measurement[1,0:ITERATIONS],c='g',s=5)
        plt.scatter(estimate_xy[0,0:ITERATIONS], estimate_xy[1,0:ITERATIONS],c='r', s=5, label='pose estimation')
        plt.scatter(odom_xy[0,0:ITERATIONS], odom_xy[1,0:ITERATIONS], c='orange', s=5, label='odometry')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("PF Odometry & Pose Estimation")
        plt.legend()

        plt.pause(.001)
        plt.ioff()
        plt.show()

    # print("Average time: ", sum(time_list) / repeat_times)
    # print("Average numbers of collision points: ", sum(collision_list) / repeat_times)
    # print("Average sum of squared error: ", sum(error_list) / repeat_times)

    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()