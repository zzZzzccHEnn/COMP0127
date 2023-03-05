#!/usr/bin/env python3
import numpy as np
import csv
from numpy.linalg import inv
import rospy
import rosbag
import rospkg
import os
import matplotlib.pyplot as plt
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from cw3q2.iiwa14DynKDL import Iiwa14DynamicKDL


class iiwaTrajectory(object):
    def __init__(self):
        '''
        This function initialize the node of ROS, then create the publisher and subscriber,
        and finally create the csv files for data storage.
        '''
        # Initialize node
        rospy.init_node('iiwa_traj_cw3', anonymous=True)

        # Save question number for check in main run method
        self.kdl_iiwa = Iiwa14DynamicKDL()

        # Create trajectory publisher and a checkpoint publisher to visualize checkpoints
        self.traj_pub = rospy.Publisher('/iiwa/EffortJointInterface_trajectory_controller/command', JointTrajectory,
                                        queue_size=5)

        # create the subsrciber to subscribe the current joint states
        self.traj_sub = rospy.Subscriber('/iiwa/joint_states', JointState, self.callback_JointStates)

        # create the csv file to store the data
        path = '/home/zijie/catkin_ws/src/comp0127_lab/cw3/cw3q5/src'
        file_1 = open(path + '/acceleration.csv', 'x')
        file_2 = open(path + '/time.csv', 'x')
        # file_3 = open(path + '/velocity.csv', 'x')



    def run(self):
        '''
        This function calls the 'q5()' function to get the trajectory and then publish to the robot.
        '''
        print("run q5")

        rospy.loginfo("Waiting 5 seconds for everything to load up.")
        rospy.sleep(2.0)

        # define the trajectory and publish to the robot
        traj = self.q5()
        traj.joint_names = ["iiwa_joint_1", "iiwa_joint_2", "iiwa_joint_3", "iiwa_joint_4", "iiwa_joint_5", "iiwa_joint_6", "iiwa_joint_7"]
        traj.header.stamp = rospy.Time.now()
        self.traj_pub.publish(traj)
        

    def q5(self):
        '''
        This function calls the 'load_targers()' function, then create the trajectory of
        the robot according to the message from bag file, and finally return the trajectory to 
        'run()' function.
        '''
        targets_pos, targets_velo, targets_acc = self.load_targets()
        self.time_goal = []

        traj = JointTrajectory()
        t = 0
        dt = 10
        for i in range(targets_pos.shape[1]):
            traj_point = JointTrajectoryPoint()
            # the effort in bag file is empty list, so provide the same empty list to
            # the trajectory
            traj_point.effort = []
            traj_point.positions = targets_pos[:,i]
            traj_point.velocities = targets_velo[i]
            traj_point.accelerations = targets_acc[i]
            t = t + dt
            self.time_goal.append(t)
            traj_point.time_from_start.secs = t
            traj.points.append(traj_point)

        assert isinstance(traj, JointTrajectory)
        return traj



    def load_targets(self):
        '''
        This function load the messages from the bag file
        '''
        rospack = rospkg.RosPack()
        path = rospack.get_path('cw3q5')

        # Initialize arrays for checkpoint transformations and joint positions
        target_joint_positions = np.zeros((7, 3))
        
        # Load path for selected question
        bag = rosbag.Bag(path + '/bag/cw3q5.bag')
        
        hardcoded_joint_targets = []
        velocities = []
        accelerations = []

        for msg in bag.read_messages():
            # print((msg[1].points[0]))
            for i in range(target_joint_positions.shape[1]):
                hardcoded_joint_targets.append(msg[1].points[i].positions)
                velocities.append(msg[1].points[i].velocities)
                accelerations.append(msg[1].points[i].accelerations)

        for i in range(target_joint_positions.shape[1]):
            target_joint_positions[:,i] = hardcoded_joint_targets[i]

        bag.close()

        # store the target positions
        self.position_goal_1 = target_joint_positions[:,0]
        self.position_goal_2 = target_joint_positions[:,1]
        self.position_goal_3 = target_joint_positions[:,2]

        return target_joint_positions, velocities, accelerations

    def callback_JointStates(self, msg):
        '''
        This function is the callback funciton of the subscriber. The position, velocity,
        effort and time are stored, then call the 'acceleration()' function to calculate the 
        'q_dd'. When the time is larger than 37s, 'plotting' function will be called.
        '''
        # store the joint states
        self.curr_pos = msg.position
        self.curr_velo = msg.velocity
        self.curr_effort = msg.effort
        self.time = rospy.get_time()

        # plot the acceleration
        if self.time > 37.0:
            self.plotting()

        # calculate the acceleration
        q_dd = self.acceleration()
        print('acceleration = ', q_dd)
        
        # store the acceleration to the csv file
        with open('/home/zijie/catkin_ws/src/comp0127_lab/cw3/cw3q5/src/acceleration.csv', 'a') as file_1:
            csv_writer_1 = csv.writer(file_1)
            csv_writer_1.writerow(q_dd)

        # store the time to the csv file
        with open('/home/zijie/catkin_ws/src/comp0127_lab/cw3/cw3q5/src/time.csv', 'a') as file_2:
            csv_writer_2 = csv.writer(file_2)
            csv_writer_2.writerow([self.time])

        # with open('/home/zijie/catkin_ws/src/comp0127_lab/cw3/cw3q5/src/velocity.csv', 'a') as file_3:
        #     csv_writer_3 = csv.writer(file_3)
        #     csv_writer_3.writerow(self.curr_velo)

        
        print('time = ', self.time)
        # print('position = ', self.curr_pos)
        # print('velocity = ', self.curr_velo)
        print('______________________________________')
        print('______________________________________')


    def acceleration(self):
        '''
        This function calculate the acceleration 'q_dd'. Some functions from KDL are called in this 
        function to get the dynamic component. Then use the forward kinematics to calculate the 'q_dd'.
        '''
        # get the dynamic components from KDL
        B = np.array(self.kdl_iiwa.get_B(self.curr_pos))
        C_times_qdot = np.array(self.kdl_iiwa.get_C_times_qdot(self.curr_pos, self.curr_velo))
        G = np.array(self.kdl_iiwa.get_G(self.curr_pos))

        # initialize the target_torque and q_dd
        q_dd = np.ones((7,1))

        # use forward kinematics to calculate the acceleration of the joints
        q_dd = inv(B) @ (self.curr_effort - C_times_qdot - G)

        return q_dd

    def plotting(self):
        '''
        This function will plot the acceleration over time when it is called.
        '''
        path = '/home/zijie/catkin_ws/src/comp0127_lab/cw3/cw3q5/src'

        time = []
        with open(path+ '/time.csv', 'r') as file_1:
            csv_reader = csv.reader(file_1)
            for data in csv_reader:
                time.append(float(data[0]))

        a_1 = []
        a_2 = []
        a_3 = []
        a_4 = []
        a_5 = []
        a_6 = []
        a_7 = []
        with open(path+ '/acceleration.csv', 'r') as file_2:
            csv_reader = csv.reader(file_2)
            for data in csv_reader:
                a_1.append(float(data[0]))
                a_2.append(float(data[1]))
                a_3.append(float(data[2]))
                a_4.append(float(data[3]))
                a_5.append(float(data[4]))
                a_6.append(float(data[5]))
                a_7.append(float(data[6]))
    
        fig, ax = plt.subplots()
        ax.plot(time, a_1, label='Joint_1')
        ax.plot(time, a_2, label='Joint_2')
        ax.plot(time, a_3, label='Joint_3')
        ax.plot(time, a_4, label='Joint_4')
        ax.plot(time, a_5, label='Joint_5')
        ax.plot(time, a_6, label='Joint_6')
        ax.plot(time, a_7, label='Joint_7')
        fig.suptitle('Acceleration of Joints')
        ax.set_xlabel('Time - t')
        ax.set_ylabel('Acceleration - q_dd')
        ax.legend(loc='lower right')
        fig.show()
        # plt.savefig('a.png', dpi=1200)
        plt.show()



if __name__ == '__main__':
    try:
        iiwa_planner = iiwaTrajectory()
        iiwa_planner.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


