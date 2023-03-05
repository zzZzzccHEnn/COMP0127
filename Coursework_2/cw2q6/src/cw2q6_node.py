#!/usr/bin/env python3
import numpy as np
import rospy
import rosbag
import rospkg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cw2q4.youbotKineKDL import YoubotKinematicKDL
import PyKDL
from visualization_msgs.msg import Marker


class YoubotTrajectoryPlanning(object):
    def __init__(self):
        # Initialize node
        rospy.init_node('youbot_traj_cw2', anonymous=True)

        # Save question number for check in main run method
        self.kdl_youbot = YoubotKinematicKDL()

        # Create trajectory publisher and a checkpoint publisher to visualize checkpoints
        self.traj_pub = rospy.Publisher('/EffortJointInterface_trajectory_controller/command', JointTrajectory,
                                        queue_size=5)
        self.checkpoint_pub = rospy.Publisher("points_and_lines", Marker, queue_size=100)

    def run(self):
        """This function is the main run function of the class. When called, it runs question 6 by calling the q6()
        function to get the trajectory. Then, the message is filled out and published to the /command topic.
        """
        print("run q6a")
        rospy.loginfo("Waiting 5 seconds for everything to load up.")
        rospy.sleep(2.0)
        traj = self.q6()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = ["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"]
        self.traj_pub.publish(traj)

    def q6(self):
        """ This is the main q6 function. Here, other methods are called to create the shortest path required for this
        question. Below, a general step-by-step is given as to how to solve the problem.
        Returns:
            traj (JointTrajectory): A list of JointTrajectory points giving the robot joint positions to achieve in a
            given time period.
        """
        # Steps to solving Q6.
        # 1. Load in targets from the bagfile (checkpoint data and target joint positions).
        # 2. Compute the shortest path achievable visiting each checkpoint Cartesian position.
        # 3. Determine intermediate checkpoints to achieve a linear path between each checkpoint and have a full list of
        #    checkpoints the robot must achieve. You can publish them to see if they look correct. Look at slides 39 in lecture 7
        # 4. Convert all the checkpoints into joint values using an inverse kinematics solver.
        # 5. Create a JointTrajectory message.

        # Your code starts here ------------------------------

        # load 4 checkpoints
        target_cart_tf, target_joint_positions = self.load_targets()

        
        # get the order of the checkpoints to get the shortest path
        sorted_order, min_dist = self.get_shortest_path(target_cart_tf)

        # print the sorted order of checkpoints
        print('sorted_order is',sorted_order)

        # add intermediate checkpoints between the exist checkpoints
        full_checkpoint_tfs = self.intermediate_tfs(sorted_order, target_cart_tf, 6)

        # visualise the checkpoints
        self.publish_traj_tfs(full_checkpoint_tfs)

        # convert the transformation matrix to the pose
        q_checkpoints = self.full_checkpoints_to_joints(full_checkpoint_tfs,target_joint_positions[:,0])
        print(q_checkpoints.shape[1])

        # ________________________
        traj = JointTrajectory()
        t = 0
        dt = 3
        for i in range(q_checkpoints.shape[1]):
            traj_point = JointTrajectoryPoint()
            traj_point.positions = q_checkpoints[:, i]
            t = t + dt
            traj_point.time_from_start.secs = t
            traj.points.append(traj_point)
        # ________________________

        # Your code ends here ------------------------------

        assert isinstance(traj, JointTrajectory)
        return traj

    def load_targets(self):
        """This function loads the checkpoint data from the 'data.bag' file. In the bag file, you will find messages
        relating to the target joint positions. You need to use forward kinematics to get the goal end-effector position.
        Returns:
            target_cart_tf (4x4x5 np.ndarray): The target 4x4 homogenous transformations of the checkpoints found in the
            bag file. There are a total of 5 transforms (4 checkpoints + 1 initial starting cartesian position).
            target_joint_positions (5x5 np.ndarray): The target joint values for the 4 checkpoints + 1 initial starting
            position.
        """
        # Defining ros package path
        rospack = rospkg.RosPack()
        path = rospack.get_path('cw2q6')

        # Initialize arrays for checkpoint transformations and joint positions
        target_joint_positions = np.zeros((5, 5))
        # Create a 4x4 transformation matrix, then stack 6 of these matrices together for each checkpoint
        target_cart_tf = np.repeat(np.identity(4), 5, axis=1).reshape((4, 4, 5))

        # Load path for selected question
        bag = rosbag.Bag(path + '/bags/data.bag')
        # Get the current starting position of the robot
        target_joint_positions[:, 0] = self.kdl_youbot.kdl_jnt_array_to_list(self.kdl_youbot.current_joint_position)
        # Initialize the first checkpoint as the current end effector position
        target_cart_tf[:, :, 0] = self.kdl_youbot.forward_kinematics(target_joint_positions[:, 0])

        # Your code starts here ------------------------------

        # initialise the hardcoded_joint_targets
        hardcoded_joint_targets = []

        for msg in bag.read_messages():
            hardcoded_joint_targets.append(msg[1].position)

        # store the hardcoded joint targets 
        for i in range(target_joint_positions.shape[1]-1):
            target_joint_positions[:,i+1] = hardcoded_joint_targets[i]

        # convert the joint targets and store into 'target_cart_tf'
        for i in range(target_joint_positions.shape[1]):
            target_cart_tf[:, :, i] = self.kdl_youbot.forward_kinematics(target_joint_positions[:, i])
        
        # Your code ends here ------------------------------

        # Close the bag
        bag.close()

        assert isinstance(target_cart_tf, np.ndarray)
        assert target_cart_tf.shape == (4, 4, 5)
        assert isinstance(target_joint_positions, np.ndarray)
        assert target_joint_positions.shape == (5, 5)

        return target_cart_tf, target_joint_positions

    def get_shortest_path(self, checkpoints_tf):
        """This function takes the checkpoint transformations and computes the order of checkpoints that results
        in the shortest overall path.
        Args:
            checkpoints_tf (np.ndarray): The target checkpoints transformations as a 4x4x5 numpy ndarray.
        Returns:
            sorted_order (np.array): An array of size 5 indicating the order of checkpoint
            min_dist:  (float): The associated distance to the sorted order giving the total estimate for travel
            distance.
        """

        # Your code starts here ------------------------------

        # initialise the sorted order where the first checkpoint was defined
        sorted_order = [0]

        # initialise the original order without '0'
        origin_order = []
        for i in range(checkpoints_tf.shape[2]-1):
            origin_order.append(i+1)

        # store the positions of the checkpoints
        position = np.zeros((3,checkpoints_tf.shape[2]))
        for _ in range(checkpoints_tf.shape[2]):
            position[:,_] = checkpoints_tf[0:3,3,_]

        # search for the shortest checkpoint order
        for i in range(checkpoints_tf.shape[2]-1):
            # initialise the sorted order where the first checkpoint was defined
            sorted_order = [0]
            sorted_order_1 = sorted_order.copy()
            temp_1 = origin_order.copy()
            # append the second checkpoint
            sorted_order_1.append(temp_1[i])
            temp_1.remove(temp_1[i])
            
            for j in range(len(temp_1)):
                sorted_order_2 = sorted_order_1.copy()
                temp_2 = temp_1.copy()
                # append the third checkpoint
                sorted_order_2.append(temp_2[j])
                temp_2.remove(temp_2[j])

                for k in range(len(temp_2)):
                    sorted_order_3 = sorted_order_2.copy()
                    temp_3 = temp_2.copy()
                    # append the forth checkpoint
                    sorted_order_3.append(temp_3[k])
                    temp_3.remove(temp_3[k])
                    # append the last checkpoint
                    sorted_order_3.append(temp_3[-1])

                    # calculate the total distance
                    min_dist = 10e6
                    temp_dist = np.sqrt(np.sum((position[:,sorted_order_3[0]]-position[:,sorted_order_3[1]])**2)) 
                    + np.sqrt(np.sum((position[:,sorted_order_3[1]]-position[:,sorted_order_3[2]])**2))
                    + np.sqrt(np.sum((position[:,sorted_order_3[2]]-position[:,sorted_order_3[3]])**2))
                    + np.sqrt(np.sum((position[:,sorted_order_3[3]]-position[:,sorted_order_3[4]])**2))

                    # compare with the exist minimun distance
                    if temp_dist < min_dist:
                        sorted_order = np.array(sorted_order_3)
                        min_dist = temp_dist
                
        # Your code ends here ------------------------------

        assert isinstance(sorted_order, np.ndarray)
        assert sorted_order.shape == (5,)
        assert isinstance(min_dist, float)

        return sorted_order, min_dist

    def publish_traj_tfs(self, tfs):
        """This function gets a np.ndarray of transforms and publishes them in a color coded fashion to show how the
        Cartesian path of the robot end-effector.
        Args:
            tfs (np.ndarray): A array of 4x4xn homogenous transformations specifying the end-effector trajectory.
        """
        id = 0
        for i in range(0, tfs.shape[2]):
            marker = Marker()
            marker.id = id
            marker.ns = "checkpoints"
            id += 1
            marker.header.frame_id = 'base_link'
            marker.header.stamp = rospy.Time.now()
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0 + id * 0.05
            marker.color.b = 1.0 - id * 0.05
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = tfs[0, -1, i]
            marker.pose.position.y = tfs[1, -1, i]
            marker.pose.position.z = tfs[2, -1, i]
            self.checkpoint_pub.publish(marker)

    def intermediate_tfs(self, sorted_checkpoint_idx, target_checkpoint_tfs, num_points):
        """This function takes the target chec       print('shape of q[0]', q[0].shape)point transforms and the desired order based on the shortest path sorting, 
        and calls the decoupled_rot_and_trans() function.
        Args:
            sorted_checkpoint_idx (list): List describing order of checkpoints to follow.
            target_checkpoint_tfs (np.ndarray): the state of the robot joints. In a youbot those are revolute
            num_points (int): Number of intermediate points between checkpoints.
        Returns:
            full_checkpoint_tfs: 4x4x(4xnum_points + 5) homogeneous transformations matrices describing the full desired
            poses of the end-effector position.
        """

        # Your code starts here ------------------------------
        
        # initialise the full_chechpoint_tfs
        full_checkpoint_tfs = target_checkpoint_tfs[:,:,0]

        # create the temp matrix to store the intermidiate checkpoint tfs
        tfs = np.zeros((4,4,num_points))
        
        # get the intermediate checkpoints
        for i in range(sorted_checkpoint_idx.shape[0]-1):
            temp = self.decoupled_rot_and_trans(target_checkpoint_tfs[:,:,sorted_checkpoint_idx[i]], target_checkpoint_tfs[:,:,sorted_checkpoint_idx[i+1]], num_points)
            full_checkpoint_tfs = np.dstack((full_checkpoint_tfs, temp))
            full_checkpoint_tfs = np.dstack((full_checkpoint_tfs, target_checkpoint_tfs[:,:,sorted_checkpoint_idx[i+1]]))
            
        # Your code ends here ------------------------------
       
        return full_checkpoint_tfs

    def decoupled_rot_and_trans(self, checkpoint_a_tf, checkpoint_b_tf, num_points):
        """This function takes two checkpoint transforms and computes the intermediate transformations
        that follow a straight line path by decoupling rotation and translation.
        Args:
            checkpoint_a_tf (np.ndarray): 4x4 transformation describing pose of checkpoint a.
            checkpoint_b_tf (np.ndarray): 4x4 transformation describing pose of checkpoint b.
            num_points (int): Number of intermediate points between checkpoint a and checkpoint b.
        Returns:,0
            tfs: 4x4x(num_points) homogeneous transformations matrices describing the full desired
            poses of the end-effector position from checkpoint a to checkpoint b following a linear path.
        """

        # Your code starts here ------------------------------

        # extract the position of the start configuration
        Ps = checkpoint_a_tf[0:3,3]
        # extract the rotation of the strat configuration
        Rs = checkpoint_a_tf[0:3,0:3]

        # extract the position of the final configuration
        Pf = checkpoint_b_tf[0:3,3]
        # extract the rotation of the final configuration
        Rf = checkpoint_b_tf[0:3,0:3]

        delta = 1/(num_points+1)

        # initialise the transformation matrix
        tran = np.zeros((4,4))
        tran[3,3]=1

        # initialise the tfs matrix
        tfs = np.zeros((4,4,num_points))

        for i in range(1,num_points+1):
            temp = tran.copy()
            Pt = Ps + i*delta*(Pf-Ps)
            # Rt = Rs + i*delta*(Rf-Rs)
            Rt = Rs@((Rs.T @ Rf)*(i*delta))
            temp[0:3,0:3] = Rt
            temp[0:3,3] = Pt

            tfs[:,:,i-1] = temp

        # Your code ends here ------------------------------

        return tfs

    def full_checkpoints_to_joints(self, full_checkpoint_tfs, init_joint_position):
        """This function takes the full set of checkpoint transformations, including intermediate checkpoints, 
        and computes the associated joint positions by calling the ik_position_only() function.
        Args:
            full_checkpoint_tfs (np.ndarray, 4x4xn): 4x4xn transformations describing all the desired poses of the end-effector
            to follow the desired path.
            init_joint_position (np.ndarray):A 5x1 array for the initial joint position of the robot.
        Returns:
            q_checkpoints (np.ndarray, 5xn): For each pose, the solution of the position IK to get the joint position
            for that pose.
        """
        
        # Your code starts here ------------------------------

        # get teh initial configuration of the manipulator
        init_joint_position = self.kdl_youbot.kdl_jnt_array_to_list(self.kdl_youbot.current_joint_position)
        q_checkpoints = np.array(init_joint_position).reshape(5,1)

        # apply pre rotation to the joint_1 and joint_5
        q_checkpoints[0,0] = q_checkpoints[0,0] + np.pi*1
        q_checkpoints[2,0] = q_checkpoints[2,0] - np.pi*0.05
        q_checkpoints[4,0] = q_checkpoints[4,0] + np.pi*0.75

        # calculate the joint position by inverse kinematics
        for i in range(full_checkpoint_tfs.shape[2]-1):
            temp = np.array(self.ik_position_only(full_checkpoint_tfs[:,:,i+1], q_checkpoints[:,i]))
            # print('No.', i+1, 'count ', self.count, ', the error is ', self.err)
            q_checkpoints = np.hstack((q_checkpoints,temp[0]))

        # Your code ends here ------------------------------

        return q_checkpoints

    def ik_position_only(self, pose, q0):
        """This function implements position only inverse kinematics.
        Args:
            pose (np.ndarray, 4x4): 4x4 transformations describing the pose of the end-effector position.
            q0 (np.ndarray, 5x1):A 5x1 array for the initial starting point of the algorithm.
        Returns:
            q (np.ndarray, 5x1): The IK solution for the given pose.
            error (float): The Cartesian error of the solution.
        """
        # Some useful notes:
        # We are only interested in position control - take only the position part of the pose as well as elements of the
        # Jacobian that will affect the position of the error.

        # Your code starts here ---------------------- --------
        
        # define the step size
        alpha = 0.3

        # initial the joint data
        prev = q0.copy()

        # initialise the counter
        self.count = 0

        # gradient decent
        while True:
            delta = (pose[0:3,3].reshape(3,1)-(self.kdl_youbot.forward_kinematics(prev)[0:3,3]).reshape(3,1))
            q = np.array(prev).reshape(5,1) + alpha*(self.kdl_youbot.get_jacobian(prev)[0:3,:].T)@delta
            prev = q.copy()

            self.count += 1

            if (np.linalg.norm((self.kdl_youbot.get_jacobian(prev)[0:3,:].T)@delta) < 8.5*10e-6):
                error = np.linalg.norm((self.kdl_youbot.get_jacobian(prev)[0:3,:].T)@delta)
                self.err = error
                break

        # Your code ends here ------------------------------

        return q, error

    @staticmethod
    def list_to_kdl_jnt_array(joints):
        """This converts a list to a KDL jnt array.
        Args:
            joints (joints): A list of the joint values.
        Returns:
            kdl_array (PyKDL.JntArray): JntArray object describing the joint position of the robot.
        """
        kdl_array = PyKDL.JntArray(5)
        for i in range(0, 5):
            kdl_array[i] = joints[i]
        return kdl_array


if __name__ == '__main__':
    try:
        youbot_planner = YoubotTrajectoryPlanning()
        youbot_planner.run()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
