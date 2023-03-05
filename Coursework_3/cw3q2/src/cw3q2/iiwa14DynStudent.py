#!/usr/bin/env python3

import numpy as np
from cw3q2.iiwa14DynBase import Iiwa14DynamicBase


class Iiwa14DynamicRef(Iiwa14DynamicBase):
    def __init__(self):
        super(Iiwa14DynamicRef, self).__init__(tf_suffix='ref')

    def forward_kinematics(self, joints_readings, up_to_joint=7):
        """This function solve forward kinematics by multiplying frame transformation up until a specified
        joint. Reference Lecture 9 slide 13.
        Args:
            joints_readings (list): the state of the robot joints.
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematics.
                Defaults to 7.
        Returns:
            np.ndarray The output is a numpy 4*4 matrix describing the transformation from the 'iiwa_link_0' frame to
            the selected joint frame.
        """

        assert isinstance(joints_readings, list), "joint readings of type " + str(type(joints_readings))
        assert isinstance(up_to_joint, int)

        T = np.identity(4)
        # iiwa base offset
        T[2, 3] = 0.1575

        # 1. Recall the order from lectures. T_rot_z * T_trans * T_rot_x * T_rot_y. You are given the location of each
        # joint with translation_vec, X_alpha, Y_alpha, Z_alpha. Also available are function T_rotationX, T_rotation_Y,
        # T_rotation_Z, T_translation for rotation and translation matrices.
        # 2. Use a for loop to compute the final transformation.
        for i in range(0, up_to_joint):
            T = T.dot(self.T_rotationZ(joints_readings[i]))
            T = T.dot(self.T_translation(self.translation_vec[i, :]))
            T = T.dot(self.T_rotationX(self.X_alpha[i]))
            T = T.dot(self.T_rotationY(self.Y_alpha[i]))

        assert isinstance(T, np.ndarray), "Output wasn't of type ndarray"
        assert T.shape == (4, 4), "Output had wrong dimensions"

        return T

    def get_jacobian_centre_of_mass(self, joint_readings, up_to_joint=7):
        """Given the joint values of the robot, compute the Jacobian matrix at the centre of mass of the link.
        Reference - Lecture 9 slide 14.

        Args:
            joint_readings (list): the state of the robot joints.
            up_to_joint (int, optional): Specify up to what frame you want to compute the Jacobian.
            Defaults to 7.

        Returns:
            jacobian (numpy.ndarray): The output is a numpy 6*7 matrix describing the Jacobian matrix defining at the
            centre of mass of a link.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7
        # 1. Compute the forward kinematics (T matrix) for all the joints.
        # 2. Compute forward kinematics at centre of mass (T_cm) for all the joints.
        # 3. From the computed forward kinematic and forward kinematic at CoM matrices,
        # extract z, z_cm (z axis of the rotation part of T, T_cm) and o, o_cm (translation part of T, T_cm) for all links.
        # 4. Based on the computed o, o_cm, z, z_cm, fill J_p and J_o matrices up until joint 'up_to_joint'. Apply equations at slide 15, Lecture 9.
        # 5. Fill the remaining part with zeroes and return the Jacobian at CoM.

        # Your code starts here ----------------------------
        
        # intialise the jacobian
        jacobian = np.zeros((6,7))

        # calculate the p_li
        P_li = self.forward_kinematics_centre_of_mass(joint_readings, up_to_joint)[0:3,3]

        for i in range(1,up_to_joint+1):
                jacobian[0:3,i-1] = np.cross(self.forward_kinematics(joint_readings,i-1)[0:3,2], (P_li-self.forward_kinematics(joint_readings,i-1)[0:3,3]))
                jacobian[3:,i-1] = self.forward_kinematics(joint_readings,i-1)[0:3,2].copy()

        # Your code ends here ------------------------------

        assert jacobian.shape == (6, 7)
        return jacobian

    def forward_kinematics_centre_of_mass(self, joints_readings, up_to_joint=7):
        """This function computes the forward kinematics up to the centre of mass for the given joint frame.
        Reference - Lecture 9 slide 14.
        Args:
            joints_readings (list): the state of the robot joints.
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematicks.
                Defaults to 5.
        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix describing the pose of frame_{up_to_joint} for the
            centre of mass w.r.t the base of the robot.
        """
        T= np.identity(4)
        T[2, 3] = 0.1575

        T = self.forward_kinematics(joints_readings, up_to_joint-1)
        T = T.dot(self.T_rotationZ(joints_readings[up_to_joint-1]))
        T = T.dot(self.T_translation(self.link_cm[up_to_joint-1, :]))

        return T

    def get_B(self, joint_readings):
        """Given the joint positions of the robot, compute inertia matrix B.
        Args:
            joint_readings (list): The positions of the robot joints.

        Returns:
            B (numpy.ndarray): The output is a numpy 7*7 matrix describing the inertia matrix B.
        """
        B = np.zeros((7, 7))
        # Some useful steps:
        # 1. Compute the jacobian at the centre of mass from second joint to last joint
        # 2. Compute forward kinematics at centre of mass from second to last joint
        # 3. Extract the J_p and J_o matrices from the Jacobian centre of mass matrices
        # 4. Calculate the inertia tensor using the rotation part of the FK centre of masses you have calculated
        # 5. Apply the the equation from slide 16 lecture 9        
	    # Your code starts here ------------------------------
        
        for _ in range(1,len(joint_readings)+1):
            # get the jacobian of the link
            jacobian = self.get_jacobian_centre_of_mass(joint_readings,_)
            j_p = jacobian[0:3,:]
            j_o = jacobian[3:,:]

            # define the mass of the link
            m = self.mass[_-1]
            
            # define the rotation matrix from the frame G_i to the base frame
            R = self.forward_kinematics_centre_of_mass(joint_readings,up_to_joint=_)[0:3,0:3]

            # define the inertia tensor for the parallel frame
            I_parallel = np.diag(self.Ixyz[_-1])

            B = B + (m*(j_p.T @ j_p) + j_o.T @ (R @ I_parallel @ R.T) @ j_o)

        # Your code ends here ------------------------------
        
        return B

    def get_C_times_qdot(self, joint_readings, joint_velocities):
        """Given the joint positions and velocities of the robot, compute Coriolis terms C.
        Args:
            joint_readings (list): The positions of the robot joints.
            joint_velocities (list): The velocities of the robot joints.

        Returns:
            C (numpy.ndarray): The output is a numpy 7*1 matrix describing the Coriolis terms C times joint velocities.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7
        assert isinstance(joint_velocities, list)
        assert len(joint_velocities) == 7
        # Some useful steps:
        # 1. Create a h_ijk matrix (a matrix containing the Christoffel symbols) and a C matrix.
        # 2. Compute the derivative of B components for the given configuration w.r.t. joint values q. Apply equations at slide 19, Lecture 9.
        # 3. Based on the previously obtained values, fill h_ijk. Apply equations at slide 19, Lecture 9.
        # 4. Based on h_ijk, fill C. Apply equations at slide 19, Lecture 9.

        # Your code starts here ------------------------------  

        # define the delta
        delta = 10e-8

        n = len(joint_readings)

        # initalise the h_ijk and C
        h_ijk = np.zeros((n,n,n))
        C = np.zeros((n,n))
        
        for k in range(n):
            delta_q_k = np.zeros(n).tolist()
            delta_q_k[k] = delta
            for j in range(n):
                for i in range(n):
                    delta_q_i = np.zeros(n).tolist()
                    delta_q_i[i] = delta
                    h_1 = (self.get_B((np.array(joint_readings)+np.array(delta_q_k)).tolist())[i,j] - self.get_B(joint_readings)[i,j])/delta
                    h_2 = 0.5 * (self.get_B((np.array(joint_readings)+np.array(delta_q_i)).tolist())[j,k] - self.get_B(joint_readings)[j,k])/delta
                    h_ijk[i,j,k] = h_1 - h_2

        for k in range(n):
            C = C + h_ijk[:,:,k] * joint_velocities[k]
        
        C = C @ joint_velocities

        # Your code ends here ------------------------------

        assert isinstance(C, np.ndarray)
        assert C.shape == (7,)
        return C

    def get_G(self, joint_readings):
        """Given the joint positions of the robot, compute the gravity matrix g.
        Args:
            joint_readings (list): The positions of the robot joints.

        Returns:
            g (numpy.ndarray): The output is a numpy 7*1 numpy array describing the gravity matrix g.
        """
        assert isinstance(joint_readings, list)
        assert len(joint_readings) == 7
        # Some useful steps:
        # 1. Compute the Jacobian at CoM for all joints.
        # 2. Use the computed J_cm to fill the G matrix. This method is actually different from what seen in the lectures.
        # 3. Alternatvely, you can compute the P matrix and use it to fill the G matrix based on formulas presented at slides 17 and 22, Lecture 9.
        # Your code starts here ------------------------------

        # initialise the delta
        delta = 10e-9

        n = len(joint_readings)

        # intialise the g
        g = np.zeros(7)

        # initialise the g0^T
        g_0T = np.array([[0,0,-self.g]])

        for i in range(n):
            # calculate P(q+delta)
            p_plus_delta = 0

            q_plus_delta = np.zeros(7).tolist()
            q_plus_delta[i] = delta
            q_plus_delta = (np.array(joint_readings) + np.array(q_plus_delta)).tolist()

            g_i = 0
            for j in range(n):
                p_plus_delta = - (self.mass[j] * g_0T @ self.forward_kinematics_centre_of_mass(q_plus_delta,j+1)[0:3,-1])
                p_q = - (self.mass[j] * g_0T @ self.forward_kinematics_centre_of_mass(joint_readings,j+1)[0:3,-1])
                temp = (p_plus_delta - p_q) / delta
                g_i = g_i + temp

            g[i] = g_i

        # Your code ends here ------------------------------

        assert isinstance(g, np.ndarray)
        assert g.shape == (7,)
        return g
