#!/usr/bin/env python3

import numpy as np
from cw2q4.youbotKineBase import YoubotKinematicBase


class YoubotKinematicStudent(YoubotKinematicBase):
    def __init__(self):
        super(YoubotKinematicStudent, self).__init__(tf_suffix='student')

        # Set the offset for theta --> This was updated on 23/11/2022. Feel free to use your own code.
        youbot_joint_offsets = [170.0 * np.pi / 180.0,
                                -65.0 * np.pi / 180.0,
                                146 * np.pi / 180,
                                -102.5 * np.pi / 180,
                                -167.5 * np.pi / 180]

        # Apply joint offsets to dh parameters
        self.dh_params['theta'] = [theta + offset for theta, offset in
                                   zip(self.dh_params['theta'], youbot_joint_offsets)]

        # Joint reading polarity signs
        self.youbot_joint_readings_polarity = [-1, 1, 1, 1, 1]

    def forward_kinematics(self, joints_readings, up_to_joint=5):
        """This function solve forward kinematics by multiplying frame transformation up until a specified
        frame number. The frame transformation used in the computation are derived from dh parameters and
        joint_readings.
        Args:
            joints_readings (list): the state of the robot joints. In a youbot those are revolute
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematics.
                Defaults to 5.
        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix describing the pose of frame_{up_to_joint}
                w.r.t the base of the robot.
        """
        assert isinstance(self.dh_params, dict)
        assert isinstance(joints_readings, list), "joint readings of type " + str(type(joints_readings))
        assert isinstance(up_to_joint, int)
        assert up_to_joint >= 0
        assert up_to_joint <= len(self.dh_params['a'])

        T = np.identity(4)

	# --> This was updated updated on 23/11/2022. Feel free to use your own code.
        joints_readings = [sign * angle for sign, angle in zip(self.youbot_joint_readings_polarity, joints_readings)]

        for i in range(up_to_joint):
            A = self.standard_dh(self.dh_params['a'][i],
                                 self.dh_params['alpha'][i],
                                 self.dh_params['d'][i],
                                 self.dh_params['theta'][i] + joints_readings[i])
            T = T.dot(A)

        assert isinstance(T, np.ndarray), "Output wasn't of type ndarray"
        assert T.shape == (4, 4), "Output had wrong dimensions"
        return T

    def get_jacobian(self, joint):
        """Given the joint values of the robot, compute the Jacobian matrix. Coursework 2 Question 4a.
        Reference - Lecture 5 slide 24.

        Args:
            joint (list): the state of the robot joints. In a youbot those are revolute

        Returns:
            Jacobian (numpy.ndarray): NumPy matrix of size 6x5 which is the Jacobian matrix.
        """
        assert isinstance(joint, list)
        assert len(joint) == 5

        # Your code starts here ----------------------------

        # For your solution to match the KDL Jacobian, z0 needs to be set [0, 0, -1] instead of [0, 0, 1], 
        # since that is how its defined in the URDF.
        # Both are correct.

        # initialise the jacobian matrix
        jacobian = np.ones((6,5))

        # calculate the transformation for the P
        P = self.forward_kinematics(joint, 5)[0:3,3]
        P0 = np.zeros((3))

        Z0 = np.array([[0,0,-1]])

        for _ in range(len(joint)):
            T = self.forward_kinematics(joint, _)
            jacobian[0:3,_] = np.cross(T[0:3,2],(P-T[0:3,3]))
            jacobian[3:,_] = T[0:3,2]

        jacobian[0:3,0]=np.cross(Z0,(P-P0))
        jacobian[3:,0]=Z0[0].tolist()
        # print('test', Z0[0].tolist())

        # Your code ends here ------------------------------

        assert jacobian.shape == (6, 5)
        return jacobian

    def check_singularity(self, joint):
        """Check for singularity condition given robot joints. Coursework 2 Question 4c.
        Reference Lecture 5 slide 30.

        Args:
            joint (list): the state of the robot joints. In a youbot those are revolute

        Returns:
            singularity (bool): True if in singularity and False if not in singularity.

        """
        assert isinstance(joint, list)
        assert len(joint) == 5
        
        # Your code starts here ----------------------------
        # calculate the Jacobian matrix
        J = self.get_jacobian(self, joint)

        # initialise the singularity 
        singularity = True 

        if J.shape[1]==6:
            if np.linalg.det(J)==0:
                singularity = True
            else:
                singularity = False
        elif J.shape[1]!=6:
            if np.linalg.det(J.T @ J)==0:
                singularity = True
            else:
                singularity = False

        # Your code ends here ------------------------------

        assert isinstance(singularity, bool)
        return singularity
