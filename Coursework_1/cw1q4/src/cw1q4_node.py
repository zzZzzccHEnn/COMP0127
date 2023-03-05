#!/usr/bin/env python3

from urllib import response
import rospy
import numpy as np

# TODO: Include all the required service classes
# your code starts here -----------------------------
from cw1q4_srv.srv import quat2rodrigues
from cw1q4_srv.srv import quat2rodriguesResponse

from cw1q4_srv.srv import quat2zyx
from cw1q4_srv.srv import quat2zyxResponse
# your code ends here -------------------------------




def convert_quat2zyx(request):
    # TODO complete the function
    """Callback ROS service function to convert quaternion to Euler z-y-x representation

    Args:
        request (quat2zyxRequest): cw1q4_srv service message, containing
        the quaternion you need to convert.

    Returns:
        quat2zyxResponse: cw1q4_srv service response, in which 
        you store the requested euler angles 
    """
    assert isinstance(request, quat2zyxRequest)

    # Your code starts here ----------------------------
    qx = request.q.x
    qy = request.q.y
    qz = request.q.z
    qw = request.q.w

    euler = quat2zyxResponse()

    #euler.x is roll (x-axis rotation)
    euler.x.data = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx**2+qy**2))

    #euler.y is pitch (y-axis rotation)
    if abs(2*(qw*qy-qz*qx)) >= 1:
        euler.y.data = np.copysign(np.pi/2, 2*(qw*qy-qz*qx))
    else:
        euler.y.data = np.asiodrigun(2*(qw*qy-qz*qx))

    #euler.z is yaw (z-axis rotation)
    euler.z.data = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy**2+qz**2))

    response = euler
    # Your code ends here ------------------------------

    assert isinstance(response, quat2zyxResponse)
    return response


def convert_quat2rodrigues(request):
    # TODO complete the function

    """Callback ROS service function to convert quaternion to rodrigues representation
    
    Args:
        request (quat2rodriguesRequest): cw1q4_srv service message, containing
        the quaternion you need to convert

    Returns:
        quat2rodriguesResponse: cw1q4_srv service response, in which 
        you store the requested rodrigues representation 
    """
    assert isinstance(request, quat2rodriguesRequest)

    # Your code starts here ----------------------------
    qx = request.q.x
    qy = request.q.y
    qz = request.q.z
    qw = request.q.w

    # theta = arcos((r11+r22+r33 - 1)/2)
    # vec_U = 1/(2 sin(theta)) [r32-r23] [r13-r31] [r21-r12]

    R11 = 1-2*(qy**2)-2*(qz**2)
    R22 = 1-2*(qx**2)-2*(qz**2)
    R33 = 1-2*(qx**2)-2*(qy**2)

    R12 = 2*qx*qy-2*qz*qw
    R13 = 2*qx*qz+2*qy*qw
    R21 = 2*qx*qy+2*qz*qw
    R23 = 2*qy*qz-2*qx*qw
    R31 = 2*qx*qz-2*qy*qw
    R32 = 2*qy*qz+2*qx*qw

    theta = np.arcos((R11+R22+R33-1)/2)

    rod = quat2rodriguesResponse()

    rod.x.data = (R32-R23)/(2*np.sin(theta)) * theta
    rod.y.data = (R13-R31)/(2*np.sin(theta)) * theta
    rod.z.data = (R21-R12)/(2*np.sin(theta)) * theta

    response = rod

    # Your code ends here ------------------------------

    assert isinstance(response, quat2rodriguesResponse)
    return response

def rotation_converter():
    rospy.init_node('rotation_converter')

    #Initialise the services
    rospy.Service('quat2rodrigues', quat2rodrigues, convert_quat2rodrigues)
    rospy.Service('quat2zyx', quat2zyx, convert_quat2zyx)

    rospy.spin()


if __name__ == "__main__":
    rotation_converter()
