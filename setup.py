"""
Inicializa o workspace do robotics toolbox no iPython.
"""

import numpy as np
from roboticstoolbox import ctraj
from spatialmath import SE3
from spatialmath.base import *
from roboticstoolbox import xplot ,ET2, DHRobot, RevoluteDH, PrismaticDH


def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the cross product between two vectors.
    """
    return np.cross(a, b)


def rr_robot(L1=1, L2=1):

    
    e1 = RevoluteDH(a = L1)
    e2 = RevoluteDH(a = L2)

    rob = DHRobot([e1,e2], name = 'RR')
    return rob


def jacobian(q1, q2, L1=1, L2=1):
    """
    Compute the Jacobian matrix for the RR robot's forward kinematics.

    Parameters:
    q1 (float): Joint angle q1 in radians.
    q2 (float): Joint angle q2 in radians.
    L1 (float): Length of link 1.
    L2 (float): Length of link 2.

    Returns:
    numpy.ndarray: The 2x2 Jacobian matrix.
    """

    # --- TODO --- #
    # Creates the Jacobian matrix for the RR robot
    # (only considering position x y)

    J11 = None
    J12 = None
    J21 = None
    J22 = None

    J = np.array([[J11, J12], [J21, J22]])

    return J


def resolved_rate_control_2r():
    """
    Resolved rate control for the RR robot.
    """
    rr = rr_robot()

    q0 = np.array([-np.pi / 3, np.pi / 2])
    #print(q0)

    TE1 = rr.fkine(q0)
    #print(TE1)
    TE2 = SE3.Trans(-0.5, 0.5, 0) @ TE1
    #print(TE2)

    t = np.arange(0, 2, 0.02)
    #print(t)

    Ts = ctraj(TE1, TE2, t)
    #print(Ts)
    xplot(t, Ts.t, labels="x y z")
    xplot(t, Ts.rpy("xyz"), labels="roll pitch yaw")

    q = np.zeros((len(t), 2))
    #print(q)

    rr.plot(q)

    # Computing position error
    pos = rr.fkine(q0).t[:2]
    print(np.linalg.norm(pos - Ts[-1].t[:2]))


def rrr_robot(L1=1, L2=1, L3=1):
    # --- TODO --- #
    # Creates the RRR robot using the DHRobot class #
    raise NotImplementedError("rrr_robot not implemented yet!")


def dh_transform(theta, d, a, alpha):
    # --- TODO --- #
    # Creates the DH transformation matrix #
    raise NotImplementedError("dh_transform not implemented yet!")


def T01(q1):
    # --- TODO --- #
    # Creates the T01 matrix #
    raise NotImplementedError("T01 not implemented yet!")


def T12(q2):
    # --- TODO --- #
    # Creates the T12 matrix #
    raise NotImplementedError("T12 not implemented yet!")


def T23(q3):
    # --- TODO --- #
    # Creates the T23 matrix #
    raise NotImplementedError("T23 not implemented yet!")


def geometric_jacobian(q):
    # --- TODO --- #
    # Creates the geometric Jacobian matrix for the RRR planar arm #

    P1 = None
    P2 = None
    P3 = None

    PE = None

    JO1 = None
    JP1 = None

    JO2 = None
    JP2 = None

    JO3 = None
    JP3 = None

    return np.array([JP1, JP2, JP3]).T


def resolved_rate_control_3r():
    # --- TODO --- #
    # Resolved rate control for the RRR robot.
    raise NotImplementedError("resolved_rate_control_3r not implemented yet!")
