import time
import numpy as np
import matplotlib.pyplot as plt
from roboticstoolbox import ctraj
from spatialmath import SE3
from spatialmath.base import *
from roboticstoolbox import jsingu,xplot ,ET2, DHRobot, RevoluteDH, PrismaticDH


def plots(qs,erro,t):

    plt.plot(erro,t)
    plt.xlabel('Tempo')
    plt.ylabel('Erro')
    plt.legend()
    plt.title("Erro X Tempo")
    plt.show()
    # Extraia as posições de cada junta em listas separadas
    theta1_positions = [pos[0] for pos in qs]
    theta2_positions = [pos[1] for pos in qs]

    # Crie um gráfico para a junta 1
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, theta1_positions, label='Junta 1')
    plt.xlabel('Tempo')
    plt.ylabel('Ângulo da Junta 1 (theta1)')
    plt.legend()

    # Crie um gráfico para a junta 2
    plt.subplot(2, 1, 2)
    plt.plot(t, theta2_positions, label='Junta 2')
    plt.xlabel('Tempo')
    plt.ylabel('Ângulo da Junta 2 (theta2)')
    plt.legend()

    # Exiba os gráficos
    plt.tight_layout()
    plt.title("Junta x Tempo")
    plt.show()

def rr_robot(L1=1, L2=1):

    e1 = RevoluteDH(a = L1)
    e2 = RevoluteDH(a = L2)

    rob = DHRobot([e1,e2], name = 'RR')
    return rob

def trajetoria():
    rr = rr_robot()

    q0 = np.array([-np.pi / 3, np.pi / 2])
    #print(q0)

    TE1 = rr.fkine(q0)
    print("Pose inicial:\n" ,TE1)
    TE2 = SE3.Trans(0,-0.5, 0) @ TE1
    print("Pose final:\n", TE2)

    t = np.arange(0, 5, 0.02)
    #print(t)

    Ts = ctraj(TE1, TE2, t)
    #print(Ts)
    xplot(t, Ts.t, labels="x y z")
    xplot(t, Ts.rpy("xyz"), labels="roll pitch yaw")
    plt.show()
    return Ts


def resolved_rate_control_2r(L1 = 1,L2=1):
    """
    Resolved rate control for the RR robot.
    """
    rr = rr_robot()
   
    Ts = trajetoria()

    t = np.arange(0, 5, 0.02)

    # Taxa de atualização de controle
    dt = 0.01

    # Posição inicial do manipulador (ângulos iniciais das juntas)
    theta1 = 0.0
    theta2 = 0.0

    erro = []
    qs = []

    # Loop ao longo da trajetória
    for i in range(len(Ts)):

        #rr.plot(q = [theta1,theta2])
        # Obtenha a posição atual do efetuador
        T = Ts[i]
        #print(T)
        x = T.t[0]
        y = T.t[1]
        #print(x,y)

        # Calcule o erro cartesiano
        error_x = x - (L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2))
        error_y = y - (L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2))

        # Calcule as velocidades das juntas usando a matriz Jacobiana
        J = np.array([[-L1*np.sin(theta1) - L2*np.sin(theta1 + theta2),-L2*np.sin(theta1 + theta2)],
                      [L1*np.cos(theta1) + L2*np.cos(theta1 + theta2),L2*np.cos(theta1 + theta2)]])
        #print(J)
        #print(jsingu(J))

        # Resolva a equação de controle: J * [dtheta1, dtheta2] = [error_x, error_y]
        determinant = np.linalg.det(J)
        if abs(determinant) < 1e-6:
            lambda_value = 0.01 # Ajuste este valor conforme necessário
            J_reg = J + lambda_value * np.identity(J.shape[0])
            joint_velocities = np.linalg.solve(J_reg, np.array([error_x, error_y]))

        else:
            # Resolver o sistema normalmente.
            joint_velocities = np.linalg.solve(J, np.array([error_x, error_y]))

        # Atualize os ângulos das juntas
        theta1 += joint_velocities[0] * dt
        theta2 += joint_velocities[1] * dt

        q = [theta1,theta2]
        qs.append(q)
        #rr.plot(q)

        pos = rr.fkine(q).t[:2]
        k = np.linalg.norm(pos - Ts[-1].t[:2])
        #print(k)
        erro.append(k)

        # Aguarde um tempo para a próxima iteração
        time.sleep(dt)

    print("q = {}".format([theta1,theta2]))
    #print(np.rad2deg(theta1), np.rad2deg(theta2))

    rr.teach([theta1,theta2])
    plots(qs,erro,t)
    

def main():
    resolved_rate_control_2r()