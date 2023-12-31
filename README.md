# Robotic_AB2_Part2

## Questão 1:

Sendo o braço RR Planar definido da base para atuador expresso pela matriz de transformação:

Sabendo que a tabela DH do braço RR planas é:

| j | θⱼ  |  dⱼ |   aⱼ   | ⍺ⱼ   |
| --|-----|-----|--------|------|
| 1 | θ1  |  0  |   L1   | 0.0° |
| 2 | θ2  |  0  |   L2   | 0.0° |

Podemos calcular a Pose final da base até o atuador pelas transformações:

$`^jT_{j+1}=\begin{bmatrix}\cos \theta _j&-\sin \theta _j\cos \alpha _j&\sin \theta _j\sin \alpha _j&a_j\cos \theta _j\\
\sin \theta _j&\cos \theta _j\cos \alpha _j&-\cos \theta _j\sin \alpha _j&a_j\sin \theta _j\\
0&\sin \alpha _j&\cos \alpha _j&d_j\\
0&0&0&1\end{bmatrix}`$

Onde a Pose final e dada por:

<div align="center">

$`^0T_2=^0T_1\cdot^1T_2`$

$`^0T_2=\begin{bmatrix}Cos(\theta1)&-Sen(\theta1)&0&L_1Cos(\theta1)\\
Sen(\theta1)&Cos(\theta1)&0&L_1Sen(\theta1)\\
0&0&1&0\\
0&0&0&1\end{bmatrix}\begin{bmatrix}Cos(\theta2)&-Sen(\theta2)&0&L_2Cos(\theta2)\\
Sen(\theta2)&Cos(\theta2)&0&L_2Sen(\theta2)\\
0&0&1&0\\
0&0&0&1\end{bmatrix}`$

$`^0T_2=\begin{bmatrix}
cos(\theta1)*cos(\theta2) - sin(\theta1)*sin(\theta2) & sin(\theta1)*(-cos(\theta2)) - cos(\theta1)*sin(\theta2) & 0 & -L2 sin(\theta1) sin(\theta2) + L2 cos(\theta1) cos(\theta2) + L1 cos(\theta1)\\
sin(\theta1)*cos(\theta2) + cos(\theta1)*sin(\theta2) & cos(\theta1)*cos(\theta2) - sin(\theta1)*sin(\theta2) & 0 & L2*sin(\theta1)*cos(\theta2) + L2*cos(\theta1)*sin(\theta2) + L1*sin(\theta1)\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1\end{bmatrix}`$

</div>

Temos, podemos expressa x e y:


<div align="center">

$`x =L1*cos(\theta1) + L2*cos(\theta1 + \theta2)`$

$`y = L1*sin(\theta1) + L2*sin(\theta1 + \theta2)`$

</div>

De forma analitica pode expresa a matriz jacobiana

<div align="center">

$`J(q) = \frac{\partial}{\partial(q)}K(q)=\begin{bmatrix}\frac{\partial x}{\partial(θ1)}&\frac{\partial x}{\partial(θ2)}\\\frac{\partial y}{\partial(θ1)}&\frac{\partial y}{\partial(θ2)}\end{bmatrix}`$

</div>

Efetuando as derivadas:

$`\frac{\partial x}{\partial(θ1)} =  -L1*sin(θ1) - L2*sin(θ2 + θ1)`$

$`\frac{\partial x}{\partial(θ2)} = -L2*sin(θ2 + θ1)`$

$`\frac{\partial y}{\partial(θ1)} = L1*cos(θ1) + L2*cos(θ2 + θ1)`$

$`\frac{\partial y}{\partial(θ2)} =  L2*cos(θ2 + θ1)`$

$`J(q) =\begin{bmatrix}-L1*sin(θ1) - L2*sin(θ2 + θ1)&-L2*sin(θ2 + θ1)\\ L1*cos(θ1) + L2*cos(θ2 + θ1)&  L2*cos(θ2 + θ1)\end{bmatrix}`$

### Letra A:

Modelando o braço RR planar atraves da blibioteca ToolBox do Peter Corke:

```
def rr_robot(L1=1, L2=1):

    e1 = RevoluteDH(a = L1)
    e2 = RevoluteDH(a = L2)

    rob = DHRobot([e1,e2], name = 'RR')
    return rob
```

Podemos calcular a trajetoria da pose TE1 para pose TE2 ao longo do tempo atraves da função ctraj , sendo TE1 e TE2 pertecendo no espaço especial euclidiano 3 é contido dentro do  espaço de trabalho. A tragetoria será composta por um array contendo as pose do intervalo de tempo, entre TE1 e TE2

```
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
```


**Saída:**

```
Pose inicial:
    0.866    -0.5       0         1.366  
   0.5       0.866     0        -0.366  
   0         0         1         0      
   0         0         0         1      

Pose final:
    0.866    -0.5       0         1.366  
   0.5       0.866     0        -0.866  
   0         0         1         0      
   0         0         0         1
```

<div style="display: flex;">
  <a name="figura1"></a>
  <img src="Figure_1.png" alt="q+" style="width: 47%;">
  <a name="figura2"></a>
  <img src="Figure_2.png" alt="" style="width: 45%;">
</div>

É possivel ver com se trata de robô planar contido no plano XY, não a mudação de z e nem rotação em z.

Com a tragetoria calculada, é podemos calcular o algoritmo do **resolved rate control** atraves da "matrix jacobiana" ou "Jacobiano" que usarar a tragetoria e a velocidade das juntas para minimizar o erro e o angulo até a pose final, na qual muitas vezes é ultilizado para achar combinação de **q** para uma pose sem ser necessario fazer a cinematica inversa do robô.

```
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
   
```

**Saída:**

```
q = [-1.1427465763113172, 1.3818157670603861]
```

Note que a parti de **resolved rate control** podemos encontrar a cinemática inversa do robô de maneira fácil sem utilizar as equações(vista anteriormente na lista 3), para achar os ângulos correspondentes $`\theta1`$ e $`\theta2`$ para uma pose final, note, também que como o controlador tende a diminuir o erro, não temos duas possíveis combinações de juntas para uma solução, já que ele sempre buscara o melhor caminho a parti da trajetória calculada. Mais um ponto negativo do método e a presença de posáveis singularidade na qual impede que o braço alcance o pose especificamente, isto e causado devido a det(J) = 0, como no caso acima o pose final deveria ser x =  1.366  e y = -0.866, já o ponto mais próximo é x = 1.387 e y = -0.673, como visto na figura abaixo:


<div style="display: flex; justify-content: center;">
  <a name="figura1"></a>
  <img src="Figure_5.png" alt="q+" style="width: 47%; text-align: center;">
</div>

Para melhor a aproximação podemos modificar Lambda ou aumenta o tempo para que melhor a integração do erro

### Letra B:

A matriz Jacobiana descreve como as pequenas mudanças nas variáveis de um sistema afetam suas saídas, sendo útil em análises de sistemas dinâmicos e otimização. A inversa da matriz Jacobiana é frequentemente usada para encontrar as mudanças nas variáveis que levam a uma mudança desejada nas saídas. Por outro lado, a transposta da matriz Jacobiana simplesmente reorganiza suas entradas, não sendo uma substituição direta para a inversa. A escolha entre as três depende do contexto específico do problema que se está resolvendo.

### Letra C:

Podemos ver o gráfico do erro e da juntas em função do tempo, sendo:

```
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
    plt.ylabel(' ngulo da Junta 1 (theta1)')
    plt.legend()

    # Crie um gráfico para a junta 2
    plt.subplot(2, 1, 2)
    plt.plot(t, theta2_positions, label='Junta 2')
    plt.xlabel('Tempo')
    plt.ylabel(' ngulo da Junta 2 (theta2)')
    plt.legend()

    # Exiba os gráficos
    plt.tight_layout()
    plt.title("Junta x Tempo")
    plt.show()
```
**Saída:**

<div style="display: flex;">
  <a name="figura1"></a>
  <img src="Figure_3.png" alt="q+" style="width: 47%;">
  <a name="figura2"></a>
  <img src="Figure_4.png" alt="" style="width: 45%;">
</div>

Onde no gráfico 1 podemos ver que o erro se aproxima muito de zero e as juntas são rotacionada gradivamente até alcança a pose correta.

## Questão 2:

$`^0T_3=^0T_1\cdot^1T_2\cdot^2T_3`$

$`^0T_3=\begin{bmatrix}Cos(\theta1)&-Sen(\theta1)&0&L_1Cos(\theta1)\\
Sen(\theta1)&Cos(\theta1)&0&L_1Sen(\theta1)\\
0&0&1&0\\
0&0&0&1\end{bmatrix}\begin{bmatrix}Cos(\theta2)&-Sen(\theta2)&0&L_2Cos(\theta2)\\
Sen(\theta2)&Cos(\theta2)&0&L_2Sen(\theta2)\\
0&0&1&0\\
0&0&0&1\end{bmatrix}\begin{bmatrix}Cos(\theta3)&-Sen(\theta3)&0&L_3Cos(\theta3)\\
Sen(\theta3)&Cos(\theta3)&0&L_3Sen(\theta3)\\
0&0&1&0\\
0&0&0&1\end{bmatrix}`$

$`^0T_3=\begin{bmatrix}
cos(\theta1 + \theta2 + \theta3 ) & -sin(\theta1 + \theta2 +\theta1) & 0 & L_1*cos(\theta1) + L_2*cos(\theta1 + \theta2) + L_3*cos(\theta1 + \theta2 + \theta3)\\
sin(\theta1 + \theta2 +\theta1) & cos(\theta1 + \theta2 + \theta3 ) & 0 & L_1*sin(\theta1) + L_2*sin(\theta1 + \theta2) + L_3*sin(\theta1 + \theta2 + \theta3)\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1\end{bmatrix}`$

Sendo a matriz jacobiana dada pelo metodo geometrico, por:

$`J = \begin{bmatrix}
J_{P1}&...&J_{Pn}\\
J_{O1}&...&J_{On}
\end{bmatrix}`$

Para juntas revolução:
$`\begin{bmatrix}
J_{Pi}\\
J_{Oi}
\end{bmatrix} = \begin{bmatrix}
z_{i-1}x(p-p_{i-1})\\ z_{i-1}
\end{bmatrix}`$

É para juntas Prismatica: 

$`\begin{bmatrix}
J_{Pi}\\
J_{Oi}
\end{bmatrix} = 
\begin{bmatrix}
z_{i-1}\\ 0
\end{bmatrix}`$

com i = 1,2,3 ... n

Para manipulador RRR planar, temos:

$`\bar p_0 = \begin{bmatrix}0\\0\\0\\1\end{bmatrix}`$

e

$`p = \begin{bmatrix}
L1cos(\theta1) + L2cos(\theta1 + \theta2) + L3cos(\theta1  + \theta2  + \theta3)\\
L1sin(\theta1) + L2sin(\theta1 + \theta2) + L3sin(\theta1  + \theta2  + \theta3)
\end{bmatrix}`$

$`J(q) = \begin{bmatrix}
z_{0}x(p-p_{0})&z_{1}x(p-p_{1})&z_{2}x(p-p_{2})\\ 
z_{0} &z_{1}& z_{2}
\end{bmatrix}`$

$`p_0 = ^0T_0*\bar p_0 \begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
0&0&0&1
\end{bmatrix}*\begin{bmatrix}0\\0\\0\\1\end{bmatrix} = \begin{bmatrix}0\\0\\0\end{bmatrix}`$

e  

$`z_0 = \begin{bmatrix} 0\\0\\1\end{bmatrix}`$

$`p_1 = ^0T_1*\bar p_0 \begin{bmatrix}
cos(\theta1)&-sin(\theta1)&0&L1cos((\theta1))\\
sin(\theta1)&cos(\theta1)&0&L1sin(\theta1)\\
0&0&1&0\\
0&0&0&1
\end{bmatrix}*\begin{bmatrix}L1cos(\theta1)\\L1sin(\theta1)\\0\\1\end{bmatrix} = \begin{bmatrix}L1cos(\theta1)\\L1sin(\theta1)\\0\end{bmatrix}`$

e  

$`z_1 = \begin{bmatrix} 0\\0\\1\end{bmatrix}`$

$`p_2 = ^0T_2*\bar p_0 \begin{bmatrix}
cos(\theta2)&-sin(\theta2)&0&L1cos(\theta1) +L2cos(\theta2)\\
sin(\theta2)&cos(\theta2)&0&L1sin(\theta1) + L2sin(\theta2)\\
0&0&1&0\\
0&0&0&1
\end{bmatrix}*\begin{bmatrix}L1cos(\theta1)\\L1sin(\theta1)\\0\\1\end{bmatrix} = \begin{bmatrix}L1cos(\theta1) +L2cos(\theta2)\\L1sin(\theta1) + L2sin(\theta2)\\0\end{bmatrix}`$

e  

$`z_2 = \begin{bmatrix} 0\\0\\1\end{bmatrix}`$

Temos que:

$`z_0 x (p - p_0) = \begin{bmatrix}
-L1sin(\theta1) -L2sin(\theta1 + \theta2) -L3sin(\theta1 + \theta2 + \theta3)\\
L1cos(\theta1) + L2cos(\theta1 + \theta2) +L3cos(\theta1 + \theta2 + \theta3)\\
0
\end{bmatrix}`$

$`z_1 x (p - p_1) = \begin{bmatrix}
 -L2sin(\theta1 + \theta2) -L3sin(\theta1 + \theta2 + \theta3)\\
L2cos(\theta1 + \theta2) +L3cos(\theta1 + \theta2 + \theta3)\\
0
\end{bmatrix}`$

$`z_2 x (p - p_2) = \begin{bmatrix}
-L3sin(\theta1 + \theta2 + \theta3)\\
L3cos(\theta1 + \theta2 + \theta3)\\
0
\end{bmatrix}`$

Portanto temos que a matriz jacobiana e dada por:

$`J(q) = \begin{bmatrix}-L1sin(\theta1) -L2sin(\theta1 + \theta2) -L3sin(\theta1 + \theta2 + \theta3) &-L2sin(\theta1 + \theta2) -L3sin(\theta1 + \theta2 + \theta3) & -L3sin(\theta1 + \theta2 + \theta3)\\
L1cos(\theta1) + L2cos(\theta1 + \theta2) +L3cos(\theta1 + \theta2 + \theta3)&L2cos(\theta1 + \theta2) +L3cos(\theta1 + \theta2 + \theta3)&L3cos(\theta1 + \theta2 + \theta3)\\
0&0&0\\
0&0&0\\
0&0&0\\
1&1&1
\end{bmatrix}`$

Como só tres linhas não são nulas, podemos simplificar a matriz jacobiana por:

$`J(q) = \begin{bmatrix}-L1sin(\theta1) -L2sin(\theta1 + \theta2) -L3sin(\theta1 + \theta2 + \theta3) &-L2sin(\theta1 + \theta2) -L3sin(\theta1 + \theta2 + \theta3) & -L3sin(\theta1 + \theta2 + \theta3)\\
L1cos(\theta1) + L2cos(\theta1 + \theta2) +L3cos(\theta1 + \theta2 + \theta3)&L2cos(\theta1 + \theta2) +L3cos(\theta1 + \theta2 + \theta3)&L3cos(\theta1 + \theta2 + \theta3)\\
1&1&1
\end{bmatrix}`$

Modelando o braço RR planar atraves da blibioteca ToolBox do Peter Corke:

```
def rrr_robot(L1=1, L2=1,L3 = 1):

    e1 = RevoluteDH(a = L1)
    e2 = RevoluteDH(a = L2)
    e3 = RevoluteDH(a = L3)

    rob = DHRobot([e1,e2,e3], name = 'RRR')
    return rob
```



```
def trajetoria():
    ef trajetoria():
    rrr = rrr_robot()

    q0 = np.array([-np.pi/3,np.pi/2,0])
    #print(q0)

    TE1 = rrr.fkine(q0)
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
```


**Saída:**

```
Pose inicial:
    0.866    -0.5       0         2.232     
   0.5       0.866     0         0.134      
   0         0         1         0          
   0         0         0         1          

Pose final:
    0.866    -0.5       0         2.232     
   0.5       0.866     0        -0.366   
   0         0         1         0       
   0         0         0         1     
```

<div style="display: flex;">
  <a name="figura1"></a>
  <img src="Figure_6.png" alt="q+" style="width: 47%;">
  <a name="figura2"></a>
  <img src="Figure_7.png" alt="" style="width: 45%;">
</div>


```
def resolved_rate_control_3r(L1 = 1,L2=1, L3= 1):
    """
    Resolved rate control for the RR robot.
    """
    rr = rrr_robot()
   
    Ts = trajetoria()

    t = np.arange(0, 5, 0.02)

    # Taxa de atualização de controle
    dt = 0.01

    # Posição inicial do manipulador (ângulos iniciais das juntas)
    theta1 = 0.0
    theta2 = 0.0
    theta3 = 0.0

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
        wz = T.angvec()[0]
        #print(wz)

        # Calcule o erro cartesiano
        error_x = x - (L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2) + L3 * np.cos(theta1 + theta2 +  theta3))
        error_y = y - (L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2) + L3 * np.sin(theta1 + theta2 +  theta3))
        erro_wz = wz -  np.cos(theta1 + theta2 +  theta3)
        #print(erro_wz)

        # Calcule as velocidades das juntas usando a matriz Jacobiana
        J = np.array([[-L1*np.sin(theta1) -L2*np.sin(theta1 + theta2) -L3*np.sin(theta1 + theta2 + theta3) ,-L2*np.sin(theta1 + theta2) -L3*np.sin(theta1 + theta2 + theta3) , -L3*np.sin(theta1 + theta2 + theta3)],[L1*np.cos(theta1) + L2*np.cos(theta1 + theta2) +L3*np.cos(theta1 + theta2 + theta3),L2*np.cos(theta1 + theta2) +L3*np.cos(theta1 + theta2 + theta3),L3*np.cos(theta1 + theta2 + theta3)],[1,1,1]])
        #print(J)
        #print(jsingu(J))

        # Resolva a equação de controle: J * [dtheta1, dtheta2] = [error_x, error_y]
        determinant = np.linalg.det(J)
        if abs(determinant) < 1e-6:
            lambda_value = 0.01 # Ajuste este valor conforme necessário
            J_reg = J + lambda_value * np.identity(J.shape[0])
            joint_velocities = np.linalg.solve(J_reg, np.array([error_x, error_y, erro_wz]))

        else:
            # Resolver o sistema normalmente.
            joint_velocities = np.linalg.solve(J, np.array([error_x, error_y, erro_wz]))

        # Atualize os ângulos das juntas
        theta1 += joint_velocities[0] * dt
        theta2 += joint_velocities[1] * dt
        theta3 += joint_velocities[2] * dt

        q = [theta1,theta2,theta3]
        qs.append(q)
        #rr.plot(q)

        pos = rr.fkine(q).t[:2]
        k = np.linalg.norm(pos - Ts[-1].t[:2])
        #print(k)
        erro.append(k)

        # Aguarde um tempo para a próxima iteração
        time.sleep(dt)

    print("q = {}".format([theta1,theta2,theta3]))
    #print(np.rad2deg(theta1), np.rad2deg(theta2))

    rr.teach(q)
    plots(qs,erro,t)
   
```

**Saída:**

```
q = [-0.27254193637398016, 1.18717917098305, -1.742768894051488]
```

Note que da mesma forma da questão antertior, temos que atraves do **resolved rate control** cinemática inversa do robô de maneira fácil
Note que a parti de **resolved rate control**  sendo que neste caso ultilizamos $`\omega_z`$ como uma variavel de controle para achar $`\theta3`$`. Observe também que por causa do ajuste feito na matriz jacobiana por det(J) = 0, temos um erro de aproximação da pose desejada para pose real. 


<div style="display: flex; justify-content: center;">
  <a name="figura1"></a>
  <img src="Figure_8.png" alt="q+" style="width: 47%; text-align: center;">
</div>

Para melhor a aproximação podemos modificar Lambda ou aumenta o tempo para que melhor a integração do erro

Podemos ver o gráfico do erro e da juntas em função do tempo, sendo:

```
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
    plt.ylabel(' ngulo da Junta 1 (theta1)')
    plt.legend()

    # Crie um gráfico para a junta 2
    plt.subplot(2, 1, 2)
    plt.plot(t, theta2_positions, label='Junta 2')
    plt.xlabel('Tempo')
    plt.ylabel(' ngulo da Junta 2 (theta2)')
    plt.legend()

    # Exiba os gráficos
    plt.tight_layout()
    plt.title("Junta x Tempo")
    plt.show()
```
**Saída:**

<div style="display: flex;">
  <a name="figura1"></a>
  <img src="Figure_9.png" alt="q+" style="width: 47%;">
  <a name="figura2"></a>
  <img src="Figure_10.png" alt="" style="width: 45%;">
</div>

Onde no gráfico 1 podemos ver que o erro se aproxima muito de zero e as juntas são rotacionada gradivamente até alcança a pose correta.
