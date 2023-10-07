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
Sen(\theta1)&Cos(\theta1)&0&L_1S_1\\
0&0&1&0\\
0&0&0&1\end{bmatrix}\begin{bmatrix}Cos(\theta2)&-Sen(\theta2)&0&L_2Cos(\theta2)\\
Sen(\theta2)&Cos(\theta2)&0&L_2S_2\\
0&0&1&0\\
0&0&0&1\end{bmatrix}`$

$`^0T_2=\begin{bmatrix}Cos(\theta1)Cos(\theta2)&Sen(\theta1)Sen(\theta2)&0&L_1Cos(\theta1)+L_2Cos(\theta1)Cos(\theta2)\\
Sen(\theta1)Sen(\theta2)&Cos(\theta1)Cos(\theta2)&0&L_1Sen(\theta1)+L_2Sen(\theta1)Sen(\theta2)\\
0&0&1&0\\
0&0&0&1\end{bmatrix}`$

</div>

Temos, podemos expressa x e y:


<div align="center">

$`x = L_1Cos(\theta1)+L_2Cos(\theta1)Cos(\theta2)`$

$`y = L_1Sen(\theta1)+L_2Sen(\theta1)Sen(\theta2)`$

</div>

De forma analitica pode expresa a matriz jacobiana 

<div align="center">

$`J(q) = \frac{\partial}{\partial(q)}K(q)=\begin{bmatrix}\frac{\partial x}{\partial(θ1)}&\frac{\partial x}{\partial(θ2)}\\\frac{\partial y}{\partial(θ1)}&\frac{\partial y}{\partial(θ2)}\end{bmatrix}`$

</div>

Efetuando as derivadas:

$`\frac{\partial x}{\partial(θ1)} =  -(L_1 + L_2 cos(\theta2)) sin(θ1)`$

$`\frac{\partial x}{\partial(θ2)} =  -L cos(θ1) sin(θ2)`$

$`\frac{\partial y}{\partial(θ1)} =  cos(θ)(L1 + sin(θ2))`$

$`\frac{\partial y}{\partial(θ2)} =  cos(θ2)sin(θ1)`$

$`J(q) =\begin{bmatrix}-(L_1 + L_2 cos(\theta2)) sin(θ1)&-L cos(θ1) sin(θ2)\\cos(θ)(L1 + sin(θ2))& cos(θ2)sin(θ1)\end{bmatrix}`$

### Letra A:

