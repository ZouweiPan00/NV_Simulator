# 旋转坐标系定义

哈密顿量记为

$$
H_0 = 
\begin{bmatrix}
    \omega_1 & {} & {} & {} & {} & {} & {} & {} & {}       \\
    {} & \omega_2 & {} & {} & {} & {} & {} & {} & {}       \\
    {} & {} & \omega_3 & {} & {} & {} & {} & {} & {}       \\
    {} & {} & {} & \omega_4 & {} & {} & {} & {} & {}       \\
    {} & {} & {} & {} & \omega_5 & {} & {} & {} & {}       \\
    {} & {} & {} & {} & {} & \omega_6 & {} & {} & {}       \\
    {} & {} & {} & {} & {} & {} & \omega_7 & {} & {}       \\
    {} & {} & {} & {} & {} & {} & {} & \omega_8 & {}       \\
    {} & {} & {} & {} & {} & {} & {} & {}       & \omega_9 \\
\end{bmatrix}
$$

取了$\hbar=1$。以加$|4\rangle$到$|7\rangle$的微波脉冲为例，因为是微波，操控电子自旋，对核自旋的影响可以忽略，相互作用哈密顿量为

$$
H_1 = -\gamma_e B_1 \cos(\omega t + \phi) S_x \otimes \mathbb{I}_3 = \sqrt{2}\Omega \cos(\omega t + \phi) S_x \otimes \mathbb{I}_3
$$

其中$\Omega$是Rabi频率，$\omega$是微波频率，$\phi$是微波相位。电子自旋算符$S_x$在$\{|0\rangle, |+1\rangle, |-1\rangle\}$基底下的矩阵表示为

$$
S_x = \frac{1}{\sqrt{2}}
\begin{bmatrix}
    0 & 1 & 0 \\
    1 & 0 & 1 \\
    0 & 1 & 0 \\
\end{bmatrix}
,\quad
S_z = 
\begin{bmatrix}
    1 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & -1 \\
\end{bmatrix}
$$

旋转坐标系定义为

$$
V(t) = \exp\left(-i \omega t S_z \otimes \mathbb{I}_3\right) = \exp\left(-i \omega t S_z \right)\otimes \mathbb{I}_3
$$

在旋转坐标系下，哈密顿量为

$$
\begin{aligned}
\tilde{H}(t) &= V(t) H(t) V^\dagger(t) - i V(t) \frac{\mathrm{d}V^\dagger(t)}{\mathrm{d}t}  \\
&= V(t) H_0 V^\dagger(t) + V(t) H_1 V^\dagger(t) - \omega S_z \otimes \mathbb{I}_3 \\
&= \tilde{H}_0 + \tilde{H}_1(t)
\end{aligned}
$$

其中

$$
\tilde{H}_0 = V(t) H_0 V^\dagger(t) - \omega S_z \otimes \mathbb{I}_3 = \begin{bmatrix}
    \omega_1 - \omega & {} & {} & {} & {} & {} & {} & {} & {}       \\
    {} & \omega_2 - \omega & {} & {} & {} & {} & {} & {} & {}       \\
    {} & {} & \omega_3 - \omega & {} & {} & {} & {} & {} & {}       \\
    {} & {} & {} & \omega_4 & {} & {} & {} & {} & {}       \\
    {} & {} & {} & {} & \omega_5  & {} & {} & {} & {}       \\
    {} & {} & {} & {} & {}  & \omega_6  & {} & {} & {}       \\
    {} & {} & {} & {}  &  {}  &  {}  & \omega_7 + \omega  &  {}  &  {}       \\
    {} & {} & {} & {}  &  {}  &  {}  &  {}  & \omega_8 + \omega  &  {}       \\
    {} & {} & {} & {}  &  {}  &  {}  &  {}  &  {}  & \omega_9 + \omega  \\
\end{bmatrix}
$$

而

$$
\begin{aligned}
\tilde{H}_1(t) &= V(t) H_1 V^\dagger(t) = \sqrt{2}\Omega \cos(\omega t + \phi) V(t) S_x \otimes \mathbb{I}_3 V^\dagger(t) \\
&= \sqrt{2}\Omega \cos(\omega t + \phi) \begin{bmatrix}
    0 & e^{-i\omega t} & 0 \\
    e^{i\omega t} & 0 & e^{i\omega t} \\
    0 & e^{-i\omega t} & 0 \\
\end{bmatrix} \otimes \mathbb{I}_3
\end{aligned}
$$