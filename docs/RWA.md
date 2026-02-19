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
    {} & {} & {} & {} & {} & {} & \omega_7 & {} & {}       \\
    {} & {} & {} & {} & {} & {} & {} & \omega_8 & {}       \\
    {} & {} & {} & {} & {} & {} & {} & {}       & \omega_9 \\
\end{bmatrix}
$$

取了$\hbar=1$。

## 电子自旋操控例子

以加$|4\rangle$到$|7\rangle$的微波脉冲为例，因为是微波，操控电子自旋，对核自旋是强偏共振的，影响可以忽略，相互作用哈密顿量为

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
    {} & {} & {} & {} & \omega_5  & {} & {} & {} & {}       \\
    {} & {} & {} & {} & {}  & \omega_6  & {} & {} & {}       \\
    {} & {} & {} & {}  &  {}  &  {}  & \omega_7 + \omega  &  {}  &  {}       \\
    {} & {} & {} & {}  &  {}  &  {}  &  {}  & \omega_8 + \omega  &  {}       \\
    {} & {} & {} & {}  &  {}  &  {}  &  {}  &  {}  & \omega_9 + \omega  \\
\end{bmatrix}
$$

而

$$
\begin{aligned}
\tilde{H}_1(t) &= V(t) H_1 V^\dagger(t) = \sqrt{2}\Omega \cos(\omega t + \phi) V(t) S_x \otimes \mathbb{I}_3 V^\dagger(t) \\
&= \Omega \cos(\omega t + \phi) \begin{bmatrix}
    0 & e^{-i\omega t} & 0 \\
    e^{i\omega t} & 0 & e^{i\omega t} \\
    0 & e^{-i\omega t} & 0 \\
\end{bmatrix} \otimes \mathbb{I}_3\\
&= \Omega \frac{e^{i(\omega t + \phi)} + e^{-i(\omega t + \phi)}}{2} \begin{bmatrix}
    0 & e^{-i\omega t} & 0 \\
    e^{i\omega t} & 0 & e^{i\omega t} \\
    0 & e^{-i\omega t} & 0 \\
\end{bmatrix} \otimes \mathbb{I}_3\\
&\approx \frac{\Omega}{2} \begin{bmatrix}
    0 & e^{i\phi} & 0 \\
    e^{-i\phi} & 0 & e^{-i\phi} \\
    0 & e^{i\phi} & 0 \\
\end{bmatrix} \otimes \mathbb{I}_3
\end{aligned}
$$

由于在这个例子中，$\omega=\omega_4-\omega_7$，所以
$$
\tilde{H}_0 = \omega_4 \mathbb{I}_9 + \begin{bmatrix}
    \Delta_1 & {} & {} & {} & {} & {} & {} & {} & {}       \\
    {} & \Delta_2 & {} & {} & {} & {} & {} & {} & {}       \\
    {} & {} & \Delta_3 & {} & {} & {} & {} & {} & {}       \\
    {} & {} & {} & 0 & {} & {} & {} & {} & {}       \\
    {} & {} & {} & {} & \Delta_5  & {} & {} & {} & {}       \\
    {} & {} & {} & {} & {}  & \Delta_6  & {} & {} & {}       \\
    {} & {} & {} & {}  &  {}  &  {}  & 0  &  {}  &  {}       \\
    {} & {} & {} & {}  &  {}  &  {}  &  {}  & \Delta_8  &  {}       \\
    {} & {} & {} & {}  &  {}  &  {}  &  {}  &  {}  & \Delta_9  \\
\end{bmatrix}
$$

$\Delta_i$的定义如下

| $\Delta_1$ | $\Delta_2$ | $\Delta_3$ | $\Delta_5$ | $\Delta_6$ | $\Delta_8$ | $\Delta_9$ |
| --- | --- | --- | --- | --- | --- | --- |
| $\omega_1-\omega-\omega_4$ | $\omega_2-\omega-\omega_4$ | $\omega_3-\omega-\omega_4$ | $\omega_5-\omega_4$ | $\omega_6-\omega_4$ | $\omega_8+\omega-\omega_4$ | $\omega_9+\omega-\omega_4$ |
| $\omega_1-2\omega_4+\omega_7$ | $\omega_2-2\omega_4+\omega_7$ | $\omega_3-2\omega_4+\omega_7$ | $\omega_5-\omega_4$ | $\omega_6-\omega_4$ | $\omega_8-\omega_7$ | $\omega_9-\omega_7$ |
| $2D$ | $2D-Q-A-\omega_n$ | $2D-2A-2\omega_n$ | $-Q-\omega_n$ | $-2\omega_n$ | $A-Q-\omega_n$ | $2A-2\omega_n$ |

因此总哈密顿量为

$$
\tilde{H}(t) = \tilde{H}_0 + \tilde{H}_1(t) = \omega_4 \mathbb{I}_9 +
\begin{bmatrix}
    \Delta_1 & 0 & 0 & g_\phi & 0 & 0 & 0 & 0 & 0 \\
    0 & \Delta_2 & 0 & 0 & g_\phi & 0 & 0 & 0 & 0 \\
    0 & 0 & \Delta_3 & 0 & 0 & g_\phi & 0 & 0 & 0 \\
    g_\phi^\ast & 0 & 0 & 0 & 0 & 0 & g_\phi^\ast & 0 & 0 \\
    0 & g_\phi^\ast & 0 & 0 & \Delta_5 & 0 & 0 & g_\phi^\ast & 0 \\
    0 & 0 & g_\phi^\ast & 0 & 0 & \Delta_6 & 0 & 0 & g_\phi^\ast \\
    0 & 0 & 0 & g_\phi & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & g_\phi & 0 & 0 & \Delta_8 & 0 \\
    0 & 0 & 0 & 0 & 0 & g_\phi & 0 & 0 & \Delta_9
\end{bmatrix}
$$

其中
$$
g_\phi = \frac{\Omega}{2}e^{i\phi}
$$

注意对于$|4\rangle\leftrightarrow|7\rangle$的共振驱动，是$|5\rangle\leftrightarrow|8\rangle$偏共振$A$和$|6\rangle\leftrightarrow|9\rangle$偏共振$2A$的驱动。当脉冲较强时，需要考虑这两个效果。




## 核自旋操控例子

以加$|4\rangle$到$|5\rangle$的射频脉冲为例，因为是射频，操控核自旋，对电子自旋是强偏共振的，影响可以忽略，相互作用哈密顿量为

$$
H_1 = -\gamma_n B_1 \cos(\omega t + \phi) \mathbb{I}_3 \otimes I_x = \sqrt{2}\Omega \cos(\omega t + \phi) \mathbb{I}_3 \otimes I_x
$$

其中

$$
I_x = \frac{1}{\sqrt{2}}
\begin{bmatrix}
    0 & 1 & 0 \\
    1 & 0 & 1 \\
    0 & 1 & 0 \\
\end{bmatrix}
,\quad
I_z = 
\begin{bmatrix}
    1 & 0 & 0 \\
    0 & 0 & 0 \\
    0 & 0 & -1 \\
\end{bmatrix}
$$

旋转坐标系定义为

$$
V(t) = \exp\left(-i \omega t \mathbb{I}_3 \otimes I_x\right) = \mathbb{I}_3 \otimes \exp\left(-i \omega t I_x\right)
$$

直接使用上述$x$轴旋转系时，$H_0$中的$I_z$与$I_z^2$项会变成显式含时项，不便于写成定常RWA形式。下面沿用前文做法，在核自旋空间改用$z$轴旋转系：

$$
V_z(t) = \exp\left(-i \omega t \mathbb{I}_3 \otimes I_z\right)
= \mathbb{I}_3 \otimes \exp\left(-i \omega t I_z \right)
$$

在旋转坐标系下，哈密顿量为

$$
\begin{aligned}
\tilde{H}(t) &= V_z(t) H(t) V_z^\dagger(t) - i V_z(t) \frac{\mathrm{d}V_z^\dagger(t)}{\mathrm{d}t}  \\
&= V_z(t) H_0 V_z^\dagger(t) + V_z(t) H_1 V_z^\dagger(t) - \omega \mathbb{I}_3 \otimes I_z \\
&= \tilde{H}_0 + \tilde{H}_1(t)
\end{aligned}
$$

其中

$$
\tilde{H}_0 = V_z(t) H_0 V_z^\dagger(t) - \omega \mathbb{I}_3 \otimes I_z
= \begin{bmatrix}
    \omega_1 - \omega & {} & {} & {} & {} & {} & {} & {} & {} \\
    {} & \omega_2 & {} & {} & {} & {} & {} & {} & {} \\
    {} & {} & \omega_3 + \omega & {} & {} & {} & {} & {} & {} \\
    {} & {} & {} & \omega_4 - \omega & {} & {} & {} & {} & {} \\
    {} & {} & {} & {} & \omega_5 & {} & {} & {} & {} \\
    {} & {} & {} & {} & {} & \omega_6 + \omega & {} & {} & {} \\
    {} & {} & {} & {} & {} & {} & \omega_7 - \omega & {} & {} \\
    {} & {} & {} & {} & {} & {} & {} & \omega_8 & {} \\
    {} & {} & {} & {} & {} & {} & {} & {} & \omega_9 + \omega
\end{bmatrix}
$$

而

$$
\begin{aligned}
\tilde{H}_1(t) &= V_z(t) H_1 V_z^\dagger(t)
= \sqrt{2}\Omega \cos(\omega t + \phi)\, V_z(t)\, \mathbb{I}_3 \otimes I_x\, V_z^\dagger(t) \\
&= \Omega \cos(\omega t + \phi)\, \mathbb{I}_3 \otimes
\begin{bmatrix}
    0 & e^{-i\omega t} & 0 \\
    e^{i\omega t} & 0 & e^{i\omega t} \\
    0 & e^{-i\omega t} & 0 \\
\end{bmatrix}\\
&= \Omega \frac{e^{i(\omega t + \phi)} + e^{-i(\omega t + \phi)}}{2}\,
\mathbb{I}_3 \otimes
\begin{bmatrix}
    0 & e^{-i\omega t} & 0 \\
    e^{i\omega t} & 0 & e^{i\omega t} \\
    0 & e^{-i\omega t} & 0 \\
\end{bmatrix}\\
&\approx \frac{\Omega}{2}\, \mathbb{I}_3 \otimes
\begin{bmatrix}
    0 & e^{i\phi} & 0 \\
    e^{-i\phi} & 0 & e^{-i\phi} \\
    0 & e^{i\phi} & 0
\end{bmatrix}
\end{aligned}
$$

由于在这个例子中，$\omega=\omega_4-\omega_5$，所以

$$
\tilde{H}_0 = \omega_5 \mathbb{I}_9 + \begin{bmatrix}
    \Delta_1 & {} & {} & {} & {} & {} & {} & {} & {} \\
    {} & \Delta_2 & {} & {} & {} & {} & {} & {} & {} \\
    {} & {} & \Delta_3 & {} & {} & {} & {} & {} & {} \\
    {} & {} & {} & 0 & {} & {} & {} & {} & {} \\
    {} & {} & {} & {} & 0 & {} & {} & {} & {} \\
    {} & {} & {} & {} & {} & \Delta_6 & {} & {} & {} \\
    {} & {} & {} & {} & {} & {} & \Delta_7 & {} & {} \\
    {} & {} & {} & {} & {} & {} & {} & \Delta_8 & {} \\
    {} & {} & {} & {} & {} & {} & {} & {} & \Delta_9
\end{bmatrix}
$$

$\Delta_i$的定义如下

| $\Delta_1$ | $\Delta_2$ | $\Delta_3$ | $\Delta_6$ | $\Delta_7$ | $\Delta_8$ | $\Delta_9$ |
| --- | --- | --- | --- | --- | --- | --- |
| $\omega_1-\omega_4$ | $\omega_2-\omega_5$ | $\omega_3+\omega_4-2\omega_5$ | $\omega_6+\omega_4-2\omega_5$ | $\omega_7-\omega_4$ | $\omega_8-\omega_5$ | $\omega_9+\omega_4-2\omega_5$ |
| $D+\omega_e+A$ | $D+\omega_e$ | $D+\omega_e+2Q-A$ | $2Q$ | $D-\omega_e-A$ | $D-\omega_e$ | $D-\omega_e+2Q+A$ |

因此总哈密顿量为

$$
\tilde{H}(t) = \tilde{H}_0 + \tilde{H}_1(t) = \omega_5 \mathbb{I}_9 +
\begin{bmatrix}
    \Delta_1 & g_\phi & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    g_\phi^\ast & \Delta_2 & g_\phi^\ast & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & g_\phi & \Delta_3 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & g_\phi & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & g_\phi^\ast & 0 & g_\phi^\ast & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & g_\phi & \Delta_6 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & \Delta_7 & g_\phi & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & g_\phi^\ast & \Delta_8 & g_\phi^\ast \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & g_\phi & \Delta_9
\end{bmatrix}
$$

其中

$$
g_\phi = \frac{\Omega}{2}e^{i\phi}
$$

注意由于$\gamma_n$较小，驱动强度通常较弱，因此偏共振效应可以忽略。
