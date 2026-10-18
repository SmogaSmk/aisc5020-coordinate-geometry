# 数学与文字解释各个函数在做什么
## `adam.py`
`adam_direction()`:
更新 Adam 的内部矩状态并计算每一步的 Adam 方向，但不更新参数位置。最后，将所有时间步的 Adam 方向组合成一个新的矩阵返回。

`adam_step()`: 
根据当前的学习率与梯度方向，向前位移一步

`replay_directions()`:
首先接收一段已有的梯度历史，它是一个二维矩阵，其中每一行都是某个时间步的梯度向量。函数从零初始化 Adam 的一阶矩、二阶矩和时间步，然后按照时间顺序将每个梯度依次传入 `adam_direction()`，计算对应的 Adam 更新方向。该过程只更新 Adam 的内部状态，不更新参数位置。最后，函数使用 `np.stack()` 将各时间步的 Adam 方向组合成一个新的方向历史矩阵并返回。

`replay_Defect()`:
1. 接收一个历史梯度:
$ G=[ g_1, g_2, \dots, g_T ]^{\top}$, 每一行是一个时间步的梯度向量

2. 回放原始梯度历史:
``` python
base = replay_directions(history, config)
```

这里重新调用了一次`replay_directions()`，锚定要计算的原始坐标系下的Adam方向历史，为了后面的进行比较，在计算中，他会得到 $ A = [A_1, A_2, \dots, A_n]^{\top} $ 其中 $A_i$ 是每个时间步的Adam方向

3. 先旋转梯度，再回放梯度  
如果旋转矩阵可定义为
$$
Q=\begin{bmatrix}
\cos\theta & -\sin\theta \\ 
\sin\theta & \cos\theta
\end{bmatrix}
$$
在代码中，每个梯度是行向量，需要对他进行转置操作
$$
g^\top=Qg \\
(g^\top)^\top = (Qg)^{\top} = g^{\top}Q^{\top}
$$
最后推导出来的结果刚好是我们代码的原排列（行向量），所以实际点乘的时候，只需要转置旋转矩阵Q即可  
在这里我们直接旋转的原始梯度，不是Adam更新方向，`replay_directions()`帮我们实现批量旋转，得到了一个矩阵，其形式与Adam方向历史矩阵相似，但是含义不同。
4. 为了找到旋转梯度再算Adam，还是直接旋转Adam的差值，用下述代码即可完成
``` python
numerator = np.linalg.norm(transformed - base @ rotation.T, axis=1)
```
注意此处base也是按照行向量存储的Adam方向，所以需要rotation进行转置

``` mermid
                      原始梯度历史 history
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
       replay_directions             先旋转梯度
                │                  history @ rotation.T
                ▼                           │
     base：基准 Adam 方向                   ▼
                │                  replay_directions
                ▼                           │
        再旋转 Adam 方向                    ▼
     base @ rotation.T       transformed：旋转输入后的 Adam 方向
                │                           │
                └─────────────┬─────────────┘
                              ▼
                      计算二者之间的差
```
之所以为什么要看比值，是因为绝对偏差会有尺度因素影响，每一个时间步的更新大小也不同，且初始化因素的影响，不同实验可能具有不同的梯度大小，初值，问题规模，特征值

## `adam_diagnostics.py`
该文件是执行任务的文件，其使用的函数可以在`adam.py`中找到，参数配置与项目代码风格一致

IO此处直接跳过，其与主要项目文件无关，后续直接给出IO接口方便前后衔接即可

大致流程  
- 诊断A: 固定梯度历史回放  
        用Gradient Descent(下文简称GD)生成一段固定梯度历史
        分别把原梯度核旋转后的梯度输入Adam
        比较Adam输出方向是否保持旋转对应  
        这3步骤的操作，主要想解答的是: "即使输入的两段梯度严格通过旋转对应，Adam 的逐坐标归一化是否会破坏这种对应关系？"

- 诊断B: 运行普通的Adam  
        在原始问题的二次型上运行Adam + 在旋转后的二次型上运行Adam
        比较两条参数轨迹是否保持旋转对应  
        他想要回答的问题是：  
        普通 Adam 真正在两个旋转对应的问题上运行时，参数轨迹是否仍然通过旋转对应？

也就是说单独研究Adam更新公式，和观察完整优化轨迹

### 函数拆解分析
1. `canonical_gradient_history()`  
在 canonical 二次问题上运行 GD，并记录每次参数更新之前的梯度。  
我们的输入是一个二次型问题：$f(\mathbf{w})=\frac{1}{2}\mathbf{w}^\top H\mathbf{w}$, 它的梯度可以通过直接求导计算 $\nabla f(\mathbf{w})=H\mathbf{w}$, 传入的是canonical problem,这也要求Hessian矩阵应当是对角阵
$$
H=D=\begin{pmatrix}
\lambda_1 & 0 \\ 
0 & \lambda_2
\end{pmatrix}
$$

- 因为我们要找到朴素GD的History，所以必须遍历全部历史步max_steps，
按照时间顺序的简易流程如下：
$$
w_0 \xrightarrow{\text{计算}} 
g_1 \xrightarrow{\text{GD 更新}} 
w_1 \xrightarrow{\text{计算}} 
g_2 \xrightarrow{\text{GD 更新}} 
w_2.
$$
通过该流程后最后得到一组梯度矩阵（数组）
$$
G=\begin{pmatrix}
g_1^\top \\ 
g_2^\top \\ 
\vdots \\
g_T^\top
\end{pmatrix}
$$
该数组的形状为`(max_steps, dimension)` 

- 为什么用GD产生梯度
对于梯度下降更新问题和(半)正定问题：
$$
\text{update: } \mathbf{w}_{t+1}=\mathbf{w}_t-\eta \nabla f(\mathbf{w}_t) \\ 
\text{positive definite: } f(\mathbf{w})=\frac{1}{2}\mathbf{w}^\top H\mathbf{w} \\ 
\text{gradient: } \nabla f(\mathbf{w}) = H\mathbf{w}
$$
将更新公式带入到gradient中，即可得到:
$$
\mathbf{w}_{t+1}=\mathbf{w}_t-\eta H\mathbf{w}_t \\
\mathbf{w}_{t+1}=(I-\eta H)\mathbf{w}_t
$$
上述是原始坐标系中的二次问题。为避免与矩阵转置符号混淆，下面用波浪号表示旋转后的问题和变量。令 $Q$ 为正交矩阵，即 $Q^\top Q=I$，并定义
$$
\widetilde H=QHQ^\top,\qquad \widetilde{\mathbf{w}}_0=Q\mathbf{w}_0.
$$
这里 $\widetilde H$ 是旋转后问题的 Hessian，而不是 $H$ 的转置。假设第 $t$ 步的位置满足
$$
\widetilde{\mathbf{w}}_t=Q\mathbf{w}_t,
$$
那么旋转后问题的梯度为
$$
\begin{aligned}
\widetilde g_t
&=\widetilde H\widetilde{\mathbf{w}}_t \\
&=(QHQ^\top)(Q\mathbf{w}_t) \\
&=QH\mathbf{w}_t \\
&=Qg_t.
\end{aligned}
$$
将该梯度代入 GD 更新可得
$$
\begin{aligned}
\widetilde{\mathbf{w}}_{t+1}
&=\widetilde{\mathbf{w}}_t-\eta\widetilde g_t \\
&=Q\mathbf{w}_t-\eta Qg_t \\
&=Q(\mathbf{w}_t-\eta g_t) \\
&=Q\mathbf{w}_{t+1}.
\end{aligned}
$$
初始点已经满足 $\widetilde{\mathbf{w}}_0=Q\mathbf{w}_0$，因此根据数学归纳法，对所有时间步都有
$$
\widetilde{\mathbf{w}}_t=Q\mathbf{w}_t,\qquad
\widetilde g_t=Qg_t.
$$

这正是使用朴素 GD 构造梯度历史的原因。GD 使用同一个标量学习率，并且更新中没有逐坐标平方、开方或除法，所以它与正交坐标旋转相容。由 canonical 问题产生梯度历史
$$
g_1,g_2,\ldots,g_T
$$
以后，无须重新运行另一个可能产生不同轨迹的优化器，直接旋转即可得到与之严格对应的历史
$$
Qg_1,Qg_2,\ldots,Qg_T.
$$

``` mermid 
                w_t  ─────GD────→  w_{t+1}
                │                    │
                Q                    Q
                │                    │
                ▼                    ▼
                Qw_t ─────GD────→ Qw_{t+1}
```
原问题的 GD 产生 $((w_t,g_t))$。将目标函数和初始点整体旋转后，旋转问题的 GD 会产生 $((Qw_t,Qg_t))$。因此，在 replay 实验中，可以直接将原梯度历史乘以 $(Q)$，把它作为旋转问题对应的正确梯度历史，而不必重新运行旋转后的 GD

上面的论证，也就是该函数`canonical_gradient_history()`的设计的Inner motivation: 为了给Adam replay提供两段严格旋转对应的输入，所以用GD构造梯度历史序列和旋转梯度历史序列$g_t$和$Qg_t$

**但是我们的Outer Motivation还没有回答**
$$
f(w)=\frac{1}{2}w^\top Hw,
\qquad
\widetilde f(\widetilde w)=\frac{1}{2}\widetilde w^\top
(QHQ^\top)\widetilde w
$$
根据线性代数的一些知识(相似对角化)我们知道，他们有相同的
- eigenvalues
- conditional number
- minimum
- Geometry shape 
- Theoritical difficulty
这种前后行为相当于一个椭圆形的碗，建立在坐标系上，长轴为x轴，这时候旋转了一下碗，那么长轴倾斜了45°。从问题本身来看，就是摆放方向变了而已。这是从人类视角来没什么事情，但是对于计算机和算法来说，这里可能发生变化。  
对于一个单纯的正交旋转，如果优化器在这个问题上的表现不同，那么说明：优化器不仅仅看到了问题的几何形状，还受到了人为坐标系的影响，也就是说本研究想要探讨的RQ1: coordinate dependence ?

**到这里是不是结束了？** 你去构造一个原坐标系的初值，再构造一个旋转坐标的初值，让他们去执行一遍Adam，然后比较结果。

当然不是。
我们这样做相当于跳过了全部中间过程，只看到了最后的结果上的差异，缺少了"差异是如何产生的"这样的问题，展开Adam的更新全过程：
```mermid
第一步方向不同
    ↓
第一步位置不同
    ↓
第二步梯度不同
    ↓
第二步矩状态不同
    ↓
后续轨迹差异越来越复杂
```
也就是说中间的过程是机制复杂的，这时候canonical_gradient_history就可以给我们提供两种情况下的梯度历史。简单来说，就是想知道"坐标轴画法不同"导致的Adam在本质相同问题做出了不同反应，然后要用什么构造，就用canonical_gradient_history提供了两组可以供我们研究的梯度历史

2. `adam_position_history()`的  
如果理清楚了canonical_gradient_history的作用的话，这个函数其实就很清晰了，当然为了理清代码，我们还是一步一步来
```python
state = OptimizerState.initial(initial)
positions = [state.position.copy()]
```
不难发现，adam_position是有初始值的，而上一个函数是没有初始值的，因为我们这里计算的是轨迹，轨迹是有出发点的，所以初始化了实际存在的点。而gradient是存放每一步更新前的梯度，我们选择在计算过程加入梯度，因此这里没有初始值

后面就很清晰了：GD 部分记录标准坐标系中的梯度历史，并由它构造旋转后的对应梯度；Adam 部分则分别记录标准问题和旋转问题中实际运行得到的位置历史。

同样，我们还是从motivation的角度出发
**Inner Motivation**：完整记录Adam的状态位置变化
用一个简易流水线来描述，大概是这样的一个过程：
$$
x_t \rightarrow g_t=\nabla f(x_t) 
\rightarrow (m_t, v_t) \rightarrow x_{t+1}
$$
- `state.position`是当前参数$x_t$
- `gradient`是当前位置上的梯度$g_t$
- `adam_step` 更新一阶矩，二阶矩以及参数位置
- `positions.append(...)`保存更新后的$x_{t+1}$ 
使用`.copy()`则是希望记录了每个位置在对应时刻的独立历史记录，而不是一个会随状态位置发生改变的可变数组，所以用copy记录当时时刻快照

简而言之，Inner motivation可以说是想要可靠地运行Adam，并将每一步参数位置保存为互相独立的历史快照

**Outer Motivation**  
从函数外部看，我们想要帮助回答"同一个优化问题旋转坐标系以后，Adam的优化轨迹是否也只是跟着旋转？" 用数学语言表达就是：
$$
x_0, x_1, \ldots, x_T \\
x_0^{(R)}, x_1^{(R)}, \ldots, x_T^{(R)}
$$
如果Adam具有旋转等变性，那么它应当满足：
```
x_t^{(R)} = Rx_t
```
对所有时间步$\{t_{i}\}$都成立。想要所有的历史，那么我们就要有函数负责记录位置历史，然后才能逐步比较：
$$
x_t^{(R)} \quad\text{and}\quad Rx_t
$$
如果只有最终位置$x_T$，我们只能知道最终结果是否不同，却无法观察：
- 偏离从哪一步开始
- 偏离如何随着迭代累积
- 某一步是否突然产生较大误差
- 初始位置发旋转对应的关系是否正确
简单来说就是我们构建这两条轨迹是为了后续比较旋转后的实际轨迹和理论旋转轨迹$x_t^{(R)}=Rx_{t}?$

$$
0^{\circ}
$$