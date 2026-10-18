# `adam_direction`、`replay_directions` 与 `replay_defect` 对比

这三个函数位于 `src/coordinate_geometry/adam.py`，但它们处于不同的抽象层级：

- `adam_direction` 处理**单个时间步**，是 Adam 更新规则的核心；
- `replay_directions` 处理**一整段梯度历史**，反复调用 `adam_direction`；
- `replay_defect` 比较**原始坐标系与旋转坐标系**中的回放结果，衡量 Adam 的旋转不等变程度。

## 快速对比

| 函数 | 输入 | 是否维护 Adam 状态 | 是否更新参数位置 | 输出 | 主要用途 |
|---|---|---:|---:|---|---|
| `adam_direction` | 一个梯度向量、已有状态、配置 | 是，直接修改传入的 `state` | 否 | 当前一步的 Adam 方向 | 实现 Adam 单步方向计算 |
| `replay_directions` | 一段梯度历史、配置 | 是，在函数内部创建并连续更新状态 | 否 | 每个时间步的 Adam 方向历史 | 在固定梯度历史上重建 Adam 的响应 |
| `replay_defect` | 一段梯度历史、旋转矩阵、配置 | 间接维护，共进行两次独立回放 | 否 | 原始与旋转回放之间的误差指标 | 测量 Adam 对坐标旋转的依赖 |

## 1. `adam_direction`

函数签名：

```python
def adam_direction(
    state: OptimizerState,
    gradient: np.ndarray,
    config: RunConfig,
) -> np.ndarray:
```

它接收当前梯度 \(g_t\)，更新 Adam 的内部状态：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

然后进行偏差修正：

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
\]

并返回 Adam 方向：

\[
A_t=\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

### 重要性质

1. 它只处理一个时间步。
2. 它会原地修改 `state.step`、`state.first_moment` 和 `state.second_moment`。
3. 它不会修改 `state.position`。
4. 它返回的是更新方向，而不是乘过学习率后的参数变化量。

真正更新参数的是 `adam_step`：

```python
state.position -= config.learning_rate * adam_direction(state, gradient, config)
```

因此二者的职责可以理解为：

```text
adam_direction：计算往哪里走
adam_step：按照学习率真正走一步
```

## 2. `replay_directions`

函数签名：

```python
def replay_directions(
    gradients: np.ndarray,
    config: RunConfig,
) -> np.ndarray:
```

它接收形状为 `(T, d)` 的梯度历史：

```text
[
    g_1,
    g_2,
    ...
    g_T,
]
```

其中：

- `T` 是时间步数；
- `d` 是参数空间的维度；
- 每一行是一个时间步的梯度。

函数先创建一个新的全零状态：

```python
state = OptimizerState.initial(np.zeros(history.shape[1]))
```

随后按照时间顺序，把每个梯度交给 `adam_direction`：

```python
directions = [
    adam_direction(state, g, config)
    for g in history
]
```

虽然使用了列表推导式，但每次调用使用的是同一个 `state`。因此第 \(t\) 个方向会受到之前所有梯度 \(g_1,\ldots,g_t\) 的影响。

输出为：

```text
[
    A_1,
    A_2,
    ...
    A_T,
]
```

形状仍然是 `(T, d)`。

### 为什么叫 replay？

正常优化时，流程是：

```text
当前位置
  -> 计算梯度
  -> 计算 Adam 方向
  -> 更新位置
  -> 在新位置计算下一个梯度
```

而回放时，梯度已经提前给定：

```text
给定的 g_1 -> Adam 方向 A_1
给定的 g_2 -> Adam 方向 A_2
...
给定的 g_T -> Adam 方向 A_T
```

它不会更新位置，也不会根据新位置重新计算梯度。因此它不是一次真正的优化运行，而是研究：

> 当 Adam 收到这段固定的梯度序列时，它的内部状态和输出方向会怎样变化？

这可以把 Adam 更新规则本身与目标函数、参数轨迹等因素分离开。

## 3. `replay_defect`

函数签名：

```python
def replay_defect(
    gradients: np.ndarray,
    rotation: np.ndarray,
    config: RunConfig,
) -> dict[str, np.ndarray]:
```

它使用同一段梯度历史进行两次独立回放。

### 第一次：原始梯度回放

```python
base = replay_directions(history, config)
```

得到：

\[
A_t=A(g_1,\ldots,g_t)
\]

### 第二次：旋转梯度回放

```python
transformed = replay_directions(history @ rotation.T, config)
```

如果旋转矩阵记为 \(Q\)，这次回放使用的是：

\[
Qg_1,Qg_2,\ldots,Qg_T
\]

得到：

\[
A'_t=A(Qg_1,\ldots,Qg_t)
\]

### 比较两条路径

函数比较的是下面两种操作：

```text
路径一：先运行 Adam，再旋转 Adam 方向
g_t -> Adam -> A_t -> Q A_t

路径二：先旋转梯度，再运行 Adam
g_t -> Q g_t -> Adam -> A'_t
```

如果 Adam 对旋转具有等变性，那么两条路径应当相同：

\[
A(Qg_1,\ldots,Qg_t)=Q A(g_1,\ldots,g_t)
\]

但标准 Adam 使用逐坐标平方和逐坐标除法，一般情况下：

\[
(Qg)^2\neq Q(g^2)
\]

所以对一般旋转，两条路径通常不相同。

## `replay_defect` 的三个返回值

### `numerator`

```python
numerator = np.linalg.norm(
    transformed - base @ rotation.T,
    axis=1,
)
```

对应：

\[
\left\|A'_t-QA_t\right\|
\]

这是每一步的绝对旋转缺陷。

- 等于 0：两条路径完全一致；
- 越大：旋转前后得到的 Adam 方向差异越明显。

### `denominator`

```python
denominator = np.linalg.norm(base, axis=1)
```

对应：

\[
\|A_t\|
\]

它表示原始 Adam 方向的大小，用于给误差提供尺度。

### `relative_defect`

```python
relative_defect = numerator / (denominator + 1e-12)
```

对应：

\[
\frac{\|A'_t-QA_t\|}{\|A_t\|+10^{-12}}
\]

这是归一化后的相对旋转缺陷。加入 \(10^{-12}\) 是为了避免 Adam 方向接近零时发生除零。

## 三者的调用关系

```text
replay_defect
├── replay_directions（原始梯度历史）
│   ├── adam_direction(g_1)
│   ├── adam_direction(g_2)
│   └── ...
└── replay_directions（旋转后的梯度历史）
    ├── adam_direction(Qg_1)
    ├── adam_direction(Qg_2)
    └── ...
```

也就是说：

```text
adam_direction
    是最底层的单步计算

replay_directions
    是对 adam_direction 的时间序列封装

replay_defect
    是对两次 replay_directions 结果的几何比较
```

## 一个简化示例

假设有两个时间步、两个坐标：

```python
gradients = np.array([
    [2.0, 1.0],
    [1.0, 3.0],
])
```

### 直接调用 `adam_direction`

调用者需要自己维护状态：

```python
state = OptimizerState.initial(np.zeros(2))

a1 = adam_direction(state, gradients[0], config)
a2 = adam_direction(state, gradients[1], config)
```

### 使用 `replay_directions`

状态创建和逐步调用由函数完成：

```python
directions = replay_directions(gradients, config)

a1 = directions[0]
a2 = directions[1]
```

这两种方式会得到相同的方向序列。

### 使用 `replay_defect`

如果还希望判断旋转坐标后结果是否一致：

```python
defect = replay_defect(gradients, rotation, config)

absolute_errors = defect["numerator"]
direction_norms = defect["denominator"]
relative_errors = defect["relative_defect"]
```

此时重点不再是某个具体方向，而是原始和旋转后的两套方向之间有多大差异。

## 最终总结

### `adam_direction`

回答：

> 给定当前梯度和已有 Adam 状态，这一步的 Adam 方向是什么？

### `replay_directions`

回答：

> 给定一整段固定梯度历史，Adam 在每一步会输出什么方向？

### `replay_defect`

回答：

> 同一段梯度历史旋转以后，Adam 的输出是否也按照相同方式旋转？如果没有，差异有多大？

因此，这三个函数分别对应：

```text
单步方向计算 -> 梯度历史回放 -> 旋转缺陷测量
```
