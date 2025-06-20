# Log-Sum-Exp Trick 的推导步骤

**目标：** 计算 $\log\left(\sum_{i=1}^N \exp(a_i)\right)$。

**问题：** 如果直接计算 $\exp(a_i)$，当 $a_i$ 很大时可能导致**上溢（overflow）**，当 $a_i$ 很小时可能导致**下溢（underflow）**。

**核心思想：** 从求和项中提取一个公共因子，通常选择为最大的 $a_i$。

---

**推导步骤：**

**Step 1: 引入常数 $M$**

我们选择 $M = \max_{j}(a_j)$。

原始和的对数：
$$
\log\left(\sum_{i=1}^N \exp(a_i)\right)
$$

**Step 2: 改变指数项的结构**

在 $\exp(a_i)$ 内部加减 $M$：$a_i = (a_i - M) + M$。

$$
\log\left(\sum_{i=1}^N \exp((a_i - M) + M)\right)
$$

**Step 3: 利用指数的性质 $\exp(x+y) = \exp(x)\exp(y)$**

将 $M$ 从指数中分离出来：

$$
\log\left(\sum_{i=1}^N \exp(a_i - M) \cdot \exp(M)\right)
$$

**Step 4: 将公共因子 $\exp(M)$ 提取到求和符号外部**

$$
\log\left(\exp(M) \sum_{i=1}^N \exp(a_i - M)\right)
$$

**Step 5: 利用对数的性质 $\log(xy) = \log(x) + \log(y)$**

将 $\exp(M)$ 的对数分离出来：

$$
\log(\exp(M)) + \log\left(\sum_{i=1}^N \exp(a_i - M)\right)
$$

**Step 6: 利用对数的性质 $\log(\exp(x)) = x$**

简化第一项：

$$
M + \log\left(\sum_{i=1}^N \exp(a_i - M)\right)
$$

**最终结果：**

这就是 Log-Sum-Exp Trick 的最终公式。通过这种方式，`exp(a_i - M)` 中的所有指数参数都变得相对较小且不为正（最大为 0），从而避免了在计算指数时可能发生的数值溢出或下溢。