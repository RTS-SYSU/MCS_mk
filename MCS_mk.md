### Response TIme Analysis
- RTA LO Mode $R_i^{LO}$
- RTA Mode Switch $R_i^{*}$
- RTA HI Mode $R_i^{HI}$


<br>



### Importance Model

importance 不直接定义在单个 job 上，而是定义在 task-level service 上。

对于一个 LO task $\tau_i$，只要它满足 baseline $(m_i,k_i)$ constraint，就获得一个基础 importance：$I_i^{base}$。

因此，$I_i^{base}$ 表示任务最低功能被保证时获得的主要收益，如果mk无法被满足，则 $I_i^{base}=0$

在 baseline 之上，如果任务被提升为：$(m_i+x_i,k_i)$,


则额外的 $x_i$ 表示 augmented importance，$0 \le x_i \le k_i-m_i$。  

$x_i$ 提供额外 importance，但这个额外收益应该小于 $I_i^{base}$。

则任务 $\tau_i$ 在 augmented level $x_i$ 下的 importance 可以定义为：
$$
I_i(x_i)=
I_i^{base}
\left(
1+\alpha_i\frac{x_i}{k_i-m_i}
\right),
$$
其中：$0 < \alpha_i < 1$, $\alpha_i$ 表示从 baseline 提升到 full service 时，最多可以额外获得多少比例的$I_i^{base}$。


因此：

- baseline $(m_i,k_i)$ 提供主要 importance；
- augmented jobs 只提供额外 improvement；
- <mark style="background-color: #ffcdd2; color: #000;"> 恢复一个 suspended task 的 baseline service 通常比提升已有任务的 $x_i$ 更重要。


<!-- 如果 retained task $\tau_j$ 的 augmented level 从 $x_j^{old}$ 降低到 $x_j^{new}$，则 sacrificed importance 为：
$$
Loss_j=
I_j(x_j^{old}) - I_j(x_j^{new}).
$$
因此，一次 recovery 被接受的 importance 条件为：
$$
Gain_i >
\sum_{\tau_j \in \Gamma_{LO}^{*}} Loss_j.
$$
也就是：
$$
I_i^{base}>
\sum_{\tau_j \in \Gamma_{LO}^{*}}
\left(
I_j(x_j^{old}) - I_j(x_j^{new})
\right).
$$
-->

<br>

<!-- <div style="border: 1px solid #555; padding: 12px; border-radius: 8px; background-color: #e6e3e3;"> -->

### Interference Upper Bound under Augmented $(m,k)$ Pattern

对于 LO task $\tau_i$，如果 augmented pattern： $(m_i^\chi,k_i)$, 

其中 $m_i^\chi = m_j+x_j^\chi$，且 $\chi \in \{l,s,h\}.$

分别表示：

- $x_i^l$：LO mode 下的 augmented level；
- $x_i^s$：mode-switch interval 下的 degraded augmented level；
- $x_i^h$：stable HI mode 的 augmented level。

在长度为 $t$ 的时间窗口内，任务 $\tau_i$ 最多释放的 job 数量为：$N_i(t)=\left\lceil \frac{t}{T_i} \right\rceil.$

在 augmented $(m,k)$ pattern 下，$\tau_i$ 在该窗口内最多参与调度的 mandatory jobs 数量上界为：

$$
\eta_i(t,m_i^\chi,k_i)=
\left\lfloor
\frac{N_i(t)}{k_i}
\right\rfloor m_i^\chi
+
\min
\left(
N_i(t) \bmod k_i,\ m_i^\chi
\right).
$$

因此，$\tau_i$ 对任务 $\tau_j$ 在长度 $t$ 内造成的 interference upper bound 为：

$$
I_{i \rightarrow j}^{\chi}(t)
=\eta_i(t,m_i^\chi,k_i) C_i^{LO}.
$$

<!-- </div>  -->

相比于 baseline $(m_i,k_i)$ pattern，增加 $x_i^\chi$ 额外引入的 interference 为：

$$
E_{i\rightarrow j}^{\chi}(x_i^\chi,t)=
\left[
\eta_i(t,m_i+x_i^\chi,k_i)-
\eta_i(t,m_i,k_i)
\right] C_i^{LO}.
$$



令 $\mathcal{A}_i^\chi$ 表示所有会受到 $\tau_i$ 额外干扰的任务集合。
$\alpha$ 表示一个 task 在当前 priority assignment 下最多能够承受的 additional interference scale。  
对于整个系统而言，能够容忍的最大 interference 不是某一个 task 的 $\alpha$，而是所有相关 tasks 可承受 $\alpha$ 的最小值。




### Paper Flow

 A MCS task set: $\Gamma_{LO} \ \text{and} \ \Gamma_{HI}$

所有HI任务都需要通过LO和mode switch的RTA测试。所有LO任务都需要通过LO的RTA测试在mk的约束下。
 mode switch $R_i^{*}$ 可以把LO任务分为两类：
 - 可调度的 $\Gamma_{LO}^*$
 - 不可调度的 $\overline{\Gamma}_{LO}^{*}$
  
HI mode $R_i^{HI}$ 可以把LO任务分为两类：
 - 可调度的 $\Gamma_{LO}^{h}$
 - 不可调度的 $\overline{\Gamma}_{LO}^{h}$ 


 <div style="border: 1px solid #555; padding: 12px; border-radius: 8px; background-color: #e6e3e3;">
对于$\Gamma_{LO}^*$，我们使用augmented mk pattern 来提升。 通过设置一个augmented 
x，x的值是动态变化的根据不同模式的slack -> $\{x^l,x^*,x^h\}$：

In LO mode：promote 每个任务的$x^l$ 根据importance and slack

In mode switch：由于HI task 的执行时间会膨胀，导致slack减小，无法满足$x^l$， 因此需要对 $x^l$进行降级成 $x^*$, $x^l\ge x^*$， 以满足mode switch 的需求。

In HI mode：系统进入稳定的HI，carry-over job执行结束，可用的slack 增多，他有两种可选的行为：
- 增加 $x^* \to x^h$, 增加自身的x，继续提升该任务自身的 augmented service
- 减少 $x^* \to x^h$，减少自身的x，去save在mode switch 被挂起的任务。

</div>

<br>

<div style="border: 1px solid #555; padding: 12px; border-radius: 8px; background-color: #e6e3e3;">
对于 $\overline{\Gamma}_{LO}^{*}$, 我们需要尽可能的保证他的基础的mk：

In LO mode：按照 baseline mk 执行
In mode switch： 这些任务会被暂时挂起
In HI mode：被挂起的任务会被拯救到mk继续执行，或者一直被挂起直到系统重置
</div>

<br>

<div style="border: 1px solid #555; padding: 12px; border-radius: 8px; background-color: #e6e3e3;">

In stable HI mode: 此时 LO模式下释放的任务已经全部执行完毕，系统进入稳定的HI mode，那么HI mode下的调度压力相比于mode switch 会有所减小，相应的slack也会随之增加。那么可以尝试的去拯救在 mode switch下被挂起的LO任务，使得他们发挥自己最低限度mk 的功能，通过利用使用多出来的slack 和 减少某些任务 augmented $x^*$->$x^h\ge0$。被救的任务的所get的importance 应该大于牺牲的任务。

按照我们之前的方法，$x^l, \ x^*$ 可以被确定。

下面给出一个确定 $x^h$的算法


**Stable-HI Recovery Algorithm**

1. 按照importance $score_i$ 逆序，对被挂起的LO task set $\overline{\Gamma}_{LO}^{*}$ 进行排序。

2. 对于每个被挂起的任务 $\tau_i$：
   - set baseline mk；
   - 首先尝试使用 stable-HI mode下的slack；
   - 执行 stable-HI 模式下的可调度性测试;
   - 如果分配失败，则尝试通过减少低$score_i$任务的 $x^*$ 来释放 slack；

3. 仅当满足以下条件时，接受任务 $\tau_i$：
   - 所有HI tasks仍然可调度；
   - 所有此前已保留的LO task 仍满足基础 $(m,k)$ 约束；
   - 被恢复的任务满足 $(m,k)$-cosntraint；
   - 恢复该任务获得的importance收益大于牺牲的importacne收益。

4. 重复上述过程，直到不存在更多既有收益又可调度的恢复方案。

</div> 

<br>

<mark style="background-color: #ffcdd2;">
**对于importacne的一些想法**
<br>
比如importance不是对于一个job而言的，对于一个完成mk的task有一个基础的importance，提升x+1后得到额外的importance，可能是baseline的 0.1*importance？ 这样？
</mark> 


<div style="border: 1px solid #555; padding: 12px; border-radius: 8px; background-color: #e6e3e3;">

对于在mode switch 可调度的LO task $\Gamma_{LO}^{s}$，我们使用 augmented $(m,k)$ pattern 来提升它们的importance。  
通过设置一个 augmented $x$，任务的 importance 可以根据不同 mode 下的 available slack 动态变化：

$$
x_i \in \{x_i^l, x_i^*, x_i^h\}.
$$

**In LO mode:**  
promote 每个任务的 $x_i^l$ according to importance and available slack。  
此时任务 $\tau_i$  从 baseline$(m_i,k_i)$提升为：$(m_i+x_i^l,k_i).$

**In mode switch:**  
由于 HI tasks 的执行时间可能从 $C_i^{LO}$ 膨胀到 $C_i^{HI}$，available slack 会减小，因此 LO-mode 下的 $x_i^l$ 可能无法继续满足 mode-switch RTA。

因此，需要将 $x_i^l$ degraded to $x_i^*$：

$$
0 \le x_i^* \le x_i^l \le k_i-m_i.
$$

此时 $\Gamma_{LO}^{*}$ 中的任务仍然被保留，但只能获得 degraded augmented service：

$$
(m_i+x_i^*,k_i).
$$

**In stable HI mode:**  
系统进入 stable HI mode 后，mode-switch interval 中的 carry-over jobs 已经执行完成或被丢弃。相比 mode-switch interval，carry-over workload 消失，因此 stable HI mode 下可能出现新的 slack。

此时 $x_i^h$ 会被重新确定，它有两种可能的行为：

- increase $x_i^* \rightarrow x_i^h$，继续提升该任务自身的 augmented service；
- decrease $x_i^* \rightarrow x_i^h$，牺牲该任务的一部分 augmented service，用来 recover mode switch 中被挂起的 LO tasks。

注意：这里不是清空任务自己的 baseline $(m_i,k_i)$，而是最多清空 augmented part。  
也就是说，$\tau_i$ 最多可以从：

$$
(m_i+x_i^*,k_i)
$$

降到：

$$
(m_i,k_i).
$$

对应：

$$
x_i^h = 0.
$$

因此 stable HI mode 下需要满足：

$$
0 \le x_i^h \le k_i-m_i.
$$

</div>

<br>

<div style="border: 1px solid #555; padding: 12px; border-radius: 8px; background-color: #e6e3e3;">

对于 $\overline{\Gamma}_{LO}^{*}$，这些 LO tasks 在 mode-switch interval 下无法通过 $R_i^*$ 测试，因此会被 temporarily suspended。

但是它们并不是永久丢弃。我们的目标是在 stable HI mode 下，尽可能 recover 它们的 baseline $(m,k)$ service。

**In LO mode:**  
这些任务按照 baseline $(m_i,k_i)$ pattern 执行。

**In mode switch:**  
这些任务会被暂时挂起，以保证所有 HI tasks 以及 $\Gamma_{LO}^{*}$ 中 retained LO tasks 的 schedulability。

**In stable HI mode:**  
系统会尝试将这些 suspended tasks recover 到 baseline service：

$$
(m_i,k_i).
$$

如果 recovery 成功，则任务 $\tau_i$ 可以在 stable HI mode 下继续发挥最低限度的 weakly-hard 功能。  
如果 recovery 失败，则任务 $\tau_i$ 保持 suspended，直到系统回到 LO mode 或 reset。

</div>

<br>

<div style="border: 1px solid #555; padding: 12px; border-radius: 8px; background-color: #e6e3e3;">

### Stable-HI Recovery

In stable HI mode, LO-mode jobs and mode-switch carry-over jobs have completed or been discarded. Therefore, compared with the mode-switch interval, the carry-over workload disappears, and additional slack may become available.

此时可以尝试 recover 在 mode switch 阶段被挂起的 LO tasks，使它们至少恢复 baseline $(m,k)$ service。

Recovery 可以利用两类 slack：

1. stable HI mode 下自然释放出来的 free slack；
2. 通过降低 retained LO tasks 的 augmented level $x_j^*$ 释放出来的 slack。

对于一个 retained task $\tau_j \in \Gamma_{LO}^{*}$，它可以牺牲的部分是 augmented service：

$$
x_j^* \rightarrow x_j^h.
$$

最极端情况下，它可以清空自己的 augmented part：

$$
x_j^h = 0.
$$

此时 $\tau_j$ 从：

$$
(m_j+x_j^*,k_j)
$$

降为：

$$
(m_j,k_j).
$$

这个操作释放的 interference upper bound 为：

$$
Free_j(t)=
\left[
\eta_j(t,m_j+x_j^*,k_j)-
\eta_j(t,m_j,k_j)
\right]C_j^{LO}.
$$

如果只减少一部分 augmented level，从 $x_j^*$ 降到 $x_j^h$，则释放的 interference upper bound 为：

$$
Free_j(t,x_j^h)=
\left[
\eta_j(t,m_j+x_j^*,k_j)-
\eta_j(t,m_j+x_j^h,k_j)
\right]C_j^{LO}.
$$

被 recover 的 suspended task $\tau_i$ 会重新加入 baseline service：

$$
(m_i,k_i).
$$

它引入的 interference upper bound 为：

$$
Need_i(t)=
\eta_i(t,m_i,k_i)C_i^{LO}.
$$

因此，一个 retained task $\tau_j$ 清空自己的 augmented part 后，能否 recover 一个 suspended task $\tau_i$，需要检查：

$$
FreeSlack(t) + Free_j(t) \ge Need_i(t)
$$

并且重新执行 stable-HI RTA。

如果 RTA 通过，则该 recovery 是 schedulable。

</div>






