# Adam Optimizer

[Adam](https://arxiv.org/abs/1412.6980) is one of the workhorses of modern deep learning. It's an SGD-based algorithm with adaptive estimation of 1st and 2nd-order moments.

This is mainly to do some Haskell coding, so for all I know this code might suck compared to what a real Haskell coder would do, so be warned. The implementation is using the [StateT monad transformer](https://book.realworldhaskell.org/read/monad-transformers.html) to manage moments' state through Adam steps.

# Algorithm
Adam maintains two moments for each model parameter $\theta$ during the optimization process:
- First Moment (mean of the gradients): $m_t$
- Second Moment (uncentered variance of the gradients): $v_t$

**Steps:**
- Initialize moments: Set $m_0$ and $v_0$ to vectors of Os (same dimension as $\theta$ ).
- For each iteration $(\mathrm{t}=1,2, \ldots)$ :
  - Compute gradient: $g_t$, gradient of the loss $L$ with respect to parameter $\theta$ at timestep $t$.
  - Update biased first moment estimate: $m_t=\beta_1 m_{t-1}+\left(1-\beta_1\right) g_t$
  - Update biased second raw moment estimate: $v_t=\beta_2 v_{t-1}+\left(1-\beta_2\right) g_t^2$
  - Compute bias-corrected first moment estimate: $\hat{m}_t=\frac{m_t}{1-\beta_1^l}$
  - Compute bias-corrected second raw moment estimate: $\hat{v}_t=\frac{v_t}{1-\beta_2^l}$
  - Update parameters: $\theta_{t+1}=\theta_t-\frac{\alpha}{\sqrt{v_t}+\epsilon} \hat{m}_t$

**Parameters and Hyperparameters:**
- $\alpha$ : Step size.
- $\beta_1, \beta_2$ : Exponential decay rates for the moment estimates.
- $\epsilon$ : Scalar (e.g., $10^{-8}$ ) used to prevent division by zero in param updates.
