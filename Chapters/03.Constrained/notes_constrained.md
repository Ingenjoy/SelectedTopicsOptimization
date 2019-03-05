# Constrained convex optimization (1)

*Selected Topics in Mathematical Optimization*

**Michiel Stock** ([email](michiel.stock@ugent.be))

![](Figures/logo.png)

## Motivation

Many more realistic optimization problems are characterized by constraints. For example, real-word systems often satisfy conservation laws, such as conservation of mass, of atoms or of charge. When designing objects, there are practical constraints of feasible dimensions, range of operations and limitations in materials. Another example is in probability, where a solution should satisfy the axioms of probability theory (probabilities are real values between 0 and 1 and the probabilities of all events should sum to 1).

Other cases, we include constraints in our problem because they encode prior knowledge about the problem or to obtain solutions with certain desirable properties.

In this chapter we discuss convex optimization problems with linear equality constraints (constraining the solution to a linear subspace) and convex inequality constrains (constraining the solution to convex subspace). Both types of constraints result again in a convex optimization problem.

## Lagrange multipliers

Lagrange multipliers are elegant ways of finding stationary points of a function of several variables given one or more constraints. We give a short introduction based on a geometric perspective.

> IMPORTANT: most textbooks treat Lagrange multipliers from as maximization problems. Here they are treated as minimization problems to be consistent with other chapters.

### Equality constraints

Consider the following optimization problem:

$$
\min_{\mathbf{x}} f(\mathbf{x})
$$
$$
\text{subject to } g(\mathbf{x})=0\,.
$$

![Convex optimization problem with an equality constraint. Here, the constraint is nonlinear.](Figures/Lagr1.png)

For every point $\mathbf{x}$ on the surface $g(\mathbf{x})=0$, the gradient $\nabla g(\mathbf{x})$ is normal to this surface. This can be shown by considering a point $\mathbf{x}+\boldsymbol{\epsilon}$, also on the surface. If we make a Taylor expansion around $\mathbf{x}$, we have

$$
g(\mathbf{x}+\boldsymbol{\epsilon})\approx g(\mathbf{x}) + \boldsymbol{\epsilon}^\top\nabla g(\mathbf{x})\,.
$$

Given that both $\mathbf{x}$ and $\mathbf{x}+\boldsymbol{\epsilon}$ lie on the surface it follows that $g(\mathbf{x}+\boldsymbol{\epsilon})= g(\mathbf{x})$. In the limit that $||\boldsymbol{\epsilon}||\rightarrow 0$ we have that $\boldsymbol{\epsilon}^\top\nabla g(\mathbf{x})=0$. Because $\boldsymbol{\epsilon}$ is parallel to the surface $g(\mathbf{x})$, it follows that $\nabla g(\mathbf{x})$ is normal to the surface.

![The same optimization problem, with some gradients of $f(\mathbf{x})$ and $g(\mathbf{x})$ shown.](Figures/Lagr2.png)

We seek a point $\mathbf{x}^\star$ on the surface such that $f(\mathbf{x})$ is minimized. For such a point, it should hold that the gradient w.r.t. $f$ should be parallel to $\nabla g$. Otherwise, it would be possible to give a small 'nudge' to $\mathbf{x}^\star$ in the direction of $\nabla f$ to decrease the function value, which would indicate that $\mathbf{x}^\star$ is not a minimizer. This figures below illustrate this point.

![Point on the surface that is *not* a minimizer.](Figures/Lagr3.png)

![Point on the surface that is a minimizer of $f$.](Figures/Lagr4.png)

$$
\nabla f(\mathbf{x}^\star) + \nu \nabla g (\mathbf{x}^\star)=0\,,
$$
with $\nu\neq 0$ called the *Lagrange multiplier*. The constrained minimization problem can also be represented by a *Lagrangian*:
$$
L(\mathbf{x}, \nu) 	\equiv f(\mathbf{x}) + \nu g(\mathbf{x})\,.
$$
The constrained stationary condition is obtained by setting $\nabla_\mathbf{x} L(\mathbf{x}, \nu) =0$, the condition $\partial  L(\mathbf{x}, \nu)/\partial \nu=0$ leads to the constraint equation $g(\mathbf{x})=0$.

### Inequality constraints

The same argument can be made for inequality constraints, i.e. solving

$$
\min_{\mathbf{x}} f(\mathbf{x})
$$
$$
\text{subject to } g(\mathbf{x})\leq0\,.
$$

Here, two situations can arise:

- **Inactive constraint**: the minimizer of $f$ lies in the region where $g(\mathbf{x}) < 0$. This corresponds to a Lagrange multiplier $\nu=0$. Note that the solution would be the same if the constraint was not present.
- **Active constraint**: the minimizer of $f$ lies in the region where $g(\mathbf{x}) > 0$. The solution of the constrained problem will lie on the bound where $g(\mathbf{x})=0$, similar to the equality-constrained problem and corresponds to a Lagrange multiplier $\nu>0$.

Both scenarios are shown below:

![Constrained minimization problem with an active inequality constraint. Optimum lies within the region where $g(\mathbf{x})\leq 0$. ](Figures/Lagr6.png)

![Constrained minimization problem with an active inequality constraint. Optimum lies on the boundary of the region where $g(\mathbf{x})\leq 0$.](Figures/Lagr5.png)


For both cases, the product $\nu g(\mathbf{x})=0$, the solution should thus satisfy the following conditions:
$$
g(\mathbf{x}) \leq 0
$$
$$
\nu \geq 0
$$
$$
\nu g(\mathbf{x})=0\,.
$$
These are called the *Karush-Kuhn-Tucker* conditions.

It is relatively straightforward to extend this framework towards multiple constraints (equality and inequality) by using several Lagrange multipliers.

## Equality constrained convex optimization

### Problem outline

We will start with convex optimization problems with linear equality constraints:

$$
\min_\mathbf{x} f(\mathbf{x})
$$
$$
\text{subject to } A\mathbf{x}=\mathbf{b}
$$

where $f : \mathbb{R}^n \rightarrow \mathbb{R}$ is convex and twice continuously differentiable and $A\in \mathbb{R}^{p\times n}$ with a rank $p < n$.

The Lagrangian of this problem is

$$
L(\mathbf{x}, \boldsymbol{\nu}) = f(\mathbf{x}) + \boldsymbol{\nu}^\top(A\mathbf{x}-\mathbf{b})\,,
$$
with $\boldsymbol{\nu}\in\mathbb{R}^p$ the vector of Lagrange multipliers.

A point $\mathbf{x}^\star\in$ **dom** $f$ is optimal for the above optimization problem only if there is a $\boldsymbol{\nu}^\star\in\mathbb{R}^p$ such that:

$$
A\mathbf{x}^\star = \mathbf{b}, \qquad \nabla f(\mathbf{x}^\star) + A^\top\boldsymbol{\nu}^\star = 0\,.
$$

We will reuse the same toy examples from the previous chapter, but add an equality constraint to both.

- Simple quadratic problem:

$$
 \min_{\mathbf{x}} \frac{1}{2} (x_1^2 + 4 x_2^2)\\
 \text{subject to }  x_1 - 2x_2 = 3
$$

- A non-quadratic function:

$$
 \min_{\mathbf{x}}\log(e^{x_1 +3x_2-0.1}+e^{x_1 -3x_2-0.1}+e^{-x_1 -0.1})\\
 \text{subject to }  x_1 + 3x_2 = 0  
$$

![The two toy functions each with a linear constraint.](Figures/example_functions.png)

### Equality constrained convex quadratic optimization

Consider the following equality constrained convex optimization problem:

$$
\min_\mathbf{x}\frac{1}{2}\mathbf{x}^\top P \mathbf{x} + \mathbf{q}^\top \mathbf{x} + r
$$
$$
\text{subject to }  A\mathbf{x}=\mathbf{b}
$$

where $P$ is symmetric.

The optimality conditions are
$$
A\mathbf{x}^\star = \mathbf{b}, \quad P\mathbf{x}^\star+\mathbf{q} +A^\top\boldsymbol{\nu}^\star=\mathbf{0}\,,
$$
which we can write as

$$
\begin{bmatrix}
P & A^\top \\
A & 0 \\
     \end{bmatrix}
     \begin{bmatrix}
\mathbf{x}^\star\\
\boldsymbol{\nu}^\star
     \end{bmatrix}
     =
     \begin{bmatrix}
-\mathbf{q} \\
\mathbf{b}
     \end{bmatrix}\,.
$$
Note that this is a block matrix.

> If $P$ is positive-definite, the linearly constrained quadratic minimization problem has an unique solution.

Solving this linear system gives both the constrained minimizer $\mathbf{x}^\star$ as well as the Lagrange multipliers.

**Assignment 1**

1. Complete the code to solve linearly constrained quadratic systems.
2. Use this code to solve the quadratic toy problem defined above.

```python
def solve_constrained_quadratic_problem(P, q, A, b):
    """
    Solve a linear constrained quadratic convex problem.

    Inputs:
        - P, q: quadratic and linear parameters of
                the linear function to be minimized
        - A, b: system of the linear constraints

    Outputs:
        - xstar: the exact minimizer
        - vstar: the optimal Lagrange multipliers
    """
    p, n = ...  # size of the problem
    # complete this code
    # HINT: use np.linalg.solve and np.bmat
    solution = ...
    xstar = solution[:n]
    vstar = solution[n:]
    return np.array(xstar), np.array(vstar)
```

### Newton's method with equality constraints

To derive $\Delta \mathbf{x}_{nt}$ for the following equality constrained problem

$$
\min_\mathbf{x}  f(\mathbf{x})
$$
$$
\text{subject to }  A\mathbf{x}=\mathbf{b}
$$

we apply a second-order Taylor approximation at the point $\mathbf{x}$, to obtain

$$
\min_\mathbf{v} \hat{f}(\mathbf{x} +\mathbf{v}) = f(\mathbf{x}) +\nabla f(\mathbf{x})^\top \mathbf{v}+ \frac{1}{2}\mathbf{v}^\top \nabla^2 f(\mathbf{x}) \mathbf{v}
$$
$$
\text{subject to } A(\mathbf{x}+\mathbf{v})=\mathbf{b}\,.
$$

Based on the solution of quadratic convex problems with linear constraints, the Newton $\Delta \mathbf{x}_{nt}$ step is characterized by

$$
\begin{bmatrix}
 \nabla^2 f(\mathbf{x})&  A^\top \\
A & 0 \\
     \end{bmatrix}
     \begin{bmatrix}
\Delta \mathbf{x}_{nt}\\
\mathbf{w}
     \end{bmatrix}
     =
     -\begin{bmatrix}
\nabla f(\mathbf{x}) \\
A\mathbf{x}-\mathbf{b}
     \end{bmatrix}
$$

- If the starting point $\mathbf{x}^{(0)}$ is chosen such that $A\mathbf{x}^{(0)}=\mathbf{b}$, the residual term vanishes and steps will remain in the feasible region. This is the **feasible start Newton method**.
- If we choose an arbitrary $\mathbf{x}^{(0)}\in$ **dom** $f$, not satisfying the constraints, this is the **infeasible start Newton method**. It will usually converge rapidly to the feasible region (check the final solution!).

Note that when we start at a feasible point, the residual vector $-(A\mathbf{x}-\mathbf{b})$ vanishes and the path will always remain in a feasible region. Otherwise we will converge to it.

In this chapter, we will use a fixed step size. For Newton's method this usually leads to only a few extra iterations compared to an adaptive step size.

>**input** starting point $\mathbf{x}\in$ **dom** $f$ (with $A\mathbf{x}=\mathbf{b}$ if using the feasible method), tolerance $\epsilon>0$.
>
>**repeat**
>
>>    1. Compute the Newton step $\Delta \mathbf{x}_{nt}$ and decrement $\lambda(\mathbf{x})$.
>>    2. *Stopping criterion*. **break** if $\lambda^2/2\leq \epsilon$.
>>    3. *Choose step size $t$*: either by line search or fixed $t$.
>>    4. *Update*. $\mathbf{x}:=\mathbf{x}+t \Delta \mathbf{x}_{nt}$.
>
>**output** $\mathbf{x}$

Again, the convergence can be monitored using the Newton decrement:

$$
\lambda^2(\mathbf{x}) = - \Delta \mathbf{x}_{nt}^\top \nabla f(\mathbf{x})\,.
$$

The algorithm terminates when

$$
\frac{\lambda(\mathbf{x})^2}{2} < \epsilon\,.
$$

The Newton decrement also indicates whether we are in or close to the feasible region.

**Assignment 2**

1. Complete the code for the linearly constrained Newton method.
2. Use this code to find the minimum of the non-quadratic toy problem, defined above (compare a feasible and infeasible start).

```python
def linear_constrained_newton(f, x0, grad_f,
              hess_f, A, b, stepsize=0.25, epsilon=1e-3,
              trace=False):
    '''
    Newton's method for minimizing functions with linear constraints.

    Inputs:
        - f: function to be minimized
        - x0: starting point (does not have to be feasible)
        - grad_f: gradient of the function to be minimized
        - hess_f: hessian matrix of the function to be minimized
        - A, b: linear constraints
        - stepsize: step size for each Newton step (fixed)
        - epsilon: parameter to determine if the algorithm is converged
        - trace: (bool) store the path that is followed?

    Outputs:
        - xstar: the found minimum
        - x_steps: path in the domain that is followed (if trace=True)
        - f_steps: image of x_steps (if trace=True)
    '''
    assert stepsize < 1 and stepsize > 0
    x = x0  # initial value
    p, n = A.shape
    if trace: x_steps = [x.copy()]
    if trace: f_steps = [f(x0)]
    while True:
        ddfx = hess_f(x)
        dfx = grad_f(x)
        # calculate residual
        Dx, _ = solve_constrained_quadratic_problem(... # complete!
        newton_decrement = ...
        if ...  # stopping criterion
            break  # converged
        # perform step
        if trace: x_steps.append(x.copy())
        if trace: f_steps.append(f(x))
    if trace: return x, x_steps, f_steps    
    else: return x
```

## References

- Boyd, S. and Vandenberghe, L., '*[Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)*'. Cambridge University Press (2004)
- Bishop, C., *Pattern Recognition and Machine Learning*. Springer (2006)
