# Constrained convex optimization

*Selected Topics in Mathematical Optimization: 2017-2018*

**Michiel Stock** ([email](michiel.stock@ugent.be))

![](Figures/logo.png)

## Motivation

## Lagrange multipliers

Lagrange multipliers are elegant ways of finding stationary points of a function of several variables given one or more constraints. We give a short introduction from a geometric perspective.

### Equality constraints

Consider the following optimization problem:

$$
\min_{\mathbf{x}} f(\mathbf{x})\\
\text{subject to } g(\mathbf{x})=0\,.
$$

![Convex optimization problem with an equality constraint. Here, the constraint is nonlinear.](Figures/Lagr1.png)

For every point $\mathbf{x}$ on the surface $g(\mathbf{x})$, the gradient $\nabla g(\mathbf{x})=0$. This can be shown by considering a point $\mathbf{x}+\boldsymbol{\epsilon}$, also on the surface. If we make a Taylor expansion around $\mathbf{x}$, we have
$$
g(\mathbf{x}+\boldsymbol{\epsilon})\approx g(\mathbf{x}) + \boldsymbol{\epsilon}^\top\nabla g(\mathbf{x})\,.
$$

![The same optimization problem, with some gradients of $f(\mathbf{x})$ and $g(\mathbf{x})$ shown.](Figures/Lagr2.png)

Given that both $\mathbf{x}$ and $\mathbf{x}+\boldsymbol{\epsilon}$ lie on the surface it follows that $g(\mathbf{x}+\boldsymbol{\epsilon})= g(\mathbf{x})$. In the limit that $||\boldsymbol{\epsilon}||\rightarrow 0$ we have that $\boldsymbol{\epsilon}^\top\nabla g(\mathbf{x})=0$. Because $\boldsymbol{\epsilon}$ is parallel to the surface $g(\mathbf{x})$, it follows that $\nabla g(\mathbf{x})$ is normal to the surface.

We seek a point $\mathbf{x}^\star$ on the surface such that $f(\mathbf{x})$ is minimized. For such a point, it should hold that the gradient w.r.t. $f$ should be parallel to $\nabla g$. Otherwise, it would be possible to give a small 'nudge' to $\mathbf{x}^\star$ in the direction of $\nabla f$ to decrease the function value, which would indicate that $\mathbf{x}^\star$ is not a minimizer. This figures below illustrate this point.

![Point on the surface that is *not* a minimizer.](Figures/Lagr3.png)

![Point on the surface that is a minimizer of $f$.](Figures/Lagr4.png)

$$
\nabla f(\mathbf{x}^\star) + \nu \nabla g (\mathbf{x}^\star)=0\,,
$$
with $\nu\neq 0$ called the *Lagrange multiplier*.

### Inequality constraints

## Equality constrained convex optimization

We will start with convex optimization problems with linear equality constraints:

$$
\min_\mathbf{x} f(\mathbf{x}) \\
\text{subject to } A\mathbf{x}=\mathbf{b}
$$

where $f : \mathbb{R}^n \rightarrow \mathbb{R}$ is convex and twice continuously differentiable and $A\in \mathbb{R}^{p\times n}$ with a rank $p < n$.

A point $\mathbf{x}^\star\in$ **dom** $f$ is optimal for the above optimization problem only if there is a $\nu\in\mathbb{R}^p$ such that:

$$
A\mathbf{x}^\star = b, \qquad \nabla f(\mathbf{x}^\star) + A^\top\nu^\star = 0\,.
$$

We will reuse the same toy examples from the previous chapter, but add an equality constraint to both.

* Simple quadratic problem:

$$
 f(x_1, x_2)  = \frac{1}{2} (x_1^2 + 4 x_2^2)\\
 \text{subject to }  x_1 - 2x_2 = 3
$$

* A non-quadratic function:

$$
f(x_1, x_2)   = \log(e^{x_1 +3x_2-0.1}+e^{x_1 -3x_2-0.1}+e^{-x_1 -0.1})\\
 \text{subject to }  x_1 + 3x_2 = 0  
$$

![The two toy functions each with a linear constraint.](Figures/example_functions.png)

### Equality constrained convex quadratic optimization

Consider the following equality constrained convex optimization problem:

$$
\min\frac{1}{2}\mathbf{x}^\intercal P \mathbf{x} + \mathbf{q}^\intercal \mathbf{x} + r  \\
\text{subject to }  A\mathbf{x}=b
$$

where $P$ is positive definite.

The optimality conditions are
$$
A\mathbf{x}^\star = b, \quad P\mathbf{x}^\star+\mathbf{q} +A^\intercal\nu^\star=0\,,
$$
which we can write as

$$
\begin{bmatrix}
P  A^\intercal \\
A  0 \\
     \end{bmatrix}
     \begin{bmatrix}
\mathbf{x}^\star\\
\nu^\star
     \end{bmatrix}
     =
     \begin{bmatrix}
-q \\
b
     \end{bmatrix}
$$

```python
def solve_constrained_quadratic_problem(P, q, A, b):
    """
    Solve a linear constrained quadratic convex problem.
    Inputs:
        - P, q: quadratic and linear parameters of
                the linear function to be minimized
        - A, b: system of the linear constaints
    Outputs:
        - xstar: the exact minimizer
        - vstar: the optimal Lagrance multipliers
    """
    p, n = A.shape  # size of the problem
    # complete this code
    # HINT: use np.linalg.solve and np.bmat
    solution = ...
    xstar = solution[:n]
    vstar = solution[n:]
    return np.array(xstar), np.array(vstar)
```
### Eliminating equality constraints

### Newton's method with equality constraints

To derive $\Delta \mathbf{x}_{nt}$ for the following equality constrained problem

$$
\min  f(\mathbf{x}) \\
\text{subject to }  A\mathbf{x}=b
$$

we apply a second-order Taylor approximation at the point $\mathbf{x}$, to obtain

$$
\min \hat{f}(\mathbf{x} +v) = f(\mathbf{x}) +\nabla f(\mathbf{x})^\intercal v+ (1/2)v^\intercal \nabla^2 f(\mathbf{x}) v \\
\text{subject to } A(\mathbf{x}+v)=b\,.
$$

Based on the solution of quadratic convex problems with linear contrains, the Newton $\Delta \mathbf{x}_{nt}$ step is characterized by

$$
\begin{bmatrix}
 \nabla^2 f(\mathbf{x})  A^\intercal \\
A  0 \\
     \end{bmatrix}
     \begin{bmatrix}
\Delta x_{nt}\\
w
     \end{bmatrix}
     =
     -\begin{bmatrix}
\nabla f(\mathbf{x}) \\
A\mathbf{x}-b
     \end{bmatrix}
$$

Note that when we start at a feasible point, the residual vector $-(A\mathbf{x}-b)$ vanishes and the path will always remain in a feasible region. Otherwise we will converge to it.


Rather than performing line search at each step, we will use a fixed step size $\nu\in]0,1]$

>**input** starting point $x\in$ **dom** $f$ with $Ax=b$, tolerance $\epsilon>0$, stepsize $\nu$.
>
>**repeat**
>
>>    1. Compute the Newton step $\Delta x_{nt}$ and decrement $\lambda(x)$.
>>    2. *Stopping criterion*. **quit** if $\lambda^2/2\leq \epsilon$.
>>    3. *Update*. $x:=x+\nu\Delta x_{nt}$.
>
>**until** stopping criterium is satisfied.

>**output** $x$

Again, the convergence can be monitored using the Newton decrement:

$$
\lambda^2(\mathbf{x}) = - \Delta \mathbf{x}_{nt}^\top \nabla f(\mathbf{x})\,.
$$

The algorithm terminates when

$$
\frac{\lambda(\mathbf{x})^2}{2} < \epsilon\,.
$$

## Inequality constrained convex optimization

### Inequality constrained minimization problems

### Logarithmic barrier and the central path

### The barrier method

## Exercise:
