# Quadratic optimization

## Motivation

Quadratic systems are important:
- Systems close to their maximum can closely be approximated by a quadratic system, studying the minimization of quadratic systems can teach us about minimization of general convex functions.
- Quadratic systems are important in their own right! Many statistical models, graph problems, molecular models etc. can be formulated as quadratic systems!

## Warming up: one-dimensional quadratic systems

In the the scalar case, a quadratic function is given by
$$
f(x) = \frac{1}{2}px^2+qx +r\,,
$$
with $p>0$ (as we will see).

Our optimization problem is given by:
$$
\min_x\,\frac{1}{2}px^2+qx +r\,.
$$

This can easily be solved by setting the first order derivative equal to zero:

$$
\frac{\mathrm{d}f(x)}{\mathrm{d}x} = px + q \\
px^\star+q = 0 \Leftrightarrow x^\star=\frac{-q}{p}
$$
To show that this is the sole minimizer of $f(x)$, we have to prove that the second order derivative is positive in this point. This means that at that point the derivative of the function is increasing: a little to the left the function is increasing and a little to the right and the function is decreasing. We have
$$
\left.\frac{\mathrm{d}^2f(x)}{\mathrm{d}x^2}\right|_{x^\star} = p\,,
$$
so if $p>0$ then $x^\star$ is the minimizer of $f(x)$.

> coding exercise

## Towards $n$-dimensional quadratic systems

Let directly move from the one-dimensional case to the $n$-dimensional case. We will use vector notation
$$
\mathbf{x} = \begin{bmatrix}
       x_1 \\ \vdots \\ x_n
     \end{bmatrix} \in \mathbb{R}^n\,.
$$
A general $n$-dimensional linear system is given by:
$$
f(\mathbf{x}) = \mathbf{x}^\intercal P \mathbf{x} + \mathbf{q}^\intercal\mathbf{x} + r\,,
$$
with $P$ an $n\times n$ symmetric matrix, $\mathbf{q}$ an $n$-dimensional vector and $r$ a scalar.

> Why is $P$ symmetric?

So we want to solve the problem:
$$
\min_\mathbf{x}\,\mathbf{x}^\intercal P \mathbf{x} + \mathbf{q}^\intercal\mathbf{x} + r\,.
$$

The concept of a derivative is extended towards higher dimensions using the *gradient* operator:
$$
\nabla_\mathbf{x} = \begin{bmatrix}
       \frac{\partial \, }{\partial x_1} \\ \vdots \\ \frac{\partial \, }{\partial x_n}
     \end{bmatrix}\,,
$$
so that the gradient of $f(\mathbf{x})$ is given by:
$$
\nabla_\mathbf{x} f(\mathbf{x}) = \begin{bmatrix}
       \frac{\partial f{x}f(\mathbf{x}) }{\partial x_1} \\ \vdots \\ \frac{\partial f{x}f(\mathbf{x}) }{\partial x_n}
     \end{bmatrix}\,,
$$
From now on, we will drop the subscript in the gradient, unless it is not clear from context how the gradient is computed. For those not familiar to vector calculus, the most useful rules are given below.

| rule | example     |
| :------------- | :------------- |
| linearity      | $\nabla(a f(\mathbf{x}) +b g(\mathbf{x})) = a\nabla f(\mathbf{x}) +b\nabla g(\mathbf{x})$       |
| product rule | $\nabla(f(\mathbf{x}) g(\mathbf{x})) = g(\mathbf{x})\nabla f(\mathbf{x}) + f(\mathbf{x})\nabla g(\mathbf{x})$|
|chain rule|$\nabla f(g(\mathbf{x})) = \left.\frac{\partial f}{\partial g}\right|_\mathbf{x}\nabla f(\mathbf{x})$|
| quadratic term | $\nabla \frac{1}{2}\mathbf{x}^\intercal A\mathbf{x}= A\mathbf{x}$|
|linear term| $\nabla \mathbf{b}^\intercal\mathbf{x}=\mathbf{b}$|
|constant term |$\nabla c = 0$ |

The gradient of the quadratic function is
$$
\nabla f(\mathbf{x})=Q\mathbf{x} +\mathbf{q}\,,
$$
setting this to zero gives
$$
\mathbf{x}^\star=-P^{-1}\mathbf{q}\,.
$$
How do we know that $\mathbf{x}^\star$ is the minimizer of the quadratic system? For this we have to extend the concept of a second order derivative to $n$ dimensions. We define the *Hessian* as:
$$
\nabla^2 f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial^2 f(\mathbf{x})}{\partial {x_{1}}^2} & \frac{\partial^2 f(\mathbf{x})}{\partial x_1 x_2} & \ldots &  \frac{\partial^2 f(\mathbf{x})}{\partial x_1 x_n}\\
\frac{\partial^2 f(\mathbf{x})}{\partial x_1 x_2} & \frac{\partial^2 f(\mathbf{x})}{\partial {x_2}^2} & \ldots & \vdots \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f(\mathbf{x})}{\partial x_1 x_n} & \frac{\partial^2 f(\mathbf{x})}{\partial x_2 x_n} & \ldots & \frac{\partial^2 f(\mathbf{x})}{\partial x_n^2}
\end{bmatrix}\,.
$$
For the quadratic system, this boils down to
$$
\nabla^2 f(\mathbf{x}) = P\,.
$$
The condition for $\mathbf{x}^\star$ to be the minimizer of $f(\mathbf{x})$ is that the Hessian should be *positive-definite* in that point.


> A symmetric $n\times n$ matrix $A$ is positive-definite (in symbols: $A\succ0$), if for any vector $\mathbf{z}\in\mathbb{R}^n$
> $$
> \mathbf{z}^\intercal A \mathbf{z} > 0\,.
> $$

A matrix is positive-definite if (and only if) all its eigenvalues as positive.

A point $\mathbf{x}^\star$ at which the gradient vanishes is a minimizer if and only if
$$
\nabla^2 f(\mathbf{x})|_{\mathbf{x}^\star} \succ 0\,.
$$
So, for the quadratic problem, $x\star$ is the unique minimizer iff $P\succ 0$. This means that along every direction $\mathbf{v}\in \mathbb{R}^n$ to project $\mathbf{x}$, the problem reduces to a one-dimensional quadratic function with a positive second-order constant:
$$
x_v = \mathbf{v}^\intercal \mathbf{x}\\
f'(x_v) = x_v (\mathbf{v}^\intercal P \mathbf{v}) x_v + (\mathbf{v}^\intercal \mathbf{q})x_v + r\,,
$$
where $\mathbf{v}^\intercal P \mathbf{v}>0$ if $P\succ 0$, which in turn implies that $f'(x_v)$ has a minimizer.

If $P\succ 0$, the quadratic system is a *convect* function with a single minimizer. In many problems, $P$ is positive-definite, so there is a well-defined solution. We will develop this further in Chapter 2.

> Consider $L_2$ regularized ridge regression:
> $$
> \min_\mathbf{x}\, (\mathbf{y} - B\mathbf{x})^\intercal(\mathbf{y} - B\mathbf{x}) + c \mathbf{x}^\intercal\mathbf{x}\,,
> $$
> with $c>0$. Write this in the standard form of a quadratic system and show that it is convex. Give the expression for the minimizer.

## Time and memory complexity of exact solution

The exact solution for convex quadratic system hinges on solving a $n\times n$ linear system. Conventional solvers for linear systems have a time complexity of $\mathcal{O}(n^3)$. This is doable for problems of moderate size ($n<1000$), but becomes infeasible for large-scale problems.

> plot of time complexity

Storing an $n\times n$ matrix also has a memory requirement of $\mathcal{O}(n^2)$. When $n$ is too large, this cannot fit in main memory. In the remainder of this chapter, we will consider the case when $P$ is too large to work with, while matrix-vector products $P\mathbf{x}$ can be computed. Some examples of when such settings occur:
- $P=B^\intercal B$, with $B\in \mathbb{R}^{n\times p}$, with $p\ll n$.
- $P$ is a very sparse matrix.
- $P$ has a special structure so that $P\mathbf{x}$ can be computed on the fly, e.g. $P_{ij}=i^2j^3$.
- $P$ is loaded an processed as different blocks.

## Descent methods

Instead of computing the solution of a convex quadratic system in one step, we will use *descent methods*. Here, a minimizing sequence $\mathbf{x}^{(k)},\, k=1,\dots$, where
$$
\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} +t^{(k)}\Delta \mathbf{x}^{(k)}\,,
$$
with $t^{(k)}\geq 0$ called the *step size* and $\Delta \mathbf{x}^{(k)}$ called the *search direction*. Proper descent methods have that
$$
f(\mathbf{x}^{(k+1)}) < f(\mathbf{x}^{(k)})\,,
$$
except when $\mathbf{x}^{(k)}$ is optimal. In this property to hold, the search direction should satisfy
$$
(\Delta \mathbf{x}^{(k)})^\intercal \nabla f(\mathbf{x}) < 0\,.
$$

> figure!

Below is the general pseudocode of a general descent method:

> **given** a starting point $\mathbf{x}$
>
> **repeat**
>> 1. Determine descent direction $\Delta \mathbf{x}$
>> 2. *Line search*. Choose $t>0$.
>> 3. *Update*. $\mathbf{x}:=\mathbf{x} + t \Delta \mathbf{x}$.
>
> **until** stopping criterion is reached.

The stepsize can be chosen in several ways:
- **exact**: $t=\arg\min_{s\geq 0}\, f(\mathbf{x}+s\Delta \mathbf{x})$.
- **approximate**: choose a $t$ that only approximately minimizes $f(\mathbf{x}+s\Delta \mathbf{x})$.
- **decaying**: choose some decaying series, e.g. $t = \frac{1}{\alpha+k}$.
- **constant**: a constant stepsize (often done in practice).

For quadratic systems we can compute the exact stepsize, as this amounts to a simple one-dimensional quadratic problem:
$$
t=\arg\min_{s\geq 0}\, \frac{1}{2}(\mathbf{x}+s\Delta \mathbf{x})^\intercal P (\mathbf{x}+s\Delta \mathbf{x}) + (\mathbf{x}+s\Delta \mathbf{x}) \mathbf{q} + r
$$
$$
t = \frac{-(\Delta\mathbf{x})^\intercal P \mathbf{x}-(\Delta\mathbf{x})^\intercal\mathbf{q}}{(\Delta\mathbf{x})^\intercal P \Delta\mathbf{x}}
$$

> implement exact line search

## Gradient descent

### Motivation

### Convergence analysis

### Example

## Gradient descent with momentum

> *While finding the gradient of an objective function is a splendid idea, ascending the gradient directly may not be.* ~ David J.C. MacKay
