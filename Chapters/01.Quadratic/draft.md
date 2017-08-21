# Quadratic optimization

## Motivation

Quadratic systems are important:
- Systems close to their maximum can closely be approximated by a quadratic system, studying the minimization of quadratic systems can teach us about minimization of general convex functions.
- Quadratic systems are important in their own right! Many statistical models, graph problems, molecular models etc. can be formulated as quadratic systems!

## Warming up: one-dimensional systems

In the the scalar case, a quadratic function is given by
$$
f(x) = \frac{1}{2}px^2+qx +r\,,
$$
with $p>0$ (as we will see).

Our optimization problem is given by:
$$
\min_x \frac{1}{2}px^2+qx +r\,.
$$

This can easily be solved by setting the first order derivative equal to zero:

$$
\frac{\mathrm{d}f(x)}{\mathrm{d}x} = px + q \\
px^\star+q = 0 \Leftrightarrow x^\star=\frac{-q}{p}
$$
To show that this is a minimizer of $f(x)$, we have to prove that the second order derivative is positive in this point. This means 
