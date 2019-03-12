# Project convex optimization: polymerization

*Selected Topics in Mathematical Optimization*

**Michiel Stock** ([email](michiel.stock@ugent.be))

![](Figures/logo.png)

## Outline

In this project, we will study reversible polymerization of proteins using the principle of *Gibbs free energy minimization*.

Consider the protein monomer $A$, which can form dimers

$$
A + A \rightleftharpoons AA\,,
$$

trimers:

$$
A + A + A \rightleftharpoons AA + A \rightleftharpoons A + AA \rightleftharpoons AAA
$$

or, in general, polymers:

$$
i A \rightleftharpoons A_i
$$

where $A_i$ is a polymer of length $i$ . A system containing these polymers can be described by an $n$-dimensional state or concentration vector $\mathbf{x}$ containing the concentrations of each polymer. Here, the elements of this vector represent:

$$
x_i = \text{concentration of polymer of length } i
$$

This vector representation can only represent systems of polymers where the longest polymer is of length $n$, whereas the polymers can be of (theoretically) arbitrary size. **The dimension of $\mathbf{x}$ thus represents the quality of the approximation of the system.**

This concentration vector has some physical constraints. Firstly, concentrations are nonnegative, i.e. $x_i\geq 0$ for all $i$. Furthermore, the system is subjected to a *conservation of mass*. The concentrations that are possible are constrained by the initial quantity $c_0$ that is introduced to the system, as no new protein monomers of $A$ are formed, nor do any disappear from the system. So, any concentration vector should satisfy the following linear constraint:

$$
\sum_{i=1}^n ix_i = c_0\,,
$$

meaning that the total number of copies of $A$ in all polymers together is equal to $c_0$.

Polymerization is facilitated by non-covalent energetically favorable bonds. Every bond results in a binding enthalpy change of $\Delta H=-1$[^39b98177]. Hence, the enthalpy $H(\mathbf{x})$ of a given concentration vector is given by

$$
H(\mathbf{x}) = -\sum_{i=1}^n(i-1)x_i\,.
$$

By minimizing the enthalpy, the system will tend to a concentration vector with as long polymers as possible.

The equilibrium is also influenced, by a second, opposing force. The systems also tends to larger *entropy*, computed as

$$
S(\mathbf{x}) = -\sum_{i=1}^nx_i\log x_i\,.
$$

Maximizing the entropy means more free particles (more 'chaos').

As a whole, the system will try to minize its Gibbs free energy, containing both the enthalpy as well as the entropy[^2e788f85]:

$$
G(\mathbf{x}) = H(\mathbf{x}) - T\cdot S(\mathbf{x})\,,
$$

where $T$ is the temperature of the system. Here, the temperature is dimensionless with 0 meaning the absolute zero. The equilibrium concentration vector $\mathbf{x}^\star$ minimizes the Gibbs free energy.

The quantity of interest is the average length $L$ of the polymers in the system:

$$
L(\mathbf{x}) = \frac{\sum_{i=1}^nix_i}{\sum_{i=1}^nx_i}\,.
$$

It should be clear that $L^\star$ is both a function of the temperature $T$ and the amount of proteins $c_0$ in the system.

**Project assignments**

1. Formally write the Gibbs free energy minimization problem as a general constraint optimization problem. Show that it is convex.
2. Complete the functions to compute the Gibbs free energy, its gradient and its Hessian.
3. Complete the function to find a strictly feasible starting concentration vector $\mathbf{x}_0$.
4. Complete the code `compute_equilibrium`, which uses a descent method to find the equilibrium concentrations of given dimension. Its inputs are $T$, $c_0$ and an initial **feasible** $\mathbf{x}_0$. The function has an additional parameter `trace`, which (if set to `True`) also returns the steps $\mathbf{x}^{(k)}$ of your iterative algorithm.
5. Use your function to make a phase diagram showing the equilibrium average length $L^\star$ as a function of the temperature $T$ and initial concentration $c_0$. Describe what you see.


## Evaluation

Firstly, you should write code that works and solves the problem. As a second goal, your code is ideally elegant and performant. You can solve the problems using a dimension of $n=100$, which should go quite quickly even using relatively sloppy implementations. However, it is possible to compute Newton steps very efficiently *without* computing the complete Hessian. Such implementations can easily deal with sizes of $n\gg 1000$ and will receive the highest scores.

Some general guidelines:

- Write some clear and concise explanation with the figures and experiments.
- Write clean code:
  - use descriptive variable names
  - document functions
  - comment code lines which may not be obvious
- Make nice figures with title, labeled axes, legend etc.

## Hints

- Keep into mind the logarithms: $0 \times \log 0$ should evaluate to 0 (by definition), but in python `0 * np.log(0)` returns a 'nan'. Make sure to keep this in mind!
- The constraint Newton step *can* be computed efficiently in terms of time and memory complexity. Implementing this is however not a prerequisite for succeeding for the project.


[^2e788f85]: Actually, the Gibbs free energy is just the change in entropy of the universe.

[^39b98177]: In this project we will dispense with all units of energy, concentration or temperature. Likewise, all constants are nice round numbers to keep the focus on optimization. A simple rescaling would make the units in this toy example realistic.
