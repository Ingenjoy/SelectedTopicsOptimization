# Selected Topics in Mathematical Optimization

*Edition 2017-2018*

**Dr. ir. Michiel Stock**

Notes and exercises of the optimization course given in the Master of Bioinformatics Bioscience Engineering and Systems Biology (Ghent University).

Goal,...

Will be updated through the semester.

## Contents

This course consists of three main parts:
1. Continuous convex optimization problems
2. Discrete optimization problems solvable in polynomial time
3. 'Dirty problems', NP hard problems and complex problems with no guarantees on optimality and performance


1. **Minimizing quadratic systems**
  - motivation
  - exact solution, scalar case, multi-dimensional case
  - conditions for optimality
  - very large (sparse) systems and the need for iterative solutions
  - gradient descent, convergence and condition numbers
  - brief notion of conjugated gradient descent
  - gradient descent with momentum (intelligent search)
  - *application*: signal recovery
2. **Unconstrained convex problems**
  - convexity and convex problems
  - gradient descent revisited
  - steepest descent methods, coordinate descent
  - Newton's method:
    - as a special case of steepest descent
    - as a quadratic approximation of the original problem
  - quasi-Newton methods
  - numerically approximating the gradient and the Hessian
  - *application*: logistic regression
3. **Constrained convex problems**
  - quadratic systems with linear equality constraints: exact solution
  - Newton's method for convex systems with linear equality constraints
  - Convex problems with convex inequality constraints
    - geometric interpretation
    - the logarithmic barrier
    - the barrier method
  - *application*: maximum flow problems
4. **Project continuous optimization**: protein oligiomerization by minimizing the Gibbs free energy
5. **Dynamic programming**
  - main idea: counting and Fibonacci sequence
  - sequence alignment
  - RNA fold prediction
  - Bellman equation?
  - *application*: something cool
6. **Bayesian networks**: MAP estimation
7. **Shortest path algorithms**
  - greedy
  - Dijkstra's algorithm
  - A* algorithm: using a heuristic
  - minimum spanning tree?
8. **Project discrete optimization**: something really cool!
9. **NP hard problems**
  - classification
  - example problems: knapsack, TSA, ...
  - algorithms:
    - exhaustive
    - greedy
    - dynamic programming
    - branch and bound
10. **Bio-inspired optimization**
  - hill climbing?
  - simulated annealing
  - genetic algorithms
  - ...
  - *application*: finding anti-microbibial peptides
11. **Learning and optimization**
  - Bayesian optimization
  - Reinforcement Learning
  - Learning inverse mappings
  - ...
  - *application*: ...
12. **Project dirty problems**:
