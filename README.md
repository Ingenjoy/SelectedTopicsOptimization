# Selected Topics in Mathematical Optimization

*Edition 2017-2018*

**Dr. ir. Michiel Stock**

Notes and exercises of the optimization course given in the Master of Bioinformatics Bioscience Engineering and Systems Biology (Ghent University).

The goal of this course is to give students a general overview of the rich field of mathematical optimization. This course will put a particular emphasis on **practical implementations** and **performance**. After this course, students should be able to formulate problems from computational biology as optimization problems and be able to read, understand and implement new optimization algorithms.

The project exercises are done in the Python 3.x programming language. It is recommended to use [Anaconda](https://anaconda.org/anaconda/python) to install Python, the required libraries and the Jupyter notebook environment.

## Using this repository

Course notes are available in Markdown format. Slides and pdf notes will be distributed via [Minerva](http://minerva.ugent.be/). Every week we will cover a new chapter. Exercises of the chapters can be made in the Jupyter notebooks. Ideally, you work in a local clone of this repository. Students without a laptop (or curious browsers) can make the exercises in [Binder](https://mybinder.org/) by clicking on the badge below.

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/MichielStock/SelectedTopicsOptimization/master)

**It seems that Binder does not allow to save your work. This is especially important for the project assignments, which have to be handed in via the student publications in Minerva. Download your notebook after use!**

## Contents

This course consists of three main parts:
1. Continuous convex optimization problems
2. Discrete optimization problems solvable in polynomial time
3. 'Dirty problems', NP hard problems and complex problems with no guarantees on optimality and performance

More detailed, tentative overview:

1. **Minimizing quadratic systems**
  - motivation
  - exact solution, scalar case, multi-dimensional case
  - conditions for optimality
  - large (sparse) systems and the need for iterative solutions
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
  - Lagrangians and the Karush–Kuhn–Tucker conditions
  - Convex problems with convex inequality constraints
    - geometric interpretation
    - the logarithmic barrier
    - the barrier method
4. **Project continuous optimization**: protein oligomerization by minimizing the Gibbs free energy
5. **Optimal transport**:
  - motivation: the KERMIT dessert party
  - quick recap of probability distributions
  - Monge and Kantorovich formulation
  - Wasserstein distances and Geodesic displacements
  - entropic regularization and the Sinkhorn algorithm
  - *applications*:
    - comparing distributions (e.g. expression profiles)
    - color transfer
    - learning epigenetic landscapes
6. **Minimum spanning trees**:
  - graphs and basic data structures (i.e. `list`, `dict`, `set`)
  - introduction to time complexities
  - Prim's algorithm
  - Kruskal's algorithm
  - *application*: phylogenetic tree reconstruction, building a maze
7. **Shortest path algorithms**:
  - sum-product and min-sum algorithm (dynamic programming)
  - Dijkstra's algorithm
  - A* algorithm: using a heuristic
8. **Project discrete optimization**: 'A walk to the park', finding all shortest paths to a set of edges.
9. **NP hard problems**
  - classification
  - example problems: knapsack, TSA, graph coloring, Boolean satisfiability problem, TSP...
  - computational complexity: NP-complete problems
  - algorithms:
    - exhaustive / brute force
    - greedy
    - dynamic programming
    - branch and bound
  - longest common subsequence: golfing contest
10. **Bio-inspired optimization**:
  - hill climbing
  - simulated annealing
  - genetic algorithms
  - evolutionary search: CMA-ES
  - *application*: antimicrobial peptide optimization
11. **Project dirty problems**: traveling salesman problem using heuristics
12. **Learning and optimization**
  - short introduction to Bayesian optimization


## Thanks

- Bernard De Baets
- Raúl Pérez-Fernández
- Bram De Jaegher
- Tim Meese (for finding the missing link)
