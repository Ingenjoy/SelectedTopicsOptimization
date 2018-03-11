# Optimal transport

*Selected Topics in Mathematical Optimization: 2017-2018*

**Michiel Stock** ([email](michiel.stock@ugent.be))

![](Figures/logo.png)

## Motivation

### Motivating example: a party in the research group

Let's have a party in our research unit! Pastries and party hats for everyone! We ask Tinne, our laboratory manager, to make some desserts: an airy merveilleux, some delicious eclairs, a big bowl of dark chocolate mousse, a sweet passion fruit-flavored bavarois and moist carrot cake (we got to have our vegetables). If we mentally cut all these sweets into portions, we have twenty portions as shown in the table below.

![Quantities of every dessert.](../images/2017_optimal_transport/desserts.png)

Since this is academia, we respect the hierarchy: people higher on the ladder are allowed to take more dessert. The professors, Bernard, Jan and Willem each get three pieces each, our senior post-doc Hilde will take four portions (one for each of her children) and the teaching assistants are allowed two portions per person. Since Wouter is a shared teaching assistant with the Biomath research group, he can only take one (sorry Wouter).

![Number of pieces every KERMIT number can take.](../images/2017_optimal_transport/staff.png)

As engineers and mathematicians, we pride ourselves in doing things the optimal way. So how can we divide the desserts to make everybody as happy as possible? As I am preparing a course on optimization, I went around and asked which of those treats they liked. On a scale between -2 and 2, with -2 being something they hated and 2 being their absolute favorite, the desert preferences of the teaching staff is given below (students: take note!).

![Preferences of the KERMIT staff for different desserts. ](../images/2017_optimal_transport/preferences.png)

See how most people like eclairs and chocolate mousse, but merveilleus are a more polarizing dessert! Jan is lactose intolerant, so he only gave a high score to the carrot cake by default.

The task is clear: divide these desserts in such a way that people get their portions of the kinds they like the most!

## Notation and formulation


### Probability vectors and histograms

We work with *probability vectors* or *histograms* $\mathbf{a}\in\Sigma_n$, where we define
$$
\Sigma_n =\{\mathbf{a}\in \mathbb{R}^n_+\mid \sum_{i=1}^na_i=1\}\,,
$$
these are the set of nonnegative vectors that sum to one. $\Sigma_n$ is called the *probability simplex*.

Let $\mathbf{r}$$ be the vector containing the relative amount of dessert every person can eat. In this case $\mathbf{r} = [3,3,3,4,2,2,2,1]^\top$ (in general the dimension of $\mathbf{r}$ is $n$). Similarly, $\mathbf{c}$ denotes the vector of how much there is of every dessert, i.e. $\mathbf{c}=[4, 2, 6, 4, 4]^\intercal$ (in general the dimension of $\mathbf{c}$ is $m$).

TODO: add examples

### Cost matrix

### Examples

### Monge's assignment problem

The original version of optimal transport was formulated by Gaspard Monge in 1781. Here, $n=m$ and the goals is to connect $n$ sources with $n$ sinks to minimize a cost:
$$
\min_{\sigma\in\text{Perm(n)}} \frac{1}{n}C_{i,\sigma(i)}\,,
$$
with $\text{Perm(n)}$ the set of all permutation of $n$ elements.

> **Example** There are $n$ mines mining iron ore and a collection of $n$ factories. Given a distance between every mine and every factory, select one factory for every mine such that the total cost (=transportation distance) is minimized.

- The size of the search space is $n!$, for $n=100$, there are more than $10^{100}$ permutations!
- Discrete combinatorial optimization problem, no efficient algorithm.
- Restrictive: two sets to match must be of same size. How to deal with different weights?
- Need for 'soft matching'.

**Exercise 1**

In a microscopy imaging experiment we monitor ten moving cells at time $t_1$ and some time later at time $t_2$. Between these times, the cells have moved. An image processing algorithm determined the coordinates of every cell in the two images. We want to know which cell in the first corresponds to the second image. To this end, search the assignment that minimizes the sum of the squared Euclidian distances between cells from the first image versus the corresponding cell of the second image.

1. `X1` and `X2` contain the $x,y$ coordinates of the cells for the two images. Compute the matrix $C$ containing the pairwise squared Euclidian distance. You can use the function `pairwise_distances` from `sklearn`.
2. Complete the function `kantorovich_brute_force` to use brute-force search for the best permutation.
3. Make a plot connecting the cells.

### Kantorovich's relaxation

- admissible couplings

## Wasserstein distances and barycenters

- measures
