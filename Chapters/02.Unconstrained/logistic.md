# Gradient and Hessian of the regulairized logistic loss

$$
\mathcal{L}(\mathbf{w})=-\sum_{i=1}^n[y_i\log(\sigma(\mathbf{w}^\top\mathbf{x}_i))+(1-y_i)\log(1-\sigma(\mathbf{w}^\top\mathbf{x}_i))] +\lambda \mathbf{w}^\top\mathbf{w}
$$

$$
\frac{\partial \mathcal{L}(\mathbf{w})}{\partial \mathbf{w}} =  -\mathbf{X}^\top \mathbf{y} +\mathbf{X}^\top\sigma(\mathbf{X}\mathbf{w}) + 2\lambda \mathbf{w}
$$

$$
\frac{\partial^2 \mathcal{L}(\mathbf{w})}{\partial \mathbf{w}^2} = \sum_{i=1}^n[\sigma(\mathbf{w}^\top\mathbf{x}_i)(1-\sigma(\mathbf{w}^\top\mathbf{x}_i))\mathbf{x}_i\mathbf{x}_i^\top] +2\lambda {I}
$$
