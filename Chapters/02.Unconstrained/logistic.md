# Gradient and Hessian of the regulairized logistic loss

$$
\mathcal{L}(\mathbf{w})=-\sum_{i=1}^n[y_i\log(\sigma(\mathbf{w}^\intercal\mathbf{x}_i))+(1-y_i)\log(1-\sigma(\mathbf{w}^\intercal\mathbf{x}_i))] +\lambda \mathbf{w}^\intercal\mathbf{w}
$$

$$
\frac{\partial \mathcal{L}(\mathbf{w})}{\partial \mathbf{w}} =  -\mathbf{X}^\intercal \mathbf{y} +\mathbf{X}^\intercal\sigma(\mathbf{X}\mathbf{w}) + 2\lambda \mathbf{w}
$$

$$
\frac{\partial^2 \mathcal{L}(\mathbf{w})}{\partial \mathbf{w}^2} = \sum_{i=1}^n[\sigma(\mathbf{w}^\intercal\mathbf{x}_i)(1-\sigma(\mathbf{w}^\intercal\mathbf{x}_i))\mathbf{x}_i\mathbf{x}_i^\intercal] +2\lambda \mathbf{I}
$$
