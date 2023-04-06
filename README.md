# A Variational AutoEncoder - based Graph Neural Network for Bayesian Structure Learning.

How does it work?:
This model is based on this [paper](https://arxiv.org/abs/1904.10098).

To summarize, it uses a variational autoencoder to embed data samples into a latent space parametrized by an adjacency matrix, while minimizing the ELBO loss and maintaining a continuous acyclicity constraint.

[Read more in this Medium Article my teammate and I worked on](https://medium.com/@mfkhalil/64ea09a0c4)
