#! /usr/bin/env python3

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

# set bijectors to ensure positive semidefinite cholesky matrix
# softplus is y = ln(1 + exp(x)), x in R, y in R+
softplus = tfp.bijectors.Softplus()
chain = tfp.bijectors.Chain([softplus])
bijection = tfp.bijectors.TransformDiagonal(diag_bijector=chain)


class GaussianMixture(object):

    def __init__(self, k, d, name="GMM"):
        self.name = name
        self.k = k
        self.d = d

        # model features
        self.computation_graph = None
        self.parameters = {}

    def build(
        self, init=tf.variance_scaling_initializer(),
            sum_to_one=True, learning_rate=1e-2
    ):

        X = tf.placeholder(tf.float32, shape=(None, self.d), name="Input_Data")
        w = tf.get_variable("weights", shape=(self.k,), initializer=init)

        if sum_to_one:
            w = tf.nn.softmax(w)
        else:
            w = tf.nn.sigmoid(w)

        # number of elements in the cholesky matrix given d
        cholesky_shape = self.d * (self.d + 1) // 2

        mus, chols, mvns = [], [], []

        for i in range(self.k):
            mu = tf.get_variable(f"Mu_{i}", shape=(self.d,), initializer=init)
            ch = tf.get_variable(f"Chol_{i}", shape=(chol_shape,), initializer=init)
            ch = tfd.fill_triangular(ch)
            ch = bijection.forward(ch, name=f"Chol_Mat_{i}")
            chol_var = tf.get_variable(f"Chol_Var_{i}", shape=(self.d, self.d))
            # assigns cholesky matrix to node in graph for later retrieval
            tf.compat.v1.assign(chol_var, chol, name=f"Chol_Var_Assign_{i}")

            mvn = tfd.MultivariateNormalTriL(
                loc=mu, scale_tril=chol, name=f"mvn_mixture_{i}"
            )

            mus.append(mu)
            chols.append(chol)
            mvns.append(mvn)

        # setup the loss function for the model
        losses = []

        for i in range(self.k):
            losses.append(tf.multiply(w[i], mvns[i].prob(X)))
