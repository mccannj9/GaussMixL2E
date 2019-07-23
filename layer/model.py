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

        X = tf.placeholder(tf.float32, shape=(None, self.d), name="inputs")
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
            cov_1 = chols[i] @ tf.transpose(chols[i])
            for j in range(self.k):
                cov_2 = chols[j] @ tf.transpose(chols[j])
                diff = tfd.MultivariateNormalFullCovariance(
                    locs=mus[i]-mus[j], covariance_matrix=cov_1+cov_2,
                    name=f"mvn_diff_optimizer_{i}"
                )
                losses.append(diff.prob(tf.zeros(self.d)) * w[i] * w[j])

        loss_1st_term = tf.add_n(losses)

        losses = []

        for i in range(self.k):
            losses.append(tf.multiply(w[i], mvns[i].prob(X)))

        # can subtract, i.e. tf.log(tf.reduce_sum(w)) to keep weights
        # sum close to one when using sigmoid bijector on weights
        loss_2nd_term = -2 * tf.reduce_mean(tf.add_n(losses))

        final_loss = tf.add(loss_1st_term, loss_2nd_term, name="loss")
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer.minimize(final_loss)

        self.built = True
        self.computation_graph = tf.get_default_graph()

    def train(
        self, data, batch_size=None, convergence=1e-4, max_epochs=10000
    ):
        X = self.computation_graph.get_tensor_by_name("inputs:0")
        w = self.computation_graph.get_tensor_by_name("weights:0")

        mus = [
            self.computation_graph.get_tensor_by_name(
                f"Mu_{i}:0"
            ) for i in range(self.k)
        ]

        cos = [
            self.computation_graph.get_tensor_by_name(
                f"Chol_Var_Assign_{i}:0"
            ) for i in range(self.k)
        ]

        loss = self.computation_graph.get_tensor_by_name("loss:0")
        training_op = self.computation_graph.get_operation_by_name("Adam")

        with self.computation_graph.as_default():
            init = tf.global_variables_initializer()

        with tf.Session(graph=self.computation_graph) as sesh:
            sesh.run(init)

    def fit(self, data, *args, **kwargs):
        self.train(self, data, **kwargs)
