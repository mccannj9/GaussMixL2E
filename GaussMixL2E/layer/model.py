#! /usr/bin/env python3

import tensorflow as tf
import tensorflow_probability as tfp

from utils.misc import stochastic_mini_batch as smb

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
        self.built = False

        # model features
        self.computation_graph = None
        self.parameters = {}

    def build(
        self, init=tf.variance_scaling_initializer(),
            sum_to_one=True, learning_rate=1e-2
    ):

        if self.built:
            return

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
        self, data, batch_size=None, convergence=1e-4,
            max_epochs=10000, restore=False
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
            f = smb(X, data, batch_size)
            loss_prev = sesh.run(loss, feed_dict=f)

            for i in range(max_epochs):
                f = smb(X, data, batch_size)
                sesh.run(training_op, feed_dict=f)

                if i % 100 == 0:
                    loss_curr = sesh.run(loss, feed_dict=f)
                    d = np.abs(loss_curr - loss_prev)

                    if d < converged:
                        print(f"[Converged {i}]: {loss_curr}, {d}")
                        break
                    else:
                        print(f"[Epoch {i}]: {loss_curr}")
                        loss_prev = loss_curr

        # return loss and likelihood to fit method, for multiple tries
        return 0

    def fit(self, data, *args, **kwargs):
        loss, lk = self.train(self, data, **kwargs)
        return {"loss": loss, "likelihood": lk}

    def fit_predict(self, data, *args, **kwargs):
        loss, lk = self.train(data, *args, **kwargs)
        predictions = self.predict(data)

        return {
            "loss": loss, "likelihood": lk, "predictions": predictions
        }

    def predict(self, data):
        pass

    def __call__(self, *args, **kwargs):
        if not(self.built):
            self.build(*args, **kwargs)

    def __str__(self):
        return f"{self.name}: Built={self.built}, K={self.k}, D={self.d}"
