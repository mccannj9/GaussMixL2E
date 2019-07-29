#! /usr/bin/env python3

from utils.misc import stochastic_mini_batch as smb
from utils.math import likelihood, sigmoid, softmax

import numpy as np

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

        X = tf.compat.v1.placeholder(tf.float32, shape=(None, self.d), name="inputs")
        w = tf.compat.v1.get_variable("weights", shape=(self.k,), initializer=init)

        if sum_to_one:
            w = tf.nn.softmax(w)
            self.weights_bijector = softmax
        else:
            w = tf.nn.sigmoid(w)
            self.weights_bijector = sigmoid

        # number of elements in the cholesky matrix given d
        cholesky_shape = self.d * (self.d + 1) // 2

        mus, chols, mvns = [], [], []

        for i in range(self.k):
            mu = tf.compat.v1.get_variable(f"Mu_{i}", shape=(self.d,), initializer=init)
            ch = tf.compat.v1.get_variable(f"Chol_{i}", shape=(cholesky_shape,), initializer=init)
            ch = tfd.fill_triangular(ch)
            ch = bijection.forward(ch, name=f"Chol_Mat_{i}")
            chol_var = tf.compat.v1.get_variable(f"Chol_Var_{i}", shape=(self.d, self.d))
            # assigns cholesky matrix to node in graph for later retrieval
            tf.compat.v1.assign(chol_var, ch, name=f"Chol_Var_Assign_{i}")

            mvn = tfd.MultivariateNormalTriL(
                loc=mu, scale_tril=ch, name=f"mvn_mixture_{i}"
            )

            mus.append(mu)
            chols.append(ch)
            mvns.append(mvn)

        # setup the loss function for the model
        losses = []

        for i in range(self.k):
            cov_1 = chols[i] @ tf.transpose(chols[i])
            for j in range(self.k):
                cov_2 = chols[j] @ tf.transpose(chols[j])
                diff = tfd.MultivariateNormalFullCovariance(
                    loc=mus[i]-mus[j], covariance_matrix=cov_1+cov_2,
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
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer.minimize(final_loss)

        self.built = True
        self.computation_graph = tf.compat.v1.get_default_graph()

    def train(
        self, data, batch_size=None, converged=1e-4,
            max_epochs=10000, restore=False
    ):
        if not(batch_size):
            batch_size = data.shape[0]

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
            print("DATA SHAPE ", data.shape)
            f = smb(X, data, batch_size)
            print("TRAINING", f[X].shape)
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

            parameters = {
                'k': self.k,
                'd': self.d,
                'mu': np.zeros((self.k, self.d)),
                'cov': np.zeros((self.k, self.d, self.d)),
                'weights': self.weights_bijector(sesh.run(w))
            }

            for i in range(self.k):
                mu, co = sesh.run([mus[i], cos[i]])#, feed_dict=feeder)
                parameters['mu'][i, :] = mu
                parameters['cov'][i, :, :] = co @ co.T
            ll = likelihood(data, parameters)
            parameters['loss'] = loss_curr
            parameters['likelihood'] = ll

        # return loss and likelihood to fit method, for multiple tries
        return parameters, loss_curr, ll

    def fit(self, data, *args, starts=1, **kwargs):
        for i in range(starts):
            parameters, loss, ll = self.train(data, **kwargs)
            if i == 0:
                best_ll = ll
                best_parameters = parameters
                best_loss = loss
                print(f"initial run: {best_ll}")
            else:
                print(f"new run: {best_ll}")
                if ll > best_ll:
                    print(f"New best: {best_ll}")
                    best_parameters = parameters
                    best_ll = ll
                    best_loss = loss

        return {
            "parameters": best_parameters,
            "loss":best_loss,
            "likelihood": best_ll
        }

    def fit_predict(self, data, *args, **kwargs):
        parameters, loss, lk = self.train(data, *args, **kwargs)
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


if __name__ == "__main__":
    from GaussMixL2E.utils.simulate import build_toy_dataset
    data = build_toy_dataset(500)

    GMM = GaussianMixture(2, 2, name="GMM_test")

