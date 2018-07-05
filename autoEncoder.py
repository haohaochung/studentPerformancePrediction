import numpy as np
import tensorflow as tf

class autoEncoder(object):
    def __init__(self, layer_name="", input=None, n_visible=60, n_hidden=256):

        with tf.name_scope('Weights'):
            W = tf.Variable(tf.random_uniform([n_visible,n_hidden],minval=(-4*(6.0/(n_visible+n_hidden))**0.5),maxval=(
                        4*(6.0/(n_visible+n_hidden))**0.5)))
            W_prime = tf.transpose(W, name="W_prime")
        with tf.name_scope('biases'):
            b = tf.Variable(tf.zeros([1, n_hidden]))
            b_prime = tf.Variable(tf.zeros([1, n_visible]), name="b_prime")

        self.layer_name = layer_name
        self.learning_rate = 0.01
        self.training_epochs = 3000
        self.x = input
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = W
        self.W_prime = W_prime
        self.b = b
        self.b_prime = b_prime

    def masking_noise(self, X, v):
        """ Apply masking noise to data in X, in other words a fraction v of elements of X
        (chosen at random) is forced to zero.
        :param X: array_like, Input data
        :param v: int, fraction of elements to distort
        :return: transformed data
        """
        X_noise = X.copy()

        n_samples = X.shape[0]
        n_features = X.shape[1]

        for i in range(n_samples):
            mask = np.random.randint(0, n_features, int(v*n_features) )
            for m in mask:
                X_noise[i][m] = 0#tf.constant(0, tf.float32)

        return X_noise

        # Encode
    def encoder(self, input):
        return tf.nn.relu(tf.add(tf.matmul(input, self.W), self.b))

        # Decode
    def decoder(self, hidden):
        return tf.nn.relu(tf.add(tf.matmul(hidden, self.W_prime), self.b_prime))

    def train(self, lr=0.1, corruption_level = 0.3, input=None):

        if input is not None:
            self.x = input
        x = self.x

        tilde_x = self.masking_noise(x, corruption_level)

        x_feeds = tf.placeholder(tf.float32, [None, self.n_visible], name= self.layer_name+"_input")
        y = self.encoder(x_feeds)
        z = self.decoder(y)
        mse = tf.reduce_mean(tf.pow(x_feeds - z, 2))
        optimizer = tf.train.AdamOptimizer(0.05).minimize(mse)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.training_epochs):
                _, loss = sess.run([optimizer, mse], feed_dict={x_feeds:tilde_x})

                if epoch % 100 == 0:
                    print(epoch, loss)
            print(tf.add(tf.matmul(self.x, self.W), self.b).shape)
            return self.W.eval(), self.b.eval(), tf.nn.relu(tf.add(tf.matmul(self.x, self.W), self.b)).eval()

