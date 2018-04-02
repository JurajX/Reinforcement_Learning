import tensorflow as tf
import numpy as np

class REINFORCE(object):
    """
    An object implementing a reinforcement learning algorithm, REINFORCE.
    This algorithm is linear, however it can easily be extended to non-linear case.
    """
    def __init__(self, n_features, n_actions):
        """
        Create an instance of REINFORCE object, i.e. create TensorFlow graph and initialize tf.Session.

        Args:
            n_features - number of features of a state vector (input layer)
            n_actions  - number of possible actions to choose from (output layer)
        """
        self._graph = tf.Graph()
        self._writer = None

        with self._graph.as_default():
            self.state     = tf.placeholder(shape=(1,n_features), dtype=tf.float32, name='state')
            self.action    = tf.placeholder(shape=None,  dtype=tf.int32,   name='action')
            self.dis_pow_t = tf.placeholder(shape=None,  dtype=tf.float32, name='discount_power_t')
            self.ret       = tf.placeholder(shape=None,  dtype=tf.float32, name="return")
            self.lrn_rate  = tf.placeholder(shape=None,  dtype=tf.float32, name='learning_rate')

            with tf.name_scope('policy'):
                _W1 = tf.Variable(tf.truncated_normal(shape=(n_features, n_actions))*0.0001,
                                       dtype=tf.float32, name='weights')
                _l1 = tf.matmul(self.state, _W1, name='layer1')
                self._probs = tf.nn.softmax(_l1, name='action_probabilities')

            with tf.name_scope('training'):
                _optimizer = tf.train.GradientDescentOptimizer(1)
                # GradientDescentOptimizer doesn't have maximize function; max(x) = min(-x)
                self._train = _optimizer.minimize(
                    -tf.log(self._probs[0, self.action])*self.dis_pow_t*self.ret*self.lrn_rate )

            self._saver = tf.train.Saver()
            self._initializer = tf.global_variables_initializer()

            with tf.name_scope('summaries'):
                tf.summary.histogram(name='histogram_weights', values=_W1)
                self._merged_summaries = tf.summary.merge_all()

        # Now that the graph is built, create a tf.Session for it, and initialize variables.
        self._sess = tf.Session(graph=self._graph)
        self._sess.run(self._initializer)

    #-----  Training and performing

    def chooseAction(self, state):
        """
        Args:
            state - state vector of shape (1, n_features)

        Returns:
            action - integer in the range [0, n_actions)
        """
        feed_dict = {self.state: state}
        probs = self._sess.run(self._probs, feed_dict).reshape(-1)
        return np.argmax(np.random.multinomial(1, probs, 1))

    def _discountedRet(self, rewards, discount):
        discounts = [discount**i for i in range(len(rewards))]
        return np.dot(discounts, rewards)

    def train(self, states, actions, rewards, discount, lrn_rate):
        """
        Perform weights update for an entire episode.

        Args:
            states   - an array of state vectors, state vectors are of shape (1, n_features)
            actions  - an array of actions taken in the corresponding states,
                       actions are integers in the range [0, n_actions)
            rewards  - an array of rewards returned by an environment after choosing an action
            discount - discount parameter, a floating point number in the range (0, 1)
            lrn_rate - learning rate, a floating point number in the range (0, 1)
        """
        for i in range(len(rewards)):
            ret = self._discountedRet(rewards[i:], discount)
            dis = discount**(i)
            feed_dict = {self.state: states[i], self.action: actions[i], self.dis_pow_t: dis,
                         self.ret: ret, self.lrn_rate: lrn_rate}
            self._sess.run(self._train, feed_dict)

    #-----  Saving, restoring, and printing

    def save(self, PATH):
        """
        Save the policy parameter vector to the given PATH.
        """
        self._saver.save(self._sess, PATH)

    def restore(self, PATH):
        """
        Restore saved policy parameter vector from the given PATH.
        """
        self._saver.restore(self._sess, PATH)

    def printParameters(self):
        """
        Print policy parameter vector.
        """
        with self._graph.as_default():
            for var in tf.global_variables():
                print(var.name)
                val = self._sess.run(var)
                print(val)

    #-----  TensorBoard summaries

    def closeFileWriter(self):
        """
        Close tf.summary.FileWriter used for saving summaries.
        """
        try: self._writer.close()
        except: pass    # Is already closed or does not exist, i.e. _writer is None.

    def createFileWriter(self, PATH):
        """
        Try to close an open tf.summary.FileWriter and create a new one.
        """
        self.closeFileWriter()
        self._writer = tf.summary.FileWriter(logdir=PATH, graph=self._graph)

    def makeSummary(self, i):
        """
        Create and send summaries to FileWriter. If no FileWriter was initialized, print error message.
        """
        try:
            summary = self._sess.run(self._merged_summaries)
            self._writer.add_summary(summary=summary, global_step=i)
        except:
            print('FileWriter is either closed or does not exist.')
            print('Use createFileWriter member function to create a new FileWriter.')

    #-----  Closing, and resetting

    def close(self):
        """
        Close tf.Session, which releases acquired resources.
        """
        self.closeFileWriter()
        try: self._sess.close()
        except: pass                # Is already closed.

    def reset(self):
        """
        Try to close current tf.Session and start a new one.
        """
        self.close()
        self._sess = tf.Session(graph=self._graph)
        self._sess.run(self._initializer)
