import tensorflow as tf
import numpy as np

#============================================= Actor ==============================================#

class nl_Actor(object):
    """
    An object implementing actor from actor-critic algorithm (reinforcement learning).
    """
    def __init__(self, n_features, n_actions):
        """
        Create an instance of Actor object, i.e. create TensorFlow graph and initialize tf.Session.

        Args:
            n_features - number of features of a state vector (input layer)
            n_actions  - number of possible actions to choose from (output layer)
        """
        self._graph = tf.Graph()
        self._time_step = 0
        self._writer = None

        with self._graph.as_default():
            self.state    = tf.placeholder(shape=(1, n_features), dtype=tf.float32, name='state')
            self.action   = tf.placeholder(shape=None,  dtype=tf.int32,   name='action')
            self.discount = tf.placeholder(shape=None,  dtype=tf.float32, name='discount')
            self.decay    = tf.placeholder(shape=None,  dtype=tf.float32, name='decay')
            self.td_error = tf.placeholder(shape=None,  dtype=tf.float32, name='td_error')
            self.lrn_rate = tf.placeholder(shape=None,  dtype=tf.float32, name='learning_rate')

            with tf.name_scope('policy'):
                _W1 = tf.Variable(tf.truncated_normal(shape=(n_features, n_features))*0.001,
                                       dtype=tf.float32, name='weights1')
                _l1 = tf.atan(tf.matmul(self.state, _W1, name='layer1'), name='activation_fct1')

                _W2 = tf.Variable(tf.truncated_normal(shape=(n_features, n_actions))*0.001,
                                       dtype=tf.float32, name='weights2')
                _l2 = tf.matmul(_l1, _W2, name='layer2')

                _probs = tf.nn.softmax(_l2, name='action_probabilities')
                _log_probs = tf.log(_probs, name='log_probs')
                self._action = tf.multinomial(logits=_log_probs, num_samples=1, name='_action')[0,0]

            with tf.name_scope('training'):
                _optimizer = tf.train.GradientDescentOptimizer(1)
                _grads = _optimizer.compute_gradients(_log_probs[0, self.action])
                # Create eligibility traces with the same structure as gradients in grads_and_vars
                # returned from the compute_gradients member function of GradientDescentOptimizer.
                _traces = [tf.Variable(initial_value=tf.zeros(shape=grad[0].shape),
                                       trainable=False,
                                       dtype=tf.float32,
                                       name='trace')
                           for grad in _grads]
                _update_traces = [ _traces[i].assign(
                    self.discount*self.decay*_traces[i] + (self.discount**self._time_step)*_grads[i][0]
                    ) for i in range(len(_grads))]
                # The minus sign is there because apply_gradients performs minimization,
                # i.e. it multiplies the gradients with an additional minus. Moreover, using
                # _update_traces instead of _traces performs traces update and uses the result.
                _new_grads = [ ( -self.lrn_rate*self.td_error*_update_traces[i], _grads[i][1] )
                               for i in range(len(_grads)) ]
                self._train = _optimizer.apply_gradients(_new_grads)

            with tf.name_scope('reset'):
                self._reset_traces = [ trace.assign(tf.zeros(shape=trace.shape)) for trace in _traces ]

            self._saver = tf.train.Saver()
            self._initializer = tf.global_variables_initializer()

            with tf.name_scope('summaries'):
                tf.summary.histogram(name='histogram_weights1', values=_W1)
                tf.summary.histogram(name='histogram_weights2', values=_W2)
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
        action = self._sess.run(self._action, feed_dict)
        return action

    def train(self, state, action, discount, decay, td_error, lrn_rate):
        """
        Perform a weights update for a state-action pair.

        Args:
            state    - state vector of shape (1, n_features)
            action   - integer in the range [0, n_actions)
            discount - discount parameter, a floating point number in the range (0, 1)
            decay    - decay factor, a floating point number in the range (0, 1)
            tf_error - TD error returned from Critic class
            lrn_rate - learning rate, a floating point number in the range (0, 1)
        """
        feed_dict = {self.state: state, self.action: action, self.discount: discount,
                     self.decay: decay, self.td_error: td_error, self.lrn_rate: lrn_rate}
        self._sess.run(self._train, feed_dict)
        self._time_step = self._time_step + 1

    def resetTraces(self):
        """
        Set eligibility traces to zeros. Needs to be called at the beginning of every episode.
        """
        self._sess.run(self._reset_traces)
        self._time_step = 0

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


#============================================= Critic =============================================#

class nl_Critic(object):
    """
    An object implementing critic from actor-critic algorithm (reinforcement learning).
    """
    def __init__(self, n_features):
        """
        Create an instance of Critic object, i.e. create TensorFlow graph and initialize tf.Session.

        Args:
            n_features     - number of features of a state vector (input layer)
        """
        self._graph = tf.Graph()
        self._writer = None

        with self._graph.as_default():
            self.state    = tf.placeholder(shape=(1, n_features), dtype=tf.float32, name='state')
            self.discount = tf.placeholder(shape=None,  dtype=tf.float32, name='discount')
            self.decay    = tf.placeholder(shape=None,  dtype=tf.float32, name='decay')
            self.td_error = tf.placeholder(shape=None,  dtype=tf.float32, name='td_error')
            self.lrn_rate = tf.placeholder(shape=None,  dtype=tf.float32, name='learning_rate')

            with tf.name_scope('value_function'):
                _W1 = tf.Variable(tf.truncated_normal(shape=(n_features, n_features))*0.001,
                                       dtype=tf.float32, name='weights1')
                _l1 = tf.atan(tf.matmul(self.state, _W1, name='layer1'), name='activation_fct1')

                _W2 = tf.Variable(tf.truncated_normal(shape=(n_features, 1))*0.001,
                                       dtype=tf.float32, name='weights2')
                _l2 = tf.matmul(_l1, _W2, name='layer2')

                self._value_fct = tf.identity(_l2, name='value_function')

            with tf.name_scope('training'):
                _optimizer = tf.train.GradientDescentOptimizer(1)
                _grads = _optimizer.compute_gradients(self._value_fct[0, 0])
                # Create eligibility traces with the same structure as gradients in grads_and_vars
                # returned from the compute_gradients member function of GradientDescentOptimizer.
                _traces = [tf.Variable(initial_value=tf.zeros(shape=grad[0].shape),
                                       trainable=False,
                                       dtype=tf.float32,
                                       name='trace')
                           for grad in _grads]
                _update_traces = [ _traces[i].assign(self.discount*self.decay*_traces[i] + _grads[i][0])
                                   for i in range(len(_grads))]
                # The minus sign is there because apply_gradients performs minimization,
                # i.e. it multiplies the gradients with an additional minus. Moreover, using
                # _update_traces instead of _traces performs traces update and uses the result.
                _new_grads = [ ( -self.lrn_rate*self.td_error*_update_traces[i], _grads[i][1] )
                               for i in range(len(_grads)) ]
                self._train = _optimizer.apply_gradients(_new_grads)

            with tf.name_scope('reset'):
                self._reset_traces = [ trace.assign(tf.zeros(shape=trace.shape)) for trace in _traces ]

            self._saver = tf.train.Saver()
            self._initializer = tf.global_variables_initializer()

            with tf.name_scope('summaries'):
                tf.summary.histogram(name='histogram_weights1', values=_W1)
                tf.summary.histogram(name='histogram_weights2', values=_W2)
                self._merged_summaries = tf.summary.merge_all()

        # Now that the graph is built, create a tf.Session for it, and initialize variables.
        self._sess = tf.Session(graph=self._graph)
        self._sess.run(self._initializer)

    #-----  Training and performing

    def computeTDerror(self, old_state, new_state, reward, discount):
        """
        Args:
            old_state - state vector of shape (1, n_features) preceding new_state
            new_state - state vector of shape (1, n_features) succeeding old_state
            reward    - rewards returned by an environment after choosing an action in the old_state
            discount  - discount parameter, a floating point number in the range (0, 1)

        Returns:
            td_error - temporal difference error used in train member functions of both Actor and Critic
        """
        old_vf = self._sess.run(self._value_fct, feed_dict={self.state: old_state})
        new_vf = self._sess.run(self._value_fct, feed_dict={self.state: new_state})
        return reward + discount*new_vf - old_vf

    def train(self, state, discount, decay, td_error, lrn_rate):
        """
        Perform a weights update for a state-action pair.

        Args:
            state    - state vector of shape (1, n_features)
            discount - discount parameter, a floating point number in the range (0, 1)
            decay    - decay factor, a floating point number in the range (0, 1)
            tf_error - TD error returned from computeTDerror method of this class
            lrn_rate - learning rate, a floating point number in the range (0, 1)
        """
        feed_dict = {self.state: state, self.discount: discount, self.decay: decay,
                     self.td_error: td_error, self.lrn_rate: lrn_rate}
        self._sess.run(self._train, feed_dict)

    def resetTraces(self):
        """
        Set eligibility traces to zeros. Needs to be called at the beginning of every episode.
        """
        self._sess.run(self._reset_traces)

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
