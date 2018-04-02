import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class ValueFunction():
    """
    The member functions of this class compute action-value function, epsGreedyPolicy,
    or perform a semi-gradient training step.
    """
    def __init__(self, in_len, out_len, degree=1):
        """
        Takes number of features in the state vector, number of actions, and polynomial degree. 
        """
        self.in_len = in_len
        self.out_len = out_len
        self.featureTransfromer = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
        self.featureTransfromer.fit(np.zeros(in_len).reshape(1,-1))
        self.weights = np.zeros((len(self.featureTransfromer.get_feature_names()), out_len))
        self.old_weights = np.zeros(self.weights.shape) # old_weights are used for dutch eligibility trace
        self.eligibility_trace = np.zeros(self.weights.shape)

    def _checkDims(self, state):
        if state.shape[0] != self.in_len:
            raise TypeError('Length of state must be equal to', self.in_len)

    def _transformState(self, state):
        self._checkDims(state)
        return self.featureTransfromer.transform(state.reshape(1,-1))[0]

    def computeVF(self, state):
        """
        Takes a state vector and returns an array contains value for each possible action.
        """
        transformed_state = self._transformState(state)
        return np.matmul(transformed_state, self.weights)

    def epsGreedyPolicy(self, state, eps):
        """
        Takes a state vector and epsilon; returns an epsilon greedy action.
        """
        probs = np.zeros(self.out_len)
        probs[np.argmax(self.computeVF(state))] = 1-eps
        probs = probs + (eps/len(probs))
        return np.argmax(np.random.multinomial(1, probs, 1))

    def softmaxPolicy(self, state, temperature = 1.0):
        """
        Choose a soft-max action with respect to the actionVF.
        It is possible to set a temperature parameter, default is 1.0.
        """
        expVF = np.exp(self.computeVF(state) / temperature)
        probs = expVF/np.sum(expVF)
        return np.argmax(np.random.multinomial(1, probs, 1))

    def trainSemiGrad(self, state, action, td_error, learning_rate):
        """
        Performs a semi-gragient training step:
        state:         a state vector in which to train
        action:        an action choosen from the state vector
        td_error:      the TD error at the current state, usually denoted delta
        learning_rate: learning rate for training, usually denoted alpha
        """
        # To derive gradient, realize that the value function is a matrix multiplication
        # of state (1,4)-matrix and weight (4,2)-matrix and gives (1,2)-matrix (two actions).
        # Gradient of this matrix multiplication w.r.t. weight vector gives two matrices
        # of shape (4,2), one for each action. For the action 0, the first column of its gradient
        # matrix is basically the state vector the other column is full of zeros; for the other
        # action the columns are interchanged.
        grad = np.zeros(self.weights.shape)
        grad[:, action] = self._transformState(state)
        self.weights = self.weights + learning_rate*td_error*grad

    def trainEligibTraceSemiGrad(self, state, action, td_error, discount, decay_factor, learning_rate):
        """
        Perform a semi-gradient training step with eligibility trace:
        state:         a state vector in which to train
        action:        an action choosen from the state vector
        td_error:      the TD error at the current state, usually denoted delta
        discount:      discount of the future rewards, usually denoted gamma
        decay_factor:  decay of trace elements, usually denoted lambda
        learning_rate: learning rate for training, usually denoted alpha
        """
        grad = np.zeros(self.weights.shape)
        grad[:, action] = self._transformState(state)
        self.eligibility_trace = discount*decay_factor*self.eligibility_trace + grad
        self.weights = self.weights + learning_rate*td_error*self.eligibility_trace

    def trainDutchTraceSemiGrad(self, state, action, td_error, discount, decay_factor, learning_rate):
        grad = np.zeros(self.weights.shape)
        grad[:, action] = self._transformState(state)
        lr = learning_rate
        ddf = discount*decay_factor
        lrddf = lr*ddf

        self.eligibility_trace = ddf*self.eligibility_trace + grad \
                                -lrddf*np.matmul(grad, np.matmul(self.eligibility_trace.T, grad))
        temp = self.weights
        self.weights = self.weights + lr*td_error*self.eligibility_trace \
                      +lr*np.matmul( self.eligibility_trace - grad, \
                                     np.matmul(self.weights.T - self.old_weights.T, grad) )
        self.old_weights = temp

    def reset(self):
        """
        Reset the weight vector and eligibility trace to zeros.
        """
        self.weights = np.zeros(self.weights.shape)
        self.old_weights = np.zeros(self.old_weights.shape)
        self.eligibility_trace = np.zeros(self.eligibility_trace.shape)

    def resetTraces(self):
        """
        Reset eligibility trace back to zeros.
        """
        self.old_weights = np.zeros(self.old_weights.shape)
        self.eligibility_trace = np.zeros(self.eligibility_trace.shape)

# this is precisely the same as trainDutchTraceSemiGrad but the matrices are flattened into vectors
    # def trainFlatDutchTraceSemiGrad(self, state, action, td_error, discount, decay_factor, learning_rate):
    #     w = self.weights.reshape(-1)
    #     o_w = self.old_weights.reshape(-1)
    #     e_t = self.eligibility_trace.reshape(-1)
    #     grad = np.zeros(self.weights.shape)
    #     grad[:, action] = self._transformState(state)
    #     grad = grad.reshape(-1)
    #
    #     lr = learning_rate
    #     ddf = discount*decay_factor
    #     lrddf = lr*ddf
    #
    #     e_t = ddf*e_t + (1 - lrddf *np.dot(e_t, grad))*grad
    #     temp = w
    #     w = w + lr*td_error*e_t + lr*np.dot(w - o_w, grad)*(e_t - grad)
    #     o_w = temp
    #
    #     self.weights = w.reshape(self.weights.shape)
    #     self.old_weights = o_w.reshape(self.old_weights.shape)
    #     self.eligibility_trace = e_t.reshape(self.eligibility_trace.shape)
