import numpy as np

class StateSpace(object):
    """
    This is a basic class to be used as a state space for GridWorld.
    To construct an instance one must provide the following:
     -shape:    shape of the world.
     -actions:  set of possible actions in every state, default ['N', 'S', 'W', 'E'].
    """
    def __init__(self, shape, actions=['N', 'S', 'W', 'E']):
        self.actions = np.array(actions)
        self.stateVF = np.zeros(shape)
        self.actionVF = np.zeros( (shape[0], shape[1], len(actions)) )

    #----------------------

    def reset(self):
        self.stateVF = np.zeros(self.stateVF.shape)
        self.actionVF = np.zeros(self.actionVF.shape)

    #----------------------

    def sample(self, ret_probs=False):
        """
        Return a random action.
        If ret_probs=True return evenly split probabilities, default is False.
        """
        if ret_probs:
            probs = np.ones(self.actions.shape)
            probs = probs/np.sum(probs)
            return probs
        else:
            action = np.random.randint( len(self.actions) )
            return self.actions[action]

    #----------------------

    def choose_action(self, probs):
        """
        Return an action based on the given probabilities, probs.
        If length of probs is not equal to the number of allowed actions, raise TypeError.
        If sum of probs differes from 1.0 by more than 0.01, raise ValueError.
        """
        if not len(probs) == len(self.actions):
            raise TypeError('Length of probs must be equal to the number of allowed actions.')
        if np.sum(probs) > 1.01 or np.sum(probs) < 0.99:
            raise ValueError('Sum of probs differes from 1.0 by more than 0.01.')
        draw = np.random.multinomial(1, probs)
        action = np.argmax(draw)
        return self.actions[action]

    #----------------------

    def greedy_policy(self, coos, ret_probs=False):
        """
        Choose a greedy action with respect to the actionVF (ties broken evenly).
        If the actionVF was not updated (is None), choose randomly.
        If ret_probs=True return the probabilities of maximizing actions, default is False.
        """
        if np.sum(self.actionVF[coos]) == 0:
            return self.sample(ret_probs=ret_probs)
        else:
            # break ties evenly among all maximizing actions
            max_val = np.amax(self.actionVF[coos])
            probs = self.actionVF[coos] == max_val
            # normalize
            probs = probs/np.sum(probs)
            if ret_probs:
                return probs
            else:
                return self.choose_action(probs)

    #----------------------

    def eps_greedy_policy(self, coos, eps=0.01, ret_probs=False):
        """
        Choose an epsilon greedy action with respect to the actionVF.
        If the actionVF was not updated (is None), choose randomly.
        If ret_probs=True return the probabilities of respective actions, default is False.
        """
        if not eps <= 1.0:
            raise ValueError('Probability eps must be less or equal to 1.')

        max_val = np.amax(self.actionVF[coos])
        probs = self.actionVF[coos] == max_val
        probs = probs/np.sum(probs)
        probs = probs*(1-eps) + (eps/len(probs))

        if ret_probs:
            return probs
        else:
            return self.choose_action(probs)

    #----------------------

    def softmax_policy(self, coos, ret_probs=False):
        """
        Choose a soft-max action with respect to the actionVF.
        If the actionVF was not updated (is None), choose randomly.
        If ret_probs=True return the probabilities of respective actions, default is False.
        """
        if np.sum(self.actionVF[coos]) == 0:
            return self.sample(ret_probs=ret_probs)
        else:
            e_aVF = np.exp(self.actionVF[coos])
            probs = e_aVF/np.sum(e_aVF)
            if ret_probs:
                return probs
            else:
                return self.choose_action(probs)

    #----------------------

    def update_stateVF(self, coos, value):
        """
        Update the value of stateVF (float).
        """
        self.stateVF[coos] = value

    #----------------------

    def update_actionVF(self, value, coos=None, action=None):
        """
        Update the value of actionVF for a given both state and action, or only action.
        In case only value is provided the entire actionVF is updated.
        If the given action is not among allowed actions, raise ValueError.
        If only state is given, value must have dimension of allowed actions.
        If only value is provided, it must have dimension of actionVF.

        """
        if coos is not None:
            if action is not None:
                if not action in self.actions:
                    raise ValueError('Given action is not among alowed actions.')
                self.actionVF[coos][self.actions == action] = value
            else:
                if value.shape != self.actionVF[coos].shape:
                    raise ValueError('When providing coos and value only, \
                    value must have the same dimension as the space of actions.')
                self.actionVF[coos] = value
        else:
            if value.shape != self.actionVF.shape:
                raise ValueError('When providing value only, \
                it must have the same dimention as action-value function.')
            self.actionVF = value

    #----------------------

    def get_actionVF(self):
        """
        Return the current actionVF.
        """
        return self.actionVF

    #----------------------

    def get_stateVF(self):
        """
        Return the current stateVF.
        """
        return self.stateVF

    #----------------------

    def get_actions(self):
        """
        Return list of allowed actions.
        """
        return self.actions
