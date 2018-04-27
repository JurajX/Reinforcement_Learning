import numpy as np

#-----  Printing scores

def printTrainingReturns(returns):
    """
    Args: list/array of returns.
    Print average returns obtained during training in four percentage groups: 0-10, 10-50, 50-90, 90-100.
    """
    length = len(returns)
    print("Performance measure: the return per episode averaged over: \n",
          "- the first 10% of the training was {0:.1f} with the standard deviation {1:.1f};\n"
          .format(np.mean(returns[:int(length*0.1)]),
                  np.sqrt(np.var(returns[:int(length*0.1)]))),
          "- 10% to 50% of the training was {0:.1f} with the standard deviation {1:.1f};\n"
          .format(np.mean(returns[int(length*0.1): int(length*0.5)]),
                  np.sqrt(np.var(returns[int(length*0.1): int(length*0.5)]))),
          "- 50% to 90% of the training was {0:.1f} with the standard deviation {1:.1f};\n"
          .format(np.mean(returns[int(length*0.5): int(length*0.9)]),
                  np.sqrt(np.var(returns[int(length*0.5): int(length*0.9)]))),
          "- the last 10% of the training was {0:.1f} with the standard deviation {1:.1f}."
          .format(np.mean(returns[int(length*0.9):]),
                  np.sqrt(np.var(returns[int(length*0.9):]))) )

def printReturns(returns):
    """
    Args: list/array of returns.
    Print average return and the standard deviation.
    """
    print("Performance measure: the return per episode averaged over {0} episodes was {1:.1f} \
with the standard deviation {2:.1f}.".format(len(returns), np.mean(returns), np.sqrt(np.var(returns))))

def printHighestLowest(returns):
    """
    Args: list/array of returns.
    Print the highest and the lowest return.
    """
    print("The highest return was {0}, and the lowest return was {1}."
          .format(np.max(returns), np.min(returns)))

#-----  Getting performance

def getPerformanceVF(env, actionVF, epsilon, episodes):
    """
    Args: gym environment, action-value function, epsilon parameter, number of episodes.
    Prints performance of passed value function.
    """
    returns = np.zeros(episodes)
    for i in range(episodes):
        state = env.reset()
        done = False
        cum_reward = 0
        while not done:
            action = actionVF.epsGreedyPolicy(state, epsilon)
            state, reward, done, _ = env.step(action)
            cum_reward = cum_reward + reward
        returns[i] = cum_reward
    printReturns(returns)
    printHighestLowest(returns)

def getPerformanceAgent(env, agent, episodes):
    """
    Args: gym environment, agent, number of episodes.
    Prints performance of the passed agent, the agent must have chooseAction member function.
    """
    returns = np.zeros(episodes)
    for i in range(episodes):
        state = env.reset()
        done = False
        cum_reward = 0
        while not done:
            action = agent.chooseAction(state[np.newaxis, :])
            state, reward, done, _ = env.step(action)
            cum_reward = cum_reward + reward
        returns[i] = cum_reward
    printReturns(returns)
    printHighestLowest(returns)

#-----  Visualising steps

def visualizeVF(env, actionVF, epsilon):
    """
    Args: gym environment, action-value function, epsilon parameter.
    Renders the steps taken by passed value function.
    """
    state = env.reset()
    done = False
    cum_reward = 0
    while not done:
        action = actionVF.epsGreedyPolicy(state, epsilon)
        state, reward, done, _ = env.step(action)
        cum_reward = cum_reward + reward
        env.render()
    print('Return of this visualization was {0}.'.format(cum_reward))

def visualizeAgent(env, agent):
    """
    Args: gym environment, action-value function.
    Renders the steps taken by the passed agent, the agent must have chooseAction member function.
    """
    state = env.reset()
    done = False
    cum_reward = 0
    while not done:
        action = agent.chooseAction(state[np.newaxis, :])
        state, reward, done, _ = env.step(action)
        cum_reward = cum_reward + reward
        env.render()
    print('Return of this visualization was {0}.'.format(cum_reward))

#-----  Visualising steps in iPython notebook

import matplotlib.pyplot as plt
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(display_animation(anim, default_mode='loop'))

def visualizeVF_inline(env, actionVF, epsilon):
    """
    Args: gym environment, action-value function, epsilon parameter.
    Renders the steps taken by passed value function.
    """
    frames = []
    state = env.reset()
    done = False
    cum_reward = 0
    while not done:
        action = actionVF.epsGreedyPolicy(state, epsilon)
        state, reward, done, _ = env.step(action)
        cum_reward = cum_reward + reward
        frames.append(env.render(mode='rgb_array'))
    print('Return of this visualization was {0}.'.format(cum_reward))
    display_frames_as_gif(frames)

def visualizeAgent_inline(env, agent):
    """
    Args: gym environment, action-value function.
    Renders the steps taken by the passed agent, the agent must have chooseAction member function.
    """
    frames = []
    state = env.reset()
    done = False
    cum_reward = 0
    while not done:
        action = agent.chooseAction(state[np.newaxis, :])
        state, reward, done, _ = env.step(action)
        cum_reward = cum_reward + reward
        frames.append(env.render(mode='rgb_array'))
    print('Return of this visualization was {0}.'.format(cum_reward))
    display_frames_as_gif(frames)
