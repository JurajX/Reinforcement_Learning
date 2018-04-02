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
