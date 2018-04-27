# Reinforcement Learning

This is a collection of IPython notebooks explaining basics of reinforcement learning. I follow a great book by Sutton and Barto. A draft of the book can be downloaded [Here](http://incompleteideas.net/book/the-book-2nd.html).

The first notebook is a very condensed but comprehensible introduction to tabular solution methods, part one of the above book. There are plenty of examples and applications of these methods on GridWorld environment with GUI. To successfully run the notebook examples you will need to have installed NumPy, PyGame, and, of course, IPython.

The second notebook takes care of approximate solution methods, again with examples but this time applied to the CartPole environment from OpenAI gym. This second notebook corresponds to the second part of the above book, and builds on the theory developed in the first notebook. To run the examples without any problems, make sure that besides the OpenAI gym you also have installed NumPy, Scikit-learn, TensorFlow, and again IPython. Moreover, if you would like to render the environment inline in the IPython notebook (e.g. when you are running the notebook on a server) you will additionally need matplotlib and JSAnimation.

Algorithms, GridWorld environment, and other dependencies are provided in separate files of respective folders; hence, you need to copy these together with the IPython notebooks. In order to download all files needed to run the first two notebooks, simply execute the following python commands. The folders with files will be placed to your current directory.
```python
import os
from six.moves import urllib

URL = "https://raw.githubusercontent.com/JurajX/Reinforcement_Learning/master/"
PATH_FILES_DIC = {
    "Tabular_Solution_Methods/":
    ["GridWorld.py", "HelperFunctions_TSM.py", "RL_with_GridWorld.ipynb", "StateSpace.py"],
    "Approximate_Solution_Methods/":
    ["ActorCritic.py", "HelperFunctions_ASM.py", "REINFORCE.py", "RL_with_Neural_Nets.ipynb",
     "ValueFunction.py", "non_linear_ActorCritic.py", "non_linear_REINFORCE.py"]}

def fetch_data(url=URL, path_files_dic=PATH_FILES_DIC):
    """Collect data from GitHub"""
    for path, files in path_files_dic.items():
        dir_path = os.path.join(".", path)
        if not os.path.isdir(dir_path): os.makedirs(dir_path)
        for file in files:
            temp_url = url+path+file
            download_path = os.path.join(dir_path, file)
            urllib.request.urlretrieve(temp_url, download_path)

if __name__ == "__main__":
    fetch_data()
```

A small comment on formatting. For some reason GitHub does not display all equations from the notebooks in a proper TeX format, however, after copying and opening (using IPython) the notebooks on your local machine, there shouldn't be any issues with displaying the equations in proper formats.

More examples are about to come in a new notebook.
