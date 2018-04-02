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
