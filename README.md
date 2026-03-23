### Installation
We only support Linux now (Ubuntu 20.04 / 22.04) with NVIDIA GPU (see the NVIDIA Isaac Sim requirements)
Check out [**`OmniGibson`**'s documentation](https://behavior.stanford.edu/omnigibson/getting_started/installation.html) to get started!
To install all the requirements, you can use pip install -r requirement.txt

### Dataset
Get the dataset from [Dataset](https://huggingface.co/datasets/serendipity800/ESRP-PD) and install it according to its requirements.
For Imitation Learning (IL) baseline, you only need to use imitation.zip
For the active Gym-compatible environment, you need the whold dataset including all scenes an meshes.

### Dataset Configuration
<pre>
rearrange_robot.py
eval_no_lstm.py
eval_model.py
new_env.py

perprocess_dataset.py
parse_json.py
split_train_valid_test_dataset.py
</pre>
These files use absoulte path, please modify them before using.

### Dataset Preprocessing
#### Mesh Textures
Before you start to run any script, run ```python -m omnigibson.create_mesh``` to create the materials of meshes.
#### Before IL
Run ```python -m omnigibson.baseline.IL.split_scenes``` to split dataset into train, valid and test.

### Examples
#### RL
- Train

  Use ```python -m omnigibson.examples.learning.rearrange_rllib ``` to try out PPO training, the trained model will be saved in the **save_models** folder at the same level as **omnigibson**.
- Test
  
  You can test the model by modifying the model and test data paths in **eval\_model.py**. ``` python -m omnigibson.examples.learning.eval_model ```. We also provide a test data processing script, **analyse\_data.py**, in **omnigibson.examples.learning**.
#### IL
- Train
  
  Use ```python -m omnigibson.baseline.IL.train_lstm ``` to train a model with LSTM, while use ``` python -m omnigibson.baseline.IL.train ``` to train a model without it. The trained model will be saved in the **new\_checkpoint** or **checkpoint** folder under the folder **IL**.
- Test
  
  Use ```python -m omnigibson.baseline.IL.evaluate ``` to test a model with LSTM, while use ``` python -m omnigibson.baseline.IL.evaluate_no_lstm ``` to test a model without it. The result will be saved in the customized path given in the above scripts.We also provide test data processing script **metrics.py** in **omnigibson.baseline.IL**

