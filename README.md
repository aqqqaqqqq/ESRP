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

### Examples
Use
<pre> python -m omnigibson.examples.learning.rearrange_rllib </pre>
to try out our ESRP's training!

