import omnigibson.utils.transform_utils as T
import torch

def f(ori):
    x = torch.tensor([1.0, 0, 0])
    trans_x = T.quat_apply(ori, x)
    return torch.atan2(trans_x[2], trans_x[0])
robot_quat1 = torch.tensor([0,-0.7,-0.7,0])
robot_quat2 = torch.tensor([0,0.7,0.7,0])
print(f(robot_quat1),f(robot_quat2))
