import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import sys, os

# 注意根据实际安装路径调整下面这一行
vendored = os.path.expanduser(
    "~/.local/share/ov/pkg/isaac-sim-4.1.0/exts/"
    "omni.isaac.ml_archive/pip_prebundle"
)
# 过滤掉 vendored 中的 torchvision 路径
sys.path = [p for p in sys.path if not p.startswith(vendored)]
from torchvision import transforms
import os
import numpy as np
import torch
from PIL import Image


def save_img(img_tensor, scene_name, filename):

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    # 2. 如果是浮点型 [0,1]，转换到 [0,255] 的 uint8
    img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)

    # 3. 用 Pillow 构造 RGB 图像并保存
    img_pil = Image.fromarray(img_uint8, mode='RGB')
    save_dir = os.path.join(os.getcwd(), scene_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}.png")
    img_pil.save(save_path)


def load_scene_data_paired(npz_path: str):
    """
    读取单个 .npz 文件中的 structured ndarray，返回 ndarray。
    """
    data = np.load(npz_path)['data']  # data 是一个 structured ndarray
    return data

class SceneDataset(Dataset):
    
    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.scene_dirs = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)
                                  if os.path.isdir(os.path.join(root_dir, d))])

        

        self.records = []
        for scene_dir in self.scene_dirs:
            npz_files = glob.glob(os.path.join(scene_dir, "*.npz"))
            for npz_path in npz_files:
                data = np.load(npz_path)['data']
                for record in data:
                    self.records.append({
                        "record": record,
                        "scene_dir": scene_dir
                    })

        self.transform_small = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
        ])
        self.transform_large = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),  # EfficientNet expects at least 224x224
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]["record"]
        scene_dir = self.records[idx]["scene_dir"]
        label = int(rec['action'])

        # Reverse the label as in your original code
        label_reverse = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
        label = label_reverse[label]

        # Load small and large images
        small_img = rec['obs'][..., :3]  # shape (128, 128, 3)
        small_img = self.transform_small(small_img)
        # import pdb; pdb.set_trace()
        scene_name = scene_dir.split('/')[-1]
        large_img_path = os.path.join(scene_dir, f"{scene_name}_target.png")  # adjust filename as needed
        large_img = Image.open(large_img_path).convert("RGB")
        large_img = self.transform_large(large_img)

        return small_img, large_img, label # 3*128*128  3*224*224
    
if __name__ == "__main__":

    dataset = SceneDataset('/home/pilab/Siqi/github/OmniGibson-Rearrange/imitation_data_train')
    # import pdb; pdb.set_trace()
    img, img2, label = dataset[0]
    print("img:", img, "img2:", img2, "label:", label)

    for i in range(50):
        save_img(dataset[i][0], "model_test", i)
        print(f"label{i}:", dataset[i][2])
        # save_img(dataset[i][1], "model_test", f"{i}_tar")

