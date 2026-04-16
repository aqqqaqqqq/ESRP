import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import sys, os
from torchvision import transforms
from PIL import Image
import math


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

# 动作反向映射
label_reverse = {0:1, 1:0, 2:3, 3:2, 4:5, 5:4}

class LSTMDataset(Dataset):
    """
    root_dir/
      scene_01/
        data.npz    # structured ndarray with fields 'obs' (128,128,4) and 'action'
        scene.png   # 512×512×4, we drop channel 4
      scene_02/
        ...
    Splits any sequence longer than max_seq_len=32 into chunks.
    Returns for each chunk:
      img_seq_small: [L, 3,128,128]
      img_seq_large: [L, 3,512,512]
      act_seq:       [L]
    """
    def __init__(self, root_dir: str, max_seq_len: int = 64, stride: int = None):
        super().__init__()
        self.max_seq_len = max_seq_len
        # 如果stride为None，使用非重叠窗口（stride = max_seq_len）
        # 如果stride < max_seq_len，使用重叠窗口
        self.stride = stride if stride is not None else max_seq_len
        self.chunks = []  # each entry: dict with npz, png, start, length
        
        # transforms
        self.tf_small = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.tf_large = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        # scan scenes and build chunk index
        for scene_name in sorted(os.listdir(root_dir)):
            scene_dir = os.path.join(root_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            npzs = glob.glob(os.path.join(scene_dir, "*.npz"))
            pngs = glob.glob(os.path.join(scene_dir, "*.png"))
            if len(npzs) != 1 or len(pngs) != 1:
                raise RuntimeError(
                    f"Scene {scene_name} must contain exactly 1 npz and 1 png, "
                    f"found {len(npzs)} npz and {len(pngs)} png"
                )
            npz_path = npzs[0]
            png_path = pngs[0]

            # load only to get sequence length
            data = np.load(npz_path)["data"]
            seq_len = data.shape[0]
            # reversed entire sequence
            # we'll slice on reversed_data
            # 使用stride创建chunks（可能重叠）
            # 例如：max_seq_len=64, stride=32 会创建: [0-63], [32-95], [64-127], ...
            start = 0
            while start < seq_len:
                length = min(self.max_seq_len, seq_len - start)
                self.chunks.append({
                    "npz": npz_path,
                    "png": png_path,
                    "start": start,
                    "length": length
                })
                start += self.stride

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        info = self.chunks[idx]
        npz_path = info["npz"]
        png_path = info["png"]
        start = info["start"]
        length = info["length"]

        # load and reverse data
        data = np.load(npz_path)["data"]
        rev = data[::-1]  # reversed along time

        # load large image once
        img_large = Image.open(png_path).convert("RGB")
        img_large = self.tf_large(img_large)  # [3,512,512]

        imgs_s, imgs_l, acts, is_fetches = [], [], [], []
        # slice this chunk
        for rec in rev[start:start+length]:
            # small obs: drop 4th channel
            obs = rec["obs"][..., :3]
            img_small = self.tf_small(obs)  # [3,128,128]
            imgs_s.append(img_small)
            # same large per step
            imgs_l.append(img_large)
            # reversed action
            a = int(rec["action"])
            acts.append(label_reverse[a])
            is_fetch = int(rec["is_fetch"])
            is_fetches.append(is_fetch)

        # stack
        img_seq_small = torch.stack(imgs_s, dim=0)  # [L,3,128,128]
        img_seq_large = torch.stack(imgs_l, dim=0)  # [L,3,128,128]
        act_seq = torch.LongTensor(acts)            # [L]
        is_fetch_seq = torch.LongTensor(is_fetches) # [L]

        return img_seq_small, img_seq_large, act_seq, is_fetch_seq


if __name__ == "__main__":
    # simple test
    ds = LSTMDataset("imitation_data_val", max_seq_len=64)
    print("Total chunks:", len(ds))
    small, large, acts, is_fetches = ds[80]
    print("Chunk lengths:", small.shape, large.shape, acts.shape, is_fetches.shape)
    for i in range(acts.shape[0]):
        print(f"step {i}, is_fetch={is_fetches[i].item()}, action={acts[i].item()}")
        save_img(small[i], "test_obs", i)
        save_img(large[i], "test_goal", i)

