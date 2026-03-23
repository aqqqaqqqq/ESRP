import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import sys, os
vendored = os.path.expanduser(
    "~/.local/share/ov/pkg/isaac-sim-4.1.0/exts/"
    "omni.isaac.ml_archive/pip_prebundle"
)
sys.path = [p for p in sys.path if not p.startswith(vendored)]
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

# class LSTMDataset(Dataset):
#     """
#     root_dir/
#       scene_01/
#         data.npz    # structured ndarray, rec['obs'] (128,128,4), rec['action']
#         scene.png   # 512x512x4, we drop第4通道
#       scene_02/
#         ...
#     返回：
#       img_seq_small: [T, 3,128,128]
#       img_seq_large: [T, 3,512,512]
#       act_seq:        [T]
#     """
#     def __init__(self, root_dir: str):
#         super().__init__()
#         self.scenes = []
#         for scene_name in sorted(os.listdir(root_dir)):
#             scene_dir = os.path.join(root_dir, scene_name)
#             if not os.path.isdir(scene_dir):
#                 continue
#             # 找 npz
#             npzs = glob.glob(os.path.join(scene_dir, "*.npz"))
#             # 找 png
#             pngs = glob.glob(os.path.join(scene_dir, "*.png"))
#             if len(npzs) != 1 or len(pngs) != 1:
#                 raise RuntimeError(f"Scene {scene_name} 下应该各有 1 个 npz 和 png，"
#                                    f"但发现 {len(npzs)} npz, {len(pngs)} png")
#             self.scenes.append({
#                 "npz": npzs[0],
#                 "png": pngs[0]
#             })

#         # 小图 pipeline：PIL→Resize→Tensor
#         self.tf_small = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((128,128)),
#             transforms.ToTensor(),        # float [0,1], CHW
#         ])
#         # 大图 pipeline：PIL→Resize→Tensor
#         self.tf_large = transforms.Compose([
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
#         ])

#     def __len__(self):
#         return len(self.scenes) # train:5922 valid:658

#     def __getitem__(self, idx):
#         info = self.scenes[idx]
#         data = np.load(info["npz"])["data"]  # structured ndarray

#         # 先加载一次大图
#         img_large = Image.open(info["png"]).convert("RGB")
#         img_large = self.tf_large(img_large)  # [3,512,512]

#         imgs_s, imgs_l, acts = [], [], []
#         # 反序遍历整个 trajectory
#         for rec in reversed(data):
#             # small obs
#             obs = rec["obs"][..., :3]            # (128,128,3)
#             img_small = self.tf_small(obs)       # [3,128,128]
#             imgs_s.append(img_small)
#             # large 同一张
#             imgs_l.append(img_large)
#             # action 反向
#             a = int(rec["action"])
#             acts.append(label_reverse[a])

#         # 拼成张量序列
#         img_seq_small = torch.stack(imgs_s, dim=0)  # [T,3,128,128]
#         img_seq_large = torch.stack(imgs_l, dim=0)  # [T,3,512,512]
#         act_seq       = torch.LongTensor(acts)      # [T]

#         return img_seq_small, img_seq_large, act_seq

# if __name__ == "__main__":
#     dataset = LSTMDataset('/home/pilab/Siqi/github/OmniGibson-Rearrange/imitation_data_train')
#     # import pdb; pdb.set_trace()
#     print("len:", len(dataset))
#     img, img2, label = dataset[0]
#     print("img:", img, "img2:", img2, "label:", label)

#     for i in range(dataset[5][2].shape[0]):
#         save_img(dataset[5][0][i], "test_obs", i)
#         save_img(dataset[5][1][i], "test_goal", i)

#         print(f"action{i}:", dataset[5][2][i])
#         # save_img(dataset[i][1], "model_test", f"{i}_tar")

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
    def __init__(self, root_dir: str, max_seq_len: int = 64):
        super().__init__()
        self.max_seq_len = max_seq_len
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
            n_chunks = math.ceil(seq_len / self.max_seq_len)
            for i in range(n_chunks):
                start = i * self.max_seq_len
                length = min(self.max_seq_len, seq_len - start)
                self.chunks.append({
                    "npz": npz_path,
                    "png": png_path,
                    "start": start,
                    "length": length
                })

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
    ds = LSTMDataset("imitation_data_val", max_seq_len=32)
    print("Total chunks:", len(ds))
    small, large, acts, is_fetches = ds[14]
    print("Chunk lengths:", small.shape, large.shape, acts.shape, is_fetches.shape)
    for i in range(acts.shape[0]):
        print(f"step {i}, is_fetch={is_fetches[i].item()}, action={acts[i].item()}")
        save_img(small[i], "test_obs", i)
        save_img(large[i], "test_goal", i)

