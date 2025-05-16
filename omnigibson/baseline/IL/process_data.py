# import os
# import numpy as np
# import torch
# from PIL import Image


# def save_img(img_tensor, scene_name, filename):

    
#     img_pil = Image.fromarray(img_tensor, mode='RGBA')
#     save_dir = os.path.join(os.getcwd(), scene_name)
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"{filename}.png")
#     img_pil.save(save_path)

# def load_scene_data_paired(scene_name, out_dir='.'):
#     out_dir = os.path.join(os.getcwd(), "imitation_data")
#     fn = os.path.join(out_dir, f'{scene_name}.npz')
#     data = np.load(fn)['data']  # data 是一个 structured ndarray
#     return data

# scene_name = "aed687f8-271d-4310-870a-f97ebafd1bec_MasterBedroom-166756"
# paired = load_scene_data_paired(f"{scene_name}")

# for idx, rec in enumerate(paired):
#     # rec 是一个 numpy.void，可通过 rec['action'], rec['img'] 访问
#     print(f"  frame {idx}: is_fetch={rec['is_fetch']}, action={rec['action']}, img.shape={rec['obs'].shape}")
#     save_img(rec['obs'], scene_name, idx)
import os
import numpy as np

def add_is_fetch_field(data: np.ndarray) -> np.ndarray:
    """
    在 structured ndarray `data` 中加入一个新字段 'is_fetch' (int8)，
    并根据 action==4/5 之间的区间标记为 1，其余标记为 0。
    """
    # 1. 构造新 dtype：在原有字段列表后面加上 ('is_fetch', 'i1')
    old_descr = data.dtype.descr
    new_descr = old_descr + [('is_fetch', 'i1')]
    new_dtype = np.dtype(new_descr)

    # 2. 创建一个空 array 并拷贝原始数据
    new_data = np.zeros(data.shape, dtype=new_dtype)
    for name in data.dtype.names:
        new_data[name] = data[name]

    # 3. 按顺序遍历，维护一个“抓取状态”开关
    fetching = False
    for i, action in enumerate(new_data['action']):
        if action == 4:
            fetching = True
            new_data['is_fetch'][i] = 1
        elif action == 5:
            fetching = False
            new_data['is_fetch'][i] = 0
        else:
            new_data['is_fetch'][i] = 1 if fetching else 0

    return new_data

def update_npz_with_is_fetch(npz_path: str, out_path: str = None):
    """
    读取 npz，给 data 增加 is_fetch 字段，然后重新保存。
    如果 out_path 为空，则覆盖原文件；否则保存到 out_path。
    """
    arr = np.load(npz_path)
    data = arr['data']
    arr.close()

    new_data = add_is_fetch_field(data)

    if out_path is None:
        out_path = npz_path

    # 用相同 key ('data') 保存
    np.savez_compressed(out_path, data=new_data)
    print(f"Saved updated npz to: {out_path}")

if __name__ == "__main__":
    train_dir = 'C:/Users/Admin/Desktop/OmniGibson-Rearrange/imitation_data_train'
    val_dir = 'C:/Users/Admin/Desktop/OmniGibson-Rearrange/imitation_data_val'
    base_dir = val_dir
    for scene_name in os.listdir(base_dir):
        # scene_name = "aed687f8-271d-4310-870a-f97ebafd1bec_MasterBedroom-166756"
        # base_dir = os.path.join(os.getcwd(), "imitation_data")
        npz_file = os.path.join(base_dir, scene_name, f"{scene_name}.npz")

        # 如果想要备份原文件，可以传入第二个参数
        # backup = os.path.join(base_dir, f"{scene_name}_bak.npz")
        # update_npz_with_is_fetch(npz_file, out_path=backup)

        # 直接修改原文件：
        update_npz_with_is_fetch(npz_file)
