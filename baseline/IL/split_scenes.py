import os
import glob
import random
import shutil

def split_scenes(root_dir, png_root_dir, train_dir, val_dir, val_ratio=0.1, seed=42):
    """
    root_dir:     放 .npz 场景文件的目录
    png_root_dir: 放场景对应 png 的根目录，结构 scene_name/*.png
    train_dir:    输出的训练集根目录
    val_dir:      输出的验证集根目录
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    # 找到所有场景 .npz
    scenes = glob.glob(os.path.join(root_dir, '*.npz'))
    scenes.sort()
    random.seed(seed)
    random.shuffle(scenes)

    n_val = max(1, int(len(scenes) * val_ratio))
    val_scenes   = scenes[:n_val]
    train_scenes = scenes[n_val:]

    def _copy_subset(scene_list, target_root):
        for npz_path in scene_list:
            scene_name = os.path.splitext(os.path.basename(npz_path))[0]
            dest_subdir = os.path.join(target_root, scene_name)
            os.makedirs(dest_subdir, exist_ok=True)

            # 拷贝 .npz
            shutil.copy(npz_path, dest_subdir)

            # 从 png_root_dir/scene_name 下拷贝第一张 png
            png_dir = os.path.join(png_root_dir, scene_name)
            if os.path.isdir(png_dir):
                png_files = glob.glob(os.path.join(png_dir, '*.png'))
                if png_files:
                    src_png = png_files[0]  # 取第一张
                    shutil.copy(src_png, dest_subdir)
                else:
                    print(f"⚠️ 目录存在，但未找到 png: {png_dir}")
            else:
                print(f"⚠️ 未找到 png 源目录: {png_dir}")

    _copy_subset(train_scenes, train_dir)
    _copy_subset(val_scenes,   val_dir)

    print(f"总场景数: {len(scenes)}, 训练场景: {len(train_scenes)}, 验证场景: {len(val_scenes)}")

def write2txt(train_dir, val_dir):
    for scene_name in os.listdir(train_dir):
        with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/train_data.txt", 'a') as f:
            f.write(f"{scene_name}" + '\n')
    for scene_name in os.listdir(val_dir):
        with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/test_data.txt", 'a') as f:
            f.write(f"{scene_name}" + '\n')

if __name__ == '__main__':
    ROOT        = 'imitation_data'           # 源 .npz 场景目录
    PNG_ROOT    = 'C:/Users/Admin/Desktop/OmniGibson-Rearrange/omnigibson/data/3d_front/scenes'           # 放各场景子文件夹的根目录
    TRAIN_DIR   = 'imitation_data_train'     # 拆分后训练集根目录
    VAL_DIR     = 'imitation_data_val'       # 拆分后验证集根目录

    split_scenes(
        root_dir     = ROOT,
        png_root_dir = PNG_ROOT,
        train_dir    = TRAIN_DIR,
        val_dir      = VAL_DIR,
        val_ratio    = 0.1,
        seed         = 42
    )
    write2txt(TRAIN_DIR, VAL_DIR)
