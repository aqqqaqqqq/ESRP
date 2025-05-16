import os
import glob
import shutil

def build_subset_from_list(
    scene_list_file: str,
    npz_root: str,
    png_root: str,
    output_root: str
):
    """
    scene_list_file: 存放场景名（不含扩展名），每行一个
    npz_root:        源 .npz 场景文件目录
    png_root:        源 png 根目录，结构 png_root/scene_name/*.png
    output_root:     输出的子集根目录
    """
    os.makedirs(output_root, exist_ok=True)

    with open(scene_list_file, 'r') as f:
        scene_names = [line.strip() for line in f if line.strip()]

    for scene_name in scene_names:
        src_npz = os.path.join(npz_root, scene_name + '.npz')
        if not os.path.isfile(src_npz):
            print(f"⚠️ 未找到 .npz 文件: {src_npz}")
            continue

        dest_subdir = os.path.join(output_root, scene_name)
        os.makedirs(dest_subdir, exist_ok=True)

        # 拷贝 .npz
        shutil.copy(src_npz, dest_subdir)

        # 拷贝第一张 png
        png_dir = os.path.join(png_root, scene_name)
        if os.path.isdir(png_dir):
            png_files = glob.glob(os.path.join(png_dir, '*.png'))
            if png_files:
                first_png = sorted(png_files)[0]
                shutil.copy(first_png, dest_subdir)
            else:
                print(f"⚠️ 目录存在，但未找到 png: {png_dir}")
        else:
            print(f"⚠️ 未找到 png 源目录: {png_dir}")

if __name__ == '__main__':
    # ———— 配置区域 ————
    NPZ_ROOT      = 'imitation_data'     # 源 .npz 场景目录
    PNG_ROOT      = 'C:/Users/Admin/Desktop/OmniGibson-Rearrange/omnigibson/data/3d_front/scenes'  # 你的 png 根目录
    TRAIN_LIST    = 'train_data.txt'          # 训练集场景名列表
    VAL_LIST      = 'valid_data.txt'            # 验证集场景名列表
    TEST_LIST     = 'test_data.txt'           # 测试集场景名列表

    TRAIN_DIR     = 'imitation_data_train_'
    VAL_DIR       = 'imitation_data_val_'
    TEST_DIR      = 'imitation_data_test_'
    # ————————————————

    build_subset_from_list(TRAIN_LIST, NPZ_ROOT, PNG_ROOT, TRAIN_DIR)
    build_subset_from_list(VAL_LIST,   NPZ_ROOT, PNG_ROOT, VAL_DIR)
    build_subset_from_list(TEST_LIST,  NPZ_ROOT, PNG_ROOT, TEST_DIR)

    print("Done.")  
