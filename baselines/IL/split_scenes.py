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

        # 文件夹名加 _1
        scene_name_with_suffix = scene_name + '_2'
        dest_subdir = os.path.join(output_root, scene_name_with_suffix)
        os.makedirs(dest_subdir, exist_ok=True)

        # 拷贝 .npz 并重命名为 scene_name_1.npz
        dest_npz = os.path.join(dest_subdir, scene_name_with_suffix + '.npz')
        shutil.copy(src_npz, dest_npz)

        # 拷贝第一张 png 并重命名
        png_dir = os.path.join(png_root, scene_name)
        if os.path.isdir(png_dir):
            png_files = glob.glob(os.path.join(png_dir, '*.png'))
            if png_files:
                first_png = sorted(png_files)[0]
                # 获取原文件名（不含扩展名）并加 _1
                png_basename = os.path.splitext(os.path.basename(first_png))[0]
                dest_png = os.path.join(dest_subdir, png_basename.replace('_target', '_2_target') + '.png')
                shutil.copy(first_png, dest_png)
            else:
                print(f"⚠️ 目录存在，但未找到 png: {png_dir}")
        else:
            print(f"⚠️ 未找到 png 源目录: {png_dir}")

if __name__ == '__main__':
    # ———— 配置区域 ————
    NPZ_ROOT      = '/home/user/Desktop/il/OmniGibson-Rearrange/imitation_data_5_2'  # 源 .npz 场景目录
    PNG_ROOT      = '/home/user/Desktop/rearrange/OmniGibson-Rearrange/omnigibson/data/3d_front/scenes'  # png 根目录
    TRAIN_LIST    = '/home/user/Desktop/rearrange/OmniGibson-Rearrange/filtered_dataset_split/train_data.txt'  # 训练集场景名列表
    VAL_LIST      = '/home/user/Desktop/rearrange/OmniGibson-Rearrange/filtered_dataset_split/valid_data.txt'  # 验证集场景名列表
    TEST_LIST     = '/home/user/Desktop/rearrange/OmniGibson-Rearrange/filtered_dataset_split/test_data.txt'  # 测试集场景名列表

    TRAIN_DIR     = '/home/user/Desktop/il/OmniGibson-Rearrange/imitation_data_train'  # 训练集输出目录
    VAL_DIR       = '/home/user/Desktop/il/OmniGibson-Rearrange/imitation_data_val'    # 验证集输出目录（包含 valid 和 test）
    # ————————————————

    print("开始处理训练集...")
    build_subset_from_list(TRAIN_LIST, NPZ_ROOT, PNG_ROOT, TRAIN_DIR)
    
    print("\n开始处理验证集（valid_data.txt）...")
    build_subset_from_list(VAL_LIST, NPZ_ROOT, PNG_ROOT, VAL_DIR)
    
    print("\n开始处理测试集（test_data.txt，合并到验证集）...")
    build_subset_from_list(TEST_LIST, NPZ_ROOT, PNG_ROOT, VAL_DIR)

    print("\nDone.")  
