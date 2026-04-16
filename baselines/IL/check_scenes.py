#!/usr/bin/env python3
"""
检查 imitation_data_5 中的 npz 文件名是否包含 filtered_dataset_split 中三个 txt 文件中的场景名字
"""
import os
from pathlib import Path
from collections import defaultdict

def load_scene_names(txt_file):
    """从 txt 文件中加载场景名（每行一个）"""
    scene_names = set()
    if not os.path.exists(txt_file):
        print(f"⚠️  文件不存在: {txt_file}")
        return scene_names
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            scene_name = line.strip()
            if scene_name:
                scene_names.add(scene_name)
    
    return scene_names

def check_npz_files(npz_dir, scene_names_set):
    """检查 npz 目录中的文件是否包含场景名"""
    npz_dir = Path(npz_dir)
    if not npz_dir.exists():
        print(f"⚠️  目录不存在: {npz_dir}")
        return
    
    all_scene_names = scene_names_set
    matched_files = []
    unmatched_files = []
    
    # 遍历所有 npz 文件
    for npz_file in npz_dir.glob('*.npz'):
        # 获取文件名（不含扩展名）
        filename_without_ext = npz_file.stem
        
        # 检查是否在场景名集合中
        if filename_without_ext in all_scene_names:
            matched_files.append(npz_file.name)
        else:
            unmatched_files.append(npz_file.name)
    
    return matched_files, unmatched_files

def main():
    # 配置路径
    filtered_dataset_dir = '/home/user/Desktop/rearrange/OmniGibson-Rearrange/filtered_dataset_split'
    npz_dir = '/home/user/Desktop/il/OmniGibson-Rearrange/imitation_data_5'
    
    txt_files = {
        'train': os.path.join(filtered_dataset_dir, 'train_data.txt'),
        'valid': os.path.join(filtered_dataset_dir, 'valid_data.txt'),
        'test': os.path.join(filtered_dataset_dir, 'test_data.txt'),
    }
    
    print("=" * 80)
    print("开始检查 npz 文件名是否包含场景名...")
    print("=" * 80)
    
    # 加载所有场景名
    all_scene_names = set()
    scene_count_by_file = {}
    
    for split_name, txt_file in txt_files.items():
        scene_names = load_scene_names(txt_file)
        scene_count_by_file[split_name] = len(scene_names)
        all_scene_names.update(scene_names)
        print(f"\n{split_name.upper()} 文件: {txt_file}")
        print(f"  场景数量: {len(scene_names)}")
    
    print(f"\n总场景数量（去重后）: {len(all_scene_names)}")
    
    # 检查 npz 文件
    print(f"\n检查 npz 目录: {npz_dir}")
    matched_files, unmatched_files = check_npz_files(npz_dir, all_scene_names)
    
    # 统计结果
    total_npz = len(matched_files) + len(unmatched_files)
    print(f"\n总 npz 文件数: {total_npz}")
    print(f"匹配的文件数: {len(matched_files)}")
    print(f"未匹配的文件数: {len(unmatched_files)}")
    print(f"匹配率: {len(matched_files)/total_npz*100:.2f}%" if total_npz > 0 else "N/A")
    
    # 显示详细信息
    print("\n" + "=" * 80)
    print("匹配的文件（前20个）:")
    print("=" * 80)
    for i, filename in enumerate(matched_files[:20], 1):
        print(f"  {i}. {filename}")
    if len(matched_files) > 20:
        print(f"  ... 还有 {len(matched_files) - 20} 个匹配的文件")
    
    if unmatched_files:
        print("\n" + "=" * 80)
        print("未匹配的文件（前20个）:")
        print("=" * 80)
        for i, filename in enumerate(unmatched_files[:20], 1):
            print(f"  {i}. {filename}")
        if len(unmatched_files) > 20:
            print(f"  ... 还有 {len(unmatched_files) - 20} 个未匹配的文件")
    
    print("\n" + "=" * 80)
    print("检查完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()

