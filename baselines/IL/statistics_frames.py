#!/usr/bin/env python3
"""
统计训练数据文件夹中所有npz文件的帧数
计算平均值和分段计数
"""
import os
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

def count_frames_in_npz(npz_path):
    """读取npz文件，返回帧数"""
    try:
        data = np.load(npz_path)
        if 'data' in data:
            frames = data['data'].shape[0]
            return frames
        else:
            print(f"⚠️  {npz_path} 中没有找到 'data' 字段")
            return None
    except Exception as e:
        print(f"⚠️  读取 {npz_path} 时出错: {e}")
        return None

def main():
    train_dir = '/home/user/Desktop/il/OmniGibson-Rearrange/imitation_data_train'
    
    print("=" * 80)
    print("统计训练数据帧数")
    print("=" * 80)
    print(f"数据目录: {train_dir}\n")
    
    if not os.path.exists(train_dir):
        print(f"❌ 目录不存在: {train_dir}")
        return
    
    # 统计信息
    frame_counts = []
    scene_frame_map = {}  # 场景名 -> 帧数
    
    # 分段计数
    segment_counts = {
        '0-32': 0,
        '32-64': 0,
        '64-128': 0,
        '>128': 0
    }
    
    # 遍历所有子文件夹
    subdirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    total_scenes = len(subdirs)
    
    print(f"找到 {total_scenes} 个子文件夹\n")
    print("正在统计...")
    
    for idx, scene_name in enumerate(subdirs, 1):
        scene_dir = os.path.join(train_dir, scene_name)
        npz_files = glob.glob(os.path.join(scene_dir, '*.npz'))
        
        if len(npz_files) == 0:
            print(f"⚠️  {scene_name}: 未找到npz文件")
            continue
        
        if len(npz_files) > 1:
            print(f"⚠️  {scene_name}: 找到多个npz文件，使用第一个")
        
        npz_path = npz_files[0]
        frames = count_frames_in_npz(npz_path)
        
        if frames is not None:
            frame_counts.append(frames)
            scene_frame_map[scene_name] = frames
            
            # 分段统计
            if frames <= 32:
                segment_counts['0-32'] += 1
            elif frames <= 64:
                segment_counts['32-64'] += 1
            elif frames <= 128:
                segment_counts['64-128'] += 1
            else:
                segment_counts['>128'] += 1
            
            if idx % 100 == 0:
                print(f"  已处理: {idx}/{total_scenes}")
    
    # 计算结果
    if len(frame_counts) == 0:
        print("\n❌ 没有找到有效的npz文件")
        return
    
    frame_counts = np.array(frame_counts)
    mean_frames = np.mean(frame_counts)
    median_frames = np.median(frame_counts)
    min_frames = np.min(frame_counts)
    max_frames = np.max(frame_counts)
    std_frames = np.std(frame_counts)
    
    # 输出结果
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    print(f"总场景数: {len(frame_counts)}")
    print(f"\n帧数统计:")
    print(f"  平均值: {mean_frames:.2f}")
    print(f"  中位数: {median_frames:.2f}")
    print(f"  最小值: {min_frames}")
    print(f"  最大值: {max_frames}")
    print(f"  标准差: {std_frames:.2f}")
    
    print(f"\n分段统计:")
    print(f"  0-32帧:   {segment_counts['0-32']:5d} ({segment_counts['0-32']/len(frame_counts)*100:.1f}%)")
    print(f"  32-64帧:  {segment_counts['32-64']:5d} ({segment_counts['32-64']/len(frame_counts)*100:.1f}%)")
    print(f"  64-128帧: {segment_counts['64-128']:5d} ({segment_counts['64-128']/len(frame_counts)*100:.1f}%)")
    print(f"  >128帧:   {segment_counts['>128']:5d} ({segment_counts['>128']/len(frame_counts)*100:.1f}%)")
    
    # 显示一些示例
    print(f"\n示例（前10个场景）:")
    for i, (scene, frames) in enumerate(list(scene_frame_map.items())[:10], 1):
        print(f"  {i:2d}. {scene}: {frames}帧")
    
    print("=" * 80)

if __name__ == '__main__':
    main()

