import re

# 日志文件路径
log_path = './omnigibson/baseline/IL/experiments/results/mobilev3_overlap_128/test.txt'

# 正则模式，提取 success、init_potential、finish_potential、all_objs、arrival_num
pattern = re.compile(
    r'success:\s*\[(True|False)\].*?'
    r'init_potential:\s*\[tensor\(([\d.]+)\)\].*?'
    r'finish_potential:\s*\[tensor\(([\d.]+)\)\].*?'
    r'all_objs:\s*\[(\d+)\].*?'
    r'arrival_num:\s*\{(\d+)\}'
)

total = 0
success_count = 0
ratio_objs = []
ratio_pot = []

with open(log_path, 'r') as f:
    for line in f:
        m = pattern.search(line)
        # import pdb;pdb.set_trace()
        if not m:
            print(line)
            continue
        total += 1
        success = m.group(1) == 'True'
        init_p = float(m.group(2))
        fin_p  = float(m.group(3))
        all_o  = int(m.group(4))
        arr_n  = int(m.group(5))

        if success:
            success_count += 1
        # 防止除零
        if all_o > 0:
            ratio_objs.append(arr_n / all_o)
        if init_p > 0:
            ratio_pot.append(fin_p / init_p)

# 计算结果
succ_prop = success_count / total if total else 0
mean_arrival_ratio = sum(ratio_objs) / len(ratio_objs) if ratio_objs else 0
mean_pot_ratio    = sum(ratio_pot)  / len(ratio_pot)  if ratio_pot    else 0

print(f'总场景数: {total}')
print(f'success=True 的场景数: {success_count}')
print(f'→ success=True 比例: {succ_prop:.4f}')
print(f'平均 arrival_num/all_objs: {mean_arrival_ratio:.4f}')
print(f'平均 finish_potential/init_potential: {mean_pot_ratio:.4f}')

# val_scene = []
# with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/train_sampled_new_new.txt", 'r') as f:
#     for line in f:

#         parts = line.strip('\n')
#         val_scene.append(parts)
        
# with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/evaluate_mobilev3_64.txt", 'r') as f:
#     for line in f:
#         parts = line.split(':', 1)
#         if len(parts[0]) > 0:
#             scene = parts[0].replace("_target.json", "")
        
#             if scene in val_scene:
#                 with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/simple_val.txt", 'a') as f1:
#                     f1.write(line)
#             else:
#                 with open("C:/Users/Admin/Desktop/OmniGibson-Rearrange/simple_test.txt", 'a') as f2:
#                     f2.write(line)