# import re
# from collections import defaultdict

# # 日志文件路径
# log_path = 'C:/Users/Admin/Desktop/OmniGibson-Rearrange/evaluate_mobilev3_accel_train.txt'

# # 正则模式
# pattern = re.compile(
#     r'success:\s*\[(True|False)\].*?'
#     r'init_potential:\s*\[tensor\(([\d.]+)\)\].*?'
#     r'finish_potential:\s*\[tensor\(([\d.]+)\)\].*?'
#     r'all_objs:\s*\[(\d+)\].*?'
#     r'arrival_num:\s*\{(\d+)\}'
# )

# # 定义分组区间的函数
# def bucket(all_o):
#     if all_o == 1:
#         return 'all_o=1'
#     elif 2 <= all_o <= 3:
#         return 'all_o=2-3'
#     else:
#         return 'all_o>=4'

# # 用于统计的结构
# stats = {
#     'all_o=1':   {'total':0, 'succ':0, 'ratios_objs':[], 'ratios_pot':[]},
#     'all_o=2-3': {'total':0, 'succ':0, 'ratios_objs':[], 'ratios_pot':[]},
#     'all_o>=4':  {'total':0, 'succ':0, 'ratios_objs':[], 'ratios_pot':[]},
# }

# with open(log_path, 'r') as f:
#     for line in f:
#         m = pattern.search(line)
#         if not m:
#             continue

#         success = (m.group(1) == 'True')
#         init_p  = float(m.group(2))
#         fin_p   = float(m.group(3))
#         all_o   = int(m.group(4))
#         arr_n   = int(m.group(5))

#         grp = bucket(all_o)
#         stats[grp]['total'] += 1
#         if success:
#             stats[grp]['succ'] += 1
#         if all_o > 0:
#             stats[grp]['ratios_objs'].append(arr_n / all_o)
#         if init_p > 0:
#             stats[grp]['ratios_pot'].append(fin_p / init_p)

# # 计算并打印结果
# for grp, data in stats.items():
#     total = data['total']
#     succ  = data['succ']
#     p_objs = (sum(data['ratios_objs']) / len(data['ratios_objs'])) if data['ratios_objs'] else 0
#     p_pot  = (sum(data['ratios_pot'])  / len(data['ratios_pot']))  if data['ratios_pot'] else 0
#     succ_rate = (succ / total) if total else 0

#     print(f'=== {grp} ===')
#     print(f'场景数          : {total}')
#     print(f'success=True 数 : {succ}')
#     print(f'success 比例    : {succ_rate:.4f}')
#     print(f'平均 arrival/all: {p_objs:.4f}')
#     print(f'平均 finish/init: {p_pot:.4f}\n')
import re

# 日志文件路径
log_path = 'C:/Users/Admin/Desktop/OmniGibson-Rearrange/evaluate_mobilev3_16.txt'

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
        if not m:
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
