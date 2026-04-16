# from omnigibson.macros import gm
# import os
# from collections import defaultdict
# import json
# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


# # threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
# # scenes_dir_path = os.path.join(threed_front_path, "scenes")
# # count_to_files = defaultdict(list)
# # total = 0
# # for entry in os.listdir(scenes_dir_path):
# #     print(f"deal: {total+1}/5549")
# #     total += 1
# #     entry_path = os.path.join(scenes_dir_path, entry)
# #     if os.path.isdir(entry_path):
# #         name = entry.split("_")[1]
# #         type = name.split("-")[0]
# #         # import pdb;pdb.set_trace()
# #         count_to_files[type].append(entry)

# # with open('C:/Users/Admin/Desktop/3D-FUTURE-model/model_info.json', 'r', encoding='utf-8') as f:
# #     cat_data = json.load(f)
# # total = 0
# # for entry in os.listdir(scenes_dir_path):
# #         print(f"deal: {total+1}/5549")
# #         total += 1
# #         entry_path = os.path.join(scenes_dir_path, entry)
# #         if os.path.isdir(entry_path):
# #             for file in os.listdir(entry_path):
# #                 if "_initial.json" in file:
# #                     filepath = os.path.join(entry_path, file)
# #                     with open(filepath, 'r', encoding='utf-8') as f:
# #                         data = json.load(f)

                    
# #                     for obj_name, obj_info in data["objects_info"]["init_info"].items():
# #                         if "furniture" in obj_name:
# #                             if obj_info["args"]["is_to_rearrange"] is True:
# #                                 name = obj_info["args"]["usd_path"].split('/')[2]
# #                                 jid = name.split('_')[0]
# #                                 # import pdb;pdb.set_trace()
# #                                 for item in cat_data:
# #                                     if item.get("model_id") == jid:

# #                                         category = item.get("super-category", None)
# #                                         if category == "Bed":
# #                                             category = item.get("category", None)
# #                                         break
                                
# #                                 count_to_files[category].append(jid)


# # 排序：按照每个 type 对应的文件数，从多到少
# # sorted_items = sorted(count_to_files.items(), key=lambda item: len(item[1]), reverse=True)

# # total = 0
# # cnt = 0
# # for type, files in sorted_items:
# #     cnt += 1
# #     print(f"\n=== type === {type} ：共 {len(files)} 个文件 ===")
# #     # for fn in files:
# #     #     print(f"  - {fn}")
# #     total += len(files)

# # print(f"\n总文件数：{total}，共 {cnt} 个不同类型")

# #     # for fn in files:
# #     #     print(f"  - {fn}")
# # print(cnt, total)

# # Paths
# thed_frent_path = gm.ThreeD_FRONT_DATASET_PATH
# scenes_dir_path = os.path.join(thed_frent_path, "scenes")

# # Prepare mapping: category -> difficulty -> list of model_ids
# count_to_files = defaultdict(lambda: defaultdict(list))
# total_entries = 0

# for entry in os.listdir(scenes_dir_path):
#     print(f"Processing: {total_entries+1}/{len(os.listdir(scenes_dir_path))}")
#     total_entries += 1
#     entry_path = os.path.join(scenes_dir_path, entry)
#     if not os.path.isdir(entry_path):
#         continue
#     name = entry.split("_")[1]
#     type = name.split("-")[0]
#     # import pdb;pdb.set_trace()

#     # Look for initial object info json
#     for file in os.listdir(entry_path):
#         obj_num = 0
#         if not file.endswith('_initial.json'):
#             continue
#         filepath = os.path.join(entry_path, file)
#         with open(filepath, 'r', encoding='utf-8') as f:
#             data = json.load(f)

#         # Iterate objects
#         for obj_name, obj_info in data.get("objects_info", {}).get("init_info", {}).items():
#             if "furniture" not in obj_name:
#                 continue
#             if obj_info.get("args", {}).get("is_to_rearrange") is not True:
#                 continue
#             obj_num += 1

#         if obj_num < 2:
#             difficulty = "Easy"
#         elif obj_num < 4:
#             difficulty = "Medium"
#         else:
#             difficulty = "Hard"

#         count_to_files[type][difficulty].append(entry)

# # Summarize and print
# # Sort categories by total count and keep top 6
# sorted_categories = sorted(
#     count_to_files.items(),
#     key=lambda item: sum(len(lst) for lst in item[1].values()),
#     reverse=True
# )[:8]

# # Summarize and print
# grand_total = 0
# for category, diff_dict in sorted_categories:
#     cat_total = sum(len(lst) for lst in diff_dict.values())
#     grand_total += cat_total
#     print(f"\n=== Category: {category} (Top 8 – Total {cat_total}) ===")
#     for difficulty, entries in diff_dict.items():
#         print(f"  - {difficulty}: {len(entries)}")

# print(f"\nGrand total for top 8 categories: {grand_total}")

# counts = []
# for cat, d in sorted_categories:
#     n_easy   = len(d.get('Easy',   []))
#     n_med    = len(d.get('Medium', []))
#     n_hard   = len(d.get('Hard',   []))
#     total    = n_easy + n_med + n_hard
#     counts.append((cat, n_easy, n_med, n_hard, total))

# # 2. 按总数降序排序
# counts.sort(key=lambda x: x[4], reverse=True)
# categories = [c[0] for c in counts]
# easy  = np.array([c[1] for c in counts])
# med   = np.array([c[2] for c in counts])
# hard  = np.array([c[3] for c in counts])
# totals= np.array([c[4] for c in counts])

# # 3. 堆叠底部
# bottom_med  = easy
# bottom_hard = easy + med

# # 4. 全局字体设为 Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'

# # 5. 绘图
# fig, ax = plt.subplots(figsize=(8, 5))
# bar_height = 0.5  # 控制柱子的“厚度”，0 < height ≤ 1

# ax.barh(categories, easy,   left=0,           height=bar_height, label='Easy',   color='#C2D6EC')
# ax.barh(categories, med,    left=bottom_med,  height=bar_height, label='Medium', color='#F1CDB1')
# ax.barh(categories, hard,   left=bottom_hard, height=bar_height, label='Hard',   color='#FBE7A3')


# ax.invert_yaxis()
# # 6. 在每条最右侧标注总数
# max_total = totals.max()
# for i, total in enumerate(totals):
#     ax.text(total + max_total*0.01,  # 向右偏移 1% 最大值
#             i,
#             str(total),
# #             va='center',
# #             ha='left',
# #             fontsize=16)

# # # 7. 轴、图例、边框
# # ax.set_xlabel('Cases', fontfamily='Times New Roman', fontsize=24)
# # ax.set_ylabel('Room Type', fontfamily='Times New Roman', fontsize=24)

# # # 这里把 size 一并放到 prop 里
# # ax.legend(
# #     loc='lower right',
# #     prop={'family': 'Times New Roman', 'size': 16},
# #     frameon=False
# # )

# # # 放大坐标刻度
# # ax.tick_params(axis='both', which='major', labelsize=16)
# # # 隐藏上边和右边框
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)

# # plt.tight_layout()
# # plt.savefig('./dataset1.eps', format='eps')

import os
import json
from collections import defaultdict

scenes_dir_path = "C:/Users/Admin/Desktop/OmniGibson-Rearrange/omnigibson/data/3d_front/scenes"
value_stats = defaultdict(list)  # key: obj_num, value: list of target_values
total_entries = 0

for entry in os.listdir(scenes_dir_path):
    print(f"Processing: {total_entries+1}/{len(os.listdir(scenes_dir_path))}")
    total_entries += 1
    entry_path = os.path.join(scenes_dir_path, entry)
    if not os.path.isdir(entry_path):
        continue

    # 查找 *_initial.json 文件
    initial_file = None
    for file in os.listdir(entry_path):
        if file.endswith('_initial.json'):
            initial_file = file
            break
    if initial_file is None:
        continue

    # 解析 initial.json 文件
    filepath = os.path.join(entry_path, initial_file)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    obj_num = 0
    for obj_name, obj_info in data.get("objects_info", {}).get("init_info", {}).items():
        if "furniture" not in obj_name:
            continue
        if obj_info.get("args", {}).get("is_to_rearrange") is not True:
            continue
        obj_num += 1

    if obj_num == 0:
        continue
    # 可选：你可以限制最大为6

    # 查找对应的 value 文件
    # 假设它和 initial_file 名字类似，只是后缀不同
    imitation_dir = "C:/Users/Admin/Desktop/OmniGibson-Rearrange/imitation_data"
    value_path = os.path.join(imitation_dir, f'{entry}.npz') # 替换为你的真实文件名
    
    if not os.path.exists(value_path):
        print(f'{value_path} does not exist.')
        continue
    import numpy as np
    data = np.load(value_path)['data']

    # 从 value_data 中提取你想要的字段，比如 target_value
    target_value = len(data)
    value_stats[obj_num].append(target_value)

# 计算平均值
print("\nAverage values by is_to_rearrange object count:")
for k in sorted(value_stats.keys()):
    avg = sum(value_stats[k]) / len(value_stats[k])
    print(f"{k} object(s): average target_value = {avg:.4f} over {len(value_stats[k])} scenes")
