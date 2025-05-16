from omnigibson.macros import gm
import os
from collections import defaultdict
import json


threed_front_path = gm.ThreeD_FRONT_DATASET_PATH
scenes_dir_path = os.path.join(threed_front_path, "scenes")
count_to_files = defaultdict(list)
# total = 0
# for entry in os.listdir(scenes_dir_path):
#     print(f"deal: {total+1}/5549")
#     total += 1
#     entry_path = os.path.join(scenes_dir_path, entry)
#     if os.path.isdir(entry_path):
#         name = entry.split("_")[1]
#         type = name.split("-")[0]
#         # import pdb;pdb.set_trace()
#         count_to_files[type].append(entry)

with open('C:/Users/Admin/Desktop/3D-FUTURE-model/model_info.json', 'r', encoding='utf-8') as f:
    cat_data = json.load(f)
total = 0
for entry in os.listdir(scenes_dir_path):
        print(f"deal: {total+1}/5549")
        total += 1
        entry_path = os.path.join(scenes_dir_path, entry)
        if os.path.isdir(entry_path):
            for file in os.listdir(entry_path):
                if "_initial.json" in file:
                    filepath = os.path.join(entry_path, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    
                    for obj_name, obj_info in data["objects_info"]["init_info"].items():
                        if "furniture" in obj_name:
                            if obj_info["args"]["is_to_rearrange"] is True:
                                name = obj_info["args"]["usd_path"].split('/')[2]
                                jid = name.split('_')[0]
                                # import pdb;pdb.set_trace()
                                for item in cat_data:
                                    if item.get("model_id") == jid:

                                        category = item.get("super-category", None)
                                        if category == "Bed":
                                            category = item.get("category", None)
                                        break
                                
                                count_to_files[category].append(jid)


# 排序：按照每个 type 对应的文件数，从多到少
sorted_items = sorted(count_to_files.items(), key=lambda item: len(item[1]), reverse=True)

total = 0
cnt = 0
for type, files in sorted_items:
    cnt += 1
    print(f"\n=== type === {type} ：共 {len(files)} 个文件 ===")
    # for fn in files:
    #     print(f"  - {fn}")
    total += len(files)

print(f"\n总文件数：{total}，共 {cnt} 个不同类型")

    # for fn in files:
    #     print(f"  - {fn}")
print(cnt, total)