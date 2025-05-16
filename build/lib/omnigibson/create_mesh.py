from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os
import json
import re
from pxr import Usd, UsdGeom, Vt, Gf, UsdShade, Sdf
from omnigibson.utils.render_utils import create_pbr_material
from omnigibson.utils.physx_utils import bind_material


# 输入 JSON 文件所在的文件夹路径
input_folder = '/home/pilab/Desktop/wq/rearrange/omnigibson/data/3d_front/3D-FRONT'
# USD 输出的基础文件夹（后续会根据 JSON 内 uid 创建子文件夹）
output_base = '/home/pilab/Desktop/wq/rearrange/omnigibson/data/3d_front/mesh'

# 遍历输入文件夹下所有的 JSON 文件
for filename in os.listdir(input_folder):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(input_folder, filename)
    print(f"正在处理文件：{json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取 JSON 文件中顶层的 uid 字段，如果不存在则用文件名（去除后缀）作为 uid
    json_uid = data.get("uid", os.path.splitext(filename)[0])
    # 针对当前 JSON 文件创建一个基础输出文件夹
    json_output_folder = os.path.join(output_base, json_uid)
    if not os.path.exists(json_output_folder):
        os.makedirs(json_output_folder)

    # 遍历 scene -> room
    scene = data.get("scene", {})
    rooms = scene.get("room", [])
    materials = data.get("material", [])
    for room in rooms:
        room_instanceid = room.get("instanceid")
        if not room_instanceid:
            continue

        # 在当前 JSON 的输出目录下，以 room 的 instanceid 命名创建子文件夹
        room_folder = os.path.join(json_output_folder, room_instanceid)
        if not os.path.exists(room_folder):
            os.makedirs(room_folder)

        # 遍历该 room 的 children
        for child in room.get("children", []):
            child_instanceid = child.get("instanceid", "")
            # 只处理 instanceid 包含 "mesh" 的 children
            if "mesh" not in child_instanceid:
                continue

            ref = child.get("ref")
            if not ref:
                print(f"警告：room {room_instanceid} 中 children 缺少 ref 字段，跳过。")
                continue

            # 在 mesh 数组中查找 uid == ref 的条目
            mesh_found = None
            for mesh_entry in data.get("mesh", []):
                mesh_type = mesh_entry["type"]
                # mesh_to_convert = ["wall", "Wall", "floor", "Floor", "Door", "Baseboard", "Window", "Hole", "Pocket"]
                # if not any(m in mesh_type for m in mesh_to_convert):
                if mesh_type == "Cabinet" or mesh_type == "Ceiling" or mesh_type == "LightBand":
                    continue

                if mesh_entry.get("uid") == ref:
                    mesh_found = mesh_entry
                    break

            if mesh_found is None:
                print(f"提示：未在 mesh 数组中找到 uid 为 {ref} 的条目，跳过。")
                continue

            # 生成 USD 文件
            # 对 uid 进行替换，确保文件名合法
            usd_uid = re.sub(r'[^a-zA-Z0-9]', '_', ref)
            usd_filename = os.path.join(room_folder, f"mesh_{usd_uid}.usd")

            # 创建新的 USD Stage
            stage = Usd.Stage.CreateNew(usd_filename)
            # 定义根节点 /World，并设置为默认 Prim
            root = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(root.GetPrim())
            # 在 /World 下创建一个以 ref 命名的 Xform
            prim_path = f"/World/mesh_{usd_uid}"
            xform = UsdGeom.Xform.Define(stage, prim_path)
            # 在该 Xform 下创建一个 Mesh 节点
            mesh_prim_path = prim_path + "/mesh"
            usd_mesh = UsdGeom.Mesh.Define(stage, mesh_prim_path)

            # 解析 "xyz" 数组，将每三个数作为一个顶点坐标
            xyz = mesh_found.get("xyz", [])
            num_points = len(xyz) // 3
            points = Vt.Vec3fArray(num_points)
            for i in range(num_points):
                x = xyz[3*i]
                y = xyz[3*i+1]
                z = xyz[3*i+2]
                ## transform pos
                points[i] = Gf.Vec3f(x, y, z)
            usd_mesh.CreatePointsAttr(points)

            # 解析 "faces" 数组，直接作为 faceIndices（假设每个面均为三角形）
            faces = mesh_found.get("faces", [])
            faceIndices = Vt.IntArray(len(faces))
            for i, idx in enumerate(faces):
                faceIndices[i] = idx
            usd_mesh.CreateFaceVertexIndicesAttr(faceIndices)
            num_faces = len(faces) // 3
            faceCounts = Vt.IntArray(num_faces)
            for i in range(num_faces):
                faceCounts[i] = 3
            usd_mesh.CreateFaceVertexCountsAttr(faceCounts)

            # 解析 "normal" 数组，每三个数作为一个法向量
            normal_vals = mesh_found.get("normal", [])
            num_normals = len(normal_vals) // 3
            normals = Vt.Vec3fArray(num_normals)
            for i in range(num_normals):
                nx = normal_vals[3*i]
                ny = normal_vals[3*i+1]
                nz = normal_vals[3*i+2]
                normals[i] = Gf.Vec3f(nx, ny, nz)
            usd_mesh.CreateNormalsAttr(normals)

            # 如有需要，可解析 UV 数据（此处省略）
            # scope_path = prim_path + "/Looks"
            # stage.DefinePrim(scope_path, "Scope")
            # mat_path = scope_path + "/Default"
            # mat = create_pbr_material(prim_path=mat_path)
            # bind_material(prim_path=mesh_prim_path, material_path=mat_path)

            # Create a material.
            material_jid = None
            material_uid = mesh_found.get("material", None)
            if material_uid is not None:
                # get material jid by uid
                for m in materials:
                    m_uid = m.get("uid", "")
                    if m_uid == material_uid:
                        material_jid = m.get("jid", "")
                        break
                if material_jid is None:
                    # 保存 USD 文件
                    stage.GetRootLayer().Save()
                    print(f"USD 文件已生成：{usd_filename}")
                    continue

                material_path = Sdf.Path(f"/World/mesh_{usd_uid}/Looks")
                material = UsdShade.Material.Define(stage, material_path)

                # Create a shader for the material.
                shader = UsdShade.Shader.Define(stage, material_path.AppendChild('PBRShader'))
                shader.CreateIdAttr('UsdPreviewSurface')

                # Create an input for the diffuse color.
                diffuse_color_input = shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f)

                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                # Create a texture for the diffuse color.
                texture_path = material_path.AppendChild('DiffuseColorTx')
                texture = UsdShade.Shader.Define(stage, texture_path)
                texture.CreateIdAttr('UsdUVTexture')

                # Set the file path of the texture.
                current_path = os.getcwd()
                # import pdb; pdb.set_trace()
                mat_path = os.path.join(current_path, f'omnigibson/data/3d_front/3D-FRONT-texture/{material_jid}/texture.png')
                texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(mat_path)

                # Connect the texture to the diffuse color input.
                texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3).ConnectToSource(diffuse_color_input)

                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), 'rgb')
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                # Bind the material to the mesh.
                UsdShade.MaterialBindingAPI(usd_mesh).Bind(material)
            # 保存 USD 文件
            stage.GetRootLayer().Save()
            print(f"USD 文件已生成：{usd_filename}")

print("批量转换完成。")
