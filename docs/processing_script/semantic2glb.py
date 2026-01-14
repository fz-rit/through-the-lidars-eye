import numpy as np
import trimesh
import matplotlib.cm as cm
import json
import pandas as pd

with open("label_maps.json", 'r') as f:
    label_maps = json.load(f)

# FOR SEMANTIC3D
csv_path = "/Users/amirhassanzadeh/Downloads/pcd_ForestSemantic_Semantic3D_3pcd_files/semantic3d_domfountain_station1_xyz_intensity_rgb_with_labels_color.csv"
out_glb = "/Users/amirhassanzadeh/Downloads/pcd_ForestSemantic_Semantic3D_3pcd_files/semantic3d_domfountain_station1_xyz_intensity_rgb_with_labels_color.glb"
dataset = "SEMANTIC3D"   # MANGROVE_ROOTS, ForestSemantic, or HARVARD_FOREST
colormap = label_maps["DATASETS"][dataset]["COLOR_TO_INDEX"]

# read data and find class ID
df = pd.read_csv(csv_path, usecols = ["X", "Y", "Z", "Classification"])

# Build xyz
X = df["X"].to_numpy(dtype=np.float32)
Y = df["Y"].to_numpy(dtype=np.float32)
Z = df["Z"].to_numpy(dtype=np.float32)
class_ids = np.asarray(df["Classification"], dtype=int)
xyz = np.vstack([X, Z, Y]).T

# Allocate color array
rgb = np.zeros((len(class_ids), 3), dtype=np.uint8)
# Assign colors by class id
for cid, color in colormap.items():
    rgb[class_ids == int(cid)] = [int(c) for c in color]

pc = trimesh.points.PointCloud(vertices=xyz, colors=rgb)
scene = trimesh.Scene(pc)
scene.export(out_glb)
print("Wrote:", out_glb)