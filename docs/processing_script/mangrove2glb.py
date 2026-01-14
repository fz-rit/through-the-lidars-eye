# use py311
from plyfile import PlyData
import numpy as np
import trimesh
import matplotlib.cm as cm
import json

with open("label_maps.json", 'r') as f:
    label_maps = json.load(f)

in_ply = "pcd_ALRSET12_7489_seg.ply"
out_glb = "pcd_ALRSET12_7489_seg.glb"
dataset = "MANGROVE_ROOTS"   # MANGROVE_ROOTS, ForestSemantic, or HARVARD_FOREST
colormap = label_maps["DATASETS"][dataset]["COLOR_TO_INDEX"]


# read data and find class ID
ply = PlyData.read(in_ply)
v = ply['vertex'].data
xyz = np.vstack([v['X'], v['Z'], v['Y']]).T.astype(np.float32)
# Read class ids
class_ids = np.asarray(v["class_id"], dtype=int)

# Allocate color array
rgb = np.zeros((len(class_ids), 3), dtype=np.uint8)
# Assign colors by class id
for color, cid in colormap.items():
    rgb[class_ids == cid] = [int(c) for c in color.split(',')]

pc = trimesh.points.PointCloud(vertices=xyz, colors=rgb)
scene = trimesh.Scene(pc)
scene.export(out_glb)
print("Wrote:", out_glb)