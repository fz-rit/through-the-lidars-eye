# pip install plyfile trimesh numpy
from plyfile import PlyData
import numpy as np
import trimesh

in_ply = "/Users/amirhassanzadeh/Downloads/pcd_ALRSET12_7489_seg.ply"
out_glb = "/Users/amirhassanzadeh/Downloads/pcd_ALRSET12_7489_seg2.glb"
scalar_name = "class_id"

ply = PlyData.read(in_ply)
v = ply['vertex'].data
xyz = np.vstack([v['X'], v['Z'], v['Y']]).T.astype(np.float32)

s = np.asarray(v[scalar_name], float)
s = (s - s.min()) / (s.max() - s.min() + 1e-12)

import matplotlib.cm as cm
rgb = (cm.get_cmap('Paired')(s)[:, :3]).astype(np.uint8)

pc = trimesh.points.PointCloud(vertices=xyz, colors=rgb)
scene = trimesh.Scene(pc)
scene.export(out_glb)
print("Wrote:", out_glb)