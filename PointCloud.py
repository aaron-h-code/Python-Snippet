pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)    # xyz: (n_pts, 3)
pcd.colors = o3d.utility.Vector3dVector(color)  # color: (n_pts, 3), range [0, 1]
o3d.io.write_point_cloud('../../test_data/sync.ply', pcd)
