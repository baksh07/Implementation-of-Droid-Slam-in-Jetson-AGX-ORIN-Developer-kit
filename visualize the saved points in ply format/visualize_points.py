import numpy as np
import open3d as o3d

if __name__ == "__main__":
    o3d.visualization.webrtc_server.enable_webrtc()
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("/home/baksh/iitdelhi/ply_files_to_show_chetan_sir/points.ply")
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                           [0, 0, 0, 1]])
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])
