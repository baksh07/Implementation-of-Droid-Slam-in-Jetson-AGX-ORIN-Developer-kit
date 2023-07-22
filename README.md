
# Implementation-of-Droid-Slam-in-Jetson-AGX-ORIN-Developer-kit
DROID SLAM is an advanced SLAM system that utilizes RGB-D cameras for real-time mapping and localization. In this repository we have implemented DROID SLAM in Jetson AGX ORIN developer kit

# Create a conda environment
```Python
conda create --name slam python=3.7.11
```

# Check Jetpack version
`cat /etc/nv_tegra_release`
```
cat /etc/nv_tegra_release
```

and below is my jetson specification


```# R35 (release), REVISION: 2.1, GCID: 32413640, BOARD: t186ref, EABI: aarch64, DATE: Tue Jan 24 23:38:33 UTC 2023```


based on your specification get to know your jetson version from this [link](https://www.stereolabs.com/blog/nvidia-jetson-l4t-and-jetpack-support/)
in my case my release version is 35.2.1 then my jetpack version will be jetpack 5.1, Next step is to visit this [link](https://elinux.org/Jetson_Zoo#PyTorch_.28Caffe2.29) and download pytorch .whl file, In my case I'm downloading Pytorch version 2.0.0

## Installing PyTorch in our envioronment


Before installing PyTorch, Excecute the below given commands,

```
sudo apt-get -y update; 
sudo apt-get -y install autoconf bc build-essential g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev;
```
```
export TORCH_INSTALL=/home/vision/Downloads/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
```


and finally install PyTorch using below command
```
python3 -m pip install --upgrade pip; python3 -m pip install aiohttp numpy=='1.19.4' scipy=='1.5.3' export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"; python3 -m pip install --upgrade protobuf; python3 -m pip install --no-cache $TORCH_INSTALL
```
## DRIOD-SLAM
After successful installation of pytorch now we can move ahead to implement DROID-SLAM in Jetson AGX developer kit, before proceding activate the envioronment which we have created before
```
conda activate slam
```
and also run the below command for install some other dependencies
```
pip3 install opencv-python rawpy einops matplotlib pandas GPUtil scikit-image scikit-learn tqdm open3d gdown
```
```
pip3 install evo --upgrade --no-binary evo
```
clone the DROID-SLAM repo with `--recursive flag`
```
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
```
move to the directory where the setup.py is located in that directory run the below code
```
python3 setup.py install
```
Download the model from this [link](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view)
you can download the dataset from the below code
```
./tools/download_sample_data.sh
```
after downloading the dataset locate demo.py and the run below given codes for excecution of DROID-SLAM
```
python3 demo.py --imagedir=data/abandonedfactory --calib=calib/tartan.txt --stride=2
```
```
python3 demo.py --imagedir=data/sfm_bench/rgb --calib=calib/eth.txt
```
```
python3 demo.py --imagedir=data/Barn --calib=calib/barn.txt --stride=1 --backend_nms=4
```
```
python3 demo.py --imagedir=data/mav0/cam0/data --calib=calib/euroc.txt --t0=150
```
```
python3 demo.py --imagedir=data/rgbd_dataset_freiburg3_cabinet/rgb --calib=calib/tum3.txt
```
To run DROID-SLAM on custom dataset follow the below steps
1. create a folder in data folder you can give any name it doesn`t matter for example i had created a folder named custom_dataset and move all the captured frames to the folder
2. create text file in calib folder and paste intrinsic parameters of the camera which you have used to captured data in my case i have used OAKD camera and i had created oak.txt for storing intrinsic parameter
3. run the below command to run DROID-SLAM on custom dataset
4. You can refer this [git](https://github.com/aartighatkesar/Camera_Calibration.git) to find intrinsic parameter`s of your camera
```
python3 demo.py --imagedir=data/custom_datset --calib=calib/oak.txt --stride=2
```
`Note: data/"folder you have named" same goes for .txt in calib folder`<br>
To capture frames for custom dataset you can use the below code 
`Note that below code is only applicable for OAK-D camera
```
import sys
import numpy as np
import cv2
import os
import glob 
import time
import depthai as dai

if __name__ == '__main__':

    pipeline = dai.Pipeline()

    camRgb = pipeline.createColorCamera()
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    camRgb.setPreviewSize(300,300)
    camRgb.initialControl.setManualFocus(128)
    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")

    camRgb.video.link(xoutRgb.input)
    
    device = dai.Device(pipeline)

    tstamps = []

    qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

    output_directory = 'frames'
    os.makedirs(output_directory, exist_ok=True)
    frame_counter = 0

    while True:
        inRgb = qRgb.tryGet()

        if inRgb is not None:
            image = inRgb.getCvFrame()
        else:
            continue

        print(image.shape)
        window_name = "image"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, image / 255.0)

        # Save the frame as an image
        output_path = os.path.join(output_directory, f'{str(frame_counter).zfill(3)}.png')
        cv2.imwrite(output_path, image)

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

```
Remember to change the output directory path <br>
To save points and posses in .ply format you can update `visualization.py` in droidslam folder as
```
import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d

from lietorch import SE3
import geom.projective_ops as pops


CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor


def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def droid_visualization(video, device="cuda:0"):
    """ DROID visualization frontend """

    torch.cuda.set_device(0)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0
    print("droid_visualization")
    save_path = '/home/vision/points/' #change where you want to save the points and poses in ply format


    droid_visualization.filter_thresh = 0.3  #0.005

    def increase_filter(vis):
        droid_visualization.filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def decrease_filter(vis):
        droid_visualization.filter_thresh *= 0.5
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True   
    
    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

        with torch.no_grad():

            with video.get_lock():
                t = video.counter.value 
                dirty_index, = torch.where(video.dirty.clone())
                dirty_index = dirty_index

            if len(dirty_index) == 0:
                return

            video.dirty[dirty_index] = False

            # convert poses to 4x4 matrix
            poses = torch.index_select(video.poses, 0, dirty_index)
            disps = torch.index_select(video.disps, 0, dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()

            images = torch.index_select(video.images, 0, dirty_index)
            images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
            points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

            thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
            
            count = droid_backends.depth_filter(
                video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)
            
            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))     
        
            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()

                if ix in droid_visualization.cameras:
                    vis.remove_geometry(droid_visualization.cameras[ix])
                    del droid_visualization.cameras[ix]

                if ix in droid_visualization.points:
                    vis.remove_geometry(droid_visualization.points[ix])
                    del droid_visualization.points[ix]

                ### add camera actor ###
                cam_actor = create_camera_actor(True)
                cam_actor.transform(pose)
                vis.add_geometry(cam_actor)
                droid_visualization.cameras[ix] = cam_actor
                
                
                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
                
                ## add point actor ###
                point_actor = create_point_actor(pts, clr)
                vis.add_geometry(point_actor)
                droid_visualization.points[ix] = point_actor

            ### Hack to save Point Cloud Data and Camnera results ###
            
            # Save points
            pcd_points = o3d.geometry.PointCloud()
            for p in droid_visualization.points.items():
                pcd_points += p[1]
            o3d.io.write_point_cloud(f"{save_path}/points.ply", pcd_points, write_ascii=False)
                
            # Save pose
            pcd_camera = create_camera_actor(True)
            for c in droid_visualization.cameras.items():
                pcd_camera += c[1]

            o3d.io.write_line_set(f"{save_path}/camera.ply", pcd_camera, write_ascii=False)

            ### end ###
            
            # hack to allow interacting with vizualization during inference
            if len(droid_visualization.cameras) >= droid_visualization.warmup:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            droid_visualization.ix += 1
            vis.poll_events()
            vis.update_renderer()

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_filter)
    vis.register_key_callback(ord("A"), decrease_filter)

    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.run()
    vis.destroy_window()
```
`Note : change save_path for your desired directory`<br>
To visualise the 3D points after saving points in .ply format you can use below code 
```
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
```
while running above code if you encounter the below error
```
[Open3D WARNING] GLFW Error: GLX: Failed to create context: GLXBadFBConfig
[Open3D WARNING] Failed to create window
[Open3D WARNING] [DrawGeometries] Failed creating OpenGL window.
```
by running ```export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6``` in your terminal you are good to go. <br>
To render the point cloud in web application visit this [git](https://github.com/mmspg/point-cloud-web-renderer.git) <br>
To render the desired pointcloud some of the things need to be taken care such as:
1. open `htdocs/pointCloudVeiwer/config/config002.json` and change the ply file for rendering the desired pointcloud
2. open `htdocs/pointCloudVeiwer/js/main.js` change line number 26 i.e
   ```
   var CONFIG = './config/config001.json';
   ```
   To
   ```
   var CONFIG = './config/config002.json';
   ```
## It is advisable to use VS-CODE for the entire process



