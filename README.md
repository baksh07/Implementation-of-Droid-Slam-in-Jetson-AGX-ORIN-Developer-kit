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
1. create a folder in data folder you can give any name it doesn`t matter for example i created a folder named custom_dataset and paste all the captured frames
2. create text file in calib folder and paste intrinsic parameters of the camera which you have used to captured data in my case i have used OAKD camera. In my case i had created oak.txt for storing intrinsic parameter
3. run the below command to run DROID-SLAM on custom dataset
```
python3 demo.py --imagedir=data/custom_datset --calib=calib/oak.txt --stride=2
```
`Note: data/"folder you have named" same goes for .txt in calib folder`
To capture frames for custom dataset you can use the below code 
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
Remember to change the output directory




