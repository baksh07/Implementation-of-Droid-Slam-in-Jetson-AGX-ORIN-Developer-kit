# Implementation-of-Droid-Slam-in-Jetson-AGX-ORIN-Developer-kit
DROID SLAM is an advanced SLAM system that utilizes RGB-D cameras for real-time mapping and localization. In this repository we have implemented DROID SLAM in Jstson AGX ORIN developer kit

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



