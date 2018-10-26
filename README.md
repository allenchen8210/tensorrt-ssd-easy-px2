# tensorrt-ssd-easy-px2 

This project not finish yet


## Environment

OS : ubuntu 16.04

OpenCV : 2.4.13

CUDA : 9.0

TensorRT : 3.0.4

Pretrained Model : VGG_VOC0712_SSD_300x300_iter_120000.caffemodel 



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
Download model : VGG_VOC0712_SSD_300x300_iter_120000.caffemodel 

### Installing
```
$ git clone git@github.com:allenchen8210/tensorrt-ssd-easy-px2.git

$ cd tensorrt-ssd-easy-px2

$ mkdir build 
```
Then you should edit your CMakeLists.txt to make program find  your TensroRT, like below
```
...
include_directories(/home/mec/JN_work/tensorrt-ssd-easy-px2_new/TensorRT-3.0.4/include)
...
target_link_libraries(inferLib /home/mec/JN_work/tensorrt-ssd-easy-px2_new/TensorRT-3.0.4/lib/libnvcaffe_parser.so)
target_link_libraries(inferLib /home/mec/JN_work/tensorrt-ssd-easy-px2_new/TensorRT-3.0.4/lib/libnvinfer.so)
target_link_libraries(inferLib /home/mec/JN_work/tensorrt-ssd-easy-px2_new/TensorRT-3.0.4/lib/libnvinfer_plugin.so)
target_link_libraries(inferLib /home/mec/JN_work/tensorrt-ssd-easy-px2_new/TensorRT-3.0.4/lib/libnvparsers.so)
...
```
Next Step, you  may build program 
```
$ cmake ..

$ make 
```

## Reference:

[saikumarGadde](https://github.com/saikumarGadde/tensorrt-ssd-easy)

[chenzhi1992](https://github.com/chenzhi1992/TensorRT-SSD)

[Teoge](https://github.com/Teoge/tensorrt-ssd-easy)
