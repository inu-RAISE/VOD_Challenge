# How to Set directory?
<h2 align='center'>
  please check <a href='https://github.com/inu-RAISE/VCOD_Challenge/blob/master/path.yaml'>path.yaml</a>.
</h2>

## Overall Folder Structure
<img src='https://github.com/inu-RAISE/VCOD_Challenge/blob/master/pic/overall_dir_structure.PNG?raw=true'></img>

You do not need to specify each folder for this folder format .In the **main.py**, directory are automatically organized. So, except for a few folders, you don't need to configure it yourself.

## Detail

### CUT series

- **path.yaml**
  - cut.car (To train and test the cut model)
  - cut.cycle (To train and test the cut model)
  - cut.save_car (save result images)
  - cut.save_cycle (save result images)

CUT is image to image translation model. To use this model, we need to set the directory.

Among the folder, it related to cut are **cut_car** and **cut_cycle**.

```
CUT_folder
|trainA
|trainB
|testA
|testB
```
Defualt direction of CUT  is A to B. i.e **A is input images and B is target images**.
### YOLO series

- **path.yaml**
  - You don't need setting because the folder name and configuration are set by itself.
  - But, you should set dataset path and parent path

YOLO is object detection model. To use this model, we need to set the directory.

Among the folder, it related to cut are **train_car,train_cycle,voc_car and voc_cycle**.
```
class_folder
| train
|____ images
|____ labels
| val
|____ images
|____ labels
```

### Weight File
If you train and test at the same time, you do not need to care about the name in the weight file, but if you use a **pretrained model**, you need to set the folder name as follows.

Please check <a href='https://github.com/inu-RAISE/VCOD_Challenge#required-weight-file-pretrained'>pretrained weight download in README</a>
```
weight
|car_best.pt # yolo weight
|cycle_best.pt # yolo weight
|cut_car
|___ latest_net_D.pth  latest_net_F.pth  latest_net_G.pth
|cut_cycle
|___ latest_net_D.pth  latest_net_F.pth  latest_net_G.pth
|models
|___ Py_efficientnet-b7_GAN_0922_25_1.pt Py_efficientnet-b7_GAN_0922_25_2.pt ... Py_efficientnet-b7_GAN_0922_25_5.pt
```
⚠️ Please check **CUT pretrained weight zip file!** After unpacking, you need to rename it. (**cut_car, cut_cycle**)
