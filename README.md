## Extra Datasets

Please downloads <a href='https://fcav.engin.umich.edu/projects/driving-in-the-matrix'>VOC Datasets</a>

## Enviornment

#### Our enviornment
- GPU NVIDIA RTS A6000 x 2
#### Prerequisites
- python3
- Linux
- NVIDA GPU (if you use pretrained model, you should have multi-gpu)
#### Getting Started
- git clone repo
```
git clone https://github.com/inu-RAISE/VCOD_Challenge VCOD && cd VCOD
pip install -r requirements.txt
```
## Setting Directory
If you use our repository, you should reset **path.yaml**

#### Required weight file
- yolov5 :  **car** weight file
- yolov5 : **cycle** weight file
- CUT : weight folder
- EfficientNet : weight file

#### Required csv file (location : path.yaml - parent)
- csv file columns
  - File : cropped image path (Please check for cropped image)
  - Class : 0 or 1 (0 is car and truck , 1 is cycle and motorbike)
  - min_x , min_y, max_x, max_y
  - v_cls : full class (included orientation label)

#### Directory Setting for Extract Local Information using YOLOv5
```
class_folder
| train
|____ images
|____ labels
| val
|____ images
|____ labels
```
#### Directory Setting for Generate Synthetic to Real Image
```
target_folder_name
|trainA
|trainB
|testA
|testB
```
Defualt direction of CUT  is A to B. i.e **A is input images and B is target images**.

## Run code

- One shot process (preprocessing - train - inference)
 ```
 python main.py --process all 
```
- training (preprocessing - train)
```
python main.py --process train
```
You can get **CUT weight folder, YOLO weight file and Classification weight file.***
- inference (preprocessing - inference)
```
python main.py --process infer
```
Please check **weight file**
- Fast Inference (Only classification)
You should already be a **csv file** and **cropped image**. All you need is a **classification weight file**
```
python classficiation_inference.py
```
## Citation

```
https://github.com/ultralytics/yolov5

@inproceedings{johnson2017driving,
  title={Driving in the Matrix: Can virtual worlds replace human-generated annotations for real world tasks?},
  author={Johnson-Roberson, Matthew and Barto, Charles and Mehta, Rounak and Sridhar, Sharath Nittur and Rosaen, Karl and Vasudevan, Ram},
  booktitle={2017 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={746--753},
  year={2017},
  organization={IEEE}
}

@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```
