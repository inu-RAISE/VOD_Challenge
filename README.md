# A 2-stage model for vehicle class and orientation detection with photo-realistic image generation
By <a href='https://github.com/winston1214'>Youngmin Kim*</a>, <a href='https://github.com/anima0729'>Donghwa Kang*</a> Hyeonboo Baek (* is co-author)

**RAISE Lab in Incheon National University**

IEEE BigData Challenge Cup 2022 <a href='https://vod2022.sekilab.global/'>Vehicle class and Orientation Detection Challenge 2022</a>

Our paper : https://ieeexplore.ieee.org/abstract/document/10020472

## Overview
<img src='https://github.com/inu-RAISE/VOD_Challenge/blob/master/pic/fig2.PNG?raw=true'></img>
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
pip install -U typing_extensions
pip install -U albumentations
```
## Setting Directory
If you use our repository, you should reset **path.yaml**

#### Required weight file (Pretrained)
- yolov5 :  <a href='https://drive.google.com/file/d/1pvCgh97NLJ0qfZpq8tUnpyosqoqX1FRa/view?usp=sharing'>**car** weight file</a>
- yolov5 : <a href='https://drive.google.com/file/d/1520eFOqhCOyvWlXdQf9GusIHkLnIRMui/view?usp=sharing'>**cycle** weight file</a>
- CUT : <a href='https://drive.google.com/file/d/1JlkQxBHxHFDGxbWjq3KR2ReoRHWAQGMc/view?usp=sharing'>**car** weight **zip** </a>
- CUT : <a href='https://drive.google.com/file/d/1H8QdT-fOEj-BvlGgc8XFiMSsWROqbrt7/view?usp=sharing'>**cycle** weight **zip** folder</a>
- EfficientNet : <a href='https://drive.google.com/drive/folders/1uPxwFl5-Eq_-2AFvZNvZ_r_Awpx95ida?usp=sharing'>classifcation weight folder (1~5) </a>

#### Required csv file (location : path.yaml - parent)
- csv file columns
  - File : cropped image path (Please check for cropped image)
  - Class : 0 or 1 (0 is car and truck , 1 is cycle and motorbike)
  - min_x , min_y, max_x, max_y
  - v_cls : full class (included orientation label)

If you want more detail about directory, please **check <a href='https://github.com/inu-RAISE/VCOD_Challenge/blob/master/docs/Setting_Dir.md'>docs/Setting_dir.md</a>**


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
@inproceedings{kim20222,
  title={A 2-Stage Model for Vehicle Class and Orientation Detection with Photo-Realistic Image Generation},
  author={Kim, Youngmin and Kang, Donghwa and Baek, Hyeongboo},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={6489--6493},
  year={2022},
  organization={IEEE}
}
```

## Reference

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
