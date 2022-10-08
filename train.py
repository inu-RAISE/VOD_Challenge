import os
from preprocessing import *
import shutil
from omegaconf import OmegaConf as OC
import pandas as pd
import argpare


# +
def yolo_preprocessing(path):
    # path = yaml_file
    parent_path = path.parent
    voc_car_path = parent_path+'/'+'voc_car/'
    voc_cycle_path = parent_path+'/'+'voc_cycle/'
    train_car_path = parent_path+'/'+'train_car/'
    train_cycle_path = parent_path+'/'+'train_cycle/'

    mk_folder(parent_path)  
    mk_folder(voc_car_path)
    mk_folder(voc_cycle_path)
    mk_folder(train_car_path)
    mk_folder(train_cycle_path)

    voc_input_label_dir = path.voc.image
    voc_input_image_dir = path.voc.label

    car_voc_output_label_dir = f'{voc_car_path}labels' # car image
    car_voc_output_image_dir = f'{voc_car_path}images' # car label
    cycle_voc_output_label_dir = f'{voc_cycle_path}labels'
    cycle_voc_output_image_dir = f'{voc_cycle_path}images'

    yolo_VOC_setting(voc_input_image_dir,voc_input_label_dir,car_voc_output_image_dir,car_voc_output_label_dir,'car') # car setting
    yolo_VOC_setting(voc_input_image_dir,voc_input_label_dir,cycle_voc_output_image_dir,cycle_voc_output_label_dir,'motorbike') # cycle setting


    input_label_dir = path.train.label
    input_image_dir = path.train.image
    car_output_label_dir = f'{train_car_path}labels' # car image
    car_output_image_dir = f'{train_car_path}images' # car label
    cycle_output_label_dir = f'{train_cycle_path}labels' # car image
    cycle_output_image_dir = f'{train_cycle_path}images' # car label

    yolo_vehicle_folder_setting(input_image_dir,input_label_dir,car_output_image_dir,car_output_label_dir,'car')
    yolo_vehicle_folder_setting(input_image_dir,input_label_dir,cycle_output_image_dir,cycle_output_label_dir,'motorbike')


    yolo_folder_setting(parent_path,car_output_image_dir,car_output_label_dir,'car') # training data (car)
    yolo_folder_setting(parent_path,car_voc_output_image_dir,car_voc_output_label_dir,'car') # voc data (car)
    yolo_folder_setting(parent_path,cycle_output_image_dir,cycle_output_label_dir,'cycle') # training data (cycle)
    yolo_folder_setting(parent_path,cycle_voc_output_image_dir,cycle_voc_output_label_dir,'cycle') # voc data (cycle)


def yolo_train(path): 
    # path = yaml_file
    project_name = path.parent + 'results/'
    mk_folder(path.parent+'weights')
    
    car_yaml_file = 'yolov5/data/car.yaml'
    car_name = 'car_train'
    cycle_yaml_file = 'yolov5/data/cycle.yaml'
    cycle_name = 'cycle_train'


    os.system(f'python yolov5/train.py --data {car_yaml_file} --weights yolov5x6.pt --epochs 20 --batch-size 32 --project {project_name} --name {car_name}')
    os.system(f'python yolov5/train.py --data {cycle_yaml_file} --weights yolov5x6.pt --epochs 20 --batch-size 32 --project {project_name} --name {cycle_name}')

    ## Arrangement Folder
    shutil.move(project_name+car_name+'/'+'weights/best.pt',path.weight.yolo_car_weight)
    shutil.move(project_name+cycle_name+'/'+'weights/best.pt',path.weight.yolo_cycle_weight)
    

def cropping_train(path):
    # path = yaml_file
    image_path = path.train.image
    label_path = path.train.label
    csv_save_path = path.parent
    crop_save_path = path.crop.all
    
    mk_folder(crop_save_path)
    csv_name = 'train_image_crop'
    csv_file_setting(image_path,label_path,csv_save_path,csv_save_path,csv_name)
    print('Train Dataset cropping Done!')
    
def cropping_test(path): 
    # path = yaml_file
    project_name = path.parent + 'results/'
    
    real_car_name = 'real_car_crop'
    real_cycle_name = 'real_cycle_crop'
    
    v1_real_image_path = path.test.v1
    v2_real_image_path = path.test.v2
    
    car_weight_path = path.weight.yolo_car_weight
    cycle_weight_path = path.weight.yolo_cycle_weight
    
    ### test set Cropping (car)
    os.system(f"python yolov5/detect.py --source {v1_real_image_path} --weights {car_weight_path} --project {project_name} --name {real_car_name} --save-crop")
    os.system(f"python yolov5/detect.py --source {v2_real_image_path} --weights {car_weight_path} --project {project_name} --name {real_car_name+'_2'} --save-crop")
    
    ### test set cropping (cycle)
    os.system(f"python yolov5/detect.py --source {v1_real_image_path} --weights {cycle_weight_path} --project {project_name} --name {real_cycle_name} --save-crop")
    os.system(f"python yolov5/detect.py --source {v2_real_image_path} --weights {cycle_weight_path} --project {project_name} --name {real_cycle_name+'_2'} --save-crop")
    
    ## Arrangement Folder
    mk_folder(path.crop.test_car)
    mk_folder(path.crop.test_cycle)
    
    for idx,real_jpg in enumerate(sorted(glob.glob(project_name + real_car_name + '/*.jpg'))):
        shutil.move(real_jpg, path.crop.test_car  + f'v1_{idx:07d}.jpg')
    for idx,real_jpg in enumerate(sorted(glob.glob(project_name + real_car_name + '_2/*.jpg'))):
        shutil.move(real_jpg, path.crop.test_car + f'v2_{idx:07d}.jpg')
    for idx,real_jpg in enumerate(sorted(glob.glob(project_name + real_cycle_name + '/*.jpg'))):
        shutil.move(real_jpg, path.crop.test_cycle + f'v1_{idx:07d}.jpg')
    for idx,real_jpg in enumerate(sorted(glob.glob(project_name + real_cycle_name + '_2/*.jpg'))):
        shutil.move(real_jpg, path.crop.test_cycle + f'v2_{idx:07d}.jpg')
    
    os.remove(project_name+real_car_name+'car')
    os.remove(project_name+real_cycle_name+'cycle')
    
def cut_folder_setting(path):
    # path = yaml_file
    
    csv_file = pd.read_csv(path.parent + 'train_image_crop.csv')
    
    car_cut_path = path.cut.car
    cycle_cut_path = path.cut.cycle
    
    mk_folder(car_cut_path)
    mk_folder(cycle_cut_path)
    mk_folder(car_cut_path + 'trainA')
    mk_folder(car_cut_path + 'trainB')
    mk_folder(car_cut_path + 'testA')
    mk_folder(car_cut_path + 'testB')
    mk_folder(cycle_cut_path + 'trainA')
    mk_folder(cycle_cut_path + 'trainB')
    mk_folder(cycle_cut_path + 'testA')
    mk_folder(cycle_cut_path + 'testB')
    
    ### synthetic image setting (trainA = testA)
    for cls, cut_train_name in zip(data['cls'],sorted(glob.glob(path.crop.all+'*.jpg'))):
        name = cut_train_name.split('/')[-1]
        
        if cls == 0: # car
            shutil.copy2(cut_train_name,car_cut_path + 'trainA/'+name)
            shutil.copy2(cut_train_name,car_cut_path + 'testA/'+name)
        else: # cycle
            shutil.copy2(cut_train_name,cycle_cut_path + 'trainA/'+name)
            shutil.copy2(cut_train_name,cycle_cut_path + 'testA/'+name)
            
    ### Real image setting (trainB = testB)
    for real_cut in sorted(glob.glob(path.crop.test_car+'*.jpg')):
        name = real_cut.split('/')[-1]
        shutil.copy2(cut_name,car_cut_path + 'trainB/'+name)
        shutil.copy2(cut_name,car_cut_path + 'testB/'+name)
        
    for real_cut in sorted(glob.glob(path.crop.test_cycle+'*.jpg')):
        name = real_cut.split('/')[-1]
        shutil.copy2(cut_name,cycle_cut_path + 'trainB/'+name)
        shutil.copy2(cut_name,cycle_cut_path + 'testB/'+name)
    print('CUT folder Setting Complete!!')
def CUT(path,file_name,cls,mode):
    # path = yaml_file
    # file name = result folder

    checkpoint = path.weight.weight_path
    
    if cls == 'car':
        root = path.cut.car
    if cls == 'cycle':
        root = path.cut.cycle
    if mode == 'train':
        os.system(f'python CUT/train.py --dataroot {root} --name {file_name}--CUT_mode CUT --phase train --n_epochs 100 --batch 2 --gpu_ids 0,1 --preprocess resize --num_threads 0 --checkpoints_dir {checkpoint}')
    if mode == 'test':
        os.system(f'python CUT/test.py --dataroot {root} --name {file_name} --CUT_mode CUT --phase test --gpu_ids 0,1 --preprocess resize --num_threads 0 --num_test 10000000')
        ## directory arrangement
        for i in tqdm(sorted(glob.glob(f'results/{file_name}/test_latest/images/fakeB/*.png'))):
            name = i.split('/')[-1]
            shutil.move(i, path.cut.save_car+name)
    
def CUT_csv(path): # Synthetic to Real image csv
    csv_file = pd.read_csv(path.parent + 'train_image_crop.csv')
    new_csv = csv_file.copy()
    car_new_csv = new_csv[new_csv['cls'] == 0].reset_index(drop=True)
    cycle_new_csv = new_csv[new_csv['cls'] == 1].reset_index(drop=True)
    
    car_new_csv['File'] = sorted(glob.glob(path.cut.save_car))
    cycle_new_csv['File'] = sorted(glob.glob(path.cut.save_cycle))
    
    car_new_csv.to_csv(path.parent + 'cut_car_image.csv',index=False)
    cycle_new_csv.to_csv(path.parent + 'cut_cycle_image.csv',index=False)
    
    csv_file = csv_file.append(car_new_csv)
    csv_file = csv_file.append(cycle_new_csv)
    csv_file.reset_index(drop=True,inplace=True)
    csv_file.to_csv(path.parent+'merge.csv',index=False)
    
    print("New Csv file Generate")


# +
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process',type=str,default='all')
    yaml_file = OC.load('path.yaml')
    opt = parser.parse_args()
    
    
    if opt.process == 'all':
        yolo_preprocessing(yaml_file)
        cropping_train(yaml_file)
        yolo_train(yaml_file)
        cropping_test(yaml_file)
        cut_folder_setting(yaml_file)
        CUT(yaml_file,'car_cut','car','train') # training CUT model (car)
        CUT(yaml_file,'cycle_cut','cycle','train') # testing CUT model (cycle)
        CUT(yaml_file,'car_cut','car','test')
        CUT(yaml_file,'cycle_cut','cycle','test')
        CUT_csv(yaml_file)
        
        
        
    
