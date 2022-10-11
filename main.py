# +
import os
import shutil
from omegaconf import OmegaConf as OC
import pandas as pd
import argparse

from preprocessing import *
# from yolo_inference import detect_save

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
    

def cropping_train(path): ## make csv
    # path = yaml_file
    image_path = path.train.image
    label_path = path.train.label
    csv_save_path = path.parent
    crop_save_path = path.crop.all
    
    mk_folder(crop_save_path)
    csv_name = 'train_image_crop'
    csv_file_setting(image_path,label_path,crop_save_path,csv_save_path,csv_name)
    print('Train Dataset cropping Done!')
    
def cropping_test(path): 
    # path = yaml_file
    project_name = path.parent + 'results/'
    ## init
    if os.path.exists(project_name):
        shutil.rmtree(project_name)
    real_car_name = 'real_car_crop'
    real_cycle_name = 'real_cycle_crop'
    
    v1_real_image_path = path.test.v1
    v2_real_image_path = path.test.v2
    
    car_weight_path = path.weight.yolo_car_weight
    cycle_weight_path = path.weight.yolo_cycle_weight
    
    car_data = 'yolov5/data/car.yaml'
    cycle_data = 'yolov5/data/cycle.yaml'
    ### test set Cropping (car)
    os.system(f"python yolov5/detect.py --source {v1_real_image_path} --weights {car_weight_path} --project {project_name} --name {real_car_name} --save-crop --data {car_data}")
    os.system(f"python yolov5/detect.py --source {v2_real_image_path} --weights {car_weight_path} --project {project_name} --name {real_car_name+'_2'} --save-crop --data {car_data}")
    
    ### test set cropping (cycle)
    os.system(f"python yolov5/detect.py --source {v1_real_image_path} --weights {cycle_weight_path} --project {project_name} --name {real_cycle_name} --save-crop --data {cycle_data}")
    os.system(f"python yolov5/detect.py --source {v2_real_image_path} --weights {cycle_weight_path} --project {project_name} --name {real_cycle_name+'_2'} --save-crop --data {cycle_data}")
    
    ## Arrangement Folder
    mk_folder(path.crop.test_car)
    mk_folder(path.crop.test_cycle)
    
    for idx,real_jpg in tqdm(enumerate(sorted(glob.glob(project_name + real_car_name + '/crops/car/*.jpg')))):
        shutil.move(real_jpg, path.crop.test_car  + f'v1_{idx:07d}.jpg')
    for idx,real_jpg in tqdm(enumerate(sorted(glob.glob(project_name + real_car_name + '_2/crops/car/*.jpg')))):
        shutil.move(real_jpg, path.crop.test_car + f'v2_{idx:07d}.jpg')
    for idx,real_jpg in tqdm(enumerate(sorted(glob.glob(project_name + real_cycle_name + '/crops/car/*.jpg')))):
        shutil.move(real_jpg, path.crop.test_cycle + f'v1_{idx:07d}.jpg')
    for idx,real_jpg in tqdm(enumerate(sorted(glob.glob(project_name + real_cycle_name + '_2/crops/car/*.jpg')))):
        shutil.move(real_jpg, path.crop.test_cycle + f'v2_{idx:07d}.jpg')
    
    shutil.rmtree(project_name+real_car_name+'/'+'crops/car')
    shutil.rmtree(project_name+real_cycle_name+'/'+'crops/cycle')
    print('Test image Cropping!')
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
    for cls, cut_train_name in tqdm(zip(csv_file['Class'],sorted(glob.glob(path.crop.all+'*.jpg')))):
        name = cut_train_name.split('/')[-1]
        
        if cls == 0: # car
            shutil.copy2(cut_train_name,car_cut_path + 'trainA/'+name)
            shutil.copy2(cut_train_name,car_cut_path + 'testA/'+name)
        else: # cycle
            shutil.copy2(cut_train_name,cycle_cut_path + 'trainA/'+name)
            shutil.copy2(cut_train_name,cycle_cut_path + 'testA/'+name)
    print('A setting Done!')
    ### Real image setting (trainB = testB)
    for real_cut in tqdm(sorted(glob.glob(path.crop.test_car+'*.jpg'))):
        name = real_cut.split('/')[-1]
        shutil.copy2(real_cut,car_cut_path + 'trainB/'+name)
        shutil.copy2(real_cut,car_cut_path + 'testB/'+name)
        
    for real_cut in tqdm(sorted(glob.glob(path.crop.test_cycle+'*.jpg'))):
        name = real_cut.split('/')[-1]
        shutil.copy2(real_cut,cycle_cut_path + 'trainB/'+name)
        shutil.copy2(real_cut,cycle_cut_path + 'testB/'+name)
    print('CUT folder Setting Complete!!')
    
    print('B setting Done!')
    
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
        os.system(f'python CUT/test.py --dataroot {root} --name {checkpoint+file_name} --CUT_mode CUT --phase test --preprocess resize --num_threads 0 --num_test 10000000')
        ## directory arrangement
        if cls == 'car':
            for i in tqdm(sorted(glob.glob(f'{checkpoint+file_name}/test_latest/images/fake_B/*.png'))):
                name = i.split('/')[-1]
                shutil.move(i, path.cut.save_car+name)
        if cls == 'cycle':
            for i in tqdm(sorted(glob.glob(f'{checkpoint+file_name}/test_latest/images/fake_B/*.png'))):
                name = i.split('/')[-1]
                shutil.move(i, path.cut.save_cycle+name)
    
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
    csv_file.to_csv(path.parent+'total.csv',index=False)
    
    print("New Csv file Generate")
    


# +
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process',type=str,default='all')
    yaml_file = OC.load('path.yaml')
    opt = parser.parse_args()
    
    print('---Folder Setting---')
    yolo_preprocessing(yaml_file) # arrangement folder for yolo
    cropping_train(yaml_file) # csv file setting & cropped image save
    print('--Done--')
    
    if opt.process == 'all':
        ## yolo setting
        yolo_train(yaml_file) # training yolo
        cropping_test(yaml_file) # cropping
        
        ## cut setting
        cut_folder_setting(yaml_file)
        CUT(yaml_file,'car_cut','car','train') # training CUT model (car)
        CUT(yaml_file,'cycle_cut','cycle','train') # training CUT model (cycle)
        CUT(yaml_file,'car_cut','car','test') # testing CUT model (cycle)
        CUT(yaml_file,'cycle_cut','cycle','test') # testing CUT model (cycle)
        CUT_csv(yaml_file) # final_csv file setting
        
        ## create submission file (Do not categorize labels)
        os.system('python yolo_inference.py')
        os.system('python classification_train.py')
        ## final inference
        os.system('python classification_inference.py')
        
    if opt.process == 'train':
        yolo_train(yaml_file) # training yolo
        cropping_test(yaml_file) # cropping
        
        ## cut setting
        cut_folder_setting(yaml_file)
        CUT(yaml_file,'cut_car','car','train') # training CUT model (car)
        CUT(yaml_file,'cut_cycle','cycle','train') # training CUT model (cycle)
        CUT_csv(yaml_file) # final_csv file setting
        os.system('python classification_train.py')
        
    if opt.process == 'infer': # using pretrained model
        print('Please Check weight file!!')
        cropping_test(yaml_file) # cropping
        cut_folder_setting(yaml_file)
        CUT(yaml_file,'cut_car','car','test') # testing CUT model (car)
        CUT(yaml_file,'cut_cycle','cycle','test') # testing CUT model (cycle)
        os.system('python yolo_inference.py') # create submission file 
        os.system('python classification_inference.py')
       
        
