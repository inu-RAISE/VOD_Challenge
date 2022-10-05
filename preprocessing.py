import os
import shutil
from tqdm import tqdm
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]
def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]

def mk_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def checking_folder(path):
    if path[-1] != '/':
        path += '/'
    return path

def yolo_VOC_setting(image_input_dir,label_input_dir,image_output_dir,label_output_dir,class_):
    '''
    image_input_dir = original VOC image directory path (e.g VOC2012/JPEGImages)
    label_input_dir = original VOC label directory path (e.g VOC2012/Annotation)
    image_output_dir = new VOC image directory path
    label_output_dir = new VOC label directory path
    class_ = 'car' or 'motorbike'
    '''
    ## folder setting
    mk_folder(image_output_dir)
    mk_folder(label_output_dir)
    ## check folder name format
    image_input_dir = checking_folder(image_input_dir)
    label_input_dir = checking_folder(label_input_dir)
    image_output_dir = checking_folder(image_output_dir)
    label_output_dir = checking_folder(label_output_dir)
    

    files = glob.glob(os.path.join(label_input_dir, '*.xml'))
    classes = [class_]

    
    for fil in tqdm(files):
        filename = fil.split('/')[-1].replace('.xml','')
        result = []
        tree = ET.parse(fil)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)
        for obj in root.findall('object'):
            label = obj.find("name").text
            if label not in classes:
                pass
            else:
                index = classes.index(label)  # 0       
                pil_bbox = [int(x.text) for x in obj.find("bndbox")]
                yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
                bbox_string = " ".join([str(x) for x in yolo_bbox])
                result.append(f"{index} {bbox_string}")

        if result:
            with open(os.path.join(label_output_dir, f"voc_{filename}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))
            shutil.copy2(image_input_dir+f"{filename}.jpg",image_output_dir+f"voc_{filename}.jpg")
    
    print(f'Complete VOC voc_{class_}Folder Setting!')

def yolo_vehicle_folder_setting(image_input_dir,label_input_dir,image_output_dir,label_output_dir,class_):
    
    '''
    image_input_dir = Provided data image directory path (e.g /data/IEEE_BigData/train_001/images/)
    label_input_dir = Provided data label directory path (e.g /data/IEEE_BigData/train_001/labels/)
    image_output_dir = new Provided data image directory path
    label_output_dir = new Provided data label directory path
    class_ = 'car' or 'motorbike'
    '''
    
    ## folder setting
    mk_folder(image_output_dir)
    mk_folder(label_output_dir)
        
    ## check folder name format
    image_input_dir = checking_folder(image_input_dir)
    label_input_dir = checking_folder(label_input_dir)
    image_output_dir = checking_folder(image_output_dir)
    label_output_dir = checking_folder(label_output_dir)
    
    
    files = glob.glob(os.path.join(label_input_dir,'*.txt'))
    for i in tqdm(files):
        with open(i,'r') as f:

            ls = f.readlines()
            cls = [i.split()[0] for i in ls]
            bbox = [i.split()[1:] for i in ls]
            bbox = sum(bbox, [])
            check = False
            name = i.split('/')[-1] # label file name (.txt)
            img_name = name.replace('.txt','.jpg') # image file name ('.jpg')
            if class_ == 'car':
                cls = list(map(lambda x: '0' if int(x)<=6 else x,cls))
                for c in range(len(cls)):
                    if int(cls[c]) <=6:
                        check = True
                        name = i.split('/')[-1]
                        with open(f'{label_output_dir}{name}','a') as t:
                            t.write(f'{cls[c]} {bbox[c:c+4][0]} {bbox[c:c+4][1]} {bbox[c:c+4][2]} {bbox[c:c+4][3]}\n')
                if check:
                    shutil.copy2(image_input_dir+img_name, image_output_dir+img_name)
            if class_ == 'motorbike':
                cls = list(map(lambda x: '0' if int(x)>6 else x,cls))
                for c in range(len(cls)):
                    if int(cls[c]) <=6:
                        check = True
                        with open(f'{label_output_dir}{name}','a') as t:
                            t.write(f'{cls[c]} {bbox[c:c+4][0]} {bbox[c:c+4][1]} {bbox[c:c+4][2]} {bbox[c:c+4][3]}\n')
                if check:
                    shutil.copy2(image_input_dir+img_name, image_output_dir+img_name)
    print(f'Complete Training {class_} Data Setting!')


def yolo_folder_setting(path,image_path,label_path,class_):
    '''
    path = image label path
    class_ = car or cycle
    
    class_folder
    | train
    |____ images
    |____ labels
    | val
    |____ images
    |____ labels
    
    '''
    path = checking_folder(path)
    image_path = checking_folder(image_path)
    label_path = checking_folder(label_path)
    
    new_path = path + class_ + '/'
    if not os.path.exists(new_path):
        os.mkdir(new_path)
        os.mkdir(new_path + 'train')
        os.mkdir(new_path + 'train/images')
        os.mkdir(new_path + 'train/labels')
        os.mkdir(new_path + 'val')
        os.mkdir(new_path + 'val/images')
        os.mkdir(new_path + 'val/labels')
        
    train_label,valid_label = train_test_split(os.listdir(label_path),test_size=.2,random_state = 42)
    
    train_image = list(map(lambda x: x.replace('labels','images'),train_label))
    train_image = list(map(lambda x: x.replace('.txt','.jpg'),train_image))
    
    valid_image = list(map(lambda x: x.replace('labels','images'),valid_label))
    valid_image = list(map(lambda x: x.replace('.txt','.jpg'),valid_image))
    
    for img,lab in zip(train_image,train_label):
        shutil.move(image_path + img,new_path+'train/images/'+img)
        shutil.move(label_path + lab,new_path+'train/labels/'+lab)
    
    for img,lab in zip(valid_image,valid_label):
        shutil.move(image_path + img,new_path+'val/images/'+img)
        shutil.move(label_path + lab,new_path+'val/labels/'+lab)
    print(f'YOLO folder setting Done!')


if __name__ == '__main__':
    
    parent_path = '/data/IEEE_BigData/sub'
    voc_car_path = parent_path+'/'+'voc_car/'
    voc_cycle_path = parent_path+'/'+'voc_cycle/'
    train_car_path = parent_path+'/'+'train_car/'
    train_cycle_path = parent_path+'/'+'train_cycle/'
    
    mk_folder(parent_path)  
    mk_folder(voc_car_path)
    mk_folder(voc_cycle_path)
    mk_folder(train_car_path)
    mk_folder(train_cycle_path)
    
    voc_input_label_dir = '/data/IEEE_BigData/VOC2012/Annotations/'
    voc_input_image_dir = '/data/IEEE_BigData/VOC2012/JPEGImages/'
    car_voc_output_label_dir = f'{voc_car_path}labels' # car image
    car_voc_output_image_dir = f'{voc_car_path}images' # car label
    cycle_voc_output_label_dir = f'{voc_cycle_path}labels'
    cycle_voc_output_image_dir = f'{voc_cycle_path}images'
    
    yolo_VOC_setting(voc_input_image_dir,voc_input_label_dir,car_voc_output_image_dir,car_voc_output_label_dir,'car') # car setting
    yolo_VOC_setting(voc_input_image_dir,voc_input_label_dir,cycle_voc_output_image_dir,cycle_voc_output_label_dir,'motorbike') # cycle setting
    
    
    input_label_dir = '/data/IEEE_BigData/train_001/labels/'
    input_image_dir = '/data/IEEE_BigData/train_001/images/'
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
