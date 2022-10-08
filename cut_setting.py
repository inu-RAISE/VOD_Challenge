import os
import shutil
from preprocessing import mk_folder
import glob

### train set cropping setting
source_path = '/data/IEEE_BigData/train_001/images'
car_weight_path = '/data/IEEE_BigData/sub/weights/car_best.pt' # file move
cycle_weight_path = '/data/IEEE_BigData/sub/weights/cycle_best.pt'


project_name = 'data/IEEE_BigData/sub/crop/'

car_name = 'car_crop' # training crop image (car)
cycle_name = 'cycle_detect' # training crop image (cycle)


### train set cropping
os.system(f"python detect.py --source {source_path} --weights {car_weight_path} --project {project_name} --name {car_name} --save-crop")
for idx,jpg in enumerate(sorted(glob.glob(project_name + car_name + '/'+'car/crops/*.jpg'))):
    shutil.move(jpg , project_name + 'car_crop/'+f'{idx:07d}.jpg')


os.system(f"python detect.py --source {source_path} --weights {cycle_weight_path} --project {project_name} --name {cycle_name} --save-crop")
for jpg in sorted(glob.glob(project_name + cycle_name + '/'+'cycle/crops/*.jpg')):
    shutil.move(jpg , project_name + 'cycle_crop/'+f'{idx:07d}.jpg')

os.remove(project_name+car_name+'car')
os.remove(project_name+cycle_name+'cycle')

### test set cropping setting (Real Image)

v1_real_image_path = '/data/IEEE_BigData/test/test_v1/'
v2_real_image_path = '/data/IEEE_BigData/test/test_v2/'

real_car_name = 'real_car_crop' 
real_cycle_name = 'real_cycle_crop'

### test set cropping (car)
os.system(f"python detect.py --source {v1_real_image_path} --weights {car_weight_path} --project {project_name} --name {real_car_name} --save-crop")
os.system(f"python detect.py --source {v2_real_image_path} --weights {car_weight_path} --project {project_name} --name {real_car_name+'_2'} --save-crop")

### test set cropping (cycle)
os.system(f"python detect.py --source {v1_real_image_path} --weights {cycle_weight_path} --project {project_name} --name {real_cycle_name} --save-crop")
os.system(f"python detect.py --source {v2_real_image_path} --weights {cycle_weight_path} --project {project_name} --name {real_cycle_name+'_2'} --save-crop")

for idx,real_jpg in enumerate(sorted(glob.glob(project_name + real_car_name + '/*.jpg'))):
    shutil.move(real_jpg, project_name + real_car_name + f'v1_{idx:07d}')

for idx,real_jpg in enumerate(sorted(glob.glob(project_name + real_car_name + '_2/*.jpg'))):
    shutil.move(real_jpg, project_name + real_car_name + f'v2_{idx:07d}')

for idx,real_jpg in enumerate(sorted(glob.glob(project_name + real_cycle_name + '/*.jpg'))):
    shutil.move(real_jpg, project_name + real_cycle_name + f'v1_{idx:07d}')

for idx,real_jpg in enumerate(sorted(glob.glob(project_name + real_cycle_name + '_2/*.jpg'))):
    shutil.move(real_jpg, project_name + real_cycle_name + f'v2_{idx:07d}')

os.remove(project_name+real_car_name+'car')
os.remove(project_name+real_cycle_name+'cycle')

### Cut folder setting

car_cut_path = '/data/IEEE_BigData/sub/cut_car/'
cycle_cut_path = '/data/IEEE_BigData/sub/cut_cycle/'
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
for cut_car_name in sorted(glob.glob(project_name + 'car_crop/*.jpg')):
    name = cut_car_name.split('/')[-1]
    shutil.copy2(cut_car_name,car_cut_path + 'trainA/'+name)
    shutil.copy2(cut_car_name,car_cut_path + 'testA/'+name)

### Real image setting (trainA = testA)
for  real_cut_name in sorted(glob.glob(project_name + 'real_car_crop/*.jpg')):
    name = real_cut_name.split('/')[-1]
    shutil.copy2(real_cut_name,car_cut_path + 'trainB/'+name)
    shutil.copy2(real_cut_name,car_cut_path + 'testB/'+name)
