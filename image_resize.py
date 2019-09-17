from PIL import Image
import os

dir1 = '/data1/LJH/paf_test/train_cnn/normal'
dir2 = '/data1/LJH/paf_test/train_target/normal'

all_img_list = os.listdir(dir1)
all_img_list.sort()

all_target_list = os.listdir(dir2)
all_target_list.sort()

for item_path in all_img_list:
    image = Image.open(item_path)
    resize_image = image.resize((640, 640))
    resize_image.save(item_path, "PNG", quality=95)

for item_path in all_target_list:
    image = Image.open(item_path)
    resize_image = image.resize((640, 640))
    resize_image.save(item_path, "PNG", quality=95)
