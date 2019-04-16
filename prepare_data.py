#!/usr/bin/env python
# coding: utf-8

# In[3]:


#run this in command line
# !python acai.py \
# --train_dir=TEMP \
# --latent=16 --latent_width=2 --depth=16 --dataset=fire_leaf


# In[1]:


import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm
import random
import os, re
from PIL import Image
from create_datasets import _encode_png, _save_as_tfrecord, _int64_feature, _bytes_feature  
from sklearn.model_selection import train_test_split
from acai import ACAI
from lib import data
import functools, math


# In[2]:


def get_images_list(input_dir):
    
    ''' 
    Returns the list of images in input_dir           
    '''   
    images_list = [os.path.join(input_dir, image_name) for image_name in os.listdir(input_dir)]
    return images_list
        


# In[40]:


def prepare_images(input_dir, output_dir, cropbox=(0, 0, 700, 600)):
    
    ''' 
    This function cuts the text at the bottom of noun project symbols 
    and covert the 3 channels to 1 
    
    kwargs:
    cropbox= (left, upper, right, lower) is the symbols frame
    boundry box, anything outside of this box is cut out           
    '''
    
    images_path = get_images_list(input_dir)
#     print(images_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for img in images_path:    
    #     (left, upper, right, lower)
        crop_box=cropbox
        original_img=Image.open(img)
        cropped_img=original_img.crop(crop_box)
        background=Image.new("RGB", cropped_img.size, (255, 255, 255))
        background.paste(cropped_img, mask=cropped_img.split()[3]) # 3 is the alpha channel
#     converting channel 3 to 1:
        converted_img = background.convert("L")
        converted_img.save(output_dir+'{}'.format(os.path.basename(os.path.normpath(img))))


# In[80]:


# prepare_images('./images/leaf-fire/', './images/leaf-fire_channel1/')


# In[3]:


def read_img(image_path, resizing_w_h):
    ''' 
    Reads and returens image in PIL.Image.Image format
    
    kwargs:

    resizing_w_h:images are resized to have the height and width of resizing_w_h   
    ''' 
    original_img = Image.open(image_path)
    resized_img = original_img.resize((resizing_w_h, resizing_w_h))
    return resized_img


def get_labels(image_path, image_type):
    
    ''' 
    Reads the image name and returns a number as its label
    
    kwargs:

    image_type: if font: it gets the font letter and returns a number between 1-28 
    '''
    
    str_label=re.search('_(.+?).png', os.path.basename(os.path.normpath(image_path))).group(1)
    
    if image_type=='font':
        return ord(str_label) - 64
        
    elif image_type=='symbol':
        
        if str_label =='fire':

            return 0
            
        elif str_label == 'leaf':

            return 1
        
def get_images_arr(input_dir, resizing_w_h, image_type='symbol'):
    
    ''' 
    Returns the images and their labels formatted in two numpy arrays with np.uint8 type
    '''
             
    imgdirs = get_images_list(input_dir)
    random.shuffle(imgdirs)
    
    images_arr=np.array([np.reshape(np.array(read_img(img, resizing_w_h), dtype=np.uint8),
                                     (resizing_w_h,resizing_w_h,1)) for img in imgdirs[:]])
    labels=[get_labels(img, image_type) for img in imgdirs[:]]

    labels_arr=np.array(labels, dtype=np.uint8)
    
    return images_arr, labels_arr


# In[39]:


def prepare_data(image_dir, resizing_w_h=28, test_size=.25):
    
    ''' 
    Gets the images piexls and their arrays, returns dictionaries of labels arrays and
    tensors of images pixels (ecnodes images arrays to tensorflow images)
    for both test and train sets

    kwargs:
    test_size: the portion images split for test set
    '''
    
    images, lables=get_images_arr(image_dir, resizing_w_h)
    
    X_train, X_test, y_train, y_test = train_test_split(images, lables,
                                                        test_size=test_size)

    test_set = {'images': X_train,
                'labels': y_train}
    train_set = {'images': X_test,
                'labels': y_test}

    train_set['images'] = _encode_png(train_set['images'])
    test_set['images'] = _encode_png(test_set['images'])
    
    return dict(train=train_set, test=test_set)


# In[92]:


LOADERS = [
    ('fire_leaf', prepare_data)
]


# In[94]:


# storing the data as as TFRecord (a sequence of binary strings)

for name, loader in LOADERS:
    print 'Preparing', name
    datas = loader('./images/leaf-fire_channel1/')
    for sub_name, data in datas.items():
        
         TFRecord file stores your data as 
        _save_as_tfrecord(data, '%s-%s' % (name, sub_name))


# In[ ]:




