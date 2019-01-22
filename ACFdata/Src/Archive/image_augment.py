# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image,ImageEnhance
import cv2
from diroperation import mkdir




def image_augment(filename,angle=None,luminance=None,Contrast=None):
    #img = Image.open(filename)
    img = cv2.imread(filename)
    name = filename.split('_')
    path = os.getcwd()
    currentdir = path.split('\\')    
    paraent_path = os.path.dirname(path)    
    if angle is not None:
        ropath = os.path.join(paraent_path,currentdir[-1]+'_rotate')
        rodirsuccess = mkdir(ropath)
        for a in angle:
#            img = img.rotate(a)
            rows,cols,channels = img.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),a,1)
            dst = cv2.warpAffine(img,M,(cols,rows))
            rotation_name = name[0]+'_ro'+str(a)
            for n in name[1:]:
                rotation_name+=('_'+n)         
                save_path = os.path.join(ropath,rotation_name)
            try:
#                img.save(save_path)
                cv2.imwrite(save_path,dst)
            except IOError:
                print("cannot rotation")
    else:
        #print("Don't do any rotation")
        pass
    
    if luminance is not None:
        lupath = os.path.join(paraent_path,currentdir[-1]+'_luminance')
        ludirsuccess = mkdir(lupath)
        for l in luminance:
            img = ImageEnhance.Brightness(img).enhance(l)
            luminance_name = name[0]+'_lu'+str(l)
            for n in name[1:]:
                luminance_name+=('_'+n)         
                save_path = os.path.join(lupath,luminance_name)
            try:
                img.save(save_path)
            except IOError:
                print("cannot enhance luminance")
    else:
        #print("Don't do any lumiance enhancement")
        pass
        
    if Contrast is not None:
        copath = os.path.join(paraent_path,currentdir[-1]+'_contrast')
        codirsuccess = mkdir(copath)
        for C in Contrast:
            img = ImageEnhance.Contrast(img).enhance(C)
            contrast_name = name[0]+'_co'+str(C)
            for n in name[1:]:
                contrast_name+=('_'+n)         
                save_path = os.path.join(copath,contrast_name)
            try:
                img.save(save_path)
            except IOError:
                print("cannot enhance contrast")
    else:
        #print("Don't do any contrast enhancement")
        pass
        
    
    
    
file_dir = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_database\\dataset1\\IR_train7'
file_dir1 = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_database\\dataset1\\IR_train7_rotate'
file_dir2 = 'D:\\pythonworkspace\\TensorflowTraining\\exercises\\Shen\\Practice\\ACFdata\\IR_database\\dataset1\\IR_train7_rotate_luminance'
os.chdir(file_dir)

angle = [i for i in range(360)[::10]]
luminance = [i/10 for i in range(20)[1:][::5][1:]]
print(luminance)

print(angle)
#
for file in os.listdir(file_dir):
    image_augment(file,angle=angle)

#os.chdir(file_dir1)
#
#for file in os.listdir(file_dir1):
#    image_augment(file,angle=None,luminance=luminance)
#
#os.chdir(file_dir2)
#
#for file in os.listdir(file_dir2):
#    image_augment(file,angle=None,luminance=None,Contrast=luminance)

