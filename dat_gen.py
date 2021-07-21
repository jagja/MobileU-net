"""
Randomly samples smaller images from larger images
"""
import os
import glob
import random
import numpy as np
import cv2

base_dir = os.getcwd()
image_dir = os.path.join(base_dir, 'raw\\images\\')
mask_dir = os.path.join(base_dir, 'raw\\masks\\')

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
dim = (224,224)
w_in = 16
h_in = w_in

rgba = 4
frac_pos = 0.1

samples = 25
subsets = 64
size = 256
# Save predictions
count = 0

image_arr=np.zeros((size,size,rgba,samples))
mask_arr=np.zeros((size,size,rgba-1,samples))

mask_stat = np.zeros((samples,subsets))

count = 0
for image_path in glob.glob(image_dir+'*.png'):
    print(image_path)
    image_arr[:,:,:,count] = np.asarray(cv2.imread(image_path, cv2.IMREAD_UNCHANGED))
    mask_arr[:,:,:,count] = np.asarray(cv2.imread(mask_dir+image_path[-7:]))
    count+=1

for i in range(samples):
    for j in range(subsets):
        flag=0
        while flag==0:
            c_x = random.randint(w_in,size-w_in)
            c_y = random.randint(h_in,size-h_in)
            image_mini = image_arr[c_x-w_in:c_x+w_in, c_y-h_in:c_y+h_in, :, i]
            mask_mini = mask_arr[c_x-w_in:c_x+w_in, c_y-h_in:c_y+h_in, :, i]
            if (np.amin(image_mini[:,:,-1])>0)&(np.mean(mask_mini)>frac_pos*np.amax(mask_mini)):
                #image_out = cv2.filter2D(image_mini[:,:,:rgba-1],-1,filter)
                image_out = cv2.resize(image_mini[:,:,:rgba-1], dim, interpolation=cv2.INTER_CUBIC)
                mask_out = cv2.resize(mask_mini, dim, interpolation=cv2.INTER_CUBIC)
                mask_stat[i,j] = np.mean(mask_mini)
                flag=1
        mask_out = 255*np.uint8(mask_out>128)
        image_out = cv2.filter2D(image_out, -1, kernel)
        cv2.imwrite('image_'+str(i)+'_'+str(j)+'.png',image_out)
        cv2.imwrite('mask_'+str(i)+'_'+str(j)+'.png',mask_out)
