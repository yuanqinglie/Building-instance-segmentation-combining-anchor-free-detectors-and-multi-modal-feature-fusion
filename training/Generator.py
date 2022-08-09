

import math
from random import shuffle

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from gaussian_radius



class Generator(object):
    def __init__(self,batch_size,train_lines,val_lines,
                input_size,num_classes,image_path,label_path,max_objects=100):
        
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.val_lines = val_lines
        self.input_size = input_size
        self.output_size = (int(input_size[0]/4) , int(input_size[1]/4))
        self.num_classes = num_classes
        self.max_objects = max_objects
        
    def get_box( self, line ):
       j=0
       box=np.zeros((len(line[1:]),5))
       for i in line[1:]:
          a =np.asarray(i.split(',')).astype(np.float32).reshape(1,5)[0]
          box[j]=[a[0],a[3],a[1],a[2],a[4]]
          j=j+1
       return box
    


    def get_random_data(self, image_path,label_path,annotation_line):

        line = annotation_line.split( )
        #image = Image.open(image_path+line[0])
        image=rasterio.open(image_path+line[0]+".TIF") #247 185 146 206
        image=np.moveaxis(image.read()[0:self.input_size[2]],0,2)
        img_label=rasterio.open(label_path+line[0]+".TIF")
        img_label=np.moveaxis(img_label.read(),0,2)[:,:,0]/255
        #image = np.uint8( (cv2.cvtColor(image[:,:,0:3], cv2.COLOR_BGR2RGB))*255 )
        box = self.get_box( line )

        return image,box,img_label

    def generate(self, train=True):
        while True:
            if train:
                # 打乱
                shuffle(self.train_lines)
                lines = self.train_lines
            else:
                shuffle(self.val_lines)
                lines = self.val_lines
                
            batch_images = np.zeros((self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2]), dtype=np.float32)
            batch_hms = np.zeros((self.batch_size, self.output_size[0], self.output_size[1], self.num_classes), dtype=np.float32)
            batch_whs = np.zeros((self.batch_size, self.max_objects, 2), dtype=np.float32)
            batch_regs = np.zeros((self.batch_size, self.max_objects, 2), dtype=np.float32)
            batch_reg_masks = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)
            batch_indices = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)
            #batch_masks = np.zeros((self.batch_size, self.input_size[0], self.input_size[1],self.max_objects), dtype=np.float32)
            batch_masks = np.zeros((self.batch_size,self.input_size[0], self.input_size[1],2))

            b = 0
            for annotation_line in lines: 


                img,y,img_label = self.get_random_data(image_path,label_path, annotation_line)


                if len(y)!=0:
                    boxes = np.array(y[:,:4],dtype=np.float32)
                    dataset= np.array(y[:,:4],dtype=np.float32)

                    boxes[:,0] = boxes[:,0]/self.input_size[1]*self.output_size[1]
                    boxes[:,1] = boxes[:,1]/self.input_size[0]*self.output_size[0]
                    boxes[:,2] = boxes[:,2]/self.input_size[1]*self.output_size[1]
                    boxes[:,3] = boxes[:,3]/self.input_size[0]*self.output_size[0]

                for i in range(len(y)):
                    bbox = boxes[i].copy()
                    bbox = np.array(bbox)
                    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_size[1] - 1)
                    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_size[0] - 1)
                    cls_id = int(y[i,-1])-1
                    
                    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    if h > 0 and w > 0:
                        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                        ct_int = ct.astype(np.int32)
                        
                        # Get heatmap
                        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                        radius = max(0, int(radius))
                        batch_hms[b, :, :, cls_id] = draw_gaussian(batch_hms[b, :, :, cls_id], ct_int, radius)
                        batch_whs[b, i] = 1. * w, 1. * h
                       
                       

                        # Compute centeroffsets
                        batch_regs[b, i] = ct - ct_int


                        # Mark index to 1 to remove 0 samples
                        batch_reg_masks[b, i] = 1
                        
                        batch_indices[b, i] = ct_int[1] * self.output_size[1] + ct_int[0]
                        """
                        # Generate masks
                        x1, y1, x2, y2 = dataset[i]

                        A= np.trunc(x1 ).astype(int)
                        B= np.trunc(y1 ).astype(int)
                        C= np.trunc(x2 ).astype(int)
                        D= np.trunc(y2 ).astype(int)
                        batch_masks[b,B:D+1,A:C+1,i]= img_label[B:D+1,A:C+1]                       
                        """
                
                
                batch_masks[b,: , : , 0 ] = (img_label == 1 ).astype(int)
                batch_masks[b,: , : , 1 ] = (img_label == 0 ).astype(int)                        
               
                  
                # origninal data BGR images
                batch_images[b] = img
                b = b + 1
                if b == self.batch_size:
                    b = 0
                    yield [batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices, batch_masks], np.zeros((self.batch_size,)) #             

                    batch_images = np.zeros((self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2]), dtype=np.float32)

                    batch_hms = np.zeros((self.batch_size, self.output_size[0], self.output_size[1], self.num_classes),
                                        dtype=np.float32)
                    batch_whs = np.zeros((self.batch_size, self.max_objects, 2), dtype=np.float32)
                    batch_regs = np.zeros((self.batch_size, self.max_objects, 2), dtype=np.float32)
                    batch_reg_masks = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)
                    batch_indices = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)
                    #batch_masks = np.zeros((self.batch_size, self.input_size[0], self.input_size[1],self.max_objects), dtype=np.float32)
                    batch_masks = np.zeros((self.batch_size,self.input_size[0], self.input_size[1], 2))
