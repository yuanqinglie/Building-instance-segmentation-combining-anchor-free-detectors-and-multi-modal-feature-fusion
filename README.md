# Building-instance-segmentation

 This is a building instance segmentation network combining multi-task deep learning and multi-modal remote sensing data including LiDAR features and optical image features. The backbone network is fed  with multispectral images based on resnet50, and the branch network is input with LiDAR products based on resnet18. In the module files, the Depthawaregate layer and Spectralawaregate layer are embedded into the residual network structure to fuse the multi-modal features. The non_localgate layer construct a cross level global context model. ACPD is a detector based on CenerNet, which introduces deformation convolution operation in supervised learning.
 
For method details, please refer to the paper "Building instance segmentation network combining anchor-free detector and multi modal feature fusion"

![image](https://user-images.githubusercontent.com/15941731/183735499-82258816-ba97-4853-9bdf-06da5c215077.png)

The metadata information  follows as table:

![metadata information](https://user-images.githubusercontent.com/15941731/183740607-24427d53-6b9d-4295-b9f9-51df8f1df82f.jpg)
![image](https://user-images.githubusercontent.com/15941731/183751773-a3bc4f2b-e411-4cb0-a6c7-0ba9a522a9da.png)



We established a building instance segmentation dataset (BISM) using multimodal remote sensing data. The BISM dataset covers 60 square kilometers in Boston, Massachusetts, USA, and contains about 39527 building objects. BISM can be downloaded via http://bismdataset.mikecrm.com/Yc5qJZD

