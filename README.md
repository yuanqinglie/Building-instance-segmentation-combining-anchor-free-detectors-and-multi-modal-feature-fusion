# Building-instance-segmentation

 This is a building instance segmentation network combining multi-task deep learning and multi-modal remote sensing data including LiDAR features and optical image features. The backbone network is fed  with multispectral images based on resnet50, and the branch network is input with LiDAR products based on resnet18. In the module files, the Depthawaregate layer and Spectralawaregate layer are embedded into the residual network structure to fuse the multi-modal features. The non_localgate layer construct a cross level global context model. ACPD is a detector based on CenerNet, which introduces deformation convolution operation in supervised learning.
 
For method details, please refer to the paper "Building instance segmentation network combining anchor-free detector and multi modal feature fusion". The code is completed by Tensorflow-Keras 2.0. Rasterio / GDAL require installation to load data.

![image](https://user-images.githubusercontent.com/15941731/183735499-82258816-ba97-4853-9bdf-06da5c215077.png)

The metadata information  follows as table:

![metadata information](https://user-images.githubusercontent.com/15941731/183740607-24427d53-6b9d-4295-b9f9-51df8f1df82f.jpg)
LiDAR/image ![image](https://user-images.githubusercontent.com/15941731/183751773-a3bc4f2b-e411-4cb0-a6c7-0ba9a522a9da.png)  ![image](https://user-images.githubusercontent.com/15941731/183759807-768f595f-91e5-4a70-b8e9-450bb7b95bdf.png) NDVI ![image](https://user-images.githubusercontent.com/15941731/183759907-ec2f2790-2f38-4940-bb91-f34c2067bbfd.png) DEM ![image](https://user-images.githubusercontent.com/15941731/183760246-e57dbba4-27bd-4482-b6e1-f2c6595210f5.png) Ground-truth






We established a building instance segmentation dataset (BISM) using multimodal remote sensing data. The BISM dataset covers 60 square kilometers in Boston, Massachusetts, USA, and contains about 39527 building objects. BISM can be downloaded via http://bismdataset.mikecrm.com/Yc5qJZD

