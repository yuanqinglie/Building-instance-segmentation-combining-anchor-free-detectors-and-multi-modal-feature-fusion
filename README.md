# Building-instance-segmentation

 This is a building instance segmentation network combining multi-task learning and multi-modal remote sensing data including lidar features and optical image features. The backbone network is input with multispectral images based on resnet50, and the branch network is input with lidar products based on resnet18. In the model file, the depth sensing layer and the spectral sensing layer are embedded into the residual network structure to fuse the multi-modal features. The non local layer builds a cross level global context model. ACPD is a detector based on cenernet, which introduces deformation convolution operation in supervised learning.

![image](https://user-images.githubusercontent.com/15941731/183735499-82258816-ba97-4853-9bdf-06da5c215077.png)
