

from keras import backend as K
from keras.layers import Layer
class Spectralawaregate(Layer):
    def __init__(self,kernelsize,stride, fmapD,**kwargs):
        self.kernelsize=kernelsize
        self.stride=stride
        self.fmapD= fmapD
        super(Spectralawaregate, self).__init__(**kwargs)

    def call(self, arg):
   
        fmapD,fmapS=arg
        B,H,W,C=K.int_shape(fmapD)


        fmapS1=AveragePooling2D(pool_size=(self.kernelsize, self.kernelsize), 
              strides=self.stride, padding='same', data_format=None)(fmapS)
        fmapS2=MaxPooling2D(pool_size=(self.kernelsize, self.kernelsize), 
              strides=self.stride, padding='same', data_format=None)(fmapS)
        fmapS= layers.add([fmapS1, fmapS2])

        fmapS=Conv2D(filters=self.kernelsize**2, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal', 
            kernel_regularizer=l2(5e-4))( fmapS) 
        fmapS = BatchNormalization()(fmapS)
        fmapS = Activation('relu')(fmapS)             

        res1=K.reshape( fmapS, [-1,H*W*self.kernelsize*self.kernelsize,1])
        res2=tf.image.extract_patches(
				  images=fmapD, 
				  sizes=[1, self.kernelsize, self.kernelsize, 1], 
				  strides=[1, self.stride, self.stride, 1], 
				  rates=[1, 1, 1, 1], 
				  padding='SAME')
   
        res2=K.reshape(res2,[-1,H*W*self.kernelsize*self.kernelsize,C])
   
        refmap = K.reshape(res1*res2, [-1,self.kernelsize*H,self.kernelsize*W,C])
        refmap = Conv2D(C, (3, 3), strides=(3, 3), use_bias=False)(refmap)
        refmap = BatchNormalization()(refmap)
        refmap = Activation('relu')(refmap)  
        refmap = layers.add([refmap,fmapD])
        return refmap
    def compute_output_shape(self):

        B,H,W,C=K.int_shape(self.fmapD)
      
        return (B, H, W, C )
