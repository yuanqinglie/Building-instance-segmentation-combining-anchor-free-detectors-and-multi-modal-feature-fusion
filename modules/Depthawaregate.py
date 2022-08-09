

from keras import backend as K
from keras.layers import Layer
class Depthawaregate(Layer):

    def __init__(self,fmapS,**kwargs):

       self.kernelsize=3
       self.stride=1
       self.a=-2
       self.fmapS=fmapS      
       super(Depthawaregate, self).__init__(**kwargs)

    def spatial_attention(self, channel_refined_feature):
          maxpool_spatial = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(channel_refined_feature)
          avgpool_spatial = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(channel_refined_feature)
          max_avg_pool_spatial = tf.concat([maxpool_spatial, avgpool_spatial],axis=-1)
          return Conv2D(filters=1, kernel_size=(5, 5), padding="same", 
              activation=None, kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)

    def call(self,arg):
 
       fmapD,fmapS=arg
       fmapD=self.spatial_attention(fmapD )
       B,H,W,C=K.int_shape(fmapS)
       
       res1=tf.image.extract_patches(
				images=fmapD, 
				sizes=[1, self.kernelsize, self.kernelsize, 1], 
				strides=[1, self.stride, self.stride, 1], 
				rates=[1, 1, 1, 1], 
				padding='SAME')
       res2=tf.image.extract_patches(
				images=fmapS, 
				sizes=[1, self.kernelsize, self.kernelsize, 1], 
				strides=[1, self.stride, self.stride, 1], 
				rates=[1, 1, 1, 1], 
				padding='SAME') 
       res2=K.reshape(res2,[-1,self.kernelsize*H,self.kernelsize*W,C])
    
       res1=K.reshape( res1,[-1,H*W*self.kernelsize*self.kernelsize])
       res1_=K.reshape(K.tile(fmapD,[1,1,1,self.kernelsize**2] ), [-1,H*W*self.kernelsize*self.kernelsize])
       depthaware=tf.reshape( tf.math.exp( self.a*tf.abs(res1-res1_) ), [-1,self.kernelsize*H,self.kernelsize*W,1] )
       refmap= res2*depthaware

       refmap = Conv2D(C, (3, 3), strides=(self.kernelsize, self.kernelsize), use_bias=False)(refmap)
       refmap = BatchNormalization()(refmap)
       refmap = Activation('relu')(refmap)  
       refmap = layers.add([refmap,fmapS])
       return refmap


    def compute_output_shape(self):
        B,H,W,C=K.int_shape(self.fmapS)
        return (B, H, W, C )
