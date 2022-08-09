

class non_localgate(Layer):
     def __init__(self,pool_factors,fmapS,**kwargs):

        self.fmapS= fmapS
        self.pool_factors=pool_factors
        super(non_localgate, self).__init__(**kwargs)

     def pool_pyramid(self,feats,H,W, pool_factor,out_channel):
       pool_outs=[]

       for p in pool_factor:
      
          pool_size = strides = [int(np.round(float(H)/p)),int(np.round(float(W)/p))]
          pooled = AveragePooling2D(pool_size=pool_size,strides=strides,padding='same')(feats)
          pooled = tf.image.resize(pooled,(p,p))    
          pooled = K.reshape(pooled,[-1,p**2,out_channel] )
          pool_outs.append(pooled)

       pool_outs= tf.concat(pool_outs,axis=1)
       return pool_outs

     def call(self, arg ):
   
        fmapD,fmapS=arg
        B,H,W,C_=K.int_shape(fmapS)
        C=C_//4
        kernelsize=3
        stride=1
        pool_factors= [2,3,6,8]
   
        # B,H,W,C_=====> B,S,C =====>B,C,S
        fmapD1=Conv2D(C, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal', 
            kernel_regularizer=l2(5e-4))( fmapD) 
        fmapD1=self.pool_pyramid(feats=fmapD1,H=H,W=W, pool_factor=self.pool_factors, out_channel=C)
        fmapD1_=tf.transpose(fmapD1,[0,2,1] )
   
        # B,H,W,C_=====> B,HW,C 
        fmapS1=Conv2D(C, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal', 
            kernel_regularizer=l2(5e-4))( fmapS) 
        fmapS1=K.reshape(fmapS1,[-1,H*W,C])

        # [B,HW,C] * [B,C,S]=====>[B,HW,S]
        non_localS = tf.matmul(fmapS1,fmapD1_)
        non_localS = Softmax(axis=-1)(non_localS)


        # [B,HW,S]* [B,S,C]=====>[B,H,W,C]
        fS=K.reshape( tf.matmul(non_localS,fmapD1), [-1,H,W,C])
        fS=Conv2D(C_, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal', 
            kernel_regularizer=l2(5e-4))( fS) 
        fS = BatchNormalization()(fS)       
        fS=layers.add( [ fS,fmapS] )
        return fS 

     def compute_output_shape(self):

        B,H,W,C=K.int_shape(self.fmapS)
      
        return (B, H, W, C )
