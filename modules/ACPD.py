
from keras import backend as K
from keras.layers import Layer
class ACPD(Layer):
    def __init__(self, fmap, **kwargs):
      
       self.filters=64 
       self.kernelsize=3 
       self.dilated_rates=1
       self.B = K.int_shape(fmap)[0]
       self.H = K.int_shape(fmap)[1]
       self.W = K.int_shape(fmap)[2]
       super(ACPD, self).__init__(**kwargs)

    def call(self,arg):
       fmap,y2=arg
       
       B,H,W,C = K.int_shape(fmap) #y2.get_shape().as_list()
       
       x= K.clip( tf.sort(y2[:,:,:,0:4],direction='DESCENDING',axis=-1)[:,:,:,0:2], 1, W-1  )
       y= K.clip( tf.sort(y2[:,:,:,4: ],direction='DESCENDING',axis=-1)[:,:,:,0:2], 1, H-1 )
       rx=self.dilated_rates*self.kernelsize # dilated rates of abscissa
       ry=self.dilated_rates*self.kernelsize # ditated rates of ordinate
      
       w= tf.reduce_sum(x, axis =-1, keepdims=True ) 
       h= tf.reduce_sum(y, axis =-1, keepdims=True ) 

       # create coordinates grid
       row= K.reshape(K.arange(0,H, dtype=tf.float32),[-1,1])
       row=  K.expand_dims( K.tile( row, [1,W]), -1)
       row = K.reshape(row,[H*W,1])

       column= K.reshape( K.arange(0,W, dtype=tf.float32), [1,-1])
       column=  K.expand_dims( K.tile( column, [H,1]), -1)
       column = K.reshape(column,[H*W,1])

       # get the dilated grid coordinates
       x_1=column-1-K.reshape(w/rx, [-1,H*W,1])
       x_0=column+K.reshape(w*1e-10, [-1,H*W,1])
       x_2=column+1+K.reshape(w/rx, [-1,H*W,1])
       x = tf.concat([x_1, x_0, x_2], axis=-1) 
       x = tf.reshape(x,[-1,H,W,1])
       x = K.tile(x,[1,1,1,self.kernelsize])

       y_1= row-1-K.reshape(h/ry, [-1,H*W,1])
       y_0=row+K.reshape(h*1e-10, [-1,H*W,1])
       y_2= row+1+ K.reshape(h/ry, [-1,H*W,1])
       y=tf.concat([y_1, y_0, y_2], axis=-1)
       y= tf.reshape(y,[-1,H,W,1])
       y = K.tile(y,[1,1,1,self.kernelsize])


       # create coordinates of interploation

       x_left=K.clip(tf.math.floor(x),0,W-1)
       x_right=K.clip(tf.math.ceil(x),0,W-1)
       y_top=K.clip(tf.math.floor(y),0,H-1)
       y_bottom=K.clip(tf.math.ceil(y),0,H-1)
       # calculate coordinates increment of interploation 
       x_right_x=K.tile( K.expand_dims( tf.reshape(x_right-x,[-1,W*H*self.kernelsize*self.kernelsize]),-1), [1,1,C])
       x_left_x=K.tile( K.expand_dims(tf.reshape(x-x_left,[-1,W*H*self.kernelsize*self.kernelsize]),-1), [1,1,C])
       y_top_y=K.tile( K.expand_dims(tf.reshape(y-y_top,[-1,W*H*self.kernelsize*self.kernelsize]),-1), [1,1,C])
       y_bottom_y=K.tile( K.expand_dims( tf.reshape(y_bottom-y,[-1,W*H*self.kernelsize*self.kernelsize]),-1), [1,1,C])


       # indices of interplolation
       left_top=tf.cast(tf.reshape(y_top*W+x_left, [-1,W*H*self.kernelsize*self.kernelsize]),dtype=tf.int32)
       right_top=tf.cast(tf.reshape(y_top*W+x_right, [-1,W*H*self.kernelsize*self.kernelsize]),dtype=tf.int32)
       left_bottom=tf.cast(tf.reshape(y_bottom*W+x_left, [-1,W*H*self.kernelsize*self.kernelsize]),dtype=tf.int32)
       right_bottom=tf.cast(tf.reshape(y_bottom*W+x_right, [-1,W*H*self.kernelsize*self.kernelsize]),dtype=tf.int32)
       # get feature value from interplation coordinates
       fmap=tf.reshape(fmap,[-1,W*H,C])
       f_left_top=tf.gather(fmap, left_top, batch_dims=1)
       f_right_top=tf.gather(fmap, right_top, batch_dims=1)
       f_left_bottom=tf.gather(fmap, left_bottom, batch_dims=1)
       f_right_bottom=tf.gather(fmap, right_bottom, batch_dims=1)
       # calculate bilinear interpolation
       f_resize=(x_right_x*f_left_top+
            x_left_x*f_left_bottom)*y_bottom_y +(x_right_x*f_right_top+
            x_left_x*f_right_bottom)*y_top_y
       f_resize=K.reshape(f_resize,[-1,H*self.kernelsize,W*self.kernelsize,C])
       
       f_resize= Conv2D(filters=self.filters, kernel_size= self.kernelsize, 
                         strides=(self.kernelsize,self.kernelsize), padding='valid', 
                         use_bias=False, kernel_initializer='he_normal', 
                         kernel_regularizer=l2(5e-4))(f_resize)
       f_resize = BatchNormalization()(f_resize)
       f_resize= Activation('relu')(f_resize)

       return f_resize

    def compute_output_shape(self, input_shape):
        return (self.B, self.H, self.W, self.filters )

#Test ACPD
    image=rasterio.open('/test3/image/241.TIF') #247 185 146 206
    image=np.moveaxis(image.read()[0:3],0,2)
    

    photo = tf.cast(np.expand_dims(image,
                                   axis=0),dtype=tf.float32)
    photo = Lambda(lambda x: tf.image.resize(x, (128, 128)))(photo)
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', 
                kernel_regularizer=l2(5e-4))(photo)
    x=y2
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = Conv2D(8, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    #f_resize = Deformoffset( [x, y2])
    acpd=ACPD(x )
    featuremap= acpd([x, y2])
    plt.imshow( K.eval(featuremap)[0][:,:,30])
