
def Mean_WH(y2):
  x=tf.sort(y2[:,:,:,0:4],direction='DESCENDING',axis=-1)[:,:,:,0:2]
  x=tf.expand_dims(tf.math.reduce_mean(x,axis=-1 ), axis=-1)
  y=tf.sort(y2[:,:,:,4: ],direction='DESCENDING',axis=-1)[:,:,:,0:2]
  y=tf.expand_dims(tf.math.reduce_mean(y,axis=-1 ),axis=-1)
  y2_=tf.concat([x, y], axis=-1) 
  return  y2_

def centernet_head(image_input,num_classes):
    #-------------------------------#
    #   Decoder
    #-------------------------------#

    C5,C4,C3,C2 = ResNet50(image_input)
   
    # Feature fusion Pyramid
    x=Conv2D(1024, 1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))( C5)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x=Lambda(lambda x: tf.image.resize(x, (32, 32)))(x)
    C4=layers.add([C4, x])

    x=Conv2D(512, 1, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))( C4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x=Lambda(lambda x: tf.image.resize(x, (64, 64)))(x)
    C3=layers.add([C3, x])


    nonlocalg=non_localgate( [2,4,6,8],C2)
    x= nonlocalg([C4, C2] )
    C2=layers.add([C2, x])

    x = Dropout(rate=0.5)(C2)

    """
    num_filters = 256
    # 16, 16, 2048  ->  32, 32, 256 -> 64, 64, 128 -> 128, 128, 64

    for i in range(3):
        # upsampling
        x = Conv2DTranspose(num_filters // pow(2, i), (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    """

    # Get the features from 128,128,64 layers
    # wh header
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = Conv2D(8, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2) 
    y2_= Lambda(Mean_WH, name='Mean_WH')(y2)

 

    # hm header enhance
    
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)

    #---------------#
    #get_roipooling=DeformROIPooling( filters=64, pooled_height=3, pooled_width=3 )
    #y1_=get_roipooling([y1,y2]) 
    #---------------#
    #y1_= Lambda( Deformoffset, name='Deformoffset')( [ y1, y2]) 
    acpd=ACPD(y1 )
    y1_= acpd([y1, y2])
    fusion = layers.add([y1, y1_]) #
    
    y1 = Conv2D(32, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(fusion)#fusion
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(num_classes, 1,padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)
    
    # wh rectification
    y2_re = Conv2D(2, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(fusion)
    y2  = layers.add([y2_,y2_re]) #
    #y2=y2_


    # reg header
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)


    # binary classification
    size_before4 = tf.keras.backend.int_shape(image_input)
    featurere = Lambda(lambda xx:tf.image.resize(xx,size_before4[1:3]))(x) #resize_images
    featurere = Conv2D(2, 3, padding='same', use_bias=False, 
                          kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(featurere)                       
    featurere= Reshape((-1,2))(featurere)                       
    featurere = Softmax(axis=-1)(featurere)
    """
    #  Prototype mask
    x_ = CoordinateChannel2D()(x)
    mask_coeff = Conv2D(100, 3, padding='same',activation='tanh', use_bias=False, 
                          kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x_)      
    featurere = Conv2D(100, 3, padding='same',activation='relu', use_bias=False, 
                          kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x_)                       
    featurere= Reshape((K.int_shape(featurere)[1]*K.int_shape(featurere)[2],100))(featurere)                       
    featurere = Softmax(axis=1)(featurere)
    """
    return y1, y2, y3, featurere
