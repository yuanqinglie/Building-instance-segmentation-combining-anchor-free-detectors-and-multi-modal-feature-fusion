

#Test modules
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
    y2 = Conv2D(64, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    plt.subplot(1, 3, 1)
    plt.title("Depthawaregate")

    depare=Depthawaregate(x)
    fmaps=depare( [x, y2] )
    plt.imshow( K.eval(fmaps)[0][:,:,2])
    
    plt.subplot(1, 3, 2)
    plt.title("Spectralawaregate")
    specaware=Spectralawaregate( 3,1,x)
    fmaps=specaware( [x, y2])

    plt.imshow( K.eval(fmaps)[0][:,:,2])

    plt.subplot(1, 3, 3)
    plt.title("non_localgate")
    nonlocalg=non_localgate( [2,4,6,8],x)
    fmaps= nonlocalg([x, y2] )
    plt.imshow( K.eval(fmaps)[0][:,:,2])
