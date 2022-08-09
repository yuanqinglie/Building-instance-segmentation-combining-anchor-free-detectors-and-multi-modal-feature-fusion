

# model loss 
def focal_loss(hm_pred, hm_true):
    #-------------------------------------------------------------------------#
    # find PT and PN in each image

    #-------------------------------------------------------------------------#
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    #-------------------------------------------------------------------------#

    neg_weights = tf.pow(1 - hm_true, 4)

    #-------------------------------------------------------------------------#
    #   Focal loss
    #-------------------------------------------------------------------------#
    pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-6, 1.)) * tf.pow(1 - hm_pred, 2) * pos_mask
    neg_loss = -tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-6, 1.)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    #-------------------------------------------------------------------------#
    #   normalization
    #-------------------------------------------------------------------------#
    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss


def reg_l1_loss(y_pred, y_true, indices, mask):
    #b = tf.shape(y_pred)[0]

    #k = tf.shape(indices)[1]
    #c = tf.shape(y_pred)[-1]
    #y_pred = tf.reshape(y_pred, (b, -1, c))
    y_pred = tf.reshape(y_pred, (-1, K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]))

    indices = tf.cast(indices, tf.int32)
    y_pred = tf.gather(y_pred, indices, batch_dims=1)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
    return reg_loss

def binary_crossentropy(y_true, y_pred):
    size = tf.keras.backend.int_shape(y_true)
    y_true = tf.reshape( y_true, [-1,size[1]*size[2],2] )
    loss = K.binary_crossentropy(y_true,y_pred) #K.categorical_crossentropy(y_true,y_pred)
    loss=K.mean(loss)
    return loss

def mask_crossentropy(y_true, y_pred,mask_coeff,indices,mask):
    # resize y_true
    size = tf.keras.backend.int_shape(mask_coeff)
    y_true = Lambda(lambda xx:tf.image.resize_images(xx,size[1:3]))(y_true)
    y_true = Reshape( (-1, K.int_shape(mask_coeff)[3]) )(y_true)

    # compute reconstructed feature using mask_coeff and prototype
    indices = tf.cast(indices, tf.int32) 
    mask_coeff = Reshape( (-1, K.int_shape(mask_coeff)[3]) )(mask_coeff)
    mask_coeff_ = tf.gather(mask_coeff, indices, batch_dims=1)
    mask_coeff_ = tf.tile(tf.expand_dims(mask_coeff_, axis=2), (1, 1, K.int_shape(y_pred)[1], 1))
    y_pred = tf.expand_dims(y_pred, axis=1)
    feature = tf.reduce_sum( y_pred*mask_coeff_, axis=-1)
    feature = tf.transpose (feature,[0,2,1] )
    
    # compute binary_crossentropy loss
    
    mask = tf.expand_dims(mask, axis=1)
    loss = K.binary_crossentropy(y_true,feature*mask)
    loss = K.mean(loss)

    return loss



def dice_coef_loss(y_true, y_pred, smooth):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice_coef=(2. * intersection + smooth) / (union + smooth)
    
    return K.mean( 1-dice_coef, axis=0)

def loss(args):

    #-----------------------------------------------------------------------------------------------------------------#
    #   hm_pred：      (batch_size, 128, 128, num_classes)
    #   wh_pred：         (batch_size, 128, 128, 2)
    #   reg_pred：center offset  (batch_size, 128, 128, 2)
    #   hm_true：heatmap      (batch_size, 128, 128, num_classes)
    #   wh_true：        (batch_size, max_objects, 2)
    #   reg_true：  (batch_size, max_objects, 2)
    #   reg_mask：        (batch_size, max_objects)
    #   indices：     (batch_size, max_objects)
    #-----------------------------------------------------------------------------------------------------------------#
    #y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input, masks
    hm_pred, wh_pred, reg_pred, hm_true,wh_true,reg_true,reg_mask,indices,mask_pred,masks_true = args
    hm_loss = focal_loss(hm_pred, hm_true)
    wh_loss = 0.1 * reg_l1_loss(wh_pred, wh_true, indices, reg_mask)
    reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)
    #mask_loss = dice_coef_loss(masks_true, mask_pred, smooth=0.01)
    mask_loss=binary_crossentropy(masks_true, mask_pred)
    #mask_loss= mask_crossentropy(masks_true, mask_pred,mask_coeff,indices,reg_mask)
    total_loss = hm_loss + wh_loss + reg_loss+3*mask_loss
    
    return total_loss
