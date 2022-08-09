


def building_instance_segmentation(input_shape, num_classes=1, backbone='resnet50', max_objects=100, mode="train"):
    assert backbone in ['resnet50', 'resnet101']
    output_size = input_shape[0] // 4
    image_input = Input(shape=input_shape)
    hm_input = Input(shape=(output_size, output_size, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))
    masks_true=Input(shape=(input_shape[0],input_shape[0],2))

    if backbone=='resnet50':

 
       y1, y2, y3, mask_pred = centernet_head(image_input, num_classes)


       if mode=="train":
            loss_ = Lambda(loss, name='centernet_loss')([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input,mask_pred,masks_true])
            model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input,masks_true], outputs=[loss_])
            return model
