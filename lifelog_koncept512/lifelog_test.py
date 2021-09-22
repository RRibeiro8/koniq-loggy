import os
import numpy as np
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import img_to_array, load_img

def fc_layers(input_layer,
              name               = 'pred',
              fc_sizes           = [2048, 1024, 256, 1],
              dropout_rates      = [0.25, 0.25, 0.5, 0],
              batch_norm         = False,
              l2_norm_inputs     = False,              
              activation         = 'relu',
              initialization     = 'he_normal',
              out_activation     = 'linear',
              test_time_dropout  = False,
              **fc_params):
    """
    Add a standard fully-connected (fc) chain of layers (functional Keras interface)
    with dropouts on top of an input layer. Optionally batch normalize, add regularizers
    and an output activation.
    * input_layer: input layer to the chain
    * name: prefix to each layer in the chain
    * fc_sizes: list of number of neurons in each fc-layer
    * dropout_rates: list of dropout rates for each fc-layer
    * batch_norm: 0 (False) = no batch normalization (BN),
                  1 = do BN for all, 2 = do for all except the last, ...
    * l2_norm_inputs: normalize the `input_layer` with L2_norm
    * kernel_regularizer: optional regularizer for each fc-layer
    * out_activation: activation added to the last fc-layer
    :return: output layer of the chain
    """
    x = input_layer
    if l2_norm_inputs:
        x = Lambda(lambda x: tf.nn.l2_normalize(x, 1))(input_layer)

    assert dropout_rates is None or (len(fc_sizes) == len(dropout_rates)),\
           'Each FC layer should have a corresponding dropout rate'

    if activation.lower() == 'selu':
        dropout_call = AlphaDropout
    else:
        dropout_call = Dropout
        
    for i in range(len(fc_sizes)):
        if i < len(fc_sizes)-1:
            act = activation
            layer_type = 'fc%d' % i
        else:
            act  = out_activation
            layer_type = 'out'
        x = Dense(fc_sizes[i], activation=act, 
                  name='%s_%s' % (name, layer_type),
                  kernel_initializer=initialization, 
                  **fc_params)(x)
        if batch_norm > 0 and i < ( len(fc_sizes)-(batch_norm-1) ):
            x = BatchNormalization(name='%s_bn%d' % (name, i))(x)
        if dropout_rates is not None and dropout_rates[i] > 0:
            do_call = dropout_call(dropout_rates[i], name = '%s_do%d' % (name, i))            
            if test_time_dropout:
                x = do_call(x, training=True)
            else:
                x = do_call(x)
    return x


def main():

    imgs_path = '../images_test'
    
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(384,512,3))
    feats = base_model.layers[-1]

    gap = GlobalAveragePooling2D(name="final_gap")(feats.output)
    model = Model(inputs=base_model.input, outputs=gap)

    head = fc_layers(model.output, name='fc',
                                fc_sizes      = [2048, 1024, 256, 1], 
                                dropout_rates = [0.25, 0.25, 0.5, 0], 
                                batch_norm    = 2)
    
    model = Model(inputs = model.input, outputs = head)

    model.load_weights('./models/koncep512-trained-model.h5')

    model.summary()

    for f in os.listdir(imgs_path):
        im_path = os.path.join(imgs_path,f)
        img = load_img(path=im_path, target_size=(384,512))
        x = img_to_array(img)

        pre_img = preprocess_input(x)
        batch = np.expand_dims(pre_img, 0)

        y_pred = model.predict(batch).squeeze()
        
        print("##### Image: ", f)
        print(f'Predicted score: {y_pred:.{2}f}')
        
    return 0

if __name__ == "__main__":
    
    main()