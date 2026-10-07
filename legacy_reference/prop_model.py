import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def conv_block(x, num_filters,verbose=False):
    if verbose:
        print('------ Convolution Block Started ------')
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    if verbose:    
        print ('After Conv2D 3x3')
        print(x.shape)
    x = BatchNormalization()(x)
    if verbose:
    
        print ('After Batch Norm')
        print(x.shape)
    x = Activation("relu")(x)
    if verbose:
    
        print ('After Activation Relu')
        print(x.shape)
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    if verbose:
    
        print ('After Conv2D 3x3')
        print(x.shape)
    x = BatchNormalization()(x)
    if verbose:
    
        print ('After Batch Norm')
        print(x.shape)
    x = Activation("relu")(x)
    if verbose:
    
        print ('After Activation Relu')
        print(x.shape)
        print('------ Convolution Block Ended ------')
    return x
def portable_bridge(x,num_filters,n,skip_x,skip_x_1,skip_x_2,verbose=False):
    x = conv_block(x, num_filters[-1])
    num_filters.reverse()
    skip_x.reverse()
    if verbose:
    
        print('------ Bridge Started ------')
    for i in range(len(num_filters)-n):
        x = UpSampling2D((2, 2))(x)
        if verbose:
            print ('After Upsampling 2x2')
            print(x.shape)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        if verbose:
    
            print ('After Concat')
            print(x.shape)
        x = conv_block(x, num_filters[i])
        if verbose:
    
            print ('After Convolution Block')
            print(x.shape)
        skip_x_1.append(x)
    skip_x_1.reverse()
    for i in range(len(num_filters)-n):
        x = conv_block(x, num_filters[i])
        if verbose:
    
            print ('After Convolution Block')
            print(x.shape)
        xs = skip_x_1[i]
        x = Concatenate()([x, xs])
        if verbose:
    
            print ('After Concat')
            print(x.shape)
        skip_x_2.append(x)
        x = MaxPool2D((2, 2))(x)
        if verbose:
    
            print ('After MaxPool 2x2')
            print(x.shape)
    skip_x_2.reverse()
    if verbose:
    
        print('Bridge Ended')
    return x,skip_x,skip_x_2


def build_model(size = 256,num_filters = [ 32, 64,128],verbose=False):
    inputs = Input((size, size, 3))
    n=1
    skip_x = []
    skip_x_1 = []
    skip_x_2 = []
    x = inputs
    if verbose:
    
        print('')
        print (' ------- Encoding Started -----')
    ## Encoder
    for i,f in enumerate(num_filters):
        x = conv_block(x, f)
        if verbose:
    
            print('After Conv Block')
            print(x.shape)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)
        if verbose:
    
            print('After Max Pooling')
            print(x.shape)
        x=Dropout(0.3)(x)
        if verbose:
    
            print('After DropOut')
            print(x.shape)
    if verbose:
    
        print('')
        print('------- Bridging -------')
    ## Bridge
    x,skip_x,skip_x_2=portable_bridge(x,num_filters,n,skip_x,skip_x_1,skip_x_2)
    if verbose:
    
        print('After Portable Bridge')
        print(x.shape)
    ## Decoder
    if verbose:
    
        print('')
        print (' ------- Decoding Started -------')
    for i in range(len(num_filters)-1):
        x = UpSampling2D((2, 2))(x)
        if verbose:
    
            print('After Upsampling')
            print(x.shape)
        xs = skip_x_2[i]
        if verbose:
    
            print('After Skip Connection')
            print(x.shape,' -> ',xs.shape)
        x = Concatenate()([x, xs])
        if verbose:
    
            print('x after concatenation')
            print(x.shape)
        x = conv_block(x, num_filters[i])
        if verbose:
    
            print('After Conv Block')
            print(x.shape)
        
    x = UpSampling2D((2, 2))(x)
    if verbose:
    
        print('After Up Sampling')
        print(x.shape)
    xs = skip_x[-1]
    
    x = Concatenate()([x, xs])
    if verbose:
    
        print('After Concatenation')
        print(x.shape, ' -> ',xs.shape)
    x=Dropout(0.1)(x)
    if verbose:
    
        print('After Dropout ')
        print(x.shape) 
    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    if verbose:
    
        print('Output after Conv 2D ')
        print(x.shape) 
    x = Activation("sigmoid")(x)
    if verbose:
    
        print('After Activation Sigmoid')
        print(x.shape) 
    return Model(inputs, x)

def get_model(size = 256,num_filters = [ 32, 64,128],verbose=False):
    model = build_model(size = size,num_filters =num_filters,verbose=verbose)
    #model.summary()
    f_name = 'Proposed Model.png' 
    tf.keras.utils.plot_model(
    model,
    to_file=f_name,
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False,
    )
    print('Model Saved ->',f_name)
    return model

# get_model()   
