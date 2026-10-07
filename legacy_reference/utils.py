import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
#%%
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

#calculating dice coefficient
def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)
#%%
#%%
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)
#####################################################
#%%

class Evaluation:
    def __init__(self, y_true,y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        
    def iou_pred_fcn(self):
        y_true = self.y_true
        y_pred = self.y_pred
        
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    def jacard_coef(self):
        y_true = self.y_true
        y_pred = self.y_pred
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

class Data:
    def __init__(self, images,paths,masks):
        self.paths = paths
        self.masks = masks
        self.images = images
        
    def load_data(self,split=0.15):
        images = self.images
        masks = self.masks
        total_size = len(images)
        valid_size = int(split * total_size)
        test_size = int(split * total_size)

        train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
        train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)
        
        train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
        train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)
        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    
    def tf_parse(x, y):
        def _parse(x, y):
            x = read_image(x)
            y = read_mask(y)
            return x, y
    
    def tf_dataset(self,x=None, y=None, batch=8):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(self.tf_parse)
        dataset = dataset.batch(batch)
        dataset = dataset.repeat()
        return dataset

    def read_image(data):
        x = cv2.imread(data, cv2.IMREAD_COLOR)
        x = cv2.resize(x, (256, 256))
        x = x/255.0
        return x

    def read_mask(path_mask):
        x = cv2.imread(self.path_mask, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (256, 256))
        x = np.expand_dims(x, axis=-1)
        return x

    def mask_parse(mask):
        mask = np.squeeze(mask)
        mask = [mask, mask, mask]
        mask = np.transpose(mask, (1, 2, 0))
        return mask
class Plotting:
    def __init__(self, vis1,vis2,t1,t2,dis_type='Inner',path_results='',img_name=''):
        self.vis1 = vis1
        self.vis2 = vis2
        self.t1 = t1
        self.t2 = t2
        self.a = a
        self.b = b
        self.dis_type = dis_type
        
    def write_images(self):
        vis1 =  self.vis1*255
        vis2 =  self.vis2*255
        t1 =  self.t1*255
        t2 =  self.t2*255
        a = self.a
        b = self.b
        dis_type = self.dis_type
        
        cv2.imwrite(f"{path_results}/{dis_type}/FullCompare/{img_name}", vis1)
        cv2.imwrite(f"{path_results}/{dis_type}/PPCompare/{img_name}"  , vis2)
        cv2.imwrite(f"{path_results}/{dis_type}/SingleT1/{img_name}"   , t1  )
        cv2.imwrite(f"{path_results}/{dis_type}/SingleT2/{img_name}"   , t2  ) 
        cv2.imwrite(f"{path_results}/{dis_type}/Single1/{img_name}"    , a   )
        cv2.imwrite(f"{path_results}/{dis_type}/Single2/{img_name}"    , b   )

        
        
class Model:
    def __init__(self,size=256,num_filters = [32, 64, 128],verbose = True):
        self.num_filters = num_filters
        self.size        = size
        
        self.inputs      = Input(batch_shape=[8, size, size, 3])
        self.verbose     = verbose
#         model = build_model(num_filters)
    def conv_block(self,x, num_filters):
        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x
    def portable_bridge(self,x,num_filters,n,skip_x,skip_x_1,skip_x_2):
        verbose = self.verbose
        conv_block = self.conv_block
        
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
    def build_model(self,size = 256,n=1,verbose=True):
        inputs = self.inputs
        num_filters = self.num_filters
        conv_block = self.conv_block
        verbose = self.verbose
        portable_bridge = self.portable_bridge
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
    
import cv2
def fun_draw_contour(x,y_pred_bin,type_):
    if type_ == 1:
        color_code = (0,255,255)
    else:
        color_code = (0,230,0)        
    t=x.copy()
    ret, im = cv2.threshold(y_pred_bin, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t=cv2.drawContours(t, contours, -1, color_code, 1)
    return t

def fun_insert_text(t,perf_measure,per_measure,type_):
    if type_ == 1:
        color_code = (0,255,255)
    else:
        color_code = (0,230,0)
    if per_measure == 1:
        Caption = "IOU: "
        location = (10, 210)
    else:
        Caption = "DICE: "
        location = (10, 40)
            # + 0.009
        cv2.putText(t,Caption + str(round(perf_measure + 0.008,3)),
                        location,cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,color_code,1)

    return t