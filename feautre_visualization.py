import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = VGG16()
model.summary()

SINGLE_OUTPUT = False
CHECK_FILTERS = False

def Visualize(layer_no=1, n_filters=6):
    filters, bias = model.layers[layer_no].get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    ix = 1
    for i in range(n_filters):
        f = filters[:, :, :, i]
        for j in range(3):
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(f[:,:,j], cmap = 'gray')
            ix += 1
    plt.show()


# ============= To visualize filters =============
if CHECK_FILTERS:
    Visualize(1)      
# ================================================




# ============= To visualize feature maps =============
if SINGLE_OUTPUT:
    # For single output
    model_2 = tf.keras.Model(inputs = model.inputs, outputs = model.layers[1].output)   # Truncate the model & obtain the output.  
else:
    # For multiple outputs.
    outputs_feature_maps = [2, 5, 9, 13, 17]
    output = [model.layers[i].output for i in outputs_feature_maps]
    model_2 = tf.keras.Model(inputs = model.inputs, outputs = output)

img = tf.keras.preprocessing.image.load_img('cat.jpg', target_size = (224, 224))
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis = 0)  # Add an axis to the img, which enables the data to be process by the truncated model.
img = tf.keras.applications.vgg16.preprocess_input(img)
feature_maps = model_2.predict(img)

if SINGLE_OUTPUT:
    # For single output
    square = 8
    ix = 1  # index
    plt.figure(figsize = (12, 8))
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0,:,:,ix-1],cmap = 'gray')
            ix += 1
    plt.show()
else:
    # For single output
    square = 8
    for fmap in feature_maps:
        ix = 1  # index
        plt.figure(figsize = (12, 8))
        for _ in range(square):
            for _ in range(square):
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                
                plt.imshow(fmap[0,:,:,ix-1],cmap = 'gray')
                ix += 1
        display = 'Feature map of ' + str(outputs_feature_maps[feature_maps.index(fmap)]) + 'th layer'
        save_name = str(outputs_feature_maps[feature_maps.index(fmap)]) + 'th_layer'
        plt.savefig(save_name)
        plt.suptitle(display,  fontsize = 20) 
        # print(outputs_feature_maps[feature_maps.index(fmap)])
        plt.show()
# ================================================