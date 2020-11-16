import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import keras_metrics as km
from tensorflow import keras as K
from utils import dice_coef, dice_coef_loss, jaccard_distance, preprocess_dataset
from predict import run_model

#  Parameters
preprocess = False 
tiff_stack = False
img_size = (600, 600, 700)
target_size = (256,256,256)

# Input image and mask directories
image_path = '/media/sweene01/SSD/ROI_testing/'
output_path = 'Predictions/'

# Preprocess RSOM data?
if preprocess:
    preprocess_dataset(im_path = image_path, 
                       img_size = img_size, 
                       target_size = target_size,
                       stack = tiff_stack, 
                       save_tiffs = True)

#  Load model
model = K.models.load_model('3Dunet_rsom.hdf5', 
                            custom_objects={'dice_coef_loss': dice_coef_loss,
                                            'dice_coef': dice_coef,
                                            'jaccard_distance': jaccard_distance,
                                            'binary_precision': km.binary_precision(),
                                            'binary_recall': km.binary_recall()})

#  Print model summary
model.summary()

# Generate predictions from model
predictions = run_model(model, 
                        img_path = image_path, 
                        img_size = img_size, 
                        target_size = target_size,
                        pred_dir = output_path)