import os
import numpy as np
import pandas as pd
from skimage import io
from customDataGen import DataGenerator
from utils import resize_volume, load_volume

def run_model(model, img_path, img_size, target_size, pred_dir, batch_size=1, 
              n_classes=2):
    
    params = {'img_dim': img_size,
              'dim': target_size,
              'batch_size': batch_size,
              'im_path': img_path,
              'n_classes': n_classes,
              'testing': True}
    
    files = os.listdir(img_path)
    
    test_generator = DataGenerator(files, **params)
    prediction = model.predict_generator(test_generator, 
                                         steps = len(files), 
                                         verbose = 50)
    
    prediction = (255 * prediction).astype('uint8')

    for i in range(prediction.shape[0]):
        
        print('Saving %s ...' %(files[i]))
        
        # Upscale image
        arr = resize_volume(prediction[i,0,].astype('uint8'), img_size)
        io.imsave(pred_dir+"{file}_upscaled.tiff".format(file=files[i]), 
                  np.transpose(arr,(2,0,1)), 
                  bigtiff=False)
        
        np.save(pred_dir+"{file}.npy".format(file=files[i]), prediction[i,0,:,:,:])

        # Save CNN prediction
        io.imsave(pred_dir+"{file}_prediction.tiff".format(file=files[i]), 
                  np.transpose(prediction[i,0,:,:,:],(2,0,1)), 
                  bigtiff=False)


    return prediction

def analyse_predictions(test_set, pred_path, mask_path, vess_path, name='Other'):
    
    pred_path = os.path.join(pred_path, 'Output/Predictions/')

    result_roi_pred = np.zeros([len(test_set)],dtype='float32')
    result_vess_gt = np.zeros([len(test_set)],dtype='float32')
    result_vess_pred = np.zeros([len(test_set)],dtype='float32')
    
    df = pd.DataFrame({'MouseID': test_set})
    df = df.MouseID.str.split('_',expand=True)
    #  Remove motion correction phrase
    df[4] = df[4].str.replace('mcCorr1','')
    #Remove any addition phrase (e.g. leg)
    df[4] = df[4].str.replace(r'[^\d.]+','')
    
    for i in range(len(test_set)):
        
        print('Analysing ... %s' %test_set[i])
        
        # Dataset training directory
        train_path = os.path.join(mask_path,test_set[i])
        

        #  Load upscaled prediction
        file = test_set[i] + '_upscaled'
        filepath = os.path.join(pred_path, file)
        roi_pred = load_volume(filepath, stack=True, ext='.tiff', datatype='uint8')
        
        #  Count no. of pixels in ROI
        result_roi_pred[i] = np.count_nonzero(roi_pred)
        
        # Analyse blood volume
        file = test_set[i] + '_vessel_sato_mask'
        filepath = os.path.join(train_path, file)
        vess_gt = load_volume(filepath, stack=True, ext='.tif', datatype='uint8')

        # Find location of vessels
        idx = np.nonzero(vess_gt)
        for (m,n,l) in zip(*idx):
            if roi_pred[m-1, n-1, l-1] > 0:
                result_vess_pred[i] = result_vess_pred[i] + 1.0
                
    result_roi_pred = 100 * result_roi_pred / roi_pred.size
    
    result_vess_gt = 100 * result_vess_gt / vess_gt.size
    result_vess_pred = 100 * result_vess_pred / vess_gt.size
    
    filename = 'Tumour_Analysis.csv'
    df = pd.DataFrame({'Full ID': test_set,
                       'Time_Stamp': df.iloc[:,1],
                       'Date': df.iloc[:,2],
                       'Animal ID': df.iloc[:,3]+df.iloc[:,4],
                       'ROI Volume Fraction (CNN)': result_roi_pred,
                       'Blood Volume Fraction (CNN)': result_vess_pred})
    df.to_csv(filename)