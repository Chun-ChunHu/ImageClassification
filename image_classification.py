# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:39:32 2023

@author: user
"""
# Load packages
import cv2
import numpy as np
from tqdm import tqdm 
from skimage import feature
from sklearn import metrics
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier

# Path set
root_path = 'C:\\Hu\\0_IMIS\\5_course\\DeepLearning\\'
data_path = root_path + 'data\\'

# Split required data and decide model input size 
train_part = []
train, test, val = ([],[],[])
train_images, train_labels, test_images, test_labels, val_images, val_labels = ([],[],[],[],[],[])
data_type = ['train', 'test', 'val']
size = 240

# Function for extracting HOG feature from images
def FeatureExtractorHOG(img, size):
        img = cv2.imread(data_path + globals()[data_type[i]][j][0], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, size))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR is the default type that imread loaded
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (hog, hog_img) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys',
                                       visualize=True, transform_sqrt=True)
        # cv2.imshow('Image', img)
        # cv2.imshow('HOG Image', hog_image)
        return hog
# Function for caculating TOP-1, TOP-5 score of prediction
def top1_5_score(test_labels, prediction, model):
    label = model.classes_
    top1_score = 0
    top5_score = 0
    for i in tqdm(range(len(test_labels))):
        top5_ans = np.argpartition(prediction[i], -5)[-5:]
        if int(test_labels[i]) in label[top5_ans]:
            top5_score = top5_score + 1
        if int(test_labels[i]) == label[np.argmax(prediction[i])]:
            top1_score = top1_score + 1
    # print(top1_score/len(test_labels) , top5_score/len(test_labels))
    return top1_score/len(test_labels) , top5_score/len(test_labels)     
 
# Read category from txt file
for i in range(len(data_type)):
    globals()[data_type[i]] = np.genfromtxt(data_path + data_type[i] + '.txt', delimiter=' ', dtype = 'str', skip_header=1)                          
   
# Split images and labels
for i in range(len(data_type)):
    for j in range(len(globals()[data_type[i]])):
        img = FeatureExtractorHOG(globals()[data_type[i]][j][0], size)
        label = globals()[data_type[i]][j][1]
        globals()[data_type[i] + '_images'].append(img)
        globals()[data_type[i] + '_labels'].append(int(label))       
               
for i in range(len(data_type)):     
    globals()[data_type[i] + '_images'] = np.array(globals()[data_type[i] + '_images'])
    globals()[data_type[i] + '_labels'] = np.array(globals()[data_type[i] + '_labels'])

# ---------------------- XGBOOST model ------------------------------
def xgboost_model(train_images, train_labels, test_images, test_labels, train_or_not):
        if (train_or_not == False):
            import joblib
            model = joblib.load(open(data_path + 'xgboost_model.pkl', 'rb'))
        else:
            model = xgb.XGBClassifier()
            model.fit(train_images, train_labels) 

        xgboost_prediction = model.predict_proba(test_images)
        
        return model, xgboost_prediction
# ---------------------- ADABOOST model ------------------------------
def adaboost_model(train_images, train_labels, test_images, test_labels, train_or_not):
        if(train_or_not == False):
            import joblib
            model = joblib.load(open(data_path + 'adaboost_model.pkl', 'rb'))
        else:
            model = AdaBoostClassifier(n_estimators = 100, random_state = 96)
            model.fit(train_images, train_labels)

        adaboost_prediction = model.predict_proba(test_images)

        return model, adaboost_prediction
# ---------------------- SVC model ------------------------------
def svc_model(train_images, train_labels, val_images, val_labels, test_images, test_labels, train_or_not):
        if(train_or_not == False):
            import joblib
            model = joblib.load(open(data_path + 'svm_model.pkl', 'rb'))
        else:
            # Define an SVM model
            model = SVC(kernel='poly', degree=3, gamma='auto', C=1)
            model.fit(train_images, train_labels)

        svc_prediction = model.predict_proba(test_images)

        return model, svc_prediction
    
# Train and get the models and predictions
# ---------------------- XGBOOST model ------------------------------
xgboost_model, xgboost_prediction = xgboost_model(train_images, train_labels, test_images, test_labels, train_or_not = False)
xgboost_results = top1_5_score(test_labels, xgboost_prediction, xgboost_model)
print('TOP-1 Accuracy for Xgboost Model:%f\nTOP-5 Accuracy for Xgboost Model:%f' 
      % (xgboost_results[0], xgboost_results[1]))
# ---------------------- ADABOOST model ------------------------------
adaboost_model, adaboost_prediction = adaboost_model(train_images, train_labels, test_images, test_labels, train_or_not = True)
adaboost_results = top1_5_score(test_labels, adaboost_prediction, adaboost_model)
print('TOP-1 Accuracy for Adaboost Model:%f\nTOP-5 Accuracy for Adaboost Model:%f' 
      % (adaboost_results[0], adaboost_results[1]))
# ---------------------- SVM model ------------------------------
svc_model, svc_prediction = svc_model(train_images, train_labels, val_images, val_labels, test_images, test_labels, train_or_not = True)
svc_results = top1_5_score(test_labels, svc_prediction, svc_model)
print('TOP-1 Accuracy for SVC Model:%f\nTOP-5 Accuracy for SVC Model:%f' 
      % (svc_results[0], svc_results[1]))

# Confusion Matrix - verify accuracy of each class
# cm = confusion_matrix(test_labels, xgboost_prediction)
#print(cm)
# sns.heatmap(cm, annot=True)







