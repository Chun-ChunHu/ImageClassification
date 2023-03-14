<!--
###################################################################################################################
＃　this file is to decribe the data extraction process and Model select in image_classification.py
###################################################################################################################
function in used:

function FeatureExtractorHOG is used to extract HOG image feature from datasets. Image size is set to be 240*240.

function top1_5_score is used to calculate the TOP-1, TOP-5 Accuracy from the prediction of each model.
###################################################################################################################
model selection:

function xgboost_model contains the XGBClassifier provided from xgboost package, the function train the dataset and output prediction of the model. If you trained the model previously, joblib package is able to input the model into variable. 

function adaboost_model include the AdaBoostClassifier from sklearn.ensemble package. Hyperparameter 'n_estimators' and random seed 'random_state' can also be changed to provide a better prediction.

function svc_model is provided by sklearn.svm package. Kernel function 'poly' is used, other kernel function such as 'linear' and 'rbf' can also provide smiliar result. the data degree = 3,  gamma is set 'auto' in this case.
 --> 







