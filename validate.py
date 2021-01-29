import csv
from os import path
import numpy as np


def checking_file_exits(predicted_test_Y_file_path):
    if not path.exists(predicted_test_Y_file_path):
        raise Exception("Couldn't find '"+predicted_test_Y_file_path+"' file")


def checking_format(test_X_file_path,predicted_test_Y_file_path):
    with open(predicted_test_Y_file_path,'r') as file:
        file_reader=csv.reader(file)
        pred_Y=np.array(list(file_reader))
        file.close()
        test_X=np.genfromtxt(test_X_file_path,delimiter=',',dtype=np.float64,skip_header=1)
    if pred_Y.shape!=(test_X.shape[0],1):
        raise Exception("Output format is not proper")

def check_weighted_f1_score(actual_test_Y_file_path,predict_test_Y_file_path):
    pred_Y=np.genfromtxt(predict_test_Y_file_path,delimiter=',',dtype=np.int)
    actual_Y=np.genfromtxt(actual_test_Y_file_path,delimiter=',',dtype=np.int)
    from sklearn.metrics import f1_score
    weighted_f1_score = f1_score(actual_Y, pred_Y, average = 'weighted')
    return weighted_f1_score
        
def validate(test_X_file_path,actual_test_Y_file_path):
    predicted_test_Y_file_path="predicted_test_Y_knn.csv"
    checking_file_exits(predicted_test_Y_file_path)
    checking_format(test_X_file_path,predicted_test_Y_file_path)
    weighted_f1_score=check_weighted_f1_score(actual_test_Y_file_path, predicted_test_Y_file_path)
    print("Weighted F1 score", weighted_f1_score)
    
