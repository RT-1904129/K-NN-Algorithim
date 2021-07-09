import numpy as np
import csv
import sys
import math
from validate import validate

mean_null_list=[]
All_mini_maxi_mean_normalize_value=[]
threshold1=0.0
threshold2=0.6
selected_feature_list_for_testing=[0, 2, 4, 5]
mean_null_list_for_testing=[[10.813225  , 10.4554858 , 14.084725  , 10.1512    , 31.78294375,12.6915625 , 10.43719375]]
All_mini_maxi_mean_normalize_value_for_testing=[[10.0, 11.193999999999999, 10.813225], [10.453393, 10.459375, 10.4554858], [13.306, 15.241, 14.084725], [10.0, 11.863, 10.1512], [30.943, 32.335, 31.782943749999998], [11.629, 14.488, 12.6915625], [10.087, 11.05, 10.43719375]]

def import_data_for_testing(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X

def replace_null_values_with_mean(X):
    mean_of_nan=np.nanmean(X,axis=0)
    mean_null_list.append(mean_of_nan)
    index=np.where(np.isnan(X))
    X[index]=np.take(mean_of_nan,index[1])
    return X

def mean_normalize(X, column_indices):
    mini=np.min(X[:,column_indices],axis=0)
    maxi=np.max(X[:,column_indices],axis=0)
    mean=np.mean(X[:,column_indices],axis=0)
    All_mini_maxi_mean_normalize_value.append([mini,maxi,mean])
    X[:,column_indices]=(X[:,column_indices]-mean)/(maxi-mini)
    return X

def get_correlation_matrix(X,class_Y):
    given_X=np.hstack((class_Y.reshape(len(X),1),X))
    num_vars = len(given_X[0])
    m = len(X)
    correlation_matix = np.zeros((num_vars,num_vars))
    for i in range(0,num_vars):
        for j in range(i,num_vars):
            mean_i = np.mean(given_X[:,i])
            mean_j = np.mean(given_X[:,j])
            std_dev_i = np.std(given_X[:,i])
            std_dev_j = np.std(given_X[:,j])
            numerator = np.sum((given_X[:,i]-mean_i)*(given_X[:,j]-mean_j))
            denominator = (m)*(std_dev_i)*(std_dev_j)
            corr_i_j = numerator/denominator    
            correlation_matix[i][j] = corr_i_j
            correlation_matix[j][i] = corr_i_j
    return correlation_matix


def select_features(corr_mat, T1, T2):
    filter_feature=[]
    m=len(corr_mat)
    for i in range(1,m):
        if(abs(corr_mat[i][0])>T1):
            filter_feature.append(i-1)
    removed_feature=[]
    n=len(filter_feature)
    select_features=list(filter_feature)
    for i in range(0,n):
        for j in range(i+1,n):
            f1=filter_feature[i]
            f2=filter_feature[j]
            if (f1 not in removed_feature) and (f2 not in removed_feature):
                if(abs(corr_mat[f1][f2])>T2):
                    select_features.remove(f2)
                    removed_feature.append(f2)
                    
    return select_features


def data_processing(class_X,class_Y) :
    X=replace_null_values_with_mean(class_X)
    for i in range(class_X.shape[1]):
        X=mean_normalize(X,i)
    
    correlation_matrix= get_correlation_matrix(X,class_Y)
    selected_feature_list=select_features(correlation_matrix,threshold1,threshold2)
    X=X[:,selected_feature_list]
    #we will uncomment it for getting selected feature list which we will use in predict.py
    #print(selected_feature_list)
    return X

def import_data_for_training():
    train_Data_X=np.genfromtxt("train_X_knn.csv", delimiter=',', dtype=np.float64, skip_header=1)
    train_Data_Y=np.genfromtxt("train_Y_knn.csv", delimiter=',', dtype=np.int)
    train_Data_X=data_processing(train_Data_X,train_Data_Y)
    train_Data_X=train_Data_X.tolist()
    train_Data_Y=train_Data_Y.tolist()
    return train_Data_X,train_Data_Y


def replace_null_values_with_mean_for_testing(X):
    mean_of_nan=mean_null_list_for_testing[0]
    index=np.where(np.isnan(X))
    X[index]=np.take(mean_of_nan,index[1])
    return X

def mean_normalize_for_testing(X, column_indices):
    mini,maxi,mean=All_mini_maxi_mean_normalize_value_for_testing[column_indices]
    X[:,column_indices]=(X[:,column_indices]-mean)/(maxi-mini)
    return X


def data_processing_for_testing(X_test) :
    X_test=replace_null_values_with_mean_for_testing(X_test)
    for i in range(X_test.shape[1]):
        X_test=mean_normalize_for_testing(X_test,i)
    X_test=X_test[:,selected_feature_list_for_testing]
    return X_test

def compute_ln_norm_distance(vector1, vector2, n):
    sum=0
    for i in range(len(vector2)):
        sum=sum+abs((vector2[i]-vector1[i])**(n))
    return (sum)**(1/n)

def find_k_nearest_neighbors(train_X, test_example, k, n):
    list_of_distance=[]
    count_index=0
    for point in train_X:
        distance=compute_ln_norm_distance(point,test_example,n)
        list_of_distance.append((distance,count_index))
        count_index+=1
    list_of_distance.sort()
    nearest_point=[]
    for i in range(k):
        nearest_point.append(list_of_distance[i][1])
    return nearest_point

def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    test_Y = []
    for test_elem_x in test_X:
        top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k,n)
        top_knn_labels = []

        for i in top_k_nn_indices:
            top_knn_labels.append(train_Y[i])
        Y_values = list(set(top_knn_labels))

        max_count = 0
        most_frequent_label = -1
        for y in Y_values:
            count = top_knn_labels.count(y)
            if(count > max_count):
                max_count = count
                most_frequent_label = y

        test_Y.append(most_frequent_label)
    return test_Y

def calculate_accuracy(predicted_Y, actual_Y):
    count=0
    for i in range(len(predicted_Y)):
        if (predicted_Y[i]==actual_Y[i]):
            count+=1
    return (count/len(actual_Y))

def get_best_k_using_validation_set(train_X, train_Y, validation_split_percent,n):
    length_training_Data_X=math.floor(((float(100-validation_split_percent))/100)*len(train_X))
    training_Data_X=train_X[0:length_training_Data_X]
    testing_Data_X=train_X[length_training_Data_X:]
    training_Data_Y=train_Y[0:length_training_Data_X]
    actual_Y=train_Y[length_training_Data_X:]
    k_values_list=[x for x in range(1,len( training_Data_X)+1)]
    list_of_accuracy_with_k=[]
    for k_value in k_values_list:
        predicted_Y=classify_points_using_knn(training_Data_X, training_Data_Y, testing_Data_X, k_value,n)
        accuracy=calculate_accuracy(predicted_Y,actual_Y)
        list_of_accuracy_with_k.append((accuracy,k_value))
    
    list_of_accuracy_with_k.sort(reverse=True)
    lower_k=0
    for i in range(len(list_of_accuracy_with_k)):
        if list_of_accuracy_with_k[i][0]==list_of_accuracy_with_k[0][0]:
            lower_k=list_of_accuracy_with_k[i][1]
        else:
            break
    return lower_k

def predict_target_values(test_X):
    train_Data_X,train_Data_Y=import_data_for_training()
    #Uncomment it when its requirement over)
    #print(mean_null_list)
    #print(All_mini_maxi_mean_normalize_value)
    test_X=data_processing_for_testing(test_X)
    n=2
    k_value=get_best_k_using_validation_set(train_Data_X,train_Data_Y,30,n)
    predicted_Y=classify_points_using_knn(train_Data_X,train_Data_Y, test_X,k_value,n)
    return predicted_Y
    

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X = import_data_for_testing(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    write_to_csv_file(np.array(pred_Y), "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    #test_X_file_path = sys.argv[1]
    test_X_file_path="test_X_knn.csv"
    predict(test_X_file_path)
    validate(test_X_file_path, actual_test_Y_file_path="test_Y_knn.csv")