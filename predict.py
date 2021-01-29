import numpy as np
import csv
import sys
import math
from validate import validate

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X

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
    train_Data_X=np.genfromtxt("train_X_knn.csv", delimiter=',', dtype=np.float64, skip_header=1)
    train_Data_X=train_Data_X.tolist()
    train_Data_Y=np.genfromtxt("train_Y_knn.csv", delimiter=',', dtype=np.int)
    train_Data_Y=train_Data_Y.tolist()
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
    test_X = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    write_to_csv_file(np.array(pred_Y), "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv")