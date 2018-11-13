import numpy as np
from math import sqrt
from collections import Counter
import pandas as pd
import random
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

iterations = 100
concept_drift_data = {}
model_accuracy_cd = {}
model_accuracy = {}
concept_drift_detection = {}
aggr_const = 1 #Used in compressData
threshold = [0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1] #Used in conceptDrift
#threshold = [0.5]#Used in conceptDrift
concept_drift_data = {}
model_accuracy_cd = {}
model_accuracy = {}
concept_drift_detection = {}
for i in range(len(threshold)):
    concept_drift_data[i] = []
    model_accuracy_cd[i] = []
    model_accuracy[i] = []
    concept_drift_detection[i] = []
test_size = 0.2
max_test_cases = 100
file_name = 'sea.data1.txt'

def compressData(data,class_values):
    new_data = {}
    for i in class_values:
        new_data[i]=[]
    count = 0
    flag = 0
    for group in data:
        temp = [0 for _ in range(len(data[group][0]))]
        count = 0
        for features in data[group]:
            flag = 1
            if(count is aggr_const):
                flag = 0
                for i in range(len(features)):
                    temp[i] += features[i]
                if(count != 0): 
                    for i in range(len(features)):
                        temp[i] = temp[i]/count
                new_data[group].append(temp)
                temp = [0 for _ in range(len(data[group][0]))]
                count = 0
            else:
                for i in range(len(features)):
                    temp[i] += features[i]
            count += 1
        if(count != 0 and flag is 1): 
            for i in range(len(data[group][0])):
                temp[i] = temp[i]/count
            new_data[group].append(temp)
    return new_data
    
def getCentroid(data,class_values):
    centroid = {}
    for i in class_values:
        centroid[i]=[]
    for group in data:
        temp = [0 for _ in range(len(data[group][0]))]
        for features in data[group]:
            for i in range(len(features)):
                temp[i] += features[i]
            
        train_set_len = len(data[group])
        if(train_set_len != 0):
            for i in range(len(temp)):
                temp[i] = temp[i]/train_set_len
        centroid[group].append(temp)
    return centroid

def conceptDrift(data, centroid, test_data, class_values):
    distances = {}
    for i in class_values:
        distances[i]=[]
    countIn = 0
    total = 0
    count = 0
    decision_value = {}
    cd = []
    for i in class_values:
        decision_value[i]=[]
    for group in centroid:
        countIn = 0
        total = 0
        for features in centroid[group]:
            eculidean_distance = np.linalg.norm(np.array(features) - np.array(test_data))
            distances[group].append(eculidean_distance)
        for features in data[group]:
            eculidean_distance = np.linalg.norm(np.array(features) - np.array(test_data))
            if eculidean_distance <= distances[group][0]:
                countIn += 1
            total += 1
        decision_value[group].append(countIn/total)
    for th in threshold:
        count = 0
        for dv in decision_value:
            if decision_value[dv][0] > th:
                count += 1
        if count is len(decision_value):
            cd.append(1)
        else:
            cd.append(0)
    return cd

def centroid_classifier(predict, centroid):
    distances = []
    for group in centroid:
        for features in centroid[group]:
            eculidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([eculidean_distance, group])
    votes = [i[1] for i in sorted(distances)[:(len(centroid))]]
    vote_result = Counter(votes).most_common(1)[0][0]
    
    return vote_result

def plot_graph(train_set,idm='1'):
    plt.clf()
    print('Plotting graph of training examples')
    colors = ['red','blue','cyan','yellow','black','magenta','orange','brown','purple','olive','pink']
    cl = {}
    itr = 0
    for i in class_values:
    	cl[i]=[colors[itr]]
    	itr += 1
    markers = ['o','x']
    mk = {}
    itr = 0
    for i in class_values:
    	mk[i]=[markers[0]]
    	itr += 1
    labels = ['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9']
    lab = {}
    itr = 0
    for i in class_values:
    	lab[i]=[labels[itr]]
    	itr += 1
    fig = plt.figure()
    #ax1 = fig.add_subplot(211, projection = '3d')
    ax2 = fig.add_subplot(111)
    for i in train_set:
        newx = {}
        for m in range(len(train_set[i][0])):
            newx[m] = []
        for s in train_set[i]:
            for k in range(len(s)):
                newx[k].append(s[k])
	      #ax1.scatter(newx, newy, newz, color = cl[i], marker = mk[i], label = lab[i])
        #print(newx)
        data_mat = np.column_stack((newx[0],newx[1],newx[2]))
        #print(data_mat)
        data_mat_std = StandardScaler().fit_transform(data_mat)
        features = data_mat_std.T
    
        covariance_mat = np.cov(features)
        eig_vals, eig_vecs = np.linalg.eig(covariance_mat)
        proj_x = data_mat_std.dot(eig_vecs.T[np.argmax(eig_vals)])
        eig_vals[np.argmax(eig_vals)] = 0
        proj_y = data_mat_std.dot(eig_vecs.T[np.argmax(eig_vals)])
        newy_ = [0 for _ in proj_x]
        ax2.scatter(proj_x,proj_y,color = cl[i], label = lab[i], s = 5)
    plt.legend()
    #plt.xticks(np.linspace(proj_x[np.argmin(proj_x)], proj_x[np.argmax(proj_x)], 5))
    #plt.yticks(np.linspace(proj_y[np.argmin(proj_y)], proj_y[np.argmax(proj_y)], 15))
    plt.savefig('Training_Set'+idm+'.png')
    plt.clf()

def plot_graph2(x=[0],y=[0],idm=1,xlabel='',ylabel='',x2=[0],y2=[0]):
    plt.clf()
    plt.plot(x,y,label = 'With CD Detection')
    plt.plot(x2,y2, label = 'Without CD Detection')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.linspace(0, x[np.argmax(x)], 10))
    plt.yticks(np.linspace(0, y[np.argmax(y)], 15))
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.savefig('result_graph'+str(idm)+'.png')

plot_count = 0
for itr in range(iterations):
    '''df = pd.read_csv("breast-cancer-wisconsin.data.txt")
    df.replace('?',-99999, inplace=True)
    df.drop(['id'], 1, inplace=True)'''
    print('Iteration ID:',itr+1,'----------------------------------------------')
    print('Reading the File')
    df = pd.read_csv(file_name)
    #df.replace('?',-99999, inplace=True)
    #df.drop(['id'], 1, inplace=True)

    print('Converting all values')
    full_data = df.astype(float).values.tolist()

    print('Shuffling the data')
    random.shuffle(full_data)

    print('Setting the test size and defining')

    print('First: train_data',end=' ')
    train_data = full_data[:-int(test_size*len(full_data))]
    print(len(train_data))

    print('Second: test_data',end=' ')
    test_data = full_data[-int(test_size*len(full_data)):]
    print(len(test_data))

    class_values = list(set([i[-1] for i in train_data]))

    print('Creating test_set and train_set')
    train_set = {}
    test_set = {}
    for i in class_values:
        train_set[i]=[]
        test_set[i]=[]

    print('Generating train_set: ',end='')
    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    aggr = 0
    for i in class_values:
        aggr += len(train_set[i])
        print(i,': ',len(train_set[i]),end=', ')
    print('Total = ',aggr)
    #print(len(train_set[0])+len(train_set[1]))

    print('Generating test_set: ',end='')
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
    aggr = 0
    for i in class_values:
        aggr += len(test_set[i])
        print(i,': ',len(test_set[i]),end=', ')
    print('Total = ',aggr)
    #print(len(test_set[0])+len(test_set[1]))
    if plot_count is 0:
        plot_count = 1
        plot_graph(train_set)
    correct = [0 for _ in range(len(threshold))]
    total = [0 for _ in range(len(threshold))]
    correct1 = [0 for _ in range(len(threshold))]
    total1 = [0 for _ in range(len(threshold))]
    concept_drift_count = [0 for _ in range(len(threshold))]
    print('Getting the centroid:',end='')
    centroid = getCentroid(train_set,class_values)
    print(centroid)
    print('Compressing Data')
    compressed_data = {}
    if aggr_const is not 1:
        compressed_data = compressData(train_set,class_values)
    else:
        compressed_data = train_set
    #compressed_data = train_set
    #print('Original Data Len/Compessed Data Len: ',len(train_set[0]),'+',len(train_set[1]),'/',len(compressed_data[0]),'+',len(compressed_data[1]))
    #print('Caliberating accuracy /with ',len(test_set[0])+len(test_set[1]),' test cases')
    print('Caliberating Accuracy:')
    for group in test_set:
        temp_count = 0
        for data in test_set[group]:
            if(temp_count is max_test_cases):
                break
            temp_count += 1
            temp = conceptDrift(compressed_data, centroid, data, class_values)
            tp_count = 0
            for tp in temp:
                if tp is 0:
                    vote = centroid_classifier(data, centroid)
                    if group == vote:
                        correct[tp_count] += 1
                        correct1[tp_count] += 1
                    train_set[group].append(data)
                    centroid = getCentroid(train_set, class_values)
                    total[tp_count] += 1
                    total1[tp_count] += 1
                else:
                    concept_drift_count[tp_count] += 1
                    vote = centroid_classifier(data, centroid)
                    if group == vote:
                        correct1[tp_count] += 1
                    total1[tp_count] += 1
                tp_count += 1
    for th in range(len(threshold)):
        print('THRESHOLD: ',threshold[th])
        print('Concept Drift Count:',concept_drift_count[th])
        concept_drift_data[th].append(concept_drift_count[th])
        if correct[th] is 0 or total[th] is 0:
            temp1 = 1
        else:
            temp1 = correct[th]/total[th]
        print('Model Accuracy including check for Concept Drift:', temp1)
        model_accuracy_cd[th].append(temp1)
        print('Model Accuracy not including check for Concept Drift:', correct1[th]/total1[th] if total1[th] is not 0 else 1)
        model_accuracy[th].append(correct1[th]/total1[th] if total1[th] is not 0 else 1)
        #print('Concept Drift Detection Accuracy: Detected Concept Drift/Actual Concept Drit',concept_drift_count[th]/((total1[th] - correct1[th]) if (total1[th] - correct1[th]) is not 0 else 1))
        concept_drift_detection[th].append(concept_drift_count[th]/((total1[th] - correct1[th]) if (total1[th] - correct1[th]) is not 0 else 1))
    
    itr += 1
for th in range(len(threshold)):
    print('-------------------------------------------------------------------------------')
    print('THRESHOLD: ',threshold[th])
    print('Average Concept Drift Count: ',sum(concept_drift_data[th])/len(concept_drift_data[th]))
    print('Average Model Accuracy including check for Concept Drift: ',sum(model_accuracy_cd[th])/len(model_accuracy_cd[th]))
    print('Average Model Accuracy not including check for Concept Drift: ',sum(model_accuracy[th])/len(model_accuracy[th]))
    #print('Average Concept Drift Detection Accuracy: Detected Concept Drift/Actual Concept Drit ',sum(concept_drift_detection[th])/len(concept_drift_detection[th]))
    x = [i+1 for i in range(iterations)]
    y = model_accuracy_cd[th]
    x2 = x
    y2 = model_accuracy[th]
    plot_graph2(x,y,threshold[th],'Iterations','Accuracy', x2,y2)

for iterat in range(iterations):
    x = threshold
    y = [model_accuracy_cd[i][iterat] for i in range(len(threshold))]
    x2 = threshold
    y2 = [model_accuracy[i][iterat] for i in range(len(threshold))]
    plot_graph2(x,y,iterat+1,'Threshold','Accuracy',x2,y2)
