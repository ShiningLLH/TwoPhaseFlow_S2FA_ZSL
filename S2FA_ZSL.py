from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader
from Siamese_S2FA import SiameseNetwork
import itertools
import torch.optim
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix
import pandas as pd
import random


def center_data(X):     # Normalization
    X_means = np.mean(X, axis=1)
    X_std = np.std(X, axis=1)
    X_norm = (X - X_means[:, np.newaxis])/X_std[:, np.newaxis]
    return X_norm, X_means, X_std

def creat_dataset_problem1(test_index, attribute_matrix1):  # Zero-shot case for transition states
    # Training data
    path = './GW_mat_data/Dataset1/'
    data1 = loadmat(path + 'data_bubble.mat')['data']
    data2 = loadmat(path + 'data_plug.mat')['data']
    data3 = loadmat(path + 'data_slug.mat')['data']
    data4 = loadmat(path + 'data_wave.mat')['data']
    data5 = loadmat(path + 'data_st.mat')['data']
    data6 = loadmat(path + 'data_ann.mat')['data']
    attribute_matrix = attribute_matrix1.values
    traindata = np.row_stack([data1, data2, data3, data4, data5, data6])
    n1 = data1.shape[0]
    attribute1 = [attribute_matrix[0, :]] * n1
    n2 = data2.shape[0]
    attribute2 = [attribute_matrix[1, :]] * n2
    n3 = data3.shape[0]
    attribute3 = [attribute_matrix[2, :]] * n3
    n4 = data4.shape[0]
    attribute4 = [attribute_matrix[3, :]] * n4
    n5 = data5.shape[0]
    attribute5 = [attribute_matrix[4, :]] * n5
    n6 = data6.shape[0]
    attribute6 = [attribute_matrix[5, :]] * n6
    train_attributelabel = np.row_stack([attribute1, attribute2, attribute3, attribute4, attribute5, attribute6])

    # Test data
    data7 = loadmat(path + 'data_b2s.mat')['data'][490:980,:]
    data8 = loadmat(path + 'data_p2s.mat')['data'][490:980,:]
    data9 = loadmat(path + 'data_s2a.mat')['data'][490:980,:]
    data10 = loadmat(path + 'data_st2w.mat')['data'][490:980,:]
    data11 = loadmat(path + 'data_w2a.mat')['data'][490:980,:]
    data_list_test = [data7, data8, data9, data10, data11]
    testdata = []
    testlabel = []
    for item in test_index:
        testdata.append(data_list_test[item-6])
        testlabel += [item] * (data_list_test[item-6].shape[0])
    testlabel = [i + 1 for i in testlabel]
    testdata = np.row_stack(testdata)
    return traindata, train_attributelabel, testdata, testlabel

def creat_dataset_problem2(test_index, attribute_matrix1):  # Zero-shot case for dangerous states
    # Training data
    path = './GW_mat_data/Dataset1/'
    data1 = loadmat(path + 'data_bubble.mat')['data']
    data2 = loadmat(path + 'data_plug.mat')['data']
    data3 = loadmat(path + 'data_slug.mat')['data']
    data4 = loadmat(path + 'data_wave.mat')['data']
    data5 = loadmat(path + 'data_st.mat')['data']
    data6 = loadmat(path + 'data_ann.mat')['data']
    attribute_matrix = attribute_matrix1.values
    data_1_1 = loadmat(path + 'data_1-1.mat')['data']
    data_2_2 = loadmat(path + 'data_2-2.mat')['data']
    data_1_6 = loadmat(path + 'data_1-6.mat')['data']
    data_5_6 = loadmat(path + 'data_5-6.mat')['data']
    data_5_1 = loadmat(path + 'data_5-1.mat')['data']
    data_1_9 = loadmat(path + 'data_1-9.mat')['data']
    train_index = list(set(np.arange(6)) - set(test_index))
    test_index.sort()
    train_index.sort()
    data_list_train = [data1, data2, data3, data4, data5, data6]
    train_attributelabel = []
    traindata = []
    for item in train_index:
        train_attributelabel += [attribute_matrix[item, :]] * (data_list_train[item].shape[0])
        traindata.append(data_list_train[item])
    train_attributelabel = np.row_stack(train_attributelabel)
    traindata = np.row_stack(traindata)

    # Test data
    data_list_test = [data_1_1, data_2_2, data_1_6, data_5_6, data_5_1, data_1_9]
    testlabel = []
    testdata = []
    for item in test_index:
        testdata.append(data_list_test[item])
        testlabel += [item] * (data_list_test[item].shape[0])
    testlabel = [i + 1 for i in testlabel]
    testdata = np.row_stack(testdata)
    return traindata, train_attributelabel, testdata, testlabel

def creat_dataset_problem3(test_index, attribute_matrix1):  # Zero-shot case for offset conditions
    # Training data
    path = './GW_mat_data/Dataset2/'
    data1 = loadmat(path + 'data_bubble.mat')['data']
    data2 = loadmat(path + 'data_plug.mat')['data']
    data3 = loadmat(path + 'data_slug.mat')['data']
    data4 = loadmat(path + 'data_wave.mat')['data']
    data5 = loadmat(path + 'data_st.mat')['data']
    data6 = loadmat(path + 'data_ann.mat')['data']
    attribute_matrix = attribute_matrix1.values
    traindata = np.row_stack([data1, data2, data3, data4, data5, data6])
    n1 = data1.shape[0]
    attribute1 = [attribute_matrix[0, :]] * n1
    n2 = data2.shape[0]
    attribute2 = [attribute_matrix[1, :]] * n2
    n3 = data3.shape[0]
    attribute3 = [attribute_matrix[2, :]] * n3
    n4 = data4.shape[0]
    attribute4 = [attribute_matrix[3, :]] * n4
    n5 = data5.shape[0]
    attribute5 = [attribute_matrix[4, :]] * n5
    n6 = data6.shape[0]
    attribute6 = [attribute_matrix[5, :]] * n6
    train_attributelabel = np.row_stack([attribute1, attribute2, attribute3, attribute4, attribute5, attribute6])

    # Test data
    data7 = loadmat(path + 'data_1-1.mat')['data']
    data8 = loadmat(path + 'data_2-2.mat')['data']
    data9 = loadmat(path + 'data_1-6.mat')['data']
    data10 = loadmat(path + 'data_5-6.mat')['data']
    data11 = loadmat(path + 'data_5-1.mat')['data']
    data12 = loadmat(path + 'data_1-9.mat')['data']
    data_list_test = [data7, data8, data9, data10, data11, data12]
    testdata = []
    testlabel = []
    for item in test_index:
        testdata.append(data_list_test[item-6])
        testlabel += [item] * (data_list_test[item-6].shape[0])
    testlabel = [i + 1 for i in testlabel]
    testdata = np.row_stack(testdata)
    return traindata, train_attributelabel, testdata, testlabel

# Attribute prediction
def pre_attribute_model(model, trainfeature, train_attributelabel, testfeature):
    print('Attribute prediction model: '+model)
    model_dict = {'SVC_linear': SVC(kernel='linear'), 'lr': LogisticRegression(), 'SVC_rbf': SVC(kernel='rbf'),
                  'rf': RandomForestClassifier(n_estimators=100), 'Ridge': Ridge(alpha=1), 'NB': GaussianNB(),
                  'Lasso': Lasso(alpha=0.1)}
    res_list = []
    for i in range(train_attributelabel.shape[1]):
        clf = model_dict[model]
        if max(train_attributelabel[:, i]) != 0:
            clf.fit(trainfeature, train_attributelabel[:, i])
            res = clf.predict(testfeature)
        else:
            res = np.zeros(testfeature.shape[0])
        res_list.append(res.T)
    test_pre_attribute = np.mat(np.row_stack(res_list)).T
    return test_pre_attribute

# Flow state prediction
def pre_state_label(test_pre_attribute, attribute_matrix1, test_index):
    attribute_matrix = pd.DataFrame(attribute_matrix1.values)
    attribute_matrix2 = attribute_matrix.iloc[test_index,:]
    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        pre_res = test_pre_attribute[i, :]
        loc = (np.sum(np.square(attribute_matrix2.values - pre_res), axis=1)).argmin()
        label_lis.append(attribute_matrix2.index[loc]+1)
    label_lis = np.array(np.row_stack(label_lis))
    return label_lis

# Confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)

    # Sample numbers of each class
    unique_classes = np.unique(y_true)
    class_counts = np.bincount(y_true)
    class_counts = class_counts[unique_classes]

    # Prediction accuracy
    cm_normalized = cm.astype('float') / class_counts[:, np.newaxis]
    Average_accuracy = accuracy_score(y_true, y_pred)
    print("Average_accuracy:", Average_accuracy)

    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Confusion Matrix (Normalized)')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = cm_normalized.max() / 2.
    for i, j in ((i, j) for i in range(cm_normalized.shape[0]) for j in range(cm_normalized.shape[1])):
        plt.text(j, i, "{:.2f}".format(cm_normalized[i, j]),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    random.seed(42)
    Training_Flag = 1       # Need training? 0-No, 1-Yes
    Problem_index = 1       # Select ZSL case
    if Problem_index == 1:
        print("================= Problem 1: ZSL for Transition States =================")
        Fea_Attri_model = 'SVC_rbf'  # 'NB','SVC_rbf','rf','Lasso','SVC_linear','Ridge','lr'
        attribute_matrix = pd.read_excel('./gw_attribute1.xlsx', index_col='no')
        testindex = [6, 7, 8, 9]    # 6-b2s,7-p2s,8-s2a,9-st2w
        traindata, train_attributelabel, testdata, testlabel = creat_dataset_problem1(testindex,attribute_matrix)
        classes = ['State 7', 'State 8', 'State 9', 'State 10']
    elif Problem_index == 2:
        print("================= Problem 2: ZSL for Unknown Typical States =================")
        Fea_Attri_model = 'lr'
        attribute_matrix = pd.read_excel('./gw_attribute.xlsx', index_col='no')
        testindex = [2, 3]       # 0-bubble, 1-plug, 2-slug, 3-wave, 4-stratified, 5-annular
        traindata, train_attributelabel, testdata, testlabel = creat_dataset_problem2(testindex,attribute_matrix)
        classes = ['State 3', 'State 4']
    elif Problem_index == 3:
        print("================= Problem 3: ZSL for Typical States in Unknown Working Conditions =================")
        Fea_Attri_model = 'SVC_rbf'
        attribute_matrix = pd.read_excel('./gw_attribute.xlsx', index_col='no')
        testindex = [0, 1, 2, 3, 4, 5]
        traindata, train_attributelabel, testdata, testlabel = creat_dataset_problem3(testindex,attribute_matrix)
        classes = ['State 1-2', 'State 2-2', 'State 3-2', 'State 4-2', 'State 5-2', 'State 6-2']

    # Data normalization
    traindata, X_means, X_std = center_data(traindata.T)
    traindata = traindata.T
    testdata = (testdata.T - X_means[:, np.newaxis])/X_std[:, np.newaxis]
    testdata = testdata.T

    # Hyper-parameter setting
    dimension_fea = 10
    S2FANet = SiameseNetwork()

    epochs = 100
    batch_size = 490
    learning_rate = 0.001
    para_cov = 1
    para_slowness = 0.05
    para_steadiness = 0.01
    para_error = 1

    num1 = traindata.shape[0]
    num2 = traindata.shape[1]
    X_train = traindata
    Y_train = train_attributelabel
    X_train, Y_train = torch.FloatTensor(X_train), torch.FloatTensor(Y_train)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

    if Training_Flag == 1:
        print("================= S2FA Siamese Network Training =================")
        criterion_attribute = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(S2FANet.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(S2FANet.parameters(), lr=learning_rate, momentum=0.9)

        '''Training'''
        loss_history = []
        I = torch.eye(dimension_fea)

        # S2FA
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        for epoch in range(epochs):
            for data in train_loader:
                X_data, Y_data = data[0], data[1]
                X1, Y1 = X_data[0:batch_size-2], Y_data[0:batch_size-2]
                X2 = X_data[1:batch_size-1]
                X3 = X_data[2:batch_size]
                output1, output2, output3, output_attri = S2FANet(X1, X2, X3)
                
                loss_error = criterion_attribute(output_attri, Y1)    # Attribute loss
                Cov = (output1.T @ output1)/(batch_size-1)            # Covariance loss
                loss_cov = (Cov-I).norm(p="fro")
                loss_slowness = (output1-output2).norm(p="fro")            # Slowness loss
                loss_steadiness = (output1-2*output2+output3).norm(p="fro")            # Steadiness loss
                loss = para_cov * loss_cov + para_slowness * loss_slowness + para_steadiness * loss_steadiness + para_error * loss_error            # Total loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 1 == 0:
                print(
                    "Epoch number {}\n Loss_slowness {}\n Loss_steadiness {}\n Loss_cov {}\n Loss_error {}\n Total_loss {}\n".format(epoch,
                                                                                                                     loss_slowness.item(),
                                                                                                                     loss_steadiness.item(),
                                                                                                                     loss_cov.item(),
                                                                                                                     loss_error.item(),
                                                                                                                     loss.item()))
                loss_history.append(loss.item())

        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label='Training Loss', color='blue')
        plt.title('Training Loss over Steps')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        if Problem_index == 1:
            torch.save(S2FANet.state_dict(), 'weights/Problem1_S2FA.pkl')
        elif Problem_index == 2:
            torch.save(S2FANet.state_dict(), 'weights/Problem2_S2FA.pkl')
        elif Problem_index == 3:
            torch.save(S2FANet.state_dict(), 'weights/Problem3_S2FA.pkl')

    elif Training_Flag == 0:
        pass

    print("================= S2FA Siamese Network Testing =================")
    # Load network weights
    net_feature_trained = SiameseNetwork()
    if Problem_index == 1:
        net_feature_trained.load_state_dict(torch.load(r'./weights/Problem1_S2FA.pkl'))
    elif Problem_index == 2:
        net_feature_trained.load_state_dict(torch.load(r'./weights/Problem2_S2FA.pkl'))
    elif Problem_index == 3:
        net_feature_trained.load_state_dict(torch.load(r'./weights/Problem3_S2FA.pkl'))

    # Load test data
    num1 = testdata.shape[0]
    num2 = testdata.shape[1]
    X_test = testdata[0:num1,0:num2].astype(float)
    Y_test = testdata[0:num1:,0]
    X_test, Y_test = torch.FloatTensor(X_test), torch.LongTensor(Y_test)
    X1_test, Y1 = X_test[0:num1-2], Y_test[0:num1-2]
    X2_test = X_test[1:num1-1]
    X3_test = X_test[2:num1]

    # Feature extraction
    output1, output2, output3, output_attri_test = net_feature_trained(X1_test, X2_test, X3_test)
    output1_train, output2_train, output3_train, output_attri_train = net_feature_trained(X_train, X_train, X_train)
    trainfeature = output1_train.detach().numpy()
    testfeature = output1.detach().numpy()

    # Attribute prediction
    print("Attribute prediction...")
    test_pre_attribute = pre_attribute_model(Fea_Attri_model, trainfeature, train_attributelabel, testfeature)

    # State identification
    print("Flow state identification...")
    label_lis = pre_state_label(test_pre_attribute, attribute_matrix, testindex)
    test_pre_attribute = np.array(test_pre_attribute)
    Mean_attribute = np.average(test_pre_attribute, axis=0)

    # Confusion matrix
    testlabel = testlabel[2:len(testlabel)]
    plot_confusion_matrix(testlabel, label_lis, classes)