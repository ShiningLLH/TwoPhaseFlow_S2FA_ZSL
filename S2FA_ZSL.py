from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.io import loadmat
from Siamese_S2FA import SiameseNetwork
from sklearn.metrics import confusion_matrix
import pandas as pd
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def creat_dataset_problem(problem_type, test_index, attribute_matrix):
    attribute_matrix = attribute_matrix.values
    if problem_type == 1:
        path = './GW_mat_data/Dataset1/'
        train_data_files = ['data_bubble.mat', 'data_plug.mat', 'data_slug.mat', 'data_wave.mat', 'data_st.mat',
                      'data_ann.mat']
        train_data_list = [loadmat(path + file)['data'] for file in train_data_files]
        train_data = np.row_stack(train_data_list)
        train_attributelabel = np.row_stack(
            [np.tile(attribute_matrix[i, :], (train_data_list[i].shape[0], 1)) for i in range(len(train_data_list))])

        test_data_files = ['data_b2s.mat', 'data_p2s.mat', 'data_s2a.mat', 'data_st2w.mat']
        test_data_list = [loadmat(path + file)['data'][500:700, :] for file in test_data_files]
        test_data = np.row_stack(test_data_list)
        test_label = []
        for item in test_index:
            test_label += [item + 1] * (test_data_list[item - 6].shape[0])

    elif problem_type == 2:
        path = './GW_mat_data/Dataset1/'
        data_files = ['data_bubble.mat', 'data_plug.mat', 'data_slug.mat', 'data_wave.mat', 'data_st.mat',
                            'data_ann.mat']

        train_data_list = [loadmat(path + file)['data'] for file in data_files]
        test_data_list = [loadmat(path + file)['data'][0:300] for file in data_files]

        train_index = list(set(np.arange(6)) - set(test_index))

        train_data = []
        train_attributelabel = []
        for item in train_index:
            train_data.append(train_data_list[item])
            train_attributelabel.append(np.tile(attribute_matrix[item, :], (train_data_list[item].shape[0], 1)))
        train_data = np.row_stack(train_data)
        train_attributelabel = np.row_stack(train_attributelabel)

        test_data = []
        test_label = []
        for item in test_index:
            test_data.append(test_data_list[item])
            test_label += [item + 1] * test_data_list[item].shape[0]
        test_data = np.row_stack(test_data)

    elif problem_type == 3:
        path = './GW_mat_data/Dataset2/'
        train_data_files = ['data_bubble.mat', 'data_plug.mat', 'data_slug.mat', 'data_wave.mat', 'data_st.mat',
                      'data_ann.mat']
        train_data_list = [loadmat(path + file)['data'] for file in train_data_files]
        train_data = np.row_stack(train_data_list)
        train_attributelabel = np.row_stack(
            [np.tile(attribute_matrix[i, :], (train_data_list[i].shape[0], 1)) for i in range(len(train_data_list))])

        # 1-1 bubble, 2-2 plug, 1-6 slug, 5-6 wave, 5-1 stratified, 1-9 annular
        test_data_files = ['data_1-1.mat', 'data_2-2.mat', 'data_1-6.mat', 'data_5-6.mat', 'data_5-1.mat', 'data_1-9.mat']
        test_data_list = [loadmat(path + file)['data'][0:300] for file in test_data_files]
        test_data = np.row_stack(test_data_list)
        test_label = []
        for item in test_index:
            test_label += [item + 1] * (test_data_list[item].shape[0])

    return train_data, train_attributelabel, test_data, test_label


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

def pre_state_label(test_pre_attribute, attribute_matrix1, test_index):
    attribute_matrix = pd.DataFrame(attribute_matrix1.values)
    attribute_matrix2 = attribute_matrix.iloc[test_index, :]
    label_lis = []
    for i in range(test_pre_attribute.shape[0]):
        pre_res = test_pre_attribute[i, :]
        loc = (np.sum(np.square(attribute_matrix2.values - pre_res), axis=1)).argmin()
        label_lis.append(attribute_matrix2.index[loc] + 1)
    label_lis = np.array(np.row_stack(label_lis))
    return label_lis

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

def train_siamese_network(S2FANet, train_loader, criterion_attribute, optimizer, dimension_fea, epochs, para_steadiness, para_cov, para_error):
    S2FANet.train()
    I = torch.eye(dimension_fea)
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in train_loader:
            X_data, Y_data = data[0], data[1]

            X1, Y1 = X_data[:-2], Y_data[:-2]
            X2 = X_data[1:-1]
            X3 = X_data[2:]

            # Forward pass
            output1, output2, output3, output_attri = S2FANet(X1, X2, X3)

            # loss
            loss_error = criterion_attribute(output_attri, Y1)
            Cov = (output1.T @ output1) / (output1.shape[0] - 1)
            loss_cov = (Cov - I).norm(p="fro")
            loss_slowness = (output1 - output2).norm(p="fro")
            loss_steadiness = (output1 - 2 * output2 + output3).norm(p="fro")
            total_loss = loss_slowness + para_steadiness * loss_steadiness + para_cov * loss_cov + para_error * loss_error

            # backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Total_loss: {round(avg_epoch_loss, 4)} | loss_cov: {round(loss_cov.item(), 4)}"
                  f" | loss_slowness: {round(loss_slowness.item(), 4)} | loss_error:{round(loss_error.item(), 4)}")

    # plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss', color='blue')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_model(S2FANet, X_test, X_train):
    S2FANet.eval()
    with torch.no_grad():
        output1, _, _, _ = S2FANet(X_test, X_test, X_test)
        output1_train, _, _, _ = S2FANet(X_train, X_train, X_train)
        trainfeature = output1_train.detach().numpy()
        testfeature = output1.detach().numpy()
    return trainfeature, testfeature


if __name__ == '__main__':

    set_random_seed()
    S2FANet = SiameseNetwork()
    Problem_type = 3       # Select ZSL case
    
    if Problem_type == 1:
        print("================= Problem 1: ZSL for Transition States =================")
        Fea_Attri_model = 'rf'  # 'NB','SVC_rbf','rf','Lasso','SVC_linear','Ridge','lr'
        attribute_matrix = pd.read_excel('./gw_attribute1.xlsx', index_col='no')
        test_index = [6, 7, 8, 9]    # 6-b2s,7-p2s,8-s2a,9-st2w
        classes = ['State 7', 'State 8', 'State 9', 'State 10']
        
    elif Problem_type == 2:
        print("================= Problem 2: ZSL for Unknown Typical States =================")
        Fea_Attri_model = 'rf'
        attribute_matrix = pd.read_excel('./gw_attribute.xlsx', index_col='no')
        test_index = [2, 3]       # 0-bubble, 1-plug, 2-slug, 3-wave, 4-stratified, 5-annular
        classes = ['State 3', 'State 4']
        
    elif Problem_type == 3:
        print("================= Problem 3: ZSL for Typical States in Unknown Working Conditions =================")
        Fea_Attri_model = 'rf'
        attribute_matrix = pd.read_excel('./gw_attribute.xlsx', index_col='no')
        test_index = [0, 1, 2, 3, 4, 5]
        classes = ['State 1-2', 'State 2-2', 'State 3-2', 'State 4-2', 'State 5-2', 'State 6-2']

    else:
        raise ValueError("Invalid Problem Type.")

    # Create dataset
    train_data, train_attributelabel, test_data, test_label = creat_dataset_problem(Problem_type, test_index, attribute_matrix)

    # Parameter setting
    dimension_fea = 10
    epochs = 300
    batch_size = 64
    learning_rate = 0.001
    para_steadiness = 2
    para_cov = 2
    para_error = 2

    X_train = torch.FloatTensor(train_data)
    Y_train = torch.FloatTensor(train_attributelabel)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

    criterion_attribute = nn.MSELoss()
    optimizer = optim.Adam(S2FANet.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-5)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # training
    print("================= S2FA Siamese Network Training =================")
    train_siamese_network(S2FANet, train_loader, criterion_attribute, optimizer, dimension_fea, epochs, para_steadiness, para_cov, para_error)

    # testing
    print("================= S2FA Siamese Network Testing =================")
    X_test = torch.tensor(test_data, dtype=torch.float32)
    trainfeature, testfeature = test_model(S2FANet, X_test, X_train)

    # Attribute prediction
    print("Attribute prediction...")
    test_pre_attribute = pre_attribute_model(Fea_Attri_model, trainfeature, train_attributelabel, testfeature)

    # State identification
    print("Flow state identification...")
    pre_label_lis = pre_state_label(test_pre_attribute, attribute_matrix, test_index)

    # Confusion matrix
    plot_confusion_matrix(test_label, pre_label_lis, classes)
