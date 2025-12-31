from matplotlib import pyplot as plt
from cnn_capsnet_model import *
# from trip_model import *
import numpy as np
from pytorchtools import EarlyStopping
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import json
from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
import logging
from sklearn.preprocessing import MinMaxScaler
logging.basicConfig(level=logging.INFO)

t_name = 'ames_275_11td_100'
txt_name = 'ames_275'
project_file_path = "/home/cpj/ybj/force_file"
# project_file_path = 'D:/yanbujian/PyCharm Community Edition 2023.1/xiangmu1/xiangmu1.2/force file'
# 'D:/yanbujian/PyCharm Community Edition 2023.1/xiangmu1/xiangmu1.2/force file'
# "/home/lxq/CPJ/force_file"
# "/home/cpj/ybj/force_file"
print(os.path.exists(project_file_path), project_file_path)
# training_files_path = "{}/training_files".format(project_file_path)
# result_files_path = "{}/result_files".format(project_file_path)trained
trained_models_path = "{}/{}/{}/trained_models".format(project_file_path, txt_name,t_name)
best_model = "{}/best_network.pth".format(trained_models_path)
# '/home/cpj/ybj/force_file_data/CYP2C9_json/'
# '/home/lxq/CPJ/force_file_data/CYP2C9_json/'  '/home/lxq/CPJ/force_file_data/{}_json/'.format(txt_name)'
# output_path ='/home/lxq/CPJ/force_file_data/{}_all_json/{}_11td_json'.format(t_name, txt_name)
# output_path ='/home/lxq/CPJ/force_file_data/ames_all_json/{}_dayu4_11td_json'.format(t_name)
output_path ='/home/cpj/ybj/force_file_data/ames_all_json/{}_json'.format(t_name)
# output_path = 'D:/yanbujian/PyCharm Community Edition 2023.1/xiangmu1/xiangmu1.2/herg_braga_set/mini4'
# weightdc = 0.003
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
r_number = 2020
def setup_seed(seed):
    random.seed((seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class Mol_Dataset(Dataset):
    def __init__(self, data, train_x, val_x, test_x, train_y, val_y, test_y):
        # 定义需要读取的文件是训练集验证集还是测试集
        if data == 'train':
            self.seq, self.label = train_x, train_y
        elif data == 'val':
            self.seq, self.label = val_x, val_y
        elif data == 'test':
            self.seq, self.label = test_x, test_y

    def __getitem__(self, index):
        return self.seq[index].to(device), self.label[index]

    def __len__(self):
        return len(self.seq)


def load_data(batch_size, train_x, val_x, test_x, train_y, val_y, test_y):
    train_dataset = Mol_Dataset('train', train_x, val_x, test_x, train_y, val_y, test_y)
    train_dataloder = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = Mol_Dataset('val', train_x, val_x, test_x, train_y, val_y, test_y)
    val_dataloder = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataset = Mol_Dataset('test', train_x, val_x, test_x, train_y, val_y, test_y)
    test_dataloder = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloder, val_dataloder, test_dataloder

def check_data(data):
    if torch.isnan(data).any() or torch.isinf(data).any():
        logging.info(f"Data stats - mean: {data.mean()}, std: {data.std()}")
        raise ValueError("Input data contains NaN or infinity")

def preprocess_data(data):
    device = data.device  # 获取数据所在的设备
    original_nan_count = torch.sum(torch.isnan(data)).item()
    original_inf_count = torch.sum(torch.isinf(data)).item()
    if original_nan_count != 0 or original_inf_count != 0:
      print('NAN数量为：', original_nan_count, 'inf数量为：', original_inf_count)
    data = torch.where(torch.isnan(data), torch.tensor(0.0, device=device), data)
    data = torch.where(torch.isinf(data), torch.tensor(0.0, device=device), data)
    return data
# 假设你使用的是NumPy数组或Pandas DataFrame
# device = torch.device("cpu")
def traink(model, batch_size, train_x, val_x, test_x, train_y, val_y, test_y, weights, learning_rate, TOTAL_EPOCHS, patience):
    train_loader, val_loader, test_loader = load_data(batch_size, train_x, val_x, test_x, train_y, val_y, test_y)
    model = model.to(device)

    # criterion = CapsuleLoss()
    criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weightdc)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=trained_models_path)
    max_norm = 1.0

    losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    SE = []
    SP = []
    val_AUC = []

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        y_true_train_list = []
        y_score_train_list = []
        y_pre_train_list = []
        for i, (features, labels) in enumerate(train_loader):
            labels_0 = torch.eye(2).index_select(dim=0, index=labels).to(device)
            optimizer.zero_grad()
            features = preprocess_data(features)
            features = features.to(device)
            check_data(features)
            # features = torch.reshape(features, (-1, 1, 97, 97)).to(device)
            logits = model(features)
            # logits = preprocess_data(logits)
            check_data(logits)
            loss = criterion(logits, labels.long().to(device))
            #
            # loss = criterion(logits, labels_0.float().to(device))

            train_loss += loss.item() * len(labels_0)
            check_data(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            # print(f"Epoch {epoch}: Loss = {loss.item()}")
            correct += torch.sum(
                torch.argmax(logits, dim=1) == torch.argmax(labels_0, dim=1)).item()
            y_true_train = labels.view(-1, 1)
            y_score_train = logits[:, 1].view(-1, 1)
            for j in logits:
                y_pre_train_list.append(torch.argmax(j))
            y_true_train = y_true_train.cpu()
            y_score_train = y_score_train.cpu()
            y_true_train_list.append(y_true_train)
            y_score_train_list.append(y_score_train)
        y_true_all = torch.cat(y_true_train_list, dim=0)
        y_score_all = torch.cat(y_score_train_list, dim=0)
        y_true_all_1 = y_true_all.detach().numpy()
        y_score_all_1 = y_score_all.detach().numpy()
        y_true_all_1 = np.array(y_true_all_1)
        y_score_all_1 = np.array(y_score_all_1)
        auc = roc_auc_score(y_true_all_1, y_score_all_1)
        accuracy = 100. * correct / len(train_x)
        epoch_train_loss = (train_loss / len(train_x))
        epoch_loss = epoch_train_loss
        print('Epoch: {}, Loss: {:.5f}, Training set accuracy: {}/{} ({:.3f}%)'
              .format(epoch + 1, epoch_loss, correct, len(train_x), accuracy))
        train_acc.append(accuracy)

        y_true_all = y_true_all.flatten()
        y_pre_list = torch.stack(y_pre_train_list)
        train_perf_dict = dict()
        train_perf_dict["MCC"] = 0.0
        try:
            train_perf_dict = prec_rec_f1_acc_mcc(y_true_all, y_pre_list)
            print('train_AUC = ', auc, 'train_perf_dict', train_perf_dict)
        except:
            print("There was a problem during test performance calculation!")

        model.eval()
        val_loss = 0
        correct = 0
        TP, TN, FP, FN = 0, 0, 0, 0
        y_true_list = []
        y_score_list = []
        y_pre_list = []
        with torch.no_grad():
            for i, (features, labels) in enumerate(val_loader):
                labels_1 = torch.eye(2).index_select(dim=0, index=labels).to(device)
                features = preprocess_data(features)
                features = features.to(device)
                # features = torch.reshape(features, (-1, 1, 97, 97)).to(device)
                logits = model(features)
                # loss = criterion(logits, labels_1).item()
                loss = criterion(logits, labels.long().to(device))
                #
                # loss = criterion(logits, labels_1.float().to(device))

                val_loss += loss.item() * len(labels_1)

                for idx, i in enumerate(logits):
                    if torch.argmax(i) == torch.argmax(labels_1[idx]) and torch.argmax(i) == 1:
                        TP += 1
                    if torch.argmax(i) == torch.argmax(labels_1[idx]) and torch.argmax(i) == 0:
                        TN += 1
                    if torch.argmax(i) != torch.argmax(labels_1[idx]) and torch.argmax(i) == 1:
                        FP += 1
                    if torch.argmax(i) != torch.argmax(labels_1[idx]) and torch.argmax(i) == 0:
                        FN += 1

                correct += torch.sum(
                    torch.argmax(logits, dim=1) == torch.argmax(labels_1, dim=1)).item()

                y_true = labels.view(-1, 1)
                y_score = logits[:, 1].view(-1, 1)
                for j in logits:
                    y_pre_list.append(torch.argmax(j))
                y_true = y_true.cpu()
                y_score = y_score.cpu()
                y_true_list.append(y_true)
                y_score_list.append(y_score)
        y_true_all = torch.cat(y_true_list, dim=0)
        y_score_all = torch.cat(y_score_list, dim=0)
        auc = roc_auc_score(y_true_all, y_score_all)
        sp = TN / (TN + FP)
        se = TP / (TP + FN)

        epoch_val_loss = (val_loss / len(val_x))
        losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        epoch_loss = epoch_val_loss
        SE.append(se)
        SP.append(sp)
        val_AUC.append(auc)
        print('SE = {},SP = {}'.format(se, sp))

        y_true_all = y_true_all.flatten()
        y_pre_list = torch.stack(y_pre_list)
        val_perf_dict = dict()
        val_perf_dict["MCC"] = 0.0
        try:
            val_perf_dict = prec_rec_f1_acc_mcc(y_true_all, y_pre_list)
            print('val_AUC = ', auc, 'val_perf_dict', val_perf_dict)
        except:
            print("There was a problem during test performance calculation!")

        accuracy = 100. * correct / len(val_x)
        print('val set: val Average loss: {:.4f}, val Accuracy: {}/{} ({:.3f}%)\n val AUC: {:.3f}'
              ', val SE: {:.3f}, val SP: {:.3f}\n'.format(
            epoch_loss, correct, len(val_x), accuracy, auc, se, sp))

        val_acc.append(accuracy)
        scheduler.step(auc)

        early_stopping(auc, model)

        if early_stopping.early_stop:
            print("Early stopping")

            break

    y_true_list = []
    y_score_list = []
    y_pre_list = []
    test_loss = 0
    model.load_state_dict(torch.load(best_model))
    model.eval()
    with torch.no_grad():
        for i, (features, labels) in enumerate(test_loader):
            labels_2 = torch.eye(2).index_select(dim=0, index=labels).to(device)
            features = preprocess_data(features)
            features = features.to(device)
            # features = torch.reshape(features, (-1, 1, 97, 97)).to(device)
            logits = model(features)
            loss = criterion(logits, labels.long().to(device))
            #
            # loss = criterion(logits, labels_2.float().to(device))

            test_loss += loss * len(labels_2)
            y_true = labels.view(-1, 1)
            y_score = logits[:, 1].view(-1, 1)
            for j in logits:
                y_pre_list.append(torch.argmax(j))
            y_true = y_true.cpu()
            y_score = y_score.cpu()
            y_true_list.append(y_true)
            y_score_list.append(y_score)
        y_true_all = torch.cat(y_true_list, dim=0)
        y_score_all = torch.cat(y_score_list, dim=0)

        test_auc = roc_auc_score(y_true_all, y_score_all)
        test_loss = test_loss / len(val_x)

        y_true_all = y_true_all.flatten()
        y_pre_list = torch.stack(y_pre_list)
        test_perf_dict = dict()
        test_perf_dict["MCC"] = 0.0
        try:
            test_perf_dict = prec_rec_f1_acc_mcc(y_true_all, y_pre_list)
            print('test_AUC = ', test_auc, 'test_perf_dict', test_perf_dict)
        except:
            print("There was a problem during test performance calculation!")

    return losses, val_losses, test_loss, train_acc, val_acc, val_AUC, test_auc


def prec_rec_f1_acc_mcc(y_true, y_pred):
    performance_threshold_dict = dict()

    ### ADDED on 28 July 2020 by YIP YEW MUN ###
    y_true_tmp = []
    for each_y_true in y_true:
        #         print('each_y_true = ',each_y_true)
        y_true_tmp.append(each_y_true.item())
    y_true = y_true_tmp
    #     print('y_true = ',y_true)

    y_pred_tmp = []
    for each_y_pred in y_pred:
        # print('each_y_pred = ', each_y_pred)
        y_pred_tmp.append(each_y_pred.item())
    y_pred = y_pred_tmp
    #     print('y_true = ',y_true,'y_pred = ',y_pred)
    ### ADDED on 28 July 2020 by YIP YEW MUN ###

    precision = metrics.precision_score(y_true, y_pred, pos_label=0)
    recall = metrics.recall_score(y_true, y_pred, pos_label=0)
    f1_score = metrics.f1_score(y_true, y_pred, pos_label=0)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1, 0]).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    performance_threshold_dict["Accuracy"] = accuracy
    performance_threshold_dict["MCC"] = mcc
    performance_threshold_dict["Precision"] = precision
    performance_threshold_dict["Recall"] = recall
    performance_threshold_dict["SPE"] = spe
    performance_threshold_dict["F1-Score"] = f1_score
    performance_threshold_dict["TP"] = tp
    performance_threshold_dict["FP"] = fp
    performance_threshold_dict["TN"] = tn
    performance_threshold_dict["FN"] = fn
    return performance_threshold_dict

def load_data_from_json(output_path):
    des_list = []
    Y_train = []

    for filename in os.listdir(output_path):
        if filename.endswith(".json"):
            file_path = os.path.join(output_path, filename)
            with open(file_path, 'r') as json_file:
                data_dict = json.load(json_file)

                E_list = data_dict['E']
                E_tensor = torch.tensor(E_list, dtype=torch.float32)  # 将列表还原为Tensor张量
                Y = data_dict['Y']
                des_list.append(E_tensor)
                Y_train.append(Y)
    class_counts = dict(Counter(Y_train))
    total_samples = len(Y_train)
    weights = [total_samples / (class_counts[i] * len(class_counts)) for i in range(len(class_counts))]
    weights = torch.tensor(weights, dtype=torch.float32, device=device).clone().detach()

    return des_list, Y_train, weights


def parti_pro(des_list, Y_train):
    train_x, val_test_x, train_y, val_test_y = train_test_split(
        des_list, Y_train, test_size=0.2, random_state=0, shuffle=True)
    val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5, random_state=0,
                                                      shuffle=True)
    # , stratify = val_test_y,
    #         stratify=Y_train)

    return train_x, val_x, test_x, train_y, val_y, test_y


def k_fold(des_list, Y_train, weights, num_epochs=100, learning_rate=0.00001, batch_size=32, patience=5, dropout=0.2):
    print('正在预测的体系是', t_name)
    print("对于本体系,num_epochs={}, learning_rate={}, batch_size={}, 早停耐心 = {}, dropout={}".format(num_epochs, learning_rate,
                                                                                   batch_size, patience, dropout))
    train_x, val_x, test_x, train_y, val_y, test_y = parti_pro(des_list, Y_train)
    # model = FusionCNNModel()
    # model = Mymodel(dropout)
    model = Mymodel(dropout=0.2)
    # model = UniFormer_Light_Binary(
    #     depth=[2, 2, 2, 2],
    #     embed_dim=[32, 64, 128, 256],
    #     head_dim=16,
    #     conv_stem=True,
    #     mlp_ratio=[3, 3, 3, 3],
    #     prune_ratio=[[], [], [1, 0.5, 0.5], [0.5, 0.5]],
    #     trade_off=[[], [], [1, 0.5, 0.5], [0.5, 0.5]]
    # )
    train_loss, val_loss, test_loss, train_acc, val_acc, val_auc, test_auc = traink(model, batch_size, train_x,
                                                                                    val_x, test_x, train_y,
                                                                                    val_y, test_y, weights,
                                                                                    learning_rate, num_epochs, patience)
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.title('Loss over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # # 绘制 Accuracy 曲线
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.title('Accuracy over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()
    # plt.grid()
    # plt.show()
    index = val_auc.index(max(val_auc))

    print('\n', '#' * 10, '最终训练（验证集上AUC值最大的那个epoch的训练集的结果）验证测试（最终的测试集上的结果）结果', '#' * 10)
    print('train_loss:{:.5f}, train_acc:{:.3f}%'.format(train_loss[index], train_acc[index]))
    print('valid loss:{:.5f}, valid_acc:{:.3f}%'.format(val_loss[index], val_acc[index]))
    print('test loss:{:.5f}, test AUC:{:.3f}\n'.format(test_loss, test_auc))

    return


if __name__ == '__main__':
    setup_seed(r_number)
    print('随机种子是：{}'.format(r_number))
    des_list, Y_train, weights = load_data_from_json(output_path)
    k_fold(des_list, Y_train, weights, num_epochs=100, learning_rate=0.0001, batch_size=32, patience=5, dropout=0.2)
