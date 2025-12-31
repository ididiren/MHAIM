from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # 使用非图形界面后端
from trip_model_keshihua import *
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import json
from collections import Counter
import matplotlib.pyplot as plt

def setup_seed(seed):
    random.seed((seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2020)

t_name = 'ames_275_11td_100'
output_path ='/home/cpj/ybj/force_file_data/ames_all_json/{}_json'.format(t_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def preprocess_data(data):
    # device = data.device  # 获取数据所在的设备
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)  # 将 NaN 替换为 0.0
    return data
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
des_list_0, labels, _ = load_data_from_json(output_path)
labels = np.array(labels)
des_list_0 = np.array(des_list_0)  # 将列表转换为 NumPy 数组
des_list = des_list_0.reshape(des_list_0.shape[0], -1)  # (num_samples, 10000)
des_list = preprocess_data(des_list)
scaler = StandardScaler()
des_list = scaler.fit_transform(des_list)

# # 使用UMAP将摩根指纹降到2D
# umap_reducer = umap.UMAP(
#     n_components=2,
#     n_neighbors=15,  # 常用值，影响局部结构
#     min_dist=0.1,    # 越小，越倾向于保留局部聚类
#     metric='cosine', # 摩根指纹通常使用余弦距离
#     random_state=42
# )
# embedding = umap_reducer.fit_transform(fingerprints)
#
# # 可视化
# plt.figure(figsize=(10, 7))
#
# # 假设 `labels` 中只有两个类别，分别标记为 0 和 1
# plt.scatter(embedding[labels == 0, 0], embedding[labels == 0, 1], s=5, label='Inactive', alpha=0.7)
# plt.scatter(embedding[labels == 1, 0], embedding[labels == 1, 1], s=5, label='Active', alpha=0.7)
#
# plt.xlabel('UMAP Dimension 1')
# plt.ylabel('UMAP Dimension 2')
# plt.title('UMAP Visualization of Molecular Fingerprints')
# plt.legend()
# plt.show()
# plt.savefig('umap_visualization.png', dpi=300)  # 保存为 PNG 格式，300 dpi 的分辨率

# 设置 T-SNE 参数
tsne = TSNE(
    n_components=2,  # 降到 2D 空间
    perplexity=30,   # 平衡局部和全局结构，常用值 5-50
    learning_rate=200,  # 学习率
    max_iter=1000,      # 最大迭代次数
    random_state=42   # 固定随机种子，确保可重复性
)

# 将摩根指纹降维到 2D
embedding = tsne.fit_transform(des_list)
# print("Min value:", embedding.min(axis=0))
# print("Max value:", embedding.max(axis=0))
# print('embedding[:5]=', embedding[:5])
# 可视化
plt.figure(figsize=(10, 7))
# print('len(labels)=', len(labels))  # 检查 labels 数组的长度
# print('np.unique(labels)=', np.unique(labels))
# print('embedding.shape=', embedding.shape)
# 假设 labels 包含 0 和 1，分别表示 Inactive 和 Active
plt.scatter(embedding[labels == 0, 0], embedding[labels == 0, 1], s=15, label='Inactive',
            alpha=0.8, color=(66/255, 140/255, 164/255),edgecolors='none')
plt.scatter(embedding[labels == 1, 0], embedding[labels == 1, 1], s=15, label='Active',
            alpha=0.8, color=(246/255, 212/255, 109/255),edgecolors='none')
# plt.scatter(embedding[labels == 0][:, 0], embedding[labels == 0][:, 1], s=50, label='Inactive', color=(66/255, 140/255, 164/255))
# plt.scatter(embedding[labels == 1][:, 0], embedding[labels == 1][:, 1], s=50, label='Active', color=(246/255, 212/255, 109/255))
# print('labels[:5]=', labels[:5])
# plt.xlim(-100, 100)
# plt.ylim(-100, 100)
plt.xlabel('x')
plt.ylabel('y')
# plt.title('T-SNE Visualization of Molecular Fingerprints')
plt.legend()
plt.savefig('/home/cpj/ybj/ames275_tsne_yuan.png', dpi=300)  # 保存为 PNG 格式，300 dpi 的分辨率