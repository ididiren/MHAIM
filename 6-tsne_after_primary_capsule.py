from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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

name = 'keshihua_model'
txt_name = 'ames_275'
t_name = 'ames_275_11td_100'
project_file_path = "/home/cpj/ybj/force_file"
print(os.path.exists(project_file_path), project_file_path)
trained_models_path = "{}/{}/{}/trained_models".format(project_file_path, txt_name,name)
best_model = "{}/best_network.pth".format(trained_models_path)
output_path ='/home/cpj/ybj/force_file_data/ames_all_json/{}_json'.format(t_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
des_list, Y_train, _ = load_data_from_json(output_path)

class Mol_Dataset(Dataset):
    def __init__(self, des_list, Y_train):
        self.seq = des_list
        self.label = Y_train

    def __getitem__(self, index):
        return self.seq[index], torch.tensor(self.label[index], dtype=torch.long)

    def __len__(self):
        return len(self.seq)

# 创建 DataLoader
batch_size = 32  # 根据显存大小选择合适的批次大小
dataset = Mol_Dataset(des_list, Y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
model = Mymodel()
model.load_state_dict(torch.load(best_model))
model.to(device)
model.eval()
# 存储全连接层输出的列表
feature_maps = []
# 定义钩子函数，用于捕获 `self.linear` 最后一个层的输出
def hook_fn(module, input, output):
    feature_map = np.nan_to_num(output.cpu().numpy(), nan=0, posinf=0, neginf=0)
    feature_maps.append(feature_map)
# 注册钩子到 `self.linear` 的最后一层（假设是 `nn.Linear(512, 64)`）
# hook = model.linear[-1].register_forward_hook(hook_fn)
hook = model.pri.register_forward_hook(hook_fn)
all_labels = []  # 存储标签
with torch.no_grad():
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        model(inputs)
        all_labels.extend(labels.cpu().numpy())  # 将标签从 GPU 转移到 CPU # 保存标签
# 将钩子捕获到的所有特征拼接成UMAP的输入
reshaped_maps = [fm.reshape(fm.shape[0], -1) for fm in feature_maps]  # 对每个批次的 feature_map 进行 reshape
feature_maps = np.concatenate(reshaped_maps, axis=0) # Reshapes to (batch_size, height*width)
scaler = StandardScaler()
feature_maps = scaler.fit_transform(feature_maps)
# print("fc_outputs shape:", feature_maps.shape)
all_labels = np.array(all_labels)
# print(all_labels[:5])
# 移除钩子
hook.remove()
# 使用UMAP将64维特征降到2D
# umap_reducer = umap.UMAP(n_components=2, n_neighbors=100, spread=2.0, min_dist=0.2, metric='cosine', target_metric='categorical')
# embedding = umap_reducer.fit_transform(fc_outputs)  # (样本数, 2)
#
# # 可视化：假设标签0表示Inactive，1表示Active
# plt.figure(figsize=(10, 7))
# plt.scatter(embedding[all_labels == 0, 0], embedding[all_labels == 0, 1], s=5, label='Inactive', alpha=0.7)
# plt.scatter(embedding[all_labels == 1, 0], embedding[all_labels == 1, 1], s=5, label='Active', alpha=0.7)
# plt.xlabel('UMAP Dimension 1')
# plt.ylabel('UMAP Dimension 2')
# plt.title('UMAP Visualization of 64-dimensional FC Layer Output')
# plt.legend()
# plt.show()

# 使用 T-SNE 将 64 维特征降到 2D
tsne = TSNE(
    n_components=2,      # 降维到 2D
    perplexity=5,       # 平衡局部和全局关系（通常取 5-50）
    learning_rate=200,   # 学习率（建议 10-1000）
    n_iter=1000,         # 最大迭代次数
    random_state=42,     # 确保结果可重复
    metric='cosine'      # 使用与 UMAP 相同的距离度量
)
embedding = tsne.fit_transform(feature_maps)  # (样本数, 2)
# 可视化：假设标签0表示Inactive，1表示Active
plt.figure(figsize=(10, 7))
plt.scatter(embedding[all_labels == 0, 0], embedding[all_labels == 0, 1], s=15, label='Inactive', alpha=0.8, color=(66/255, 140/255, 164/255),edgecolors='none')
plt.scatter(embedding[all_labels == 1, 0], embedding[all_labels == 1, 1], s=15, label='Active', alpha=0.8, color=(246/255, 212/255, 109/255),edgecolors='none')
plt.xlabel('x')
plt.ylabel('y')
# plt.title('T-SNE Visualization')
plt.legend()
plt.savefig('/home/cpj/ybj/ames275_100_tsne_after_pri.png', dpi=300)  # 保存为 PNG 格式，300 dpi 的分辨率