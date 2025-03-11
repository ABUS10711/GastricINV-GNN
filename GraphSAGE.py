import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import SAGEConv


# 读取Excel文件
excel_file_path = '4442.xlsx'

if not os.path.exists(excel_file_path):
    raise FileNotFoundError(f"The file {excel_file_path} does not exist.")

try:
    df = pd.read_excel(excel_file_path)
except Exception as e:
    print(f"Error reading the Excel file: {e}")
    exit(1)

# 创建Num列作为行索引
df['Num'] = df.index


# 打印DataFrame的列以确认
print("\nDataFrame columns after adding 'Num' and 'Cath':")
print(df.columns)

# Extract features and normalize
features = df.drop(columns=['Num', 'Cath']).select_dtypes(include=['float64', 'int64'])
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)
# # 特征标准化
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(df.drop(columns=['Num', 'Cath']))

# 将标准化后的特征添加回DataFrame
df[df.columns.difference(['Num', 'Cath'])] = normalized_features

# 提取节点属性
attributes = df.to_dict(orient='list')

# 创建NetworkX图
G = nx.Graph()

# 添加节点和它们的属性到图中
for i in range(len(attributes['Num'])):
    G.add_node(
        attributes['Num'][i],
        **{key: attributes[key][i] for key in attributes if key != 'Num'}
    )

# 计算节点对之间的皮尔逊相关系数，并以此添加边（需要改）
edge_list = []
edge_weights = []

for i in range(len(attributes['Num'])):
    for j in range(i + 1, len(attributes['Num'])):
        node_i_attributes = [attributes[key][i] for key in attributes if key not in ['Num', 'Cath']]
        node_j_attributes = [attributes[key][j] for key in attributes if key not in ['Num', 'Cath']]

        # 计算皮尔逊相关系数
        correlation = pd.Series(node_i_attributes).corr(pd.Series(node_j_attributes))

        if abs(correlation) > 0.75:
            G.add_edge(attributes['Num'][i], attributes['Num'][j], weight=correlation)
            edge_list.append((attributes['Num'][i], attributes['Num'][j]))
            edge_weights.append(correlation)

# 打印节点及其属性
print("\nNodes and their attributes:")
for node, data in G.nodes(data=True):
    print(f"Node: {node}, Data: {data}")

# 打印边以检查它们是否正确添加
print("\nEdges in the graph:")
print(G.edges(data=True))

# 提取节点属性到字典中
node_attributes = {key: [] for key in attributes if key != 'Num'}
for node, data in G.nodes(data=True):
    for key, value in data.items():
        node_attributes[key].append(value)

# 将字典转换为DataFrame
df_attributes = pd.DataFrame(node_attributes)

# 打印DataFrame以检查提取的属性
print("\nExtracted node attributes:")
print(df_attributes)

# 'Cath' 是要预测的标签
labels = df_attributes['Cath'].values
features = df_attributes.drop(columns=['Cath']).values

# 数据标准化
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 创建 PyTorch Geometric 的数据对象
x = torch.tensor(features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)
edge_index = torch.tensor(edge_list).t().contiguous()
edge_weight = torch.tensor(edge_weights, dtype=torch.float)

# 检查 edge_index 是否为空
if edge_index.size(0) == 0:
    raise ValueError("Edge index is empty. Check if the graph has edges.")

data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

# Split training, validation, and testing sets
train_val_mask, test_mask = train_test_split(range(len(y)), test_size=0.2, random_state=42, shuffle=True)
train_mask, val_mask = train_test_split(train_val_mask, test_size=0.25, random_state=42, shuffle=True)  # 0.25 x 0.8 = 0.2

train_mask = torch.tensor(train_mask, dtype=torch.long)
val_mask = torch.tensor(val_mask, dtype=torch.long)
test_mask = torch.tensor(test_mask, dtype=torch.long)

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask



# class SAGE(torch.nn.Module):
#     def __init__(self, inputs, hidden_layer, outputs, nlayers=2):
#         super(SAGE, self).__init__()
#
#         self.convolution_layers = torch.nn.ModuleList()
#         self.convolution_layers.append(SAGEConv(inputs, hidden_layer))
#         self.nlayers = nlayers
#
#         for _ in range(nlayers - 2):
#             self.convolution_layers.append(SAGEConv(hidden_layer, hidden_layer))
#         self.convolution_layers.append(SAGEConv(hidden_layer, outputs))
#
#     def reset_parameters(self):
#         for conv_layer in self.convolution_layers:
#             conv_layer.reset_parameters()
#
#     def forward(self, x, adjs):
#         for i, (edge_index, _, size) in enumerate(adjs):
#             xs = []
#             x_target = x[:size[1]]
#             x = self.convolution_layers[i]((x, x_target), edge_index)
#             if i != self.nlayers - 1:
#                 x = F.relu(x)
#                 x = F.dropout(x, p=0.5, training=self.training)
#             xs.append(x)
#             layer_embeddings = [None] * 2
#             if i < 2:
#                 x_all = torch.cat(xs, dim=0)
#                 layer_embeddings[i] = x_all
#
#         return tuple(layer_embeddings)
#
#
# model = SAGE(inputs=100, hidden_layer=256, outputs=6, nlayers=2)
# num_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters : {num_params}")

#
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = torch.nn.CrossEntropyLoss()

# Define GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_features, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.6)  # Increase dropout rate

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Train and validate the model
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def validate():
    model.eval()
    out = model(data)
    loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
    return loss.item()

def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    prob = F.softmax(out, dim=1)  # 计算每个类别的概率


    acc = correct / data.test_mask.size(0)

    # 计算 MSE
    mse = mean_squared_error(data.y[data.test_mask].cpu().numpy(), pred[data.test_mask].cpu().numpy())

    # 计算混淆矩阵中的四个值
    TP = ((pred[data.test_mask] == 1) & (data.y[data.test_mask] == 1)).sum().item()
    TN = ((pred[data.test_mask] == 0) & (data.y[data.test_mask] == 0)).sum().item()
    FP = ((pred[data.test_mask] == 1) & (data.y[data.test_mask] == 0)).sum().item()
    FN = ((pred[data.test_mask] == 0) & (data.y[data.test_mask] == 1)).sum().item()

    # 计算召回率（Recall）和特异性（Specificity）
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return acc, mse, recall, specificity, prob[data.test_mask]  # 返回五个值

# Set parameters
in_features = data.num_features
hidden_channels = 64  # Increase hidden units
out_channels = len(set(labels))  # Number of unique labels should be the number of output classes
print(f"Number of output classes: {out_channels}")

model = GraphSAGE(in_features, hidden_channels, out_channels)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)  # Increase weight decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Learning rate scheduler
epochs = 400
train_losses = []
val_losses = []
test_accuracies = []

if __name__ == "__main__":
    # Training loop
    for epoch in range(epochs):
        train_loss = train()
        val_loss = validate()
        scheduler.step()  # Update learning rate each epoch
        test_acc = test()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_accuracies.append(test_acc[0])
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Test Accuracy: {test_acc}')

# 训练循环结束后进行测试
    final_test_acc, final_test_mse, final_test_recall, final_test_specificity, test_probs = test()

    print(f"Final Test Accuracy: {final_test_acc}")
    print(f"Final Test MSE: {final_test_mse}")
    print(f"Final Test Recall: {final_test_recall}")
    print(f"Final Test Specificity: {final_test_specificity}")

    # Plot training and validation loss and test accuracy
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Binarize the labels for multi-class ROC
    y_test_binarized = label_binarize(data.y[data.test_mask].cpu().numpy(), classes=range(out_channels))

    # Plotting ROC Curve for each class
    plt.figure(figsize=(10, 8))

    for i in range(out_channels):
        # Assuming binary classification
        fpr, tpr, _ = roc_curve(data.y[data.test_mask].cpu().numpy(), 1 - test_probs[:, 0].detach().cpu().numpy())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')

        # fpr, tpr, _ = roc_curve(y_test_binarized[:, i], test_probs[:, i].detach().cpu().numpy())
        # roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line for reference
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()