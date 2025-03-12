import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

def build_and_save_graph(excel_file_path='4442.xlsx', output_file='data.pt'):

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

    # 划分数据集
    train_val_mask, test_mask = train_test_split(range(len(y)), test_size=0.2, random_state=42, shuffle=True)
    train_mask, val_mask = train_test_split(train_val_mask, test_size=0.25, random_state=42, shuffle=True)  # 0.25 x 0.8 = 0.2

    train_mask = torch.tensor(train_mask, dtype=torch.long)
    val_mask = torch.tensor(val_mask, dtype=torch.long)
    test_mask = torch.tensor(test_mask, dtype=torch.long)

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # 保存数据
    torch.save(data, output_file)
    print(f"Graph data saved to {output_file}")

if __name__ == "__main__":
    build_and_save_graph()