# GraphSAGE 图分类项目

基于PyTorch Geometric实现的GraphSAGE模型，用于图结构数据分类任务。


可视化软件的官网https://gephi.org/

## 📦 项目结构

```bash
.
├── build_graph.py    # 图数据预处理与构建脚本
├── train.py          # 模型训练与评估脚本
├── rawdata.xlsx        # 原始数据文件（需自行准备）
├── requirements.txt  # 依赖库列表
└── README.md         # 项目说明文档
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric 2.0+

### 安装依赖
```bash
pip install -r requirements.txt

# 如果安装PyTorch Geometric遇到问题，建议使用官方推荐方式：
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric
```

### 数据准备
1. 将原始数据文件放在项目根目录
2. 数据应包含：
   - 数值型特征列
   - 分类标签列

### 运行流程
1. 构建图数据：
```bash
python build_graph.py
```
*生成预处理后的图数据文件 `data.pt`*

2. 训练模型：
```bash
python train.py
```
