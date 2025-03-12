import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import SAGEConv


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
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, data):
    model.eval()
    out = model(data)
    loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
    return loss.item()

def test(model, data):
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


def main(data_file='data.pt'):
    data = torch.load(data_file)

    # Set parameters
    in_features = data.num_features
    hidden_channels = 64  # Increase hidden units
    out_channels = len(torch.unique(data.y))# Number of unique labels should be the number of output classes
    print(f"Number of output classes: {out_channels}")

    model = GraphSAGE(in_features, hidden_channels, out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)  # Increase weight decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Learning rate scheduler
    epochs = 400
    train_losses = []
    val_losses = []
    test_accuracies = []

    # Training loop
    for epoch in range(epochs):
        train_loss = train(model, data, optimizer)
        val_loss = validate(model, data)
        val_loss = validate()
        scheduler.step()  # Update learning rate each epoch
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}')

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

if __name__ == "__main__":
        main()