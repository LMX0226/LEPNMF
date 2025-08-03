from sklearn import metrics
import torch as th
from param import *
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def caculate_metrics(real_score, pre_score):
    y_true = real_score#真实标签
    y_pre = pre_score#预测得分
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)#计算ROC曲线的假阳性率，真阳性率和阈值
    auc = metrics.auc(fpr, tpr)#计算roc曲线的AUC值
    #计算PR曲线的精确率，召回率和阈值
    precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u)#计算PR曲线的AUC值
    #根据预测得分确定预测标签
    y_score = [0 if j < 0.5 else 1 for j in y_pre]
    #计算准确率，F1分数，召回率和精确率
    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score)
    recall = metrics.recall_score(y_true, y_score)
    precision = metrics.precision_score(y_true, y_score)
    #将各个指标保存到列表中
    metric_result = [auc, aupr, acc, f1, precision, recall]
    return metric_result

def integ_similarity(M1,M2):
    for i in range(len(M1)):
        for j in range(len(M1)):
            if M1[i][j] == 0:
                M1[i][j] = M2[i][j]
    return M1

# 对矩阵操作拿到edge_index集合
def get_edge_index(matrix, device):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.tensor(edge_index, dtype=torch.long, device=device)

# def attenweight(q_h, k_h, adj_matrix, n, num_head):
#
#     attenweightmatrix = torch.zeros(num_head, n, n)
#
#     for head in range(num_head):
#         # 将注意力权重矩阵中对应的位置赋值
#         scores = th.matmul(q_h[head,:,:],k_h[head,:,:])
#         attenweightmatrix[head] = scores * adj_matrix  # 删除多余的维度，并根据邻接矩阵筛选非邻接节点的权重
#     return attenweightmatrix
def attenweight(q_h, k_h, edge_index, n, num_head):
    adj_matrix = torch.zeros((n, n))
    edge_index = edge_index[:, :edge_index.shape[1]//2]
    attenweightmatrix = torch.zeros(n, num_head, n)
    for src_node, dest_node in zip(edge_index[0], edge_index[1]):
        adj_matrix[src_node, dest_node] = 1
        adj_matrix[dest_node, src_node] = 1
    adj_matrix = adj_matrix.to(q_h.device)
    #sum_edge = np.sum(adj_matrix.numpy())
    for head in range(num_head):
        # 将注意力权重矩阵中对应的位置赋值
        scores = th.matmul(q_h[:, head, :],k_h[head, :, :])
        attenweightmatrix[:,head,:] = th.mul(scores, adj_matrix)  # 删除多余的维度，并根据邻接矩阵筛选非邻接节点的权重

    return attenweightmatrix
def my_softmax(tensor):
    # batch_size, rows, cols = tensor.shape

    # 找出每个矩阵中每一行中不为零的索引
    non_zero_indices = torch.nonzero(tensor, as_tuple=False)

    # 将非零值应用 softmax 操作
    non_zero_values = tensor[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]]
    softmax_values = F.softmax(non_zero_values, dim=0)

    # 创建一个与原始张量形状相同的张量，用于存储 softmax 结果
    softmax_tensor = torch.zeros_like(tensor)

    # 将 softmax 结果放回到对应的位置
    softmax_tensor[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]] = softmax_values
    return softmax_tensor
import numpy as np


def MY_layer_normalize(tensor):
    # 获取张量的形状
    k, n, _ = tensor.shape

    # 创建一个和输入张量相同大小的零张量，用于存储标准化后的结果
    normalized_tensor = torch.zeros_like(tensor, dtype=torch.float)

    # 遍历张量的每一层（每个二维矩阵）
    for i in range(k):
        matrix = tensor[i]

        # 创建一个和当前二维矩阵相同大小的零矩阵，用于存储当前层的标准化结果
        normalized_matrix = torch.zeros_like(matrix, dtype=torch.float)

        # 遍历当前二维矩阵的每一行
        for j in range(n):
            row = matrix[j]

            # 找到当前行中非零元素的索引和值
            non_zero_indices = torch.nonzero(row).squeeze()
            non_zero_values = row[non_zero_indices]

            # 如果当前行没有非零元素，则直接将零矩阵中的对应行赋值为零
            if non_zero_indices.numel() == 0:
                normalized_matrix[j] = torch.zeros_like(row)
            else:
                # 使用 PyTorch 提供的函数计算当前行非零元素的均值和标准差
                non_zero_mean = torch.mean(non_zero_values)
                non_zero_std = torch.std(non_zero_values)

                # 如果标准差为零，则手动将其替换为1e-5
                if non_zero_std == 0:
                    non_zero_std = 1e-5

                # 对当前行中的非零元素进行标准化
                normalized_row = torch.abs(non_zero_values - non_zero_mean) / non_zero_std

                # 扩展标准化后的行向量，使其与当前行的长度相匹配
                expanded_normalized_row = torch.zeros_like(row)
                expanded_normalized_row.index_put_((non_zero_indices,), normalized_row)

                # 将标准化后的结果放入对应位置
                normalized_matrix[j] = expanded_normalized_row

        # 将当前层的标准化结果放回到零张量的对应位置
        normalized_tensor[i] = normalized_matrix

    # 检查是否存在 NAN 值，若存在，则用均值替换
    if torch.isnan(normalized_tensor).any():
        mean_value = torch.mean(normalized_tensor[torch.logical_not(torch.isnan(normalized_tensor))])
        normalized_tensor[torch.isnan(normalized_tensor)] = mean_value

    return normalized_tensor


def adjacency_matrix_to_edge_list(adj_matrix):
    # 获取邻接矩阵的形状
    n_nodes = adj_matrix.shape[0]

    # 存储边的列表
    edge_list = []

    # 遍历邻接矩阵，找到所有的有向边
    for i in range(n_nodes):
        for j in range(n_nodes):  # 遍历所有元素，考虑有向图
            if adj_matrix[i, j] !=0:  # 如果有边
                edge_list.append([i, j])

    # 将边列表转换为numpy数组，形状为 [2, n]，其中 n 为边的数量
    edge_list = np.array(edge_list).T

    return edge_list