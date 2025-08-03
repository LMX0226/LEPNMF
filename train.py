import datetime

from model import *
from preData import *
from utils import *


class EarlyStopping:
    def __init__(self, patience=10, delta=0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.mode = mode
        if mode == 'min':
            self.monitor_op = np.less  # loss 下降为好
        else:  # 'max' 模式，用于跟踪像 AUC 这种指标
            self.monitor_op = np.greater

    def __call__(self, score):
        # 确保 `score` 没有梯度并转为 NumPy 数组
        score_np = score.detach().cpu().numpy()  # 确保从计算图中分离并转为 NumPy

        if self.best_score is None:
            self.best_score = score_np  # 第一次记录
        elif self.monitor_op(score_np - self.delta, self.best_score):
            self.best_score = score_np  # 更新最佳 score
            self.counter = 0  # 重置计数器
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

def fold_valid(args):
    # similarity_feature= similarity_feature_process(args)
    # edge_idx_dict , g = load_fold_data(args)
    dataset = loading(args)
    # 假设 edge_idx_dict 是一个字典，里面包含了多个张量
    edge_idx_dict = load_fold_data(args, dataset)

    # 将字典中的每个张量移动到设备上
    for key in edge_idx_dict:
        if isinstance(edge_idx_dict[key], torch.Tensor):  # 确保值是张量
            edge_idx_dict[key] = edge_idx_dict[key].to(args.device)

    n_rna = dataset['miRNA'].shape[1]
    n_dis = dataset['disease'].shape[1]

    metric_result_list = []
    metric_result_list_str = []
    flod_y_list = []
    metric_result_list_str.append('AUC    AUPR    Acc    F1    pre    recall')
    #五折
    torch.autograd.set_detect_anomaly(True)
    for i in range(args.kfolds):
        model = Gps_Module(args, n_rna, n_dis).to(args.device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        criterion = torch.nn.BCEWithLogitsLoss().to(args.device)

        print(f'###########################Fold {i + 1} of {args.kfolds}###########################')
        Record_res = []
        fold_y = []
        Record_res.append('AUC    AUPR    Acc    F1    pre    recall')
        model.train()
        #每折100轮
        for epoch in range(args.epoch):
            optimizer.zero_grad()
            #data-similarity,edge_index-edge_idx_dict,
            out = model(dataset, edge_idx_dict[str(i)]['fold_train_edges_all'], edge_idx_dict[str(i)]['fold_train_edges_80p_80n']).view(-1)
            loss = criterion(out, edge_idx_dict[str(i)]['fold_train_label_80p_80n'])
            loss.backward()
            optimizer.step()
            # validation
            test_auc, metric_result, y_true, y_score, z= valid_fold(model,
                                                                  dataset,
                                                                  edge_idx_dict[str(i)]['fold_train_edges_all'],
                                                                  edge_idx_dict[str(i)]['fold_valid_edges_20p_20n'],
                                                                  edge_idx_dict[str(i)]['fold_valid_label_20p_20n'],
                                                                  args)
            # 记录单次实验结果
            One_epoch_metric = '{:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f} '.format(*metric_result)
            Record_res.append(One_epoch_metric)
            # 当epoch结束后保留实验结果以及实验原始数据
            if epoch + 1 == args.epoch:
                metric_result_list.append(metric_result)
                metric_result_list_str.append(One_epoch_metric)
                fold_y = [str(y_true.cpu().numpy().tolist()), str(y_score.cpu().numpy().tolist())]
                flod_y_list.append(fold_y)
            # 打印单次实验的loss和验证集auc
            print('epoch {:03d} train_loss {:.8f} val_auc {:.4f} acc {:.4f} pre {:.4f}'.format(epoch, loss.item(), test_auc , metric_result[2],
                                                                                               metric_result[4]))

    arr = np.array(metric_result_list)
    averages = np.round(np.mean(arr, axis=0), 4)
    metric_result_list_str.append('平均值：')
    metric_result_list_str.append('{:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f}    {:.4f} '.format(*list(averages)))


    now = datetime.datetime.now()
    with open('平均_' + now.strftime("%Y_%m_%d_%H_%M_%S") + '_.txt', 'w') as f:
        f.write('\n'.join(metric_result_list_str))

        # 调用函数绘制ROC曲线
    plot_roc_pr_curve_cv(flod_y_list)
    return averages




def valid_fold(model, data, encodeEdge, decodeEdge, lable, args):
    model.eval()
    with torch.no_grad():
        z = model.encode(data, encodeEdge)#获取由80训练集训练后的所有节点的特征向量
        out= model.decode(z, decodeEdge).view(-1).sigmoid()#使用训练好的特征，对20测试集进行边预测
        model.train()
    metric_result = caculate_metrics(lable.cpu().numpy(), out.cpu().numpy())
    my_acu = metrics.roc_auc_score(lable.cpu().numpy(), out.cpu().numpy())
    return my_acu, metric_result, lable, out, z


def case_study(model, z, decodeEdge, data):
    out = model.decode(z, decodeEdge).view(-1).sigmoid()
    res = np.zeros((901, 877))
    j = 0
    edge_label_index = torch.tensor(decodeEdge, dtype=torch.long)

    for i in range(len(out)):
        res[edge_label_index[0][i]][edge_label_index[1][i]-901] = out[j]
        j = j + 1

    # 获取第二列数据
    second_column = res[:, 1]

    # 找出前50个最大的数值及其对应的索引
    top_50_indices = np.argsort(-second_column)[:50]
    case_name = []
    for idx in top_50_indices:
        case_name.append(data['miRNA'][idx][0])

    # 保存 case_name 到本地文件
    with open('E:\\pycharm\\py_project\\gpsMDA - 副本\\result\\colon_case_name.txt', 'w') as f:
        for name in case_name:
            f.write(f"{name}\n")

    return case_name

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import ast


def plot_roc_pr_curve_cv(fold_y_list):
    folds_results = []
    now = datetime.datetime.now()

    # 遍历 fold_y_list，计算每一折的FPR, TPR, AUC, Precision和Recall
    for y_true_str, y_pred_proba_str in fold_y_list:
        # 将字符串转换为实际的数值列表
        y_true = np.array(ast.literal_eval(y_true_str))
        y_pred_proba = np.array(ast.literal_eval(y_pred_proba_str))

        # 计算ROC曲线和AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        folds_results.append((fpr, tpr, roc_auc, precision, recall, pr_auc))

    # 开始绘制ROC曲线
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)  # 第一个子图
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')

    colors = ['b', 'g', 'r', 'c', 'm']  # 颜色列表
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    auc_values = []

    # 循环遍历每个折的结果
    for i, (fpr, tpr, roc_auc, _, _, _) in enumerate(folds_results):
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=1, label='Fold {} (AUC = {:.4f})'.format(i + 1, roc_auc))
        auc_values.append(roc_auc)  # 将每个折的AUC值添加到列表中
        # 插值计算TPR
        tpr_interpolated = np.interp(mean_fpr, fpr, tpr)
        tpr_interpolated[0] = 0.0  # 确保在FPR=0时TPR=0
        tprs.append(tpr_interpolated)

    # 计算平均AUC值
    mean_auc = np.mean(auc_values)

    # 计算平均TPR值
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    # 绘制平均ROC曲线
    plt.plot(mean_fpr, mean_tpr, color='b', linestyle='--', lw=2, label='Mean ROC (AUC = {:.4f})'.format(mean_auc))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - 5-fold Cross Validation')
    plt.legend(loc="lower right")

    # 开始绘制PR曲线
    plt.subplot(1, 2, 2)  # 第二个子图

    mean_recall = np.linspace(0, 1, 100)
    precisions = []
    pr_auc_values = []

    # 循环遍历每个折的结果
    for i, (_, _, _, precision, recall, pr_auc) in enumerate(folds_results):
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=1,
                 label='Fold {} (AUC = {:.4f})'.format(i + 1, pr_auc))
        pr_auc_values.append(pr_auc)  # 将每个折的PR AUC值添加到列表中
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))  # 插值计算Precision

    # 计算平均PR AUC值
    mean_pr_auc = np.mean(pr_auc_values)

    # 插值计算平均精确率
    mean_precision = np.mean(precisions, axis=0)

    # 绘制平均PR曲线
    plt.plot(mean_recall, mean_precision, color='b', linestyle='--', lw=2,
             label='Mean PR (AUC = {:.4f})'.format(mean_pr_auc))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - 5-fold Cross Validation')
    plt.legend(loc="lower left")

    plt.tight_layout()

    # 保存图像到本地
    plt.savefig('D:\\pycharm\\py_project\\gpsMDA - 副本\\result\\' + now.strftime("%Y_%m_%d_%H_%M_%S") + '.png')






