from sklearn.model_selection import KFold
from utils import *
import pandas as pd
from sklearn.decomposition import NMF
def loading(args):
    data = dict()

    data['all_sample'] = pd.read_csv(args.data_dir + 'all_sample.csv', header=None).iloc[:, :].values

    data['miRNA'] = pd.read_csv(args.data_dir + 'miRNA.csv', header=None).iloc[:, :].values
    data['disease'] = pd.read_csv(args.data_dir + 'disease.csv', header=None).iloc[:, :].values
    data['miRNA_disease'] = np.concatenate((data['miRNA'], data['disease']), axis=0)

    data['miRNA_disease_feature'] = pd.read_csv(args.data_dir + 'miRNA_disease_feature.csv', header=None).iloc[:,
                                    :].values
    data['miRNA_sim'] = pd.read_excel(args.data_dir + 'rna_sim.xlsx', header=None, engine='openpyxl').iloc[:, :].values
    data['disease_sim'] = pd.read_excel(args.data_dir + 'disease_sim.xlsx', header=None, engine='openpyxl').iloc[:,
                          :].values

    return data

# ***************************载入图的边数据开始**************************
def load_fold_data(args,dataset):
    n= dataset['miRNA_disease'].shape[0]
    kfolds = args.kfolds
    edge_idx_dict = dict()
    edge_index_new = make_index(dataset, dataset['all_sample'][:, :2])
    k_sim = top_k_matrix(dataset['miRNA_sim'], dataset['disease_sim'], 10)
    sim_edge = adjacency_matrix_to_edge_list(k_sim).T
    matrix = dict()

    train_pos_edges = edge_index_new[:edge_index_new.shape[0] // 2, :].T
    train_neg_edges = edge_index_new[edge_index_new.shape[0] // 2:, :].T
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=123)

    train_pos_edges = train_pos_edges.T
    train_neg_edges = train_neg_edges.T

    train_idx, valid_idx = [], []
    train_sim_idx, valid_sim_idx = [],[]
    for train_index, valid_index in kf.split(train_pos_edges):
        train_idx.append(train_index)
        valid_idx.append(valid_index)

    for (train_sim_index, vaild_sim_index) in kf.split(sim_edge):
        train_sim_idx.append(train_sim_index)
        valid_sim_idx.append(vaild_sim_index)


    for i in range(kfolds):
        edges_train_pos, edges_valid_pos = train_pos_edges[train_idx[i]], train_pos_edges[valid_idx[i]]
        fold_train_pos80 = edges_train_pos.T
        fold_valid_pos20 = edges_valid_pos.T

        edges_train_neg, edges_valid_neg = train_neg_edges[train_idx[i]], train_neg_edges[valid_idx[i]]
        fold_train_neg80 = edges_train_neg.T
        fold_valid_neg20 = edges_valid_neg.T

        edges_train_sim, edges_valid_sim = sim_edge[train_sim_idx[i]], sim_edge[valid_sim_idx[i]]
        fold_train_sim = edges_train_sim.T
        fold_valid_sim = edges_valid_sim.T

        fold_train_edges_all = np.hstack((fold_train_sim, fold_train_pos80))
        fold_train_edges_80p_80n = np.hstack((fold_train_pos80, fold_train_neg80))
        fold_train_label_80p_80n = np.hstack((np.ones(fold_train_pos80.shape[1]), np.zeros(fold_train_neg80.shape[1])))

        fold_valid_edges_all = np.hstack((fold_valid_sim, fold_valid_pos20))
        fold_valid_edges_20p_20n = np.hstack((fold_valid_pos20, fold_valid_neg20))
        fold_valid_label_20p_20n = np.hstack((np.ones(fold_valid_pos20.shape[1]), np.zeros(fold_valid_neg20.shape[1])))

        edge_idx_dict[str(i)] = {}
        matrix[str(i)] = {}


        #训练用的80
        edge_idx_dict[str(i)]["fold_train_edges_80p_80n"] = torch.tensor(fold_train_edges_80p_80n).to(torch.long).to(
            device=args.device)
        edge_idx_dict[str(i)]["fold_train_edges_all"] = torch.tensor(fold_train_edges_all).to(torch.long).to(
            device=args.device)
        edge_idx_dict[str(i)]["fold_train_label_80p_80n"] = torch.tensor(fold_train_label_80p_80n).to(torch.float32).to(
            device=args.device)
        #验证用的20
        edge_idx_dict[str(i)]["fold_valid_edges_20p_20n"] = torch.tensor(fold_valid_edges_20p_20n).to(torch.long).to(
            device=args.device)
        edge_idx_dict[str(i)]["fold_valid_edges_all"] = torch.tensor(fold_valid_edges_all).to(torch.long).to(
            device=args.device)
        edge_idx_dict[str(i)]["fold_valid_label_20p_20n"] = torch.tensor(fold_valid_label_20p_20n).to(torch.float32).to(
            device=args.device)

    return edge_idx_dict

def make_index(data, sample):
    sample_index = []
    for i in range(sample.shape[0]):
        idx = np.where(sample[i][0] == data['miRNA_disease'])
        idy = np.where(sample[i][1] == data['miRNA_disease'])
        sample_index.append([idx[0].item(), idy[0].item()])
    sample_index = np.array(sample_index)
    return sample_index

def top_k_matrix(matrix1,matrix2,k):
    top_k_idx_m = np.argpartition(matrix1, -k, axis=1)[:, -k:]
    top_k_idx_d = np.argpartition(matrix2, -k, axis=1)[:, -k:]
    n_rows_m, n_cols_m = matrix1.shape
    indicator_matrix_m = np.zeros((n_rows_m, n_cols_m), dtype=int)
    for i in range(n_rows_m):
        indices = top_k_idx_m[i]
        indicator_matrix_m[i, indices] = 1
    n_rows_d, n_cols_d = matrix2.shape
    indicator_matrix_d = np.zeros((n_rows_d, n_cols_d), dtype=int)
    for i in range(n_rows_d):
        indices = top_k_idx_d[i]
        indicator_matrix_d[i, indices] = 1
    n1, m1 = indicator_matrix_m.shape
    n2, m2 = indicator_matrix_d.shape
    # 创建新的矩阵
    merged_matrix = np.zeros((n1 + n2, n1 + n2))
    # 将第一个矩阵放入左上角
    merged_matrix[:n1, :m1] = indicator_matrix_m
    # 将第二个矩阵放入右下角
    merged_matrix[n1:, n1:] = indicator_matrix_d
    return merged_matrix

def md_adj_matrix(edge_index, n):
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        src_node = int(edge_index[0][i])
        dest_node = int(edge_index[1][i])
        adj_list[src_node].append(dest_node)
        adj_list[dest_node].append(src_node)

    # 构建邻接矩阵，包括自身连接
    adj_matrix = torch.zeros(n, n)
    for i, neighbors in enumerate(adj_list):
        adj_matrix[i, neighbors] = 1
    return adj_matrix
def atten_adj_matrix(matrix1, matrix2):
    adj_matrix = np.add(matrix1,matrix2)
    return adj_matrix
def MYNMF(args):
    data = loading(args)
    sample = data['all_sample'][:, :2]
    sample_index = []
    for i in range(sample.shape[0]):
        idx = np.where(sample[i][0] == data['miRNA'])
        idy = np.where(sample[i][1] == data['disease'])
        sample_index.append([idx[0].item(), idy[0].item()])
    sample_index = np.array(sample_index)
    adjacency_matrix = np.zeros((901, 877), dtype=int)
    for row in sample_index[:15186,]:
        src, dest = row
        adjacency_matrix[src, dest] = 1
    for row in sample_index[15186:,:]:
        src,dest = row
        adjacency_matrix[src, dest] = 0
    md_matrix = adjacency_matrix
    nmf = NMF(n_components=256, init='random', random_state=0, tol=1e-4, max_iter=1200)
    W = nmf.fit_transform(md_matrix)
    H = nmf.components_.T
    return W,H