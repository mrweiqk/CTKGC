import numpy as np
import scipy.sparse as sp
import torch

class Data:

    def __init__(self, data_dir="data/FB15k-237/", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    print(adj,adj.shape)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_adj(adj, norm=True, sparse=False):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = adj + sp.eye(adj.shape[0])
    if norm:
        adj = normalize_adj(adj)
    if sparse:
        return sparse_to_tuple(adj)
    return adj.todense()


def get_indice_graph(mask_adj, beg_indices, size, keep_r=1.0):
    indices = [beg_indices]
    if keep_r < 1.0:
        indices = np.random.choice(indices, int(indices.size*keep_r), False)
    pre_indices = set()
    indices = set(indices)
    while len(indices) < size:
        new_add = indices - pre_indices
        if not new_add:
            break
        pre_indices = indices
        candidates = get_candidates(mask_adj, new_add) - indices
        if len(candidates) > size - len(indices):
            candidates = set(np.random.choice(list(candidates), size-len(indices), False))
        indices.update(candidates)
    print('indices size:-------------->', len(indices))
    return sorted(indices)


def get_min_graph(mask_adj, beg_indices, size, keep_r=1.0):
    indices = [beg_indices]
    if keep_r < 1.0:
        indices = np.random.choice(indices, int(indices.size*keep_r), False)
    pre_indices = set()
    indices = set(indices)
    while len(indices) < size:
        new_add = indices - pre_indices
        if not new_add:
            break
        pre_indices = indices
        candidates = get_candidates(mask_adj, new_add) - indices
        if len(candidates) > size - len(indices):
            candidates = set(np.random.choice(list(candidates), size-len(indices), False))
        indices.update(candidates)
    print('indices size:-------------->', len(indices))
    return sorted(indices)

def get_candidates(adj, new_add):
    return set(adj[sorted(new_add)].sum(axis=0).nonzero()[1])




def get_h_BFS(graph_dict, beg_indices, size, neb_max_size):
    indices = [beg_indices]
    neb_1 = list(set(graph_dict[beg_indices]))

    ####填充的方法
    ind_list = []
    if len(neb_1) == 0:
        # 有大图遮掩过后，节点找不到邻居的情况
        # neb_1=[],怎么处理？ 选头实体自己
        # print('neb_1:',neb_1)
        # 全部填充0
        ind_list = np.random.choice([0], neb_max_size, True)
        # print(ind_list)
        assert len(ind_list) == neb_max_size, 'graph indices not equal neb_max_size'
        return ind_list
    if len(neb_1) > neb_max_size:
        ind_list = np.random.choice(neb_1, neb_max_size, False)
    else:
        ind_list = list(neb_1)
        pre_indices = set(indices)
        indices = set(neb_1)
        for i in range(size):
            new_add = indices - pre_indices
            if (len(ind_list) >= neb_max_size):
                break
            for new in new_add:
                new_neb = set(graph_dict[new]) - indices
                indices.update(new_neb)
                if (len(new_neb) < (neb_max_size - len(ind_list))):
                    for k in new_neb:
                        ind_list.append(k)
                else:
                    len_o = len(ind_list)
                    for k in (list(new_neb)[:(neb_max_size - len_o)]):
                        ind_list.append(k)
                    break

    if (len(ind_list) < neb_max_size):
        ##少的部分,也pad 0
        pad_len = neb_max_size - len(ind_list)
        for i in range(pad_len):
            ind_list.append(0)
    assert len(ind_list) == neb_max_size, 'graph indices not equal neb_max_size'


    return ind_list

    #####原始不填充的方法
    # ind_list = []
    # if len(neb_1)==0:
    #     # 有大图遮掩过后，节点找不到邻居的情况
    #     # neb_1=[],怎么处理？ 选头实体自己
    #     # print('neb_1:',neb_1)
    #     ind_list = np.random.choice(indices, neb_max_size, True)
    #     # print(ind_list)
    #     assert len(ind_list) == neb_max_size, 'graph indices not equal neb_max_size'
    #     return sorted(ind_list)
    # if len(neb_1) > neb_max_size:
    #     ind_list = np.random.choice(neb_1, neb_max_size, False)
    # else:
    #     ind_list = list(neb_1)
    #     pre_indices = set(indices)
    #     indices = set(neb_1)
    #     for i in range(size):
    #         new_add = indices - pre_indices
    #         if (len(ind_list) >= neb_max_size):
    #             break
    #         for new in new_add:
    #             new_neb = set(graph_dict[new]) - indices
    #             indices.update(new_neb)
    #             if(len(new_neb) < neb_max_size - len(ind_list)):
    #                 ind_list = ind_list + list(new_neb)
    #             else:
    #                 ind_list = ind_list + list(new_neb)[:(neb_max_size-len(ind_list))]
    #                 break
    #
    # if(len(ind_list) < neb_max_size):
    #     ind_list = np.random.choice(ind_list, neb_max_size, True)
    # assert len(ind_list) == neb_max_size, 'graph indices not equal neb_max_size'
    #
    #
    # return sorted(ind_list)

