import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix

def get_labels(data, dsname):
    categories_mappings = {
        'amazon-computers': ['Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories', 
                            'Computers & Tablets', 'Computer Components', 'Data Storage', 'Networking Products', 'Monitors', 'Servers', 'Tablet Replacement Parts'],
        'amazon-sports':['Other Sports', 'Golf', 'Hunting & Fishing', 'Exercise & Fitness', 'Team Sports', 'Accessories', 'Swimming', 'Leisure Sports & Game Room', 
                'Airsoft & Paintball', 'Boating & Sailing', 'Sports Medicine', 'Tennis & Racquet Sports', 'Clothing'],
    }

    amazon_type = "amazon-" + dsname
    products_category = categories_mappings[amazon_type]
    class_map = {x: i for i, x in enumerate(products_category)}
    print(class_map)
    inverse_class_map = {}
    for lb, lb_id in class_map.items():
        inverse_class_map[lb_id] = lb
    print("inverse_class_map:", inverse_class_map)

    y = data.y
    labels = [inverse_class_map[l.item()] for l in y.view(-1)]
    print("y.shape:", y.shape)
    print("y.view(-1)[:10]:", y.view(-1)[:10])
    print("labels[:10]:", labels[:10])

    return labels


def build_pygData_from_csv(dataset):
    csv_file = f"../dataset/{dataset}/{dataset}.csv"
    df = pd.read_csv(csv_file)
    category_list = df['category']
    text_list = df['text']
    label_list = df['label']
    neighbour_list = df['neighbour']

    # 创建标签到类别的映射字典
    label_to_category = dict(zip(label_list, category_list))
     
    # 获取所有类别
    category_names = [label_to_category[label] for label in sorted(label_to_category.keys())]
    x = None
    y = torch.tensor(label_list.values, dtype=torch.float)
    edge_index_list = []
    for index, neighbours in enumerate(neighbour_list):
        # 将邻居列表转换为整数
        neighbours = neighbours.strip('[]').split(',')
        neighbours = [int(neighbour.strip()) for neighbour in neighbours if neighbour.strip()]
        # 创建边索引对
        edges = [(index, neighbour) for neighbour in neighbours]
        if (index,index) not in edges:
            edges.append((index,index))
        edge_index_list.extend(edges)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    num_nodes = len(text_list)
    data = Data(x=x, edge_index=edge_index, y=y, category_names = category_names, num_nodes = num_nodes)
    print("data:", data)

    edge_index = data.edge_index
    # reshape edge_index to make sure it has shape [2, num_edges]
    row = edge_index[0].numpy()  # src
    col = edge_index[1].numpy()  # dst
    data1 = torch.ones(edge_index.shape[1]).numpy()  # assume weight = 1
    # COO sparse matrix
    num_nodes = data.num_nodes
    s = coo_matrix((data1, (row, col)), shape=(num_nodes, num_nodes))
    edge_index, edge_attr = from_scipy_sparse_matrix(s)
    print("edge_index.shape:", edge_index.shape)
    data.edge_index = edge_index
    print("data:", data)

    # set train_mask, val_mask, test_mask
    with open(f'../dataset/{dataset}/{dataset}_train_ids.txt', 'r') as f:
        ids_read = f.read()
    train_ids = [int(x) for x in ids_read.split(',')]
    with open(f'../dataset/{dataset}/{dataset}_test_ids.txt', 'r') as f:
        ids_read = f.read()
    test_ids = [int(x) for x in ids_read.split(',')]
    val_ids = []
    for i in range(data.num_nodes):
        if i not in train_ids and i not in test_ids:
            val_ids.append(i)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_ids] = True
    val_mask[val_ids] = True
    test_mask[test_ids] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    print("data:", data)

    labels = get_labels(data, dataset)

    return data, text_list, labels


def get_raw_text_computers(use_text=False, seed=0):
   return build_pygData_from_csv('computers')

def get_raw_text_sports(use_text=False, seed=0):
   return build_pygData_from_csv('sports')