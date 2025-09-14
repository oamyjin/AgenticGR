from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import json
import numpy as np
import os
import time
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
# from core.utils import time_logger


def get_labels(data):
    products_category = ['Home & Kitchen', 'Health & Personal Care', 'Beauty', 'Sports & Outdoors', 'Books', 'Patio, Lawn & Garden', 'Toys & Games', 'CDs & Vinyl', 
                             'Cell Phones & Accessories', 'Grocery & Gourmet Food', 'Arts, Crafts & Sewing', 'Clothing, Shoes & Jewelry', 'Electronics', 'Movies & TV', 
                             'Software', 'Video Games', 'Automotive', 'Pet Supplies', 'Office Products', 'Industrial & Scientific', 'Musical Instruments', 'Tools & Home Improvement', 
                             'Magazine Subscriptions', 'Baby Products', 'label 25', 'Appliances', 'Kitchen & Dining', 'Collectibles & Fine Art', 'All Beauty', 'Luxury Beauty', 'Amazon Fashion', 
                             'Computers', 'All Electronics', 'Purchase Circles', 'MP3 Players & Accessories', 'Gift Cards', 'Office & School Supplies', 'Home Improvement', 'Camera & Photo', 
                             'GPS & Navigation', 'Digital Music', 'Car Electronics', 'Baby', 'Kindle Store', 'Buy a Kindle', 'Furniture & Decor', '#508510']
    class_map = {x: i for i, x in enumerate(products_category)}
    inverse_class_map = {}
    for lb, lb_id in class_map.items():
        inverse_class_map[lb_id] = lb
    print("inverse_class_map:", inverse_class_map)
    
    y = data.y
    unique_y = set(y.view(-1).tolist())
    print("unique_y:", unique_y)
    

    labels = [inverse_class_map[l.item()] for l in y.view(-1)]
    print("y.shape:", y.shape)
    print("y.view(-1)[:10]:", y.view(-1)[:10])
    print("labels[:10]:", labels[:10])

    return labels



def get_raw_text_products(use_text=True, seed=0):
    data = torch.load('../dataset/products/ogbn-products_subset.pt')
    text = pd.read_csv('../dataset/products/ogbn-products_subset.csv')
    text = [f'Title:{ti}; Description: {cont}\n'for ti,
            cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric()
    print("data:", data)

    edge_index = data.edge_index
    row, col, _ = edge_index.coo()  # This extracts the COO format (row, col)
    row = row.numpy()
    col = col.numpy()
    data1 = torch.ones(edge_index.nnz()).numpy()  # Assuming weights are 1
    s = coo_matrix((data1, (row, col)), shape=(data.num_nodes, data.num_nodes))
    edge_index, edge_attr = from_scipy_sparse_matrix(s)
    data.edge_index = edge_index
    print("edge_index.shape:", edge_index.shape)
    print("data:", data)

    if not use_text:
        return data, None

    labels = get_labels(data)


    return data, text, labels


get_raw_text_products()