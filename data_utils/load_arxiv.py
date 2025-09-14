import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix


def cal_map(): 
    file_path = '/scratch/jl11523/GraphGPT/labelidx2arxivcategeory.csv'
    # df = pd.read_csv(file_path, compression='gzip')
    df = pd.read_csv(file_path)
    label_dict = {}
    for index, line in df.iterrows(): 
        lb = line['arxiv category'].split(' ')[-1]
        lb_new = 'cs.' + lb.upper()
        label_dict[lb_new] = line['label idx']
    print("list(label_dict):", list(label_dict))
    return label_dict

def get_labels(data):
    class_map = cal_map()
    inverse_class_map = {}
    for lb, lb_id in class_map.items():
        inverse_class_map[lb_id] = lb
    print("inverse_class_map:", inverse_class_map)
    
    # set data.label_dict
    #data.categories = list(class_map)

    y = data.y
    labels = [inverse_class_map[l.item()] for l in y.view(-1)]
    print("y.shape:", y.shape)
    print("y.view(-1)[:10]:", y.view(-1)[:10])
    print("labels[:10]:", labels[:10])

    return labels



def get_raw_text_arxiv(use_text=False, seed=0):
    # Load the OGBN-Arxiv dataset
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
    data = dataset[0]

    # Get index splits for training, validation, and test sets
    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Ensure the graph is undirected
    data.edge_index = data.adj_t.to_symmetric()
    print("data:", data)

    edge_index = data.edge_index
    # If edge_index is a SparseTensor, extract row and col using its built-in methods
    row, col, _ = edge_index.coo()  # This extracts the COO format (row, col)
    # Convert them to numpy arrays if needed
    row = row.numpy()
    col = col.numpy()
    # Now you can proceed with your logic to build the COO matrix and continue the rest of your code
    data1 = torch.ones(edge_index.nnz()).numpy()  # Assuming weights are 1
    s = coo_matrix((data1, (row, col)), shape=(data.num_nodes, data.num_nodes))
    # Convert back to PyTorch Geometric edge index format
    edge_index, edge_attr = from_scipy_sparse_matrix(s)
    print("edge_index.shape:", edge_index.shape)
    data.edge_index = edge_index
    print("data:", data)



    if not use_text:
        return data, None

    # Load mapping files
    nodeidx2paperid = pd.read_csv('../dataset/arxiv/nodeidx2paperid.csv.gz', compression='gzip')

    raw_text = pd.read_csv('../dataset/arxiv/titleabs.tsv',
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])

    # Filter out rows where 'paper id' is not numeric
    raw_text = raw_text[pd.to_numeric(raw_text['paper id'], errors='coerce').notnull()]
    raw_text['paper id'] = raw_text['paper id'].astype(int)

    # Ensure 'paper id' in nodeidx2paperid is also int
    nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype(int)

    # Merge the data on 'paper id'
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')

    # Combine titles and abstracts into a text field
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '; ' + 'Abstract: ' + ab
        text.append(t)

    labels = get_labels(data)

    return data, text, labels


# cal_map()