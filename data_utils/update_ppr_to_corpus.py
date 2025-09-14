import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm
from torch_geometric.nn import APPNP
from build_corpus import load_data
import json


def fast_ppr(edge_index, num_nodes, alpha=0.85, k_hops=5, top_k=3, batch_size=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    
    edge_index = edge_index.to(device)
    appnp = APPNP(K=k_hops, alpha=alpha).to(device)
    
    top_indices = torch.zeros((num_nodes, top_k), dtype=torch.long)
    top_values = torch.zeros((num_nodes, top_k))
    
    # batch cal ppr
    num_batches = (num_nodes + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="cal PPR batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_nodes)
        current_batch_size = end_idx - start_idx
        
        # initialize batch input
        x = torch.zeros((num_nodes, current_batch_size), device=device)
        for i in range(current_batch_size):
            x[start_idx + i, i] = 1.0
        
        # one step PPR computation
        with torch.no_grad(): # no need to compute gradients
            ppr_batch = appnp(x, edge_index)  # [num_nodes, batch_size]
        
        # for each src, get top-k neighbors
        for i in range(current_batch_size):
            node_idx = start_idx + i
            ppr = ppr_batch[:, i]
            
            # set self-loop ppr to -1 to exclude itself
            ppr[node_idx] = -1.0
            
            # find top-k
            values, indices = torch.topk(ppr, top_k)
            top_indices[node_idx] = indices.cpu()
            top_values[node_idx] = values.cpu()
        
        # release gpu memory 
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return top_indices, top_values


def gen_ppr_nodes(datasets):

    for dataset in tqdm(datasets):
        data, text, _ = load_data(dataset)
        num_nodes = len(text)
        
        # apply fast PPR
        top_indices, top_values = fast_ppr(
            data.edge_index, 
            num_nodes, 
            alpha=0.15, 
            k_hops=30, # 5
            top_k=5, # 3
            batch_size=500
        )
        
        # save sparse ppr data
        sparse_ppr_data = {
            'indices': top_indices,
            'values': top_values
        }

        data_path =  f"../dataset/{dataset}/{dataset}_ppr_top5.pt"
        torch.save(sparse_ppr_data, data_path)
        print(f"Saved top-5 personalized pagerank for {dataset} to {data_path}")


def update_to_corpus(datasets):
    # 'cora', 'arxiv', 'pubmed', 'products', 'computers'

    for dataset in tqdm(datasets):
        print(f"Processing dataset: {dataset}")

        # load ppr data
        data_path =  f"../dataset/{dataset}/{dataset}_ppr_top5.pt"
        sparse_ppr_data = torch.load(data_path)
        top_indices = sparse_ppr_data['indices']

        # load corpus
        corpus_path = f"../dataset/{dataset}/{dataset}_corpus.json"
        with open(corpus_path, "r") as f:
            corpus = json.load(f)

        # update corpus with ppr neighbors
        for i in range(len(corpus)):
            ppr_neighbors = top_indices[i].tolist()
            corpus[str(i)]['ppr_neighbors'] = ppr_neighbors

        # save updated corpus
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=4)
        
        print(f"Updated corpus with PPR neighbors for {dataset} at {corpus_path}")


def update_hopnb_to_corpus(datasets):

    # load ppr data
    import json
    import random
    import torch
    from tqdm import tqdm
    from retrieve_neighbors import get_hop1_neighbors
    from build_corpus import load_data

    for dataset in tqdm(datasets):
        print(f"Processing dataset: {dataset}")

        # load corpus
        corpus_path = f"../dataset/{dataset}/{dataset}_corpus.json"
        with open(corpus_path, "r") as f:
            corpus = json.load(f)

        data, text, labels = load_data(dataset)
        hop_neighbors, degrees = get_hop1_neighbors(data)
        # update corpus with hop neighbors
        for i in range(len(corpus)):
            corpus[str(i)]['neighbors'] = hop_neighbors[i]
        
        # save updated corpus
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=4)
        
        print(f"Updated corpus with hop neighbors for {dataset} at {corpus_path}")


datasets = ['cora'] # 'cora', 'arxiv', 'pubmed', 'products', 'computers', 'sports'
#gen_ppr_nodes(datasets)
# update_to_corpus(datasets)
update_hopnb_to_corpus(datasets)



