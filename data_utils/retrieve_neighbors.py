from tqdm import tqdm
import pdb


def get_hop1_neighbors(data):
    edge_index = data.edge_index
    num_nodes = len(data.y)
    print("num_nodes:", num_nodes)
    # get test ids
    test_ids = [i for i, x in enumerate(data.test_mask) if x]
    print("len(test_ids):", len(test_ids))

    adjacency_list = {i: set() for i in range(num_nodes)}
    print("adjacency_list:", len(adjacency_list))
    for src, dst in edge_index.T.tolist():
        adjacency_list[src].add(dst)
        adjacency_list[dst].add(src)  # undirected graph

    hop_neighbors = {} # "node_id": []
    degrees = {}
    no_nb = 0
    empty_nb_cnt = 0
    for i in tqdm(range(num_nodes)):
        direct_neighbors = list(adjacency_list[i])
        degrees[i] = len(direct_neighbors)
        # No neighbors
        if not direct_neighbors:
            hop_neighbors[i] = []
            continue
        # Filter from train or val set
        valid_neighbors = [n for n in direct_neighbors] # if n not in test_ids]
        hop_neighbors[i] = valid_neighbors
        if len(valid_neighbors) == 0:
            empty_nb_cnt += 1
    print("len(hop_neighbors):", len(hop_neighbors))
    print("no_nb:", no_nb)
    print("empty_nb_cnt:", empty_nb_cnt)
    return hop_neighbors, degrees



def get_hop_neighbors(data, hop, top_k, nb_type):
    edge_index = data.edge_index
    num_nodes = len(data.y)
    print("num_nodes:", num_nodes)
    # get test ids
    test_ids = [i for i, x in enumerate(data.test_mask) if x]
    print("len(test_ids):", len(test_ids))

    adjacency_list = {i: set() for i in range(num_nodes)}
    print("adjacency_list:", len(adjacency_list))
    for src, dst in edge_index.T.tolist():
        adjacency_list[src].add(dst)
    
    # Sort neighbors based on the specified method
    d_path = None
    if nb_type == "ppr_sorting":
        d_path = f'/scratch/jl11523/dataset-mgllm/{dataset}/{dataset}_pagerank.pt'
    elif nb_type == "sbert_cossim_sorting":
        d_path = f'/scratch/jl11523/dataset-mgllm/{dataset}/{dataset}_sbert_cossim.pt'
    elif nb_type == "clip_cossim_sorting":
        d_path = f'/scratch/jl11523/dataset-mgllm/{dataset}/{dataset}_clip_cossim.pt'
    else:
        print(f'warning: nb_type {nb_type} not supported, default to random sampling')  
    if d_path is not None:
        if not os.path.exists(d_path):
            raise ValueError(f'Error: d_path {d_path} not exists')
        data = torch.load(d_path)

    hop_neighbors = {}
    degrees = {}
    no_nb = 0
    max_len = 0
    empty_nb_cnt = 0
    for i in tqdm(range(num_nodes)):
        hop_neighbors[i] = {} # "hop1":[], "hop2":[]
        direct_neighbors = list(adjacency_list[i])
        degrees[i] = len(direct_neighbors)
        if i < 2:
            print("direct_neighbors:", direct_neighbors)
        # No neighbors
        if not direct_neighbors:
            for k in range(hop):
                hop_neighbors[i][f"hop{k+1}"] = []
            no_nb += 1
            continue
        # Sample topk from train or val set
        valid_neighbors = [n for n in direct_neighbors] # if n not in test_ids]
        # sort neighbors
        if nb_type == "ppr_sorting" or nb_type == "clip_cossim_sorting" or nb_type == "sbert_cossim_sorting":
            valid_neighbors_values = {}
            for val in valid_neighbors:
                valid_neighbors_values[val] = data[val][i] # data[val][i] is equal to data[i][val] ?
            sorted_neighbors = sorted(valid_neighbors_values.items(), key=lambda x: x[1], reverse=True) # decreasing order
            valid_neighbors = [k for k, v in sorted_neighbors] # get keys
            hop_neighbors[i]["hop1"] = valid_neighbors[:top_k] # top_k neighbors
        else:
            hop_neighbors[i]["hop1"] = valid_neighbors[:top_k] # randomly cap to top_k neighbors
        if len(valid_neighbors) == 0:
            empty_nb_cnt += 1
    print("len(hop_neighbors):", len(hop_neighbors))

    # Get additional hops if needed, from hop 2
    for k in range(1, hop):
        print(f"******Processing hop {k+1}******")
        for i in tqdm(range(num_nodes)):
            current_neighbors = hop_neighbors[i][f"hop{k}"]
            next_hop_neighbors = []
            for neighbor in current_neighbors:
                next_hop_neighbors.extend(hop_neighbors[neighbor][f"hop{k}"])
            hop_neighbors[i][f"hop{k+1}"] = list(set(next_hop_neighbors)) # remove duplicates
            current_neighbors = next_hop_neighbors  # Move to next hop level

    # clean the neighbor set: one node can only appear once in all hops
    filtered_hop_neighbors = {}
    for i in range(num_nodes):
        appeared = []
        for k in range(hop):
            filtered_neighbors = []
            for neighbor in hop_neighbors[i][f"hop{k+1}"]:
                if neighbor not in appeared and neighbor != i:
                    filtered_neighbors.append(neighbor)
                    appeared.append(neighbor)
            filtered_hop_neighbors.setdefault(i, {})[f"hop{k+1}"] = filtered_neighbors
    print("max_len of one hop neighbors:", max_len)
       
            
    print("no_nb:", no_nb)
    print("len(filtered_hop_neighbors):", len(filtered_hop_neighbors))
    print("empty_nb_cnt:", empty_nb_cnt)
    return filtered_hop_neighbors, degrees
