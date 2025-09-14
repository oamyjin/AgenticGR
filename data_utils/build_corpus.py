import json
import random
import torch
from tqdm import tqdm
from retrieve_neighbors import get_hop_neighbors
from retrieve_neighbors import get_hop1_neighbors
import pdb



def generate_emb(model_path, texts, output, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device) # sbert
    embeddings = model.encode(texts, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True)
    torch.save(embeddings, output)
    print(f"[Info] Saved embeddings to '{output}'")
    print(f"[Info] Embeddings shape: {embeddings.shape}")


def load_data(dataset, use_dgl=False, use_text=True, seed=0):
    # pdb.set_trace()
    if dataset.lower() == 'cora':
        from load_cora import get_raw_text_cora as get_raw_text
    elif dataset.lower() == 'pubmed':
        from load_pubmed import get_raw_text_pubmed as get_raw_text
        # num_classes = 3
    elif dataset.lower() == 'arxiv': # ogbn-arxiv
        from load_arxiv import get_raw_text_arxiv as get_raw_text
        # num_classes = 40
    elif dataset.lower() == 'products': # ogbn-products
        from load_products import get_raw_text_products as get_raw_text
        # num_classes = 47
    elif dataset.lower() == 'computers':
        from load_amazon import get_raw_text_computers as get_raw_text   
    elif dataset.lower() == 'sports':
        from load_amazon import get_raw_text_sports as get_raw_text    
    else:
        exit(f'Error: Dataset {dataset} not supported')
    
    data, text, labels = get_raw_text(use_text=True, seed=seed)

    return data, text, labels


# dataset_corpus
# node: "id", "text", "degree", "nb:{hop1:[], hop2:[], ...}", "label", "summarized_text"
# global: "dataset:avg_degree"

def build_corpus(dataset, max_hop=1, topk=None, gen_emb=False):
    dataset_corpus_file = f"../dataset/{dataset}/{dataset}_corpus.json"

    data, text, labels = load_data(dataset)
    print("data:", data)
    print("len(text):", len(text))

    # global info: dataset_avg_degree; .2f format
    avg_degree = round(data.edge_index.shape[1] / data.num_nodes, 2)
    # get neighbors of hops for each node
    # hop_neighbors, degrees = get_hop_neighbors(data, max_hop, topk, "todo") #"clip_cossim_sorting")
    hop_neighbors, degrees = get_hop1_neighbors(data)

    # for each node in test set, build corpus
    test_ids = [i for i, x in enumerate(data.test_mask) if x]
    train_ids = [i for i, x in enumerate(data.train_mask) if x]
    val_ids = [i for i, x in enumerate(data.val_mask) if x]
    dataset_corpus = {}
    num_nodes = len(data.y)
    for i in tqdm(range(num_nodes)): # train_ids+val_ids   
        # build one record
        one_record = {
            "id": i,
            "split": "test" if i in test_ids else ("train" if i in train_ids else "val"),
            "text": text[i],
            "summarized_text": "",
            "neighbors": hop_neighbors[i],
            "label": labels[i],
            "degree": degrees[i],
            "dataset_avg_degree": avg_degree
        }
        dataset_corpus[i] = one_record
    print("len(dataset_corpus):", len(dataset_corpus))
    with open(dataset_corpus_file, 'w') as f:
        json.dump(dataset_corpus, f, indent=4)
    print(f"Saved corpus to {dataset_corpus_file}")


    # generate and save text embeddings
    if gen_emb:
        print("="*20, "Generating text embeddings ...", "="*20)
        model_path = "/vast/jl11523/projects-local-models/all-mpnet-base-v2"
        output = f"../dataset/{dataset}/{dataset}_sbert_embeddings.pt"
        generate_emb(model_path, text, output)


if __name__ == "__main__":
    build_corpus("sports")

