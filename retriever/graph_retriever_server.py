import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import datasets
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pdb


def load_corpus(corpus_path: str):
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    return corpus

    
class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.topk = config.retrieval_topk
        self.corpus_path = config.corpus_path
        self.corpus = load_corpus(self.corpus_path)
        self.batch_size = config.retrieval_batch_size
        self.model = SentenceTransformer(config.model_name)
        # load pre-computed embeddings
        embeddings_path = self.corpus_path.replace("_corpus.json", "_sbert_embeddings.pt")
        self.corpus_embs = torch.load(embeddings_path)

    def _search(self, query: str, node_id: str, search_hop: int, num: int):
        raise NotImplementedError # Jiajin comments: actaully not been called yet

    def _batch_search(self, queries: List[str], curr_nb_ids: List[List[str]], node_ids: List[str], search_hops: List[int], num: int):
        raise NotImplementedError 

    def search(self, query: str, node_id: str, search_hop: int, num: int):
        return self._search(query, node_id, search_hop, num) # Jiajin comments: actaully not been called yet
    
    def batch_search(self, queries: List[str], curr_nb_ids: List[List[str]], node_ids: List[str], search_hops: List[int], num: int, candidate_filter: str, score_method: str):
        if candidate_filter is not None:
            self.config.candidate_filter = candidate_filter
        if score_method is not None:
            self.config.score_method = score_method
        return self._batch_search(queries, curr_nb_ids, node_ids, search_hops, num)



class SemanticSimRetriever(BaseRetriever):

    def __init__(self, config):
        super().__init__(config)

    def calculate_scores(self, corpus, node_id, cand_embs, q_emb, device):
        # semantic similarity between the query_embedding and the candidatets embedding, as the ranker
        calculater = self.config.score_method
        if calculater in ("SemQ", "SemA", "SemQA"):
            # attribute similarity between the query and query (query_only, attribute_only, query+attribute) <- decided by the searcher input
            scores = torch.nn.functional.cosine_similarity(cand_embs, q_emb.unsqueeze(0).expand_as(cand_embs), dim=1)  # [M]
            return scores
        elif calculater == "WeightedSemQA":
            # weighted sum of semantic similarity and attribute similarity
            alpha = 0.5
            beta = 1 - alpha 
            an_emb = corpus[int(node_id)]            # [d]
            scores = alpha * torch.nn.functional.cosine_similarity(cand_embs, an_emb.expand_as(cand_embs), dim=1) + \
                     beta * torch.nn.functional.cosine_similarity(cand_embs, q_emb.unsqueeze(0).expand_as(cand_embs), dim=1)
            return scores
        elif calculater == "StrucSemQA":
            print(f"[Warining] {calculater} Not implemented yet.")
            raise NotImplementedError
        else:
            raise ValueError(f"Not supported score_method: {calculater}.")
        
    

    def get_candidate_ids(self, node_id, curr_nb_id, topk):
        # Sem, Hop, PPR, Sem+Hop, Sem+PPR, Hop+PPR, Sem+Hop+PPR
        candidate_ids = []
        for nb in curr_nb_id:
            if "Sem" in self.config.candidate_filter:
                candidate_ids.extend(self.corpus[node_id]["semantic_neighbors"][:topk])
            if "Hop" in self.config.candidate_filter:
                candidate_ids.extend(self.corpus[nb]["neighbors"][:topk])
            if "PPR" in self.config.candidate_filter:
                candidate_ids.extend(self.corpus[node_id]["ppr_neighbors"][:topk])
        candidate_ids = list(set(candidate_ids))  # unique
        # print("candidate_ids:", candidate_ids)
        return candidate_ids


    def retrieve_nb_ids(self, queries: List[str], node_ids: List[str], curr_nb_ids: List[List[str]], topk: int): # node_ids: unused
        batch_query_emb = self.model.encode(
            queries, convert_to_tensor=True,
            batch_size=self.batch_size, normalize_embeddings=True
        )  # [B, d]
        # print("batch_query_emb.shape:", batch_query_emb.shape)

        corpus = self.corpus_embs
        if not torch.is_tensor(corpus):
            corpus = torch.tensor(corpus)

        device = batch_query_emb.device
        corpus = corpus.to(device)
        corpus = torch.nn.functional.normalize(corpus, p=2, dim=1)  # [N, d]

        k = min(topk, corpus.size(0))
        nb_ids = []

        for q_emb, node_id, curr_nb_id in zip(batch_query_emb, node_ids, curr_nb_ids):
            if curr_nb_id is None or len(curr_nb_id) == 0:
                raise ValueError("curr_nb_id cannot be None or empty.")
            # candidate set
            candidate_ids = self.get_candidate_ids(node_id, curr_nb_id, topk)
            if not candidate_ids:
                nb_ids.append([])
                continue
            # ranker: semantic similarity, structural similarity, combined similarity
            cand_idx = torch.as_tensor(candidate_ids, device=device, dtype=torch.long)
            # print("cand_idx:", cand_idx)
            cand_embs = corpus.index_select(0, cand_idx)  # [M, d]
            scores = self.calculate_scores(corpus, node_id, cand_embs, q_emb, device)
            k = min(topk, cand_embs.size(0))
            topk_local = torch.topk(scores, k=k).indices.tolist()
            topk_global = [candidate_ids[j] for j in topk_local]
            # print("topk_local:", topk_local, "topk_global:", topk_global)
            nb_ids.append(topk_global)

        # print(f"Retrieved neighbors for node_ids{node_ids} queries: {nb_ids}.")
        return nb_ids


    # Jiajin: no need yet
    def _search(self, query: str, node_id: str, search_hop: int, num: int):
        if num is None:
            num = self.topk
        nb_ids = self.retrieve_nb_ids(node_id, search_hop, num)
        results = [self.corpus[idx] for idx in nb_ids[0]]
        return results

    def _batch_search(self, queries: List[str], curr_nb_ids: List[List[str]], node_ids: List[str], search_hops: List[int], num: int):
        if isinstance(queries, str):
            queries = [queries]

        results = []
        for start_idx in tqdm(range(0, len(queries), self.batch_size), desc='Retrieval process: '):
            query_batch = queries[start_idx:start_idx + self.batch_size]
            curr_nb_ids_batch = curr_nb_ids[start_idx:start_idx + self.batch_size] # retrieve hop1 nbs based on ids in the requested list
            batch_nb_ids = self.retrieve_nb_ids(query_batch, node_ids, curr_nb_ids_batch, num)
            flat_nb_ids = [str(i) for i in sum(batch_nb_ids, [])]
            batch_results = [self.corpus[idx] for idx in flat_nb_ids]
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_nb_ids))]
            results.extend(batch_results)
            
            del batch_nb_ids, query_batch, flat_nb_ids, batch_results
            torch.cuda.empty_cache()
        
        return results
    





class RandomRetriever(BaseRetriever):

    def __init__(self, config):
        super().__init__(config)
        self.corpus = load_corpus(self.corpus_path)
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def retrieve_nb_ids(self, node_ids: List[str], search_hops: List[int], topk: int):
        nb_ids = []
        for n_id, hop in zip(node_ids, search_hops):
            n_id = int(n_id)
            if hop == 0:
                raise ValueError("Hop value cannot be 0.")
            else:
                if n_id >= len(self.corpus):
                    raise ValueError(f"Node ID {n_id} exceeds the number of nodes in the index ({len(self.corpus)}).")
                # retrieve neighbors from corpus
                neighbors = self.corpus[str(n_id)]["nb"][f"hop{hop}"]
                if not neighbors:
                    warnings.warn(f"No neighbors found for Node ID {n_id} at hop {hop}. Returning empty list.")
                    nb_ids.append([])
                    continue
                nb_ids.append(neighbors[:topk])
        print(f"Retrieved neighbors for {len(node_ids)} nodes: {nb_ids}.")
        return nb_ids


    def _search(self, query: str, node_id: str, search_hop: int, num: int):
        if num is None:
            num = self.topk
        nb_ids = self.retrieve_nb_ids(node_id, search_hop, num)
        results = [self.corpus[idx] for idx in nb_ids[0]]
        return results

    def _batch_search(self, queries: List[str], curr_nb_ids: List[List[str]], node_ids: List[str], search_hops: List[int], num: int):
        if isinstance(node_ids, str):
            node_ids = [node_ids]
        if isinstance(search_hops, int):
            search_hops = [search_hops]
        assert len(node_ids) == len(search_hops), "Length of node_ids and search_hops must be the same."
        if num is None:
            num = self.topk
        
        results = []
        for start_idx in tqdm(range(0, len(node_ids), self.batch_size), desc='Retrieval process: '):
            query_batch = node_ids[start_idx:start_idx + self.batch_size]
            search_hops_batch = search_hops[start_idx:start_idx + self.batch_size]
            batch_nb_ids = self.retrieve_nb_ids(query_batch, search_hops_batch, num)
            flat_nb_ids = [str(i) for i in sum(batch_nb_ids, [])]
            batch_results = [self.corpus[idx] for idx in flat_nb_ids]
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_nb_ids))]
            results.extend(batch_results)
            
            del batch_nb_ids, query_batch, flat_nb_ids, batch_results
            torch.cuda.empty_cache()
        
        return results
    

class RawSimRetriever(BaseRetriever):

    def __init__(self, config):
        super().__init__(config)
        self.corpus = load_corpus(self.corpus_path)
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        from sentence_transformers import SentenceTransformer
        model_name = "/vast/jl11523/projects-local-models/all-mpnet-base-v2"
        self.model = SentenceTransformer(model_name)
        # load pre-computed embeddings
        embeddings_path = self.corpus_path.replace("_corpus.json", "_sbert_embeddings.pt")
        self.corpus_embs = torch.load(embeddings_path)


    def retrieve_nb_ids(self, queries: List[str], topk: int):
        batch_query_emb = self.model.encode(
            queries, convert_to_tensor=True,
            batch_size=self.batch_size, normalize_embeddings=True
        )  # [B, d]

        corpus = self.corpus_embs
        if not torch.is_tensor(corpus):
            corpus = torch.tensor(corpus)
        corpus = corpus.to(batch_query_emb.device)
        corpus = torch.nn.functional.normalize(corpus, p=2, dim=1)  # [N, d]

        k = min(topk, corpus.size(0))
        nb_ids = []

        for query_emb in batch_query_emb:  # no zip()
            # [N, d] vs [d] -> extend to [N, d]
            q_expanded = query_emb.unsqueeze(0).expand_as(corpus)  # [N, d]
            scores = torch.nn.functional.cosine_similarity(corpus, q_expanded, dim=1)  # [N]
            topk_ids = torch.topk(scores, k=k).indices.tolist()
            nb_ids.append(topk_ids)

        print(f"Retrieved neighbors for {len(queries)} queries: {nb_ids}.")
        return nb_ids


    def _search(self, query: str, node_id: str, search_hop: int, num: int):
        nb_ids = self.retrieve_nb_ids([query], num)
        results = [self.corpus[idx] for idx in nb_ids[0]]
        return results

    def _batch_search(self, queries: List[str], curr_nb_ids: List[List[str]], node_ids: List[str], search_hops: List[int], num: int):
        if isinstance(queries, str):
            queries = [queries]

        results = []
        for start_idx in tqdm(range(0, len(queries), self.batch_size), desc='Retrieval process: '):
            query_batch = queries[start_idx:start_idx + self.batch_size]
            batch_nb_ids = self.retrieve_nb_ids(query_batch, num)
            flat_nb_ids = [str(i) for i in sum(batch_nb_ids, [])]
            batch_results = [self.corpus[idx] for idx in flat_nb_ids]
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_nb_ids))]
            results.extend(batch_results)
            
            del batch_nb_ids, query_batch, flat_nb_ids, batch_results
            torch.cuda.empty_cache()
        
        return results




class HopSimRetriever(BaseRetriever):

    def __init__(self, config):
        super().__init__(config)
        self.corpus = load_corpus(self.corpus_path)
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        from sentence_transformers import SentenceTransformer
        model_name = "/vast/jl11523/projects-local-models/all-mpnet-base-v2"
        self.model = SentenceTransformer(model_name)
        # load pre-computed embeddings
        embeddings_path = self.corpus_path.replace("_corpus.json", "_sbert_embeddings.pt")
        self.corpus_embs = torch.load(embeddings_path)


    def retrieve_nb_ids(self, queries: List[str], curr_nb_ids: List[List[str]], topk: int): # node_ids: unused
        batch_query_emb = self.model.encode(
            queries, convert_to_tensor=True,
            batch_size=self.batch_size, normalize_embeddings=True
        )  # [B, d]
        # print("batch_query_emb.shape:", batch_query_emb.shape)

        corpus = self.corpus_embs
        if not torch.is_tensor(corpus):
            corpus = torch.tensor(corpus)
        device = batch_query_emb.device
        corpus = corpus.to()
        corpus = torch.nn.functional.normalize(corpus, p=2, dim=1)  # [N, d]

        k = min(topk, corpus.size(0))
        nb_ids = []

        for q_emb, curr_nb_id in zip(batch_query_emb, curr_nb_ids):
            if curr_nb_id is None or len(curr_nb_id) == 0:
                raise ValueError("curr_nb_id cannot be None or empty.")
            # filter corpus to only curr_nb_id's neighbors
            candidate_ids = []
            for nb in curr_nb_id:
                candidate_ids.extend(self.corpus[nb]["neighbors"])
            candidate_ids = list(set(candidate_ids))  # unique
            # print("candidate_ids:", candidate_ids)
            if not candidate_ids:
                nb_ids.append([])
                continue
            cand_idx = torch.as_tensor(candidate_ids, device=device, dtype=torch.long)
            # print("cand_idx:", cand_idx)
            cand_embs = corpus.index_select(0, cand_idx)  # [M, d]
            scores = torch.nn.functional.cosine_similarity(cand_embs, q_emb.unsqueeze(0).expand_as(cand_embs), dim=1)  # [M]
            # print("scores:", scores)
            k = min(topk, cand_embs.size(0))
            topk_local = torch.topk(scores, k=k).indices.tolist()
            # print("topk_local:", topk_local)
            topk_global = [candidate_ids[j] for j in topk_local]
            # print("topk_global:", topk_global)
            nb_ids.append(topk_global)

        print(f"Retrieved neighbors for {len(queries)} queries: {nb_ids}.")
        return nb_ids


    def _search(self, query: str, node_id: str, search_hop: int, num: int):
        if num is None:
            num = self.topk
        nb_ids = self.retrieve_nb_ids(node_id, search_hop, num)
        results = [self.corpus[idx] for idx in nb_ids[0]]
        return results

    def _batch_search(self, queries: List[str], curr_nb_ids: List[List[str]], node_ids: List[str], search_hops: List[int], num: int):
        if isinstance(queries, str):
            queries = [queries]

        results = []
        for start_idx in tqdm(range(0, len(queries), self.batch_size), desc='Retrieval process: '):
            query_batch = queries[start_idx:start_idx + self.batch_size]
            curr_nb_ids_batch = curr_nb_ids[start_idx:start_idx + self.batch_size] # retrieve hop1 nbs based on ids in the requested list
            print("query_batch:", query_batch)
            print("curr_nb_ids_batch:", curr_nb_ids_batch)
            batch_nb_ids = self.retrieve_nb_ids(query_batch, curr_nb_ids_batch, num)
            flat_nb_ids = [str(i) for i in sum(batch_nb_ids, [])]
            print("retrieved flat_nb_ids:", flat_nb_ids)
            batch_results = [self.corpus[idx] for idx in flat_nb_ids]
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_nb_ids))]
            results.extend(batch_results)
            
            del batch_nb_ids, query_batch, flat_nb_ids, batch_results
            torch.cuda.empty_cache()
        
        return results



class HopGlobalSimRetriever(BaseRetriever):

    def __init__(self, config):
        super().__init__(config)
        self.corpus = load_corpus(self.corpus_path)
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        from sentence_transformers import SentenceTransformer
        model_name = "/vast/jl11523/projects-local-models/all-mpnet-base-v2"
        self.model = SentenceTransformer(model_name)
        # load pre-computed embeddings
        embeddings_path = self.corpus_path.replace("_corpus.json", "_sbert_embeddings.pt")
        self.corpus_embs = torch.load(embeddings_path)
        

    def retrieve_nb_ids(self, queries: List[str], node_ids: List[str], curr_nb_ids: List[List[str]], topk: int): # node_ids: unused
        batch_query_emb = self.model.encode(
            queries, convert_to_tensor=True,
            batch_size=self.batch_size, normalize_embeddings=True
        )  # [B, d]
        # print("batch_query_emb.shape:", batch_query_emb.shape)

        corpus = self.corpus_embs
        if not torch.is_tensor(corpus):
            corpus = torch.tensor(corpus)
        device = batch_query_emb.device
        corpus = corpus.to()
        corpus = torch.nn.functional.normalize(corpus, p=2, dim=1)  # [N, d]

        k = min(topk, corpus.size(0))
        nb_ids = []

        for q_emb, node_id, curr_nb_id in zip(batch_query_emb, node_ids, curr_nb_ids):
            if curr_nb_id is None or len(curr_nb_id) == 0:
                raise ValueError("curr_nb_id cannot be None or empty.")
            candidate_ids = []
            # curr_nb_id's neighbors
            for nb in curr_nb_id:
                candidate_ids.extend(self.corpus[nb]["neighbors"])
                candidate_ids.extend(self.corpus[nb]["ppr_neighbors"])  # include global by ppr
            candidate_ids = list(set(candidate_ids))  # unique
            # print("candidate_ids:", candidate_ids)
            if not candidate_ids:
                nb_ids.append([])
                continue
            cand_idx = torch.as_tensor(candidate_ids, device=device, dtype=torch.long)
            # print("cand_idx:", cand_idx)
            cand_embs = corpus.index_select(0, cand_idx)  # [M, d]
            # TODO: as an API: semantic similarity, structural similarity, combined similarity
            scores = torch.nn.functional.cosine_similarity(cand_embs, q_emb.unsqueeze(0).expand_as(cand_embs), dim=1)  # [M]
            # print("scores:", scores)
            k = min(topk, cand_embs.size(0))
            topk_local = torch.topk(scores, k=k).indices.tolist()
            # print("topk_local:", topk_local)
            topk_global = [candidate_ids[j] for j in topk_local]
            # print("topk_global:", topk_global)
            nb_ids.append(topk_global)

        print(f"Retrieved neighbors for {len(queries)} queries: {nb_ids}.")
        return nb_ids


    def _search(self, query: str, node_id: str, search_hop: int, num: int):
        if num is None:
            num = self.topk
        nb_ids = self.retrieve_nb_ids(node_id, search_hop, num)
        results = [self.corpus[idx] for idx in nb_ids[0]]
        return results

    def _batch_search(self, queries: List[str], curr_nb_ids: List[List[str]], node_ids: List[str], search_hops: List[int], num: int):
        if isinstance(queries, str):
            queries = [queries]

        results = []
        for start_idx in tqdm(range(0, len(queries), self.batch_size), desc='Retrieval process: '):
            query_batch = queries[start_idx:start_idx + self.batch_size]
            curr_nb_ids_batch = curr_nb_ids[start_idx:start_idx + self.batch_size] # retrieve hop1 nbs based on ids in the requested list
            print("query_batch:", query_batch)
            print("curr_nb_ids_batch:", curr_nb_ids_batch)
            batch_nb_ids = self.retrieve_nb_ids(query_batch, node_ids, curr_nb_ids_batch, num)
            flat_nb_ids = [str(i) for i in sum(batch_nb_ids, [])]
            print("retrieved flat_nb_ids:", flat_nb_ids)
            batch_results = [self.corpus[idx] for idx in flat_nb_ids]
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_nb_ids))]
            results.extend(batch_results)
            
            del batch_nb_ids, query_batch, flat_nb_ids, batch_results
            torch.cuda.empty_cache()
        
        return results



def get_retriever(config):
    if config.retrieval_method == "similarity":
        print("Based on similarity for subgraph nodes for retrieval.")
        return RandomRetriever(config)
    if config.retrieval_method == "raw-similarity":
        print("Based on similarity for ALL nodes for retrieval.")
        return RawSimRetriever(config)
    if config.retrieval_method == "hop-similarity":
        print("Based on similarity for hop-h nodes for retrieval.")
        return HopSimRetriever(config)
    if config.retrieval_method == "hop-global-similarity":
        print("Based on similarity for hop-h and global nodes for retrieval.")
        return HopGlobalSimRetriever(config)
    # ===================================================
    # New Version below. If useful, the above old ones (including classes) can be deleted.
    if config.retrieval_method == "semantic-similarity": # seems like we dont need config.retrieval_method  TODO: jiajin to clear
        print("Semantic similarity as the retriever.")
        return SemanticSimRetriever(config)
    else:
        print(f"No config.retrieval_method={config.retrieval_method}.")
        raise ValueError


#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "similarity", 
        retrieval_topk: int = 3,
        model_name: str = "/vast/jl11523/projects-local-models/all-mpnet-base-v2",
        corpus_path: str = "./dataset/cora/cora_corpus.json",
        faiss_gpu: bool = True,
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        candidate_filter: str = "Sem", # or "Hop" or "PPR" or "Sem+Hop" or "Sem+PPR" or "Hop+PPR" or "Sem+Hop+PPR"
        score_method: str = "SemQ"  # or "SemA" or "SemQA" or "WeightedSemQA" or "StrucSemQA(TODO)"
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.model_name = model_name
        self.corpus_path = corpus_path
        self.faiss_gpu = faiss_gpu
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.candidate_filter = candidate_filter
        self.score_method = score_method


class QueryRequest(BaseModel):
    queries: List[str]
    curr_nb_ids: List[List[str]]
    node_ids: List[str]
    search_hops: List[int]
    topk: int
    candidate_filter: Optional[str] = None  # Optional field for candidate filtering method
    score_method: Optional[str] = None  # Optional field for scoring method


app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    # Perform batch retrieval
    results = retriever.batch_search(
        queries=request.queries,
        curr_nb_ids=request.curr_nb_ids,
        node_ids=request.node_ids,
        search_hops=request.search_hops,
        num=request.topk,
        candidate_filter=request.candidate_filter,
        score_method=request.score_method
    )
    
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        resp.append(single_result)
    # print("[POST Return] retrieve_endpoint")
    return {"result": resp}



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
    parser.add_argument("--corpus_path", type=str, default="./dataset/cora/cora_corpus.json", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved nodes for one query.")
    parser.add_argument("--retriever_name", type=str, default="semantic-similarity", help="Name of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')

    args = parser.parse_args()
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk, 
        faiss_gpu=args.faiss_gpu,
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    print("Configuration:")
    for k, v in vars(config).items():
        print(f"  {k}: {v}")
    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=args.port)
