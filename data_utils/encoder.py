# sbert_single.py
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from build_corpus import load_data
import argparse
import json
import pdb


def load_text(dataset):
    corpus_path = f"../dataset/{dataset}/{dataset}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    texts = [corpus[str(i)]["text"] for i in range(len(corpus))]
    print("len(texts):", len(texts))
    print("Example text:", texts[0][:200])
    return texts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SBERT dataset encoder")
    p.add_argument("--model_path", default="/vast/jl11523/projects-local-models/all-mpnet-base-v2", help="pretrained model name or path")
    p.add_argument("--dataset", default="arxiv", help="Dataset name")
    p.add_argument("--output", default=None, help="path to output file (pt format)")
    p.add_argument("--batch_size", type=int, default=128, help="Batch size for encoding",)

    args = p.parse_args()
    if args.output is None:
        args.output = f"../dataset/{args.dataset}/{args.dataset}_sbert_embeddings.pt"

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = SentenceTransformer(args.model_path, device=device) # sbert

    # # pdb.set_trace()
    # #_, texts, _ = load_data(args.dataset) # texts is []

    # texts = load_text(args.dataset)

    # print(f"[Info] Loaded dataset '{args.dataset}' with {len(texts)} texts")
    # embeddings = model.encode(texts, convert_to_tensor=True, batch_size=args.batch_size, show_progress_bar=True)
   
    # torch.save(embeddings, args.output)
    # print(f"[Info] Saved embeddings to '{args.output}'")
    # print(f"[Info] Embeddings shape: {embeddings.shape}")

    raw = torch.load(args.output, weights_only=True)

    # Extract torch.Tensor as above → E (2-D)
    # (reuse the extraction code from Option A)
    # Now convert to NumPy:
    X = raw.detach().cpu().numpy()  # [N, d]

    cos_sim = cosine_similarity(X)  # [N, N] (watch memory!)
    print("Cosine similarity matrix shape:", cos_sim.shape)
    top_k = 10
    top_k_indices = cos_sim.argsort(axis=1)[:, -top_k-1:-1][:, ::-1]  # exclude self
    print("Top-k indices shape:", top_k_indices.shape)
    print("Example top-k indices:", top_k_indices[:2])
    # save top_k_indices
    corpus_path = f"../dataset/{args.dataset}/{args.dataset}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    for i in range(top_k_indices.shape[0]):
        corpus[str(i)]["semantic_neighbors"] = top_k_indices[i].tolist()
    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=4)
    print(f"[Info] Saved semantic neighbors to '{corpus_path}'")
