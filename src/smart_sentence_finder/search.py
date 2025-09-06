from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .embedding import load_embedder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process(
    model_name: str,
    input_query: str,
    sentences: Iterable[str],
    n: int = 5,
    device_: torch.device = device,
    batch_size: int = 32,
    max_length: int = 512,
) -> Dict[str, object]:
    """Encode query and sentences with a model and rank by cosine similarity."""
    progress_bar = tqdm(total=5, desc="Starting")

    with load_embedder(model_name, device_) as embedder:
        progress_bar.update(1)
        progress_bar.set_description("Model loaded")

        input_embedding = embedder.encode(
            [input_query], batch_size=1, max_length=max_length, normalize=True, show_progress=False
        )[0]
        progress_bar.update(1)
        progress_bar.set_description("Input query encoded")

        filtered_sentences = [s for s in sentences if len(s.split()) > 5]
        embeddings = embedder.encode(
            filtered_sentences,
            batch_size=batch_size,
            max_length=max_length,
            normalize=True,
            show_progress=True,
            progress_desc=f"Embed {model_name}",
        )
        progress_bar.update(1)
        progress_bar.set_description("Sentences encoded")

    scores = F.cosine_similarity(embeddings, input_embedding.unsqueeze(0), dim=1)
    progress_bar.update(1)
    progress_bar.set_description("Cosine similarity calculated")

    scores_cpu = scores.to(torch.float32).cpu()
    top_sentences: List[Tuple[str, float]] = sorted(
        zip(filtered_sentences, scores_cpu), key=lambda x: x[1], reverse=True
    )[:n]
    progress_bar.update(1)
    progress_bar.set_description("Sentences sorted and selected")

    print(f"Model name is: {model_name}.\n")
    print(f"Input query is: {input_query}\n")
    for i, (sentence, score) in enumerate(top_sentences):
        print(f"Ranking: {i+1} | Score: {float(score):.4f}\nSentence: {sentence}\n")

    progress_bar.close()

    # Move heavy tensors to CPU to release GPU memory
    embeddings_cpu = embeddings.to(torch.float32).cpu()
    input_embedding_cpu = input_embedding.to(torch.float32).cpu()

    return {
        "filtered": len(filtered_sentences),
        "dim": embeddings_cpu.shape[1],
        "top": top_sentences,
        "model_name": model_name,
        "input_emb": input_embedding_cpu,
        "filtered_sentences": filtered_sentences,
        "scores": scores_cpu,
        "embeds": embeddings_cpu,
    }
