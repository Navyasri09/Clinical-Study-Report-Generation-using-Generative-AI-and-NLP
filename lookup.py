import pandas as pd
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch
import numpy as np

data = pd.read_csv("Disease.csv")

# creating an instance of the SentenceTransformer model using a specific pre-trained model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


# Function to generate embeddings for passages
def generate_embeddings(passages):
    embeddings = model.encode(
        passages, convert_to_tensor=True
    )  # encode passages into embeddings
    embeddings = F.layer_norm(
        embeddings, normalized_shape=(embeddings.shape[1],)
    )  # applying layer normalization to embeddings
    matryoshka_dim = 512
    embeddings = embeddings[:, :matryoshka_dim]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


class Lookup:
    def __init__(self, corpus_data: pd.DataFrame):
        self.corpus = corpus_data
        self.corpus_embeddings = generate_embeddings(
            corpus_data["Disease"].to_list()
        )  # Generate embeddings for list of passages in corpus_data

    def most_similar_column(self, query: str) -> str:
        query_embedding = generate_embeddings(
            [query]
        )  # Generate embedding for user given query

        cosine_similarity = F.cosine_similarity(
            query_embedding, self.corpus_embeddings
        )  # To calculate cosine similarity between query embedding and corpus embeddings

        """convert the cosine_similarity(tensor) to a NumPy array.
        detach(): This method detaches the tensor from the computation graph.
        numpy(): This method converts the tensor to a NumPy array.
        flatten(): This method flattens the NumPy array into a 1-dimensional array."""

        cosine_similarity_np = cosine_similarity.detach().numpy().flatten()

        max_similarity_index = np.argmax(
            cosine_similarity_np
        )  # Finds index of maximum similarity in numpy array
        most_similar_column = self.corpus.iloc[max_similarity_index]["Disease"]
        return most_similar_column
