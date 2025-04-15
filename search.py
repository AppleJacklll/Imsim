import faiss
import numpy as np
import embedding.dino_service as dino_service
#import os

#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    epsilon = 1e-10
    return embeddings / (norms + epsilon)


def create_faiss_index_ip(embeddings: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def get_final_similarity(similarity: float) -> int:
    final_similarity = int(similarity * 100) / 100
    if final_similarity >= 99.95:
        final_similarity = 100
    elif final_similarity >= 99:
        final_similarity = 99
    elif final_similarity >= 98:
        digit = int(final_similarity * 100) % 100
        digit = (100 * (digit + 50)) / 150
        final_similarity = int(digit)
    elif final_similarity >= 97.5:
        digit = int(final_similarity * 100) % 100
        digit = (100 * (digit - 50)) / 150
        final_similarity = int(digit)
    else:
        final_similarity = 0
        
    return final_similarity


def find_similar_images(filename: str, results_count: int) -> dict:
    
    print("Finding similar images")
    candidate_embeddings = dino_service.load_embeddings()
    new_features = dino_service.extract_features(filename)
    if new_features is None:
        return {}
    
    embeddings_list = list(candidate_embeddings.values())
    embeddings = np.vstack(embeddings_list).astype('float32')
    embeddings = normalize_embeddings(embeddings)
    index = create_faiss_index_ip(embeddings)
    
    new_features = new_features.astype('float32').reshape(1, -1)
    new_features = normalize_embeddings(new_features)
    
    distances, indices = index.search(new_features, min(10, len(embeddings_list)))
    
    similarities = [100 * (1 + d) / 2 for d in distances[0]]
    results = [(list(candidate_embeddings.keys())[idx], float(similarities[i])) for i, idx in enumerate(indices[0])]

    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    
    result = {}
    for filename, similarity in results_sorted[:results_count]:
        print(f"{filename} (Similarity: {similarity:.2f}%)")
        final_similarity = get_final_similarity(similarity)
        if final_similarity > 0:
            result[filename] = final_similarity

    return result
