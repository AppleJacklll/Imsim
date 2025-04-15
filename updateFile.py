import embedding.dino_service as dino_service


def update_file(filename: str, new_filename: str):
    
    print("Updating file")
    candidate_embeddings = dino_service.load_embeddings()
    if filename in candidate_embeddings:
        print(f"Processing {new_filename}")
        features = dino_service.extract_features(new_filename)
        if features is None:
            return
        
        candidate_embeddings.pop(filename, None)
        candidate_embeddings[filename] = features
        
    dino_service.update_embeddings(candidate_embeddings)
