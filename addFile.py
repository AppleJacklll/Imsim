import embedding.dino_service as dino_service


def add_files(filenames: list):
    
    print("Adding files")
    candidate_embeddings = dino_service.load_embeddings()
    for filename in filenames:
        if filename not in candidate_embeddings:
            print(f"Processing {filename}")
            features = dino_service.extract_features(filename)
            if features is None:
                continue
            
            candidate_embeddings[filename] = features
            
    dino_service.update_embeddings(candidate_embeddings)
