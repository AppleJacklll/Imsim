import embedding.dino_service as dino_service


def delete_file(filename: str):
    
    print("Deleting file")
    candidate_embeddings = dino_service.load_embeddings()
    print(f"Processing {filename}")

    candidate_embeddings.pop(filename, None)
        
    dino_service.update_embeddings(candidate_embeddings)
