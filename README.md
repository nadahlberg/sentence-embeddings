# sentence-embeddings

## Setup
Install from PyPI

    pip install sentence-embeddings
    
## Quickstart
    
    from sentence_embeddings import Embeddings
    
    embeddings = Embeddings(vectors, texts=texts, labels=labels)
    
    # Visualize texts / labels in 3d space
    embeddings.plot_3d()
    
    # Search
    embeddings.search(query_vectors, k=5)

## How to create embeddings

