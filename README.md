# Sentence Embeddings
A tool to search and visualize sentence embeddings.

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

## Examples

Try out the demo!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ytaVFtLSIDzMQDqITiiezrC5QgZNzoKn)

![language model embeddings](https://github.com/nadahlberg/sentence-embeddings/blob/main/images/language_model_embeddings.png?raw=true)
![language model finetuned embeddings](https://github.com/nadahlberg/sentence-embeddings/blob/main/images/language_model_finetuned_embeddings.png?raw=true)
