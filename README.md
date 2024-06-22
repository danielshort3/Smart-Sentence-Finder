# Smart Sentence Finder

This repository contains a project aimed at finding the most relevant sentences to an input query using various pre-trained sentence-transformer models. The project processes large text files, segments them into sentences, cleans the sentences, and computes their relevance scores based on cosine similarity with the input query.

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Processing Text](#processing-text)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
This project leverages pre-trained sentence-transformer models to identify and rank the most relevant sentences in a text document based on a given query. The process includes loading and cleaning the text, segmenting it into sentences, and calculating relevance scores using cosine similarity.

## Models
The repository supports the following pre-trained models:
- `sentence-transformers/all-mpnet-base-v2`
- `BAAI/bge-large-en-v1.5`
- `BAAI/bge-small-en-v1.5`
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `sentence-transformers/all-distilroberta-v1`
- `sentence-transformers/paraphrase-distilroberta-base-v1`
- `sentence-transformers/distiluse-base-multilingual-cased-v2`
- `sentence-transformers/msmarco-distilbert-cos-v5`

These models are used to encode both the input query and sentences from the text to compute similarity scores.

## Processing Text
The project includes several helper functions to process the text:
- **Text Chunking**: Splits the text into manageable chunks based on character count.
- **Sentence Segmentation**: Segments chunks into individual sentences using `pysbd`.
- **Cleaning Sentences**: Removes extra spaces and normalizes the text.

The `process` function calculates relevance scores for the sentences and prints the top-ranked sentences.

## Installation
Clone the repository:
```bash
git clone https://github.com/danielshort3/smart-sentence-finder.git
cd sentence-embeddings
```

Install the required packages:
```base
pip install torch sentence-transformers pysbd
```

## Usage
Run the notebook to process a text document and find relevant sentences:

1. Place your text file in the data directory.
2. Modify the input_query variable in the notebook to your desired query.
3. Execute the cells in the notebook.
