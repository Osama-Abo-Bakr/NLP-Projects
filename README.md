# Natural Language Processing (NLP) Projects Repository

This repository contains a collection of **Natural Language Processing (NLP)** projects. Each project demonstrates the application of state-of-the-art techniques in NLP, ranging from machine translation and text classification to fine-tuning transformer models and building real-world applications. These projects serve as a comprehensive resource for anyone interested in learning and exploring NLP.

---

## Table of Contents
1. [Projects Overview](#projects-overview)
2. [Installation](#installation)
3. [Project Details](#project-details)
   - [1. Transformer Translation (Arabic to Italian)](#1-transformer-translation-arabic-to-italian)
   - [2. Text Classification with BERT](#2-text-classification-with-bert)
   - [3. GPT-2 Fine-Tuning for Text Generation](#3-gpt-2-fine-tuning-for-text-generation)
   - [4. Named Entity Recognition (NER)](#4-named-entity-recognition-ner)
   - [5. Sentiment Analysis](#5-sentiment-analysis)
   - [6. Streamlit-based NLP Applications](#6-streamlit-based-nlp-applications)
4. [Technologies Used](#technologies-used)
5. [License](#license)

---

## Projects Overview

This repository includes the following projects:
1. **Transformer Translation**: Fine-tuning a MarianMT model for Arabic to Italian translation.
2. **Text Classification**: Using BERT and fine-tuning it for multi-label classification tasks.
3. **GPT-2 Fine-Tuning**: Adapting GPT-2 for text generation tasks.
4. **NER Model**: Implementing Named Entity Recognition using SpaCy and Hugging Face Transformers.
5. **Sentiment Analysis**: Analyzing sentiment using pre-trained models on custom datasets.
6. **Streamlit Apps**: Deploying NLP models as interactive web applications for real-world use cases.

Each project folder contains its respective code, datasets, and a detailed explanation of the implementation.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/NLP-Projects.git
   cd NLP-Projects
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: Each project folder may have additional dependencies mentioned in its respective `README`.

---

## Project Details

### 1. Transformer Translation (Arabic to Italian)
**Description**: Fine-tunes the `Helsinki-NLP/opus-mt-ar-it` MarianMT model for translating Arabic to Italian using the Hugging Face `transformers` library.  
- Dataset: [Helsinki-NLP News Commentary](https://huggingface.co/datasets/Helsinki-NLP/news_commentary)  
- Model: [Helsinki-NLP/opus-mt-ar-it](https://huggingface.co/Helsinki-NLP/opus-mt-ar-it)  
- Deployment: Streamlit-based translator web app.

**Key Features**:
- Preprocessing and tokenization with MarianMT.
- BLEU score evaluation.
- Fine-tuning with Hugging Face Trainer.

**Path**: `/projects/transformer_translation_ar_it`

---

### 2. Text Classification with BERT
**Description**: Fine-tunes BERT for text classification tasks.  
- Dataset: Custom dataset with multiple labels.  
- Model: [BERT](https://huggingface.co/bert-base-uncased).

**Key Features**:
- Multi-label classification using the Hugging Face `Trainer`.
- Metrics computation (accuracy, F1 score).

**Path**: `/projects/text_classification_bert`

---

### 3. GPT-2 Fine-Tuning for Text Generation
**Description**: Fine-tunes GPT-2 to generate coherent text on custom prompts.  
- Dataset: Custom text dataset for training.  
- Model: [GPT-2](https://huggingface.co/gpt2).

**Key Features**:
- Tokenization with GPT-2 tokenizer.
- Fine-tuning with learning rate scheduling.
- Generative text outputs with temperature scaling.

**Path**: `/projects/gpt2_fine_tuning`

---

### 4. Named Entity Recognition (NER)
**Description**: Implements NER using SpaCy and fine-tuning transformer models.  
- Dataset: Annotated custom text data.  
- Tools: SpaCy, Hugging Face.

**Key Features**:
- Custom pipeline for entity extraction.
- Evaluation metrics like precision, recall, and F1 score.

**Path**: `/projects/named_entity_recognition`

---

### 5. Sentiment Analysis
**Description**: Analyzes sentiment from text using pre-trained models.  
- Dataset: Public datasets for sentiment analysis.  
- Model: Pre-trained Transformer models.

**Key Features**:
- Sentiment scoring and classification.
- Model fine-tuning for domain-specific sentiment analysis.

**Path**: `/projects/sentiment_analysis`

---

### 6. Streamlit-based NLP Applications
**Description**: A collection of Streamlit apps for deploying NLP models interactively.  
**Key Features**:
- Real-time translation.
- Text classification demos.
- Sentiment analysis dashboards.

**Path**: `/projects/streamlit_nlp_apps`

---

## Technologies Used
- **Frameworks**: Hugging Face Transformers, PyTorch, TensorFlow
- **Languages**: Python
- **Libraries**: SpaCy, NumPy, Pandas, Evaluate, SacreBLEU
- **Deployment**: Streamlit, Flask
- **Visualization**: Matplotlib, Seaborn

---

## License
This repository is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit changes: `git commit -m "Add feature"`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

For any questions or suggestions, feel free to open an issue.

---
