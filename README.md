
🧠 Next Word Prediction using NLP & Transformers
📌 Overview

This project implements a Next Word Prediction system using Natural Language Processing (NLP) and Transformer-based models (GPT-2).
It predicts the most probable next word in a sentence — similar to how typing assistants (like Google or chatbots) suggest words in real time.

The model was fine-tuned on a text dataset and deployed through a Streamlit web app for interactive predictions.

🚀 Features

✅ Predicts the next word based on user input text
✅ Fine-tuned Transformer model (GPT-2) from Hugging Face
✅ Clean, real-time interface built with Streamlit
✅ GPU-enabled Colab notebook for easy training and deployment
✅ Demonstrates text preprocessing, tokenization, and model inference


🧩 Tech Stack

| Category             | Tools / Libraries         |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| NLP Framework        | Hugging Face Transformers |
| Deep Learning        | PyTorch / TensorFlow      |
| Web App              | Streamlit                 |
| Data Handling        | Pandas, NumPy             |
| Environment          | Google Colab (GPU)        |


Next-Word-Prediction/
│
├── next_word_prediction.ipynb     # Colab notebook (training + app)
├── app.py                         # Streamlit app file
├── requirements.txt               # Dependencies
├── sample_output.png              # Screenshot of app
├── README.md                      # Project documentation
└── data/
    └── sample_text.txt            # Optional dataset

🧠 How It Works

Dataset Preparation – Load and preprocess text data (cleaning, tokenization).

Model Selection – Load pre-trained GPT-2 from Hugging Face.

Fine-Tuning – Train on custom text corpus for domain adaptation.

Prediction – Given a text input, generate the most likely next word.

Deployment – Streamlit app allows interactive next-word prediction.

🖥️ Sample Output

Input:

Artificial Intelligence is changing the

Predicted Output:

world 🌍

App Screenshot:


📈 Key Learnings

Understanding Transformer architecture and language modeling

Fine-tuning pre-trained NLP models

Deploying AI models using Streamlit

Using GPU acceleration for NLP tasks in Colab

🔮 Future Improvements

Extend model for sentence completion

Train on larger custom datasets

Add support for multilingual prediction

Integrate with chatbots or text editors

🤝 Connect With Me
