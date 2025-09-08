📰 Unsupervised News Topic Modeling

📖 Overview

Every day, thousands of news articles are published, making it impossible to manually analyze themes and trends. This project demonstrates how unsupervised learning in NLP can automatically discover hidden topics from a large corpus of news articles — no labels, no prior knowledge required.

The result: an AI-powered system for content analysis, trend detection, and information retrieval.

🛠️ The Solution

This project implements a full NLP + Machine Learning pipeline to extract and interpret latent topics in unstructured news text.

Preprocessing → Tokenization, stop-word removal, lemmatization, bag-of-words

Topic Modeling → Latent Dirichlet Allocation (LDA) for unsupervised topic discovery

Visualization → Word clouds and keyword lists to make hidden themes interpretable

This workflow highlights data cleaning, feature engineering, ML modeling, and interpretability — critical skills for any applied NLP role.

🔍 Technical Workflow

Data Preprocessing

Tokenization → split articles into words

Stop-word removal → drop irrelevant filler words

Lemmatization → unify vocabulary by reducing words to their root form

Corpus creation → dictionary + bag-of-words representation

Topic Modeling with LDA

Unsupervised algorithm to infer hidden topics

Each document is represented as a mixture of topics

Each topic = distribution of keywords with probabilities

Visualization & Interpretation

Word Clouds → highlight important keywords per topic

Top-N Keywords → provide interpretable insights for each theme

⚡ Tech Stack

Python → Core implementation

Pandas → Data ingestion + manipulation

NLTK & SpaCy → Text preprocessing (tokenization, lemmatization, stopwords)

Scikit-learn → LDA topic modeling

Matplotlib & WordCloud → Visualization of discovered topics

🚀 Getting Started

1. Clone the repository
git clone <your-repository-url>
cd <your-project-directory>

2. Install dependencies
pip install pandas scikit-learn matplotlib wordcloud


(You may also need to download NLTK/Spacy models.)

3. Run the project
   
python topic_modeling.py


🎯 Key Highlights

✅ Real-world application of unsupervised ML in NLP

✅ Extracts actionable insights from unstructured text data

✅ Strong focus on explainability via visualization

✅ Deployable for media monitoring, trend detection, and research

💻 Frontend Demo

<img width="1920" height="878" alt="image" src="https://github.com/user-attachments/assets/2529be24-c5c3-47fc-8616-6c2c92cb90ce" />
<img width="1910" height="881" alt="image" src="https://github.com/user-attachments/assets/25ac9a2c-5874-4640-9f1d-092d18c76a41" />




📜 License

This project is licensed under the MIT License.
