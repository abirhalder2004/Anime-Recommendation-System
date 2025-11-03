
# 🎌 Anime Recommendation System

## Anime_recomendation_v1


The **Anime Recommender System** is a web application that suggests anime titles based on user input and preferences. It uses a hybrid filtering approach combining **content-based filtering**, **collaborative filtering**, and **popularity-based ranking** to provide personalized and accurate anime suggestions.


Users can:
- Search for anime recommendations by name or genre.
- Filter recommendations by anime types (TV, Movie, OVA, etc.).
- Browse popular anime by type with pagination.
- View detailed project information and team member profiles.

This system aims to enhance the anime discovery experience for fans and newcomers alike.

## 📌 Features

- 🔍 **Content-Based Recommendations** using TF-IDF and Cosine Similarity.
- 🌟 **Popularity-Based Suggestions** based on normalized ratings and member counts.
- 🧠 **Hybrid Filtering** combining personalization and trending shows.
- 📊 **Type-based Filtering** for popular anime (TV, Movie, OVA).
- 💻 **Flask Web Application** with a clean, user-friendly UI.

---

## 📁 Project Directory Structure

```
anime-recommender/
├── app.py
├── templates/
│ ├── index.html
│ ├── recommendations.html
│ └── popular.html
├── static/
│ ├── css/
│ │ └── style.css
│ └── js/
│ └── script.js
├── data/
│ └── anime_data.csv
├── models/
│ └── collaborative_model.pkl
├── requirements.txt
├── Procfile
├── runtime.txt
└── README.md
```

> ℹ️ **The full project description and motivation can be found in** `templates/description.html`. It includes:
> - Purpose & scope
> - Dataset details
> - Algorithmic steps
> - Comparisons with other approaches
> - Advantages of the hybrid model
> - Future improvement scope

---
## Installation & Setup

### Prerequisites

- Python 3.7 or above  
- pip (Python package installer)  
- Git
- Flask
- pandas
- scikit-learn
- Flask==2.0.1
- pandas==1.3.3
- numpy==1.21.2
- scikit-learn==0.24.2
- requests==2.26.0
- python-dotenv==0.19.0
- Jinja2==3.0.1
- scipy==1.7.1
- gunicorn==20.1.0

## 🚀 Installation Guide

### Prerequisites
- Python 3.6+
- Git
- pip

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Prasunnandi/Anime_recomendation_v1.git
   cd Anime_recomendation_v1
   
**Create and activate virtual environment**:

 **For Linux/MacOS:**

```bash
python3 -m venv venv
source venv/bin/activate

**For Windows:**

```bash
python -m venv venv
venv\Scripts\activate

