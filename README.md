# Twitter Sentiment Analysis Project

## Project Overview
This project aims to perform sentiment analysis on Twitter data related to various technology companies and products. The goal is to classify tweets into different sentiment categories (Positive, Negative, Neutral, Irrelevant) using machine learning models. The notebook covers data loading, preprocessing, exploratory data analysis, feature engineering using TF-IDF, model training with Logistic Regression and RandomForest Classifier, and evaluation of the models.

## Dataset
The dataset used for this project is `twitter_training.csv`, which contains tweets with corresponding sentiment labels. The columns in the dataset are:
- `2401`: An identifier (dropped during preprocessing).
- `Borderlands`: Original column name for the technology company/product.
- `Positive`: Original column name for the sentiment label.
- `im getting on borderlands and i will murder you all ,`: Original column name for the tweet text.

After preprocessing, the columns are renamed to `technology_companies`, `sentiment`, and `review`.

## Methodology

### 1. Data Loading and Initial Inspection
- The dataset `twitter_training.csv` is loaded into a pandas DataFrame.
- Initial rows are displayed, and basic information (`df.info()`, `df.isna().sum()`, `df.duplicated().sum()`) is checked to understand data types, missing values, and duplicates.

### 2. Data Cleaning and Preprocessing
- **Column Renaming**: Columns are renamed for clarity: `Borderlands` to `technology_companies`, `Positive` to `sentiment`, and `im getting on borderlands and i will murder you all ,` to `review`.
- **Handling Missing Values**: Rows with null values in the `review` column are dropped, as they constitute a small percentage (less than 1%) of the data.
- **Handling Duplicates**: Duplicate rows are identified and removed from the dataset.
- **Text Cleaning Function (`clean`)**:
    - Converts text to lowercase.
    - Removes punctuation.
    - Removes English stopwords using NLTK.
- **Lemmatization and Tokenization with SpaCy (`process_batch`)**:
    - Uses `en_core_web_sm` model from SpaCy.
    - Extracts lemmas of words.
    - Filters out stopwords, punctuation, and whitespace tokens.
    - Joins the lemmatized tokens back into a string.
- A new column `cleaned` is created by applying the `clean` function, and `lemma_text` is created by applying `process_batch`.

### 3. Exploratory Data Analysis (EDA)
- **Frequency Analysis**: Most common words in the `lemma_text` are identified using `nltk.FreqDist`.
- **Word Cloud**: A word cloud is generated from the `cleaned` text to visually represent the most frequent words.
- **Review Length Distribution**: A histogram showing the distribution of tweet lengths (number of words) is plotted.

### 4. Feature Engineering (TF-IDF)
- **TF-IDF Vectorization**: The `lemma_text` column is transformed into numerical features using `TfidfVectorizer`.
    - `max_features` is set to 18000.
    - `ngram_range` is set to (1, 2) to capture unigrams and bigrams.
- `X` represents the TF-IDF features, and `y` represents the `sentiment` labels.

### 5. Model Training and Evaluation

#### A. Logistic Regression
- **Train-Test Split**: The data is split into training and testing sets (80% train, 20% test) using `stratify=y` to maintain sentiment distribution.
- **Model Initialization**: A `LogisticRegression` model is initialized with `max_iter=300`.
- **Training**: The model is trained on `X_train` and `y_train`.
- **Prediction**: Predictions are made on `x_test`.
- **Evaluation**: 
    - **Accuracy**: Calculated using `accuracy_score`.
    - **Classification Report**: Displays precision, recall, f1-score, and support for each sentiment class.
    - **Confusion Matrix**: Visualized using a heatmap to show correct and incorrect predictions for each class.

#### B. RandomForest Classifier
- **Model Initialization**: A `RandomForestClassifier` model is initialized with `n_estimators=100` and `random_state=42`.
- **Training**: The model is trained on `X_train` and `y_train`.
- **Prediction**: Predictions are made on `x_test`.
- **Evaluation**: 
    - **Accuracy**: Calculated using `accuracy_score`.
    - **Classification Report**: Displays precision, recall, f1-score, and support for each sentiment class.
    - **Confusion Matrix**: Visualized using a heatmap.

### 6. Inference on New Text
- Both trained models (Logistic Regression and RandomForest Classifier) are used to predict the sentiment of new, custom sentences.

## Project Report and Model Comparison

### Logistic Regression Model
- **Accuracy**: 0.7659
- **Classification Report Summary**:
    - **Irrelevant**: Precision (0.78), Recall (0.65), F1-score (0.71)
    - **Negative**: Precision (0.79), Recall (0.83), F1-score (0.81)
    - **Neutral**: Precision (0.72), Recall (0.75), F1-score (0.74)
    - **Positive**: Precision (0.76), Recall (0.79), F1-score (0.78)
- The model performed reasonably well, with a balanced performance across classes, although 'Irrelevant' had slightly lower recall.

### RandomForest Classifier Model
- **Accuracy**: 0.8850
- **Classification Report Summary**:
    - **Irrelevant**: Precision (0.94), Recall (0.82), F1-score (0.87)
    - **Negative**: Precision (0.90), Recall (0.91), F1-score (0.91)
    - **Neutral**: Precision (0.91), Recall (0.86), F1-score (0.88)
    - **Positive**: Precision (0.83), Recall (0.92), F1-score (0.87)
- The RandomForest Classifier significantly outperformed the Logistic Regression model in terms of overall accuracy and F1-scores across all classes. It showed strong precision for 'Irrelevant' and 'Neutral' classes and high recall for 'Negative' and 'Positive' sentiments.

### Conclusion
Based on the evaluation metrics, the **RandomForest Classifier** is the superior model for this sentiment analysis task, achieving an accuracy of **0.8850** compared to the Logistic Regression model's **0.7659**. The Random Forest model demonstrates better generalization and predictive power for classifying sentiments in the given Twitter dataset.
