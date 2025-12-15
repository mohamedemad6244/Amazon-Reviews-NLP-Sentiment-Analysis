# Amazon Reviews NLP Sentiment Analysis

A **Natural Language Processing (NLP) and Machine Learning project** that analyzes Amazon customer reviews to extract sentiment, generate insights, and predict product ratings. The project focuses on transforming large volumes of unstructured review data into actionable business intelligence.

---

## Project Overview

In large-scale e-commerce platforms such as Amazon, customer reviews play a critical role in product evaluation and business decision-making. However, the massive volume of reviews makes manual analysis impractical.

This project applies **NLP techniques and supervised machine learning models** to analyze review text, understand customer sentiment, and predict star ratings, enabling data-driven decisions for product improvement, marketing, and customer experience optimization.

---

## Business Case

### Challenges

* Missed insights hidden in textual reviews
* Uncertain or inaccurate product recommendations
* Underutilized customer feedback

### Value of the Solution

* Improve customer satisfaction and experience
* Enhance product quality using sentiment-driven insights
* Optimize business resources
* Support marketing and sales strategies

**ROI:** Automating review analysis enables faster, more accurate decision-making and improves brand reputation and customer retention.

---

## Objectives

* Analyze customer sentiment from reviews
* Improve product recommendations
* Predict review star ratings
* Detect anomalies and potential fake reviews
* Identify sentiment trends over time

---

## Dataset Overview

The dataset consists of Amazon customer reviews containing:

* Review title
* Review text
* Recommendation indicator
* Star rating (target variable)

Initial exploratory analysis was performed to understand data structure, rating distribution, and sentiment balance.

---

## Analysis & Insights

* Identified top-rated products
* Analyzed rating distributions
* Sentiment distribution:

  * Positive: ~93%
  * Negative: ~7%
* Extracted most frequently used words across reviews

---

## Methodology

### Feature Selection

The following features were used for model training:

* Review Title
* Review Text
* Review Recommendation
* Target: Star Rating

### Data Preprocessing

* Handled missing and null values in recommendation and rating fields
* Removed rows with excessive unknown data
* Cleaned and standardized text data

### NLP Pipeline

1. Remove non-alphabetic characters
2. Convert text to lowercase
3. Tokenization
4. Stop-word removal
5. Stemming
6. Vectorization using CountVectorizer
7. Model training

---

## Models Used

* **XGBoost Classifier**
* **Multinomial Naive Bayes**

Both models were trained and evaluated to compare performance in predicting review star ratings.

---

## Model Evaluation

* Training and testing accuracy were measured
* Predictions were validated using sample review examples
* Comparative evaluation was conducted between XGBoost and Naive Bayes

---

## Tools & Technologies

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* NLP libraries (NLTK)
* Jupyter Notebook

---

## Model Performance & Accuracy

* The **XGBoost Classifier** achieved the highest performance, reaching a training accuracy of approximately **82%**, making it the primary model for sentiment-based rating prediction.
* The **Naive Bayes model** provided faster training and inference, serving as a strong baseline for comparison.
* Model performance was evaluated using a **train/test split** and accuracy metrics, with qualitative validation through sample review predictions.

The results demonstrate the effectiveness of combining NLP preprocessing with ensemble learning techniques for large-scale sentiment analysis tasks.

---

## Conclusion

This project demonstrates how NLP and machine learning can be leveraged to convert unstructured customer reviews into meaningful insights. The solution provides scalable sentiment analysis and rating prediction capabilities suitable for real-world e-commerce applications.
