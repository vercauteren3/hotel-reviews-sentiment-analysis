# Hotel Reviews Sentiment — Text ML Pipeline

End-to-end machine learning pipeline predicting hotel review sentiment (positive or negative) using Booking.com review data.

---

## Project Overview
This project builds a text-based classification system to identify whether hotel reviews are positive or negative.  
It includes data cleaning, preprocessing, feature engineering, model training (Logistic Regression, Naïve Bayes, Random Forest), and model interpretation (feature importance and partial dependence plots).

The goal is to demonstrate a reproducible text analytics pipeline using Python and scikit-learn.

---

## Model Performance

| Model               | Accuracy | Balanced Accuracy |
|----------------------|-----------|------------------------|
| Logistic Regression  | 0.79      | 0.79                   |
| Naïve Bayes          | 0.77      | 0.78                  |
| Random Forest        | 0.78      | 0.77                  |

## Key Features
	•	Text preprocessing (regex cleaning, lemmatization, stopword removal)
	•	Sparse feature extraction via CountVectorizer
	•	Feature selection using Chi-square & Mutual Information
	•	Multiple models: Logistic Regression, Naïve Bayes, Random Forest
	•	Explainability: feature importance (MDI) and Partial Dependence Plots
	•	Fully modular and reproducible design

## Summary

This project applies machine learning to predict hotel review sentiment using text data from over 500,000 Booking.com reviews of European luxury hotels. The goal was to determine whether a review is positive or negative based solely on its textual content and to identify the main factors influencing customer satisfaction.

After cleaning and preprocessing the text (removing noise, lemmatizing, and combining positive and negative reviews), three models were trained and compared: Logistic Regression, Multinomial Naïve Bayes, and Random Forest. Feature selection was performed using Information Gain and Chi-Square statistics to retain the 500 most informative terms.

Results showed that all models performed similarly, each achieving around 78–80% balanced accuracy.  
Key sentiment-driving words included great, excellent, and poor, as well as hotel-specific terms like room, staff, and reception.  

The Partial Dependence Plots indicate that reviews mentioning room, reception, bed, or manager more frequently are associated with lower predicted sentiment, suggesting that these topics often appear in negative reviews (e.g., complaints about comfort, cleanliness, or service quality). In contrast, mentions of breakfast and staff correspond to slightly higher predicted sentiment, implying that guests who emphasize these aspects tend to leave more positive feedback.  

Overall, the analysis highlights that operational features, particularly those related to room quality and front-desk service, are key drivers of customer dissatisfaction, while food and hospitality aspects contribute positively to guest sentiment.

## Author
**Lucas Vercauteren** 
MSc Finance with Data Science student — University College London (2025–26)
l.j.h.vercauteren@gmail.com
