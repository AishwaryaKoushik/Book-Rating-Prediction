# Book-Rating-Prediction

A model is implemented to predict the book rating based on the dataset containing the reviews that the book has received and other relevant information. The main goal is to try different models and parameters to train the dataset. The classification problem is addressed and weighted F1 score is the metric which indicates the performance of the model. The heart of the model lies within Natural Language Processing (NLP) and its main techniques. The project flow starts with preprocessing the dataset to include the important features and analyzing these features in order to enhance the model’s performance and accuracy. NLP techniques are exploited throughout the project to comprehend the book rating based on the reviews.

For ease and simplicity, the project is broken down into 4 sections -
  1. Pre-Processing
  2. Term Frequency - Inverse Document Frequency (TF-IDF) Representation
  3. Standardization
  4. Classification
  


Two models have been used to perform classification - 
  1. Random Forest Classifier
     Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset. Instead of relying on one       decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.


     <img width="304" alt="image" src="https://github.com/AishwaryaKoushik/Book-Rating-Prediction/assets/161193220/4ab1956f-3731-425a-a4d2-29ad15efac65">

   


  2. Logistic Regression
     A statistical approach and a Machine Learning algorithm that is used for classification problems and is based on the concept of probability. Logistics regression uses the sigmoid function to return the probability of a label. The different types are - Binomial and Multinomial


     
  <img width="433" alt="image" src="https://github.com/AishwaryaKoushik/Book-Rating-Prediction/assets/161193220/800310d9-4fce-4c3c-ad1d-c86516d33f0b">



The two models are tested for their performance and accuracy. It is observes that Logistic Regression outsmarts Random Forest classifier in terms of accuracy.


  <img width="903" alt="RESULTD" src="https://github.com/AishwaryaKoushik/Book-Rating-Prediction/assets/161193220/0d1f26a1-436b-4e6c-8189-0d9262ceb0b8">


     
