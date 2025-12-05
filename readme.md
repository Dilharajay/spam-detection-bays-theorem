# Naive Bayes Spam Classifier

This project implements a Naive Bayes classifier from scratch to distinguish between spam and legitimate (ham) SMS messages. The entire implementation and analysis are contained within the `guest.ipynb` Jupyter notebook.

## Methodology

The classifier is built using a probabilistic approach based on Bayes' theorem. The core steps are:

1.  **Data Loading and Preprocessing**: The `SMSSpamCollection` dataset is loaded, which contains SMS messages labeled as either 'spam' or 'ham'. The text is preprocessed by converting it to lowercase and removing common English stop words to reduce noise.

2.  **Feature Extraction**: The preprocessed text is converted into numerical features using the `CountVectorizer` from scikit-learn. This creates a vocabulary of unique words and represents each message as a vector of word counts.

3.  **Training the Naive Bayes Model**:
    *   **Class Priors**: The prior probabilities of a message being spam (`P(Spam)`) or ham (`P(Ham)`) are calculated based on their proportions in the training dataset.
    *   **Conditional Probabilities**: For each word in the vocabulary, the conditional probabilities `P(word|Spam)` (spamicity) and `P(word|Ham)` (hamicity) are calculated. These represent the likelihood of a word appearing in a spam or ham message, respectively. **Laplace (add-1) smoothing** is applied to handle words that might not appear in the training set and avoid zero probabilities.

4.  **Prediction**: To classify a new message, the model calculates the posterior probability of it being spam and ham. This is done by combining the prior probabilities with the conditional probabilities of all the words in the message. The class with the higher posterior probability is chosen as the prediction. Log probabilities are used during calculation to prevent numerical underflow with very small probability values.

## Results

The model was trained on 70% of the dataset and evaluated on the remaining 30%. The performance on the test set is as follows:

*   **Accuracy**: 97.43%
*   **Confusion Matrix**:
    *   True Positives (Spam correctly identified): 190
    *   True Negatives (Ham correctly identified): 1444
    *   False Positives (Ham mistaken for Spam): 31
    *   False Negatives (Spam mistaken for Ham): 12
*   **Spam Metrics**:
    *   **Precision**: 85.97% (Of all messages predicted as spam, ~86% were actually spam)
    *   **Recall**: 94.06% (The model correctly identified ~94% of all actual spam messages)
    *   **F1-Score**: 89.83%

The high recall indicates that the model is very effective at catching most spam messages, while the good precision shows that it doesn't incorrectly flag too many legitimate messages as spam.

### Performance on Custom Emails

The classifier was also tested on a few custom examples:
*   `'renew your password'`: Predicted as **SPAM** (Spam Confidence: 99.98%)
*   `'renew your vows'`: Predicted as **HAM** (Spam Confidence: 0.01%)
*   `'benefits of our account'`: Predicted as **HAM** (Spam Confidence: 0.04%)
*   `'the importance of physical activity'`: Predicted as **HAM** (Spam Confidence: 0.00%)

These results demonstrate the model's ability to discern context and make reasonable predictions on unseen data.

## Conclusion

This from-scratch implementation of a Naive Bayes classifier demonstrates a powerful and effective approach for spam detection. With an accuracy of over 97% and a high F1-score for the spam class, the model proves to be reliable. The project highlights the fundamentals of text classification, including preprocessing, feature extraction, and probabilistic modeling.
