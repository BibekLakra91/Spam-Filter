# Machine Learning Model for Spam Email Classification

## Overview

This script implements a machine learning model for spam email classification using the Linear Support Vector Classification (LinearSVC) algorithm. The task involves distinguishing between 'spam' and 'ham' (non-spam) messages based on textual content.

## Usage

1. **Data Loading:**
   - The script assumes the presence of a CSV file named 'mail_data.csv' containing the relevant data for training and testing.
   - Ensure that the file is located in the correct path or provide the correct path if needed.

2. **Data Preprocessing:**
   - Missing values in the dataset are handled by replacing them with empty strings.
   - The 'Category' column is converted to numerical values: 0 for 'spam' and 1 for 'ham'.

3. **Train-Test Split:**
   - The data is split into training and testing sets using an 80-20 split.
   ```python
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
   ```

4. **Text Vectorization:**
   - TF-IDF vectorization is applied to convert text data into a numerical format.
   ```python
   feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase='True')
   X_train_features = feature_extraction.fit_transform(X_train)
   X_test_features = feature_extraction.transform(X_test)
   ```

5. **Model Training:**
   - The Linear Support Vector Classification (SVC) model is initialized with hinge loss and trained on the training set.
   ```python
   clf = LinearSVC(loss="hinge")
   clf.fit(X_train_features, Y_train)
   ```

6. **Prediction and Accuracy:**
   - The model predicts labels on the test set, and accuracy is calculated.
   ```python
   clf_prediction_on_test_data = clf.predict(X_test_features)
   clf_accuracy_on_test_data = accuracy_score(Y_test, clf_prediction_on_test_data)
   ```

7. **Results:**
   - The achieved accuracy on the test data is 98.2%.



---