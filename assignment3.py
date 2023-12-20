#sp20-bse-017
#ids
#20/12/2023
#
# Importing necessary libraries and modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Uploading the "gender-prediction.csv" file in Google Colab
from google.colab import files
uploaded = files.upload()

# Reading the CSV file into a DataFrame
import io
df = pd.read_csv(io.StringIO(uploaded['gender-prediction.csv'].decode('utf-8')))

# Displaying the first few rows of the dataset
print(df.head())

# Separating features (X) and the target variable (y)
X = df.drop('gender', axis=1)
y = df['gender']

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Making predictions
y_pred = logreg.predict(X_test_scaled)

# Evaluating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy)

# Task: Identifying two powerful attributes for gender prediction and explaining why
# Assuming 'Height' and 'Weight' are influential attributes due to significant differences between males and females.

# Task: Excluding 'Height' and 'Weight', rerunning the experiment with 80/20 train/test split, and checking for changes in results
X_excluded = X.drop(['Height', 'Weight'], axis=1)
X_train_excluded, X_test_excluded, y_train_excluded, y_test_excluded = train_test_split(X_excluded, y, test_size=0.2, random_state=42)

# Applying Logistic Regression, Support Vector Machines, and Multilayer Perceptron without the excluded attributes
lr_model_excluded = LogisticRegression()
lr_model_excluded.fit(X_train_excluded, y_train_excluded)
lr_predictions_excluded = lr_model_excluded.predict(X_test_excluded)
lr_accuracy_excluded = accuracy_score(y_test_excluded, lr_predictions_excluded)

svm_model_excluded = SVC()
svm_model_excluded.fit(X_train_excluded, y_train_excluded)
svm_predictions_excluded = svm_model_excluded.predict(X_test_excluded)
svm_accuracy_excluded = accuracy_score(y_test_excluded, svm_predictions_excluded)

mlp_model_excluded = MLPClassifier()
mlp_model_excluded.fit(X_train_excluded, y_train_excluded)
mlp_predictions_excluded = mlp_model_excluded.predict(X_test_excluded)
mlp_accuracy_excluded = accuracy_score(y_test_excluded, mlp_predictions_excluded)

# Printing results for the excluded attributes
print("\nTask 6: Results after excluding 'Height' and 'Weight':")
print("Logistic Regression Accuracy:", lr_accuracy_excluded)
print("Support Vector Machines Accuracy:", svm_accuracy_excluded)
print("Multilayer Perceptron Accuracy:", mlp_accuracy_excluded)

# Printing the number of instances incorrectly classified for each model without the excluded attributes
incorrect_lr_excluded = (y_test_excluded != lr_predictions_excluded).sum()
incorrect_svm_excluded = (y_test_excluded != svm_predictions_excluded).sum()
incorrect_mlp_excluded = (y_test_excluded != mlp_predictions_excluded).sum()

print("\nTask 6: Number of instances incorrectly classified without 'Height' and 'Weight':")
print("Logistic Regression:", incorrect_lr_excluded)
print("Support Vector Machines:", incorrect_svm_excluded)
print("Multilayer Perceptron:", incorrect_mlp_excluded)

#question 3
# Importing necessary libraries and modules
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeavePOut
from sklearn.metrics import f1_score

# Loading the dataset
data = pd.read_csv('your_dataset.csv')

# Assuming 'X' contains the features and 'y' contains the target variable
X = data.drop('gender', axis=1)
y = data['gender']

# Initializing Random Forest Classifier with chosen parameter values
random_forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Defining the number of iterations for Monte Carlo cross-validation
num_iterations = 5

# Monte Carlo cross-validation
mc_f1_scores = cross_val_score(random_forest, X, y, cv=num_iterations, scoring='f1_macro')

print(f"Monte Carlo Cross-Validation F1 Scores: {mc_f1_scores}")
print(f"Average F1 Score for Monte Carlo Cross-Validation: {mc_f1_scores.mean()}")

# Defining the number of samples to leave out for Leave P-Out cross-validation
leave_p_out = 2

# Leave P-Out cross-validation
lp_f1_scores = []
leave_p_out_cv = LeavePOut(leave_p_out)
for train_index, test_index in leave_p_out_cv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    lp_f1_scores.append(f1)

print(f"Leave P-Out Cross-Validation F1 Scores: {lp_f1_scores}")
print(f"Average F1 Score for Leave P-Out Cross-Validation: {sum(lp_f1_scores) / len(lp_f1_scores)}")



#question 4

# Importing necessary libraries and modules
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Concatenating the existing dataset and the 10 new test instances
merged_data = pd.concat([data, new_test_data], ignore_index=True)

# Assuming 'X' contains the features and 'y' contains the target variable for the merged dataset
X_merged = merged_data.drop('gender', axis=1)
y_merged = merged_data['gender']

# Separating the merged dataset into training and testing sets
# Using the first len(data) instances for training and the remaining 10 instances for testing
X_train = X_merged.iloc[:len(data)]
X_test = X_merged.iloc[len(data):]
y_train = y_merged.iloc[:len(data)]
y_test = y_merged.iloc[len(data):]

# Training Gaussian Naïve Bayes classifier using all instances from the gender prediction dataset
gnb = GaussianNB()