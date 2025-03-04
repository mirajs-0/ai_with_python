import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Reading bank.csv using pandas
data = pd.read_csv('bank.csv', delimiter=';')

# 1. Inspecting dataframe's column names and variable types
print(data.info())
print(data.head())

# 2. Selecting specific columns to create a second dataframe
df2 = data[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
# print(df2['y'].value_counts())
# print(df2.head())

# 3. Converting categorical variables to dummy numerical values
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])
print("\nDataframe with Dummy Variables")
print(df3.head())

# Converting target variable 'y' to numeric values
df3['y'] = df3['y'].map({'yes': 1, 'no': 0})

# 4. Producing heatmap of correlation coefficients for all variables
plt.figure(figsize=(15, 10))
sns.heatmap(df3.corr().round(2), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of df3 Variables')
plt.show()

"""
The heatmap shows the correlation coefficients between variables in df3. Correlation values range from -1 to 1.

Observations:
The target variable 'y' has low correlation values with most features, indicating weak linear relationships.
Some dummy variables derived from the same original categorical feature exhibit negative correlations with each other.
"""

# 5. Selecting column 'y' as target variable and other remaining columns as explanatory variables 'X'

x = df3.drop(columns=['y'])
y = df3['y']

# 6. Splitting dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)

# 7. Setting up a Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# 8. Confusion Matrix and Accuracy Score for Logistic Regression
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"Confution matrix (logistic regression): \n{conf_matrix}")
print(f"Accuracy Score (logistic regression): \n{accuracy}")

# 9. K-Nearest Neighbors Model with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)

#Evaluating the KNN Model
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Confution matrix (KNN): \n{conf_matrix_knn}")
print(f"Accuracy Score (KNN): \n{accuracy_knn}")

"""
Comparing the results:
- Accuracy score of logistic regression is 0.8974358974358975 and that of knn is 0.8735632183908046. It means the logistic regression is more accurate.

- The logistic regression has less false positives (9) compared to that of knn (28). It means that logistic regression makes fewer incorrect predications of "Yes" when the actual value is "No".

- The false negatives in also lower in case of logistic regression (107) compared to that of knn (115). It means the logistic regression makes fewer incorrect predictions of "No" when the actual value is "Yes".
"""