import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')



# Load the dataset
file_path = 'movies_data.csv'
movies_data = pd.read_csv(file_path, encoding='ISO-8859-1')

print(movies_data.columns)

# Display the first few rows and summary information of the dataset
movies_data_info = movies_data.info()
movies_data_head = movies_data.head()

movies_data_info, movies_data_head

# Create binary target for nominations (1 if nominated, 0 if not)
movies_data['Nominated'] = movies_data['Oscar and Golden Globes nominations'].apply(lambda x : 1 if x > 0 else 0)

# Create binary target for high IMDb score (1 if score >= 7.0, 0 if score < 7.0)
movies_data['High_IMDb_Score'] = movies_data['IMDb score'].apply(lambda x : 1 if x>= 7.0 else 0)

# Dropping unnecessary columns

movies_cleaned = movies_data.drop(columns = ['Movie', 'Actor 1', 'Actor 2' , 'Actor 3', 'Oscar and Golden Globes nominations', 'Oscar and Golden Globes awards' , 'IMDb score' ])

# Handling categorical columns: One-Hot Encoding for 'Genre' and 'Director'

movies_cleaned = pd.get_dummies(movies_cleaned, columns=['Genre', 'Director'] , drop_first=True)

# Define features (all columns except target) and targets (Nominated and High_IMDb_Score)

X = movies_cleaned.drop(columns=['Nominated','High_IMDb_Score'])
y_nominated = movies_cleaned['Nominated']
y_imdb = movies_cleaned['High_IMDb_Score']

X_train_nom, X_test_nom, y_train_nom, y_test_nom = train_test_split (X, y_nominated, test_size=0.3, random_state=42 )

X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb = train_test_split(X, y_imdb, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train_nom = scaler.fit_transform(X_train_nom)
X_test_nom = scaler.fit_transform(X_test_nom)
X_train_imdb = scaler.fit_transform(X_train_imdb)
X_test_imdb = scaler.fit_transform(X_test_imdb)

# Initialize models
log_reg = LogisticRegression(max_iter=1000)
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)

# Train and evaluate models for predicting nominations
log_reg.fit(X_train_nom, y_train_nom)
dec_tree_nom = decision_tree.fit(X_train_nom, y_train_nom)
rand_forest_nom = random_forest.fit(X_train_nom, y_train_nom)

# Train and evaluate models for predicting IMDb score
log_reg.fit(X_train_imdb, y_train_imdb)
dec_tree_imdb = decision_tree.fit(X_train_imdb, y_train_imdb)
rand_forest_imdb = random_forest.fit(X_train_imdb, y_train_imdb)


# Predict nominations
y_pred_log_nom = log_reg.predict(X_test_nom)
y_pred_tree_nom = dec_tree_nom.predict(X_test_nom)
y_pred_forest_nom = rand_forest_nom.predict(X_test_nom)

# Predict IMDb scores
y_pred_log_imdb = log_reg.predict(X_test_imdb)
y_pred_tree_imdb = dec_tree_imdb.predict(X_test_imdb)
y_pred_forest_imdb = rand_forest_imdb.predict(X_test_imdb)



# Generate classification reports and accuracy for both tasks
nominations_report = {
    'Logistic Regression': classification_report(y_test_nom, y_pred_log_nom, output_dict=True),
    'Decision Tree': classification_report(y_test_nom, y_pred_tree_nom, output_dict=True),
    'Random Forest': classification_report(y_test_nom, y_pred_forest_nom, output_dict=True)
}

imdb_report = {
    'Logistic Regression': classification_report(y_test_imdb, y_pred_log_imdb, output_dict=True),
    'Decision Tree': classification_report(y_test_imdb, y_pred_tree_imdb, output_dict=True),
    'Random Forest': classification_report(y_test_imdb, y_pred_forest_imdb, output_dict=True)
}

# You should use the dictionaries you created directly:
reports = {
    "Nominations - Logistic Regression": nominations_report['Logistic Regression'],
    "Nominations - Decision Tree": nominations_report['Decision Tree'],
    "Nominations - Random Forest": nominations_report['Random Forest'],
    "IMDb Score - Logistic Regression": imdb_report['Logistic Regression'],
    "IMDb Score - Decision Tree": imdb_report['Decision Tree'],
    "IMDb Score - Random Forest": imdb_report['Random Forest']
}

# Display the reports
# Instead of just returning the reports, print each one for better readability

# Print classification reports for nominations prediction
print("Nominations - Logistic Regression:")
print(classification_report(y_test_nom, y_pred_log_nom))

print("\nNominations - Decision Tree:")
print(classification_report(y_test_nom, y_pred_tree_nom))

print("\nNominations - Random Forest:")
print(classification_report(y_test_nom, y_pred_forest_nom))

# Print classification reports for IMDb score prediction
print("\nIMDb Score - Logistic Regression:")
print(classification_report(y_test_imdb, y_pred_log_imdb))

print("\nIMDb Score - Decision Tree:")
print(classification_report(y_test_imdb, y_pred_tree_imdb))

print("\nIMDb Score - Random Forest:")
print(classification_report(y_test_imdb, y_pred_forest_imdb))

# Plot confusion matrix for Random Forest on nominations
cf_matrix_nom = confusion_matrix(y_test_nom, y_pred_forest_nom)
plt.figure(figsize=(10, 7))
sns.heatmap(cf_matrix_nom, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest (Nominations)")
plt.savefig('confusion_matrix_nominations.png')  # Save plot
# plt.show() is removed

# Plot confusion matrix for Random Forest on IMDb score
cf_matrix_imdb = confusion_matrix(y_test_imdb, y_pred_forest_imdb)
plt.figure(figsize=(10, 7))
sns.heatmap(cf_matrix_imdb, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest (IMDb Score)")
plt.savefig('confusion_matrix_imdb.png')  # Save plot
# plt.show() is removed

#python moviesml.py