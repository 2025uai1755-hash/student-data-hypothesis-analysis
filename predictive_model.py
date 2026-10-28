import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Dataset Load Karein
df = pd.read_csv('cleaned_student_data_full_139.csv')

# 2. Target Variable Create Karein (1 = Placed / Offers > 0, 0 = Not Placed)
df['Placed'] = (df['Offers Received'] > 0).astype(int)

# 3. Features & Target Define Karein
features = [
    'CGPA', 
    'Academic Workload Rating (1-5)', 
    'CP Rating', 
    'Total Problems Solved', 
    'Placement Motivation Rating (1-5)', 
    'Peer Culture Influence Rating (1-5)'
]

X = df[features]
y = df['Placed']

# 4. Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Machine Learning Model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = model.predict(X_test)
print("=== Predictive Model Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Feature Importance Plot (Sabse zaroori factors konsi hain)
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='teal')
plt.title('Feature Importance for Student Placement Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Graph saved as 'feature_importance.png'")
