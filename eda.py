import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Cleaned Dataset
df = pd.read_csv('cleaned_student_data_full_139.csv')

# Style setting for charts
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("--- 1. BASIC DATA OVERVIEW ---")
print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nNumerical Summary:\n", df.describe())

# 2. UNIVARIATE ANALYSIS

# Chart A: Branch Distribution
plt.figure(figsize=(10, 5))
sns.countplot(y='Branch', data=df, order=df['Branch'].value_counts().index, palette='Blues_r')
plt.title('Distribution of Students by Branch')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('eda_branch_distribution.png')
plt.close()

# Chart B: CGPA Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['CGPA'].dropna(), kde=True, color='purple', bins=15)
plt.title('CGPA Distribution')
plt.xlabel('CGPA')
plt.tight_layout()
plt.savefig('eda_cgpa_distribution.png')
plt.close()

# 3. BIVARIATE ANALYSIS

# Chart C: Average CP Rating by Branch
plt.figure(figsize=(10, 5))
branch_cp = df.groupby('Branch')['CP Rating'].mean().sort_values(ascending=False).reset_index()
sns.barplot(x='CP Rating', y='Branch', data=branch_cp, palette='Greens_r')
plt.title('Average CP Rating by Branch')
plt.xlabel('Average CP Rating')
plt.tight_layout()
plt.savefig('eda_avg_cp_rating_by_branch.png')
plt.close()

# Chart D: Correlation Heatmap
plt.figure(figsize=(8, 6))
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png')
plt.close()

print("\nEDA Completed! All charts saved in your folder as PNG images.")
