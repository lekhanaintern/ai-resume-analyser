import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the correct path
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'resumes.csv')

print(f"Looking for dataset at: {csv_path}")

# Check if file exists
if not os.path.exists(csv_path):
    print("\n❌ ERROR: resumes.csv not found!")
    print(f"Please place the dataset file at: {csv_path}")
    print("\nOptions:")
    print("1. Download from Kaggle: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset")
    print("2. Or I can help you create a sample dataset for testing")
    exit()

# Load dataset
df = pd.read_csv(csv_path)

# Basic info
print("\n✅ Dataset loaded successfully!")
print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head(2))

# Check categories
print("\n📊 Job Categories:")
print(df['Category'].value_counts())

# Check for missing values
print("\n🔍 Missing Values:")
print(df.isnull().sum())

# Category distribution
plt.figure(figsize=(12, 6))
df['Category'].value_counts().plot(kind='bar')
plt.title('Distribution of Job Categories')
plt.xlabel('Job Role')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
chart_path = os.path.join(current_dir, 'category_distribution.png')
plt.savefig(chart_path)
print(f"\n📈 Chart saved as: {chart_path}")