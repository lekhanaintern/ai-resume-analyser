import pandas as pd
import os

print("🧹 Starting dataset cleaning...\n")

# Load the raw dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'resumes.csv')
df = pd.read_csv(csv_path)

print(f"Original shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()[:10]}...\n")

# Step 1: Keep only the columns we need
# Based on the output, we need 'Resume_str' and 'Category'
df_clean = df[['Resume_str', 'Category']].copy()

print(f"After selecting columns: {df_clean.shape}")

# Step 2: Remove rows with missing values
df_clean = df_clean.dropna()

print(f"After removing missing values: {df_clean.shape}")

# Step 3: Remove any duplicate resumes
df_clean = df_clean.drop_duplicates(subset=['Resume_str'])

print(f"After removing duplicates: {df_clean.shape}")

# Step 4: Clean the Category column (remove extra whitespace)
df_clean['Category'] = df_clean['Category'].str.strip()

# Step 5: Rename columns for clarity
df_clean.columns = ['Resume', 'Category']

# Step 6: Check category distribution
print("\n📊 Clean Category Distribution:")
print(df_clean['Category'].value_counts())

print(f"\n✅ Unique Categories: {df_clean['Category'].nunique()}")
print(f"✅ Total Clean Resumes: {len(df_clean)}")

# Step 7: Remove categories with very few samples (less than 20)
category_counts = df_clean['Category'].value_counts()
valid_categories = category_counts[category_counts >= 20].index
df_clean = df_clean[df_clean['Category'].isin(valid_categories)]

print(f"\n✅ After removing rare categories: {len(df_clean)} resumes")
print(f"✅ Final categories: {df_clean['Category'].nunique()}")

# Step 8: Save cleaned dataset
clean_csv_path = os.path.join(current_dir, 'resumes_clean.csv')
df_clean.to_csv(clean_csv_path, index=False)

print(f"\n💾 Cleaned dataset saved to: {clean_csv_path}")

# Show sample
print("\n📄 Sample resume (first 200 characters):")
print(df_clean.iloc[0]['Resume'][:200])
print(f"\nCategory: {df_clean.iloc[0]['Category']}")