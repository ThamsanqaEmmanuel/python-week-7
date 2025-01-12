import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
try:
    # Load the dataset
    df = pd.read_csv(url, header=None, names=column_names)
    print("Dataset loaded successfully.")
    
    # Display the first few rows
    print("First few rows of the dataset:")
    print(df.head())

    # Explore the structure of the dataset
    print("\nDataset Info:")
    print(df.info())

    # Check for missing values
    print("\nMissing Values in the dataset:")
    print(df.isnull().sum())

    # Clean the dataset by filling missing values or dropping them (if any)
    df = df.dropna()  # Dropping rows with missing values (if any)

except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
# Basic statistics of the numerical columns
print("\nBasic Statistics of the dataset:")
print(df.describe())

# Perform grouping based on the categorical column 'species' and compute mean of numerical columns
print("\nGroup by 'species' and calculate mean of numerical columns:")
print(df.groupby('species').mean())

# Identify any patterns or interesting findings
print("\nInteresting Pattern - Average Petal Length per Species:")
print(df.groupby('species')['petal_length'].mean())

# Task 3: Data Visualization
# Set the plot style
sns.set(style="whitegrid")

# Line Chart (e.g., trends in petal length over time or observations)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x=df.index, y="petal_length", hue="species", marker='o')
plt.title("Petal Length Over Observations")
plt.xlabel("Observation Index")
plt.ylabel("Petal Length")
plt.legend(title="Species")
plt.show()

# Bar Chart (e.g., Average petal length per species)
plt.figure(figsize=(8, 6))
sns.barplot(x='species', y='petal_length', data=df, palette='viridis')
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length")
plt.show()

# Histogram (e.g., Distribution of sepal length)
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal_length'], kde=True, bins=20, color='blue')
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot (e.g., Relationship between sepal length and petal length)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal_length', y='petal_length', data=df, hue='species', style='species', palette='Set1')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend(title="Species")
plt.show()
