import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

#Task 1
#Iris dataset from sklearn
try:
    iris_data = load_iris()
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target
    df['species'] = df['species'].map(dict(zip(range(3), iris_data.target_names)))
    print("Dataset loaded successfully.")
except Exception as e:
    print("Error loading dataset:", e)

# Display the first 5 rows
df.head()


# Check data types and missing values
print(df.info())
print("\nMissing values:\n", df.isnull().sum())


# Check data types and missing values
print(df.info())
print("\nMissing values:\n", df.isnull().sum())


#Task 2

# Basic statistics
df.describe()

# Group by species and compute the mean of features
grouped = df.groupby('species').mean()
grouped

#Task 3

#Line Chart
plt.figure(figsize=(10, 5))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.plot(subset.index, subset['petal length (cm)'], label=species)
plt.title("Petal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

#Bar Chart
plt.figure(figsize=(8, 5))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'], palette="viridis")
plt.title("Average Petal Length per Species")
plt.ylabel("Petal Length (cm)")
plt.xlabel("Species")
plt.show()


#Histogram

plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.show()

#Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette="deep")
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

