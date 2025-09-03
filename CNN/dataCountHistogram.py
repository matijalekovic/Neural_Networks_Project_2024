import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("C:\\Users\\jovan\\PycharmProjects\\neuralne2zadatak\\train.csv")

features = ["Id"]
X = df[features].values  # Numeric feature vectors

y_raw = df["label"].values  # Raw class labels (strings)

# Encode string labels into integers - neural networks work with numeric labels for classification
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_  # Save class names for interpretation later

print("Classes found:", class_names)
print("Samples per class in full dataset:")
print(pd.Series(y).value_counts().sort_index())  # Check class distribution - important to understand if data is balanced


plt.figure(figsize=(6,4))
sns.countplot(x=y_raw, order=class_names)
plt.title("Distribution of Samples by Class (Full Dataset)")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.show()