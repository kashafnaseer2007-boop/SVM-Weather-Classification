import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from datetime import datetime

# Set the visual style for all matplotlib plots
# 'seaborn-v0_8-darkgrid' gives a clean grid background with subtle colors
plt.style.use('seaborn-v0_8-darkgrid')

# Set the color palette for seaborn plots
# "husl" = Hue-Saturation-Lightness - creates vibrant, distinct colors
sns.set_palette("husl")
df = pd.read_csv("/kaggle/input/weather-type-classification/weather_classification_data.csv")
df
print("Dataset shape:", df.shape)
display(df.head())
print(" Total rows:", len(df))
display(df.tail(3))

print(df['Weather Type'].value_counts())
print(df.dtypes)
print(df.describe())
missing = df.isnull().sum()  
print(missing)

if missing.sum() == 0:
    print("\n All good - no missing data!")
else:
    print(f"\n Total missing values: {missing.sum()}")
# numbers-based features
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

num_features = ['Temperature', 'Humidity', 'Wind Speed', 
                'Precipitation (%)', 'Atmospheric Pressure', 
                'UV Index', 'Visibility (km)']

for i, feature in enumerate(num_features):
    ax = axes[i]
    df[feature].hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
    # ax=ax means "draw this histogram on THIS specific subplot"
    ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

for i in range(len(num_features), 9):
    axes[i].axis('off')

plt.tight_layout()
plt.show()
# text-based features
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

cat_features = ['Cloud Cover', 'Season', 'Location']

for i, feature in enumerate(cat_features):
    ax = axes[i]
    counts = df[feature].value_counts()
    print(f"\n{counts}")

    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("husl", len(counts)), edgecolor='black')
    
    ax.set_title(f'{feature}', fontsize=14, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
   
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
counts.index
counts.values
plt.figure(figsize=(8, 5))

weather_counts = df['Weather Type'].value_counts()
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']  # Blue, Green, Orange, Red

bars = plt.bar(weather_counts.index, weather_counts.values, color=colors, edgecolor='black', linewidth=2)

plt.title('Weather Type Distribution', fontsize=16, fontweight='bold', y=1.02)
plt.xlabel('Weather Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', alpha=0.8)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 20,
             f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()
X = df.drop('Weather Type', axis=1)  # Everything EXCEPT weather type
y = df['Weather Type']               # JUST weather type (what we predict)

print("Feature shape (X):", X.shape)
print("Target shape (y):", y.shape)
print("\n Feature columns:")
print(X.columns.tolist())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing, 80% for training
    random_state=42,    # Makes split reproducible (same every time)
    stratify=y          # Keeps same class proportions in both sets
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")
print(f"\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))  # normalize=True -> Shows percentages, not counts
# Numbers → Need scaling (SVM is sensitive to feature magnitudes)
# Text → Need encoding (SVM can't read words, needs numbers)
num_features = ['Temperature', 'Humidity', 'Wind Speed', 
                'Precipitation (%)', 'Atmospheric Pressure', 
                'UV Index', 'Visibility (km)']

cat_features = ['Cloud Cover', 'Season', 'Location']

print("Numerical features:", num_features)
print("Categorical features:", cat_features)
print(f"Total features: {len(num_features) + len(cat_features)}")
print(f"Check: Should match original {X.shape[1]} columns")
preprocessor = ColumnTransformer(                   # Structure: (name, transformer, columns)
    transformers=[
        ('num', StandardScaler(), num_features),    # Scale numerical features
        ('cat', OneHotEncoder(drop='first'), cat_features)  # Encode categorical
    ])

print("Preprocessor created!")
print(f"Steps: 1) Scale {len(num_features)} numerical features")  # [0, 0] = Spring, [1, 0] = Summer, [0, 1] = Autumn
print(f"       2) One-hot encode {len(cat_features)} categorical features")
X_train_processed = preprocessor.fit_transform(X_train)

print("Training data processed!")
print(f"Original shape: {X_train.shape}")
print(f"Processed shape: {X_train_processed.shape}")
print("\nWhat happened:")
print("- Numerical features: Scaled (mean=0, std=1)")
print("- Categorical features: One-hot encoded")
print(f"- Total features now: {X_train_processed.shape[1]}")
X_test_processed = preprocessor.transform(X_test)  # NOTICE: only transform, not fit!

print("Test data processed!")
print(f"Test shape: {X_test_processed.shape}")
print("\nImportant: Used same parameters from training")
print("- Same mean/std for scaling")
print("- Same categories for encoding")
print("- No refitting on test data!")
svm_model = SVC(
    kernel='rbf',          # Radial Basis Function - handles complex patterns
    C=1.0,                 # Regularization parameter (start moderate)
    gamma='scale',         # Kernel coefficient (auto-scaled)
    random_state=42,       # Reproducible results
    verbose=True           # Shows training progress (cool to watch!)
)

print("SVM Model Created!")
print(f"Kernel: {svm_model.kernel}")
print(f"C: {svm_model.C}")
print(f"Gamma: {svm_model.gamma}")
print("Training SVM... (this might take a minute)")
svm_model.fit(X_train_processed, y_train)

print("\nTraining complete!")
print(f"Model trained on {X_train_processed.shape[0]} samples")
print(f"With {X_train_processed.shape[1]} features each")
# Test our trained model
print("Making predictions on test data...")
y_pred = svm_model.predict(X_test_processed)

print("Predictions ready!")
print(f"First 5 predictions: {y_pred[:5]}")
print(f"First 5 actual:      {y_test.values[:5]}")
print(f"\nSample comparison:")
for i in range(5):
    print(f"  Sample {i+1}: Predicted = {y_pred[i]}, Actual = {y_test.values[i]}")
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=['Sunny', 'Rainy', 'Cloudy', 'Snowy'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                               display_labels=['Sunny', 'Rainy', 'Cloudy', 'Snowy'])

disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Weather Predictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print("🔍 Calculating feature importance...")
result = permutation_importance(svm_model, X_test_processed, y_test,
                                n_repeats=10, random_state=42)

# Get feature names after preprocessing
num_feature_names = num_features  # 7 numerical features
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
all_feature_names = list(num_feature_names) + list(cat_feature_names)

# Sort by importance
importances = result.importances_mean
sorted_idx = importances.argsort()[::-1]  # index-vise Descending order not amount-vise

print("\n🏆 Top 10 Most Important Features:")
for i in sorted_idx[:10]:
    print(f"  {all_feature_names[i]:30} → Importance: {importances[i]:.4f}")

# Plot top features
plt.figure(figsize=(10, 6))
top_n = 10
plt.barh(range(top_n), importances[sorted_idx[:top_n]][::-1])
plt.yticks(range(top_n), [all_feature_names[i] for i in sorted_idx[:top_n]][::-1])
plt.xlabel('Feature Importance')
plt.title('Top 10 Features for Weather Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
final_accuracy = accuracy_score(y_test, y_pred)
print("="*50)
print("FINAL MODEL SUMMARY")
print("="*50)
print(f"Model Type: SVM with RBF kernel")
print(f"Accuracy: {final_accuracy:.2%}")
print(f"Training Samples: {X_train.shape[0]:,}")
print(f"Test Samples: {X_test.shape[0]:,}")
print(f"Features: {X_train_processed.shape[1]} (after preprocessing)")
print(f"Support Vectors: {svm_model.n_support_.sum():,} out of {X_train.shape[0]:,}")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Save the model for future use
model_filename = f"weather_svm_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
joblib.dump(svm_model, model_filename)
print(f"\nModel saved as: {model_filename}")

# Save preprocessor too
preprocessor_filename = f"weather_preprocessor_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
joblib.dump(preprocessor, preprocessor_filename)
print(f"Preprocessor saved as: {preprocessor_filename}")

print("\nAll done! You now have a weather prediction SVM!")
