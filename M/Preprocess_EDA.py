# Step 2: Data Preprocessing and Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv('sepsis_sample_dataset.csv')

print("=== STEP 2: DATA PREPROCESSING & EDA ===\n")

# Basic statistics
print("Dataset Overview:")
print(f"Total samples: {len(df)}")
print(f"Features: {df.shape[1]-1}")
print(f"Target variable: outcome (1=survived, 0=deceased)")
print(f"Survival rate: {df['outcome'].mean():.2%}")
print(f"Mortality rate: {(1-df['outcome'].mean()):.2%}")

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Class distribution
print("\nClass Distribution:")
print(df['outcome'].value_counts().sort_index())

# Basic statistics by outcome
print("\nFeatures by Outcome:")
for col in ['age', 'heart_rate', 'systolic_bp', 'temperature', 'lactate']:
    survived = df[df['outcome']==1][col].mean()
    deceased = df[df['outcome']==0][col].mean()
    print(f"{col:15}: Survived={survived:6.1f}, Deceased={deceased:6.1f}")

# Prepare features and target
feature_columns = ['age', 'gender', 'heart_rate', 'systolic_bp', 'temperature', 
                  'respiratory_rate', 'wbc_count', 'lactate', 'sofa_score']
X = df[feature_columns]
y = df['outcome']

print(f"\nFeatures selected: {feature_columns}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train survival rate: {y_train.mean():.2%}")
print(f"Test survival rate: {y_test.mean():.2%}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preprocessing completed!")
print("- Train/test split: 80/20")
print("- Features standardized using StandardScaler")
print("- No missing values to handle")

# Save preprocessed data
np.save('X_train_scaled.npy', X_train_scaled)
np.save('X_test_scaled.npy', X_test_scaled) 
np.save('y_train.npy', y_train.values)
np.save('y_test.npy', y_test.values)

# Save feature names and scaler
import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_columns, 'feature_names.pkl')

print("\nPreprocessed data saved:")
print("- X_train_scaled.npy, X_test_scaled.npy")
print("- y_train.npy, y_test.npy")
print("- scaler.pkl, feature_names.pkl")