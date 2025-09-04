# Install XGBoost and then train the model
import subprocess
import sys

# Install xgboost
subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])

print("XGBoost installed successfully!")