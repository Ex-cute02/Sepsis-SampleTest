# Step 4: Add SHAP interpretability (our recommended approach)
import subprocess
import sys

# Install SHAP
subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])

print("SHAP installed successfully!")