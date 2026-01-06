**Overview**
This project predicts insurance premiums using machine learning models. It uses pre-trained models and scalers to estimate premiums for different user categories.

**Project Structure**
main.py: Main script to run predictions.
prediction_helper.py: Helper functions for data preprocessing and prediction.
artifacts/: Contains pre-trained models and scalers.
model_rest.joblib
model_young.joblib
scaler_rest.joblib
scaler_young.joblib

**Requirements**
Python 3.x
See requirements.txt for dependencies.

**How to Use**
Install dependencies:
pip install -r requirements.txt

**Run the main script:**
python main.py

The script will prompt for user input and display the predicted insurance premium.

**Customization**
To update models, replace the .joblib files in the artifacts/ directory.
Modify prediction_helper.py to change preprocessing or prediction logic.
