============================================================
REAL ESTATE PRICE PREDICTOR - ISTA NTIC SYBA
Developed by: youssef laachir , ayman berrhmoch , abdllah talbi , kodi kofi 
Project Category: Machine Learning & Full-Stack Development
============================================================

1. PROJECT OVERVIEW
-------------------
This application is a real estate price estimation tool built 
using Python. It uses a Machine Learning model (trained via 
Scikit-Learn) to predict house prices based on features like 
surface area, number of rooms, furnishing status, and more.

2. CORE FEATURES
----------------
- AI Prediction: Real-time price calculation.
- Modern Interface: User-friendly GUI with organized inputs.
- Database: Stores history of predictions in 'predictions.db'.
- History View: Dedicated window to review past estimations.
- Field Reset: One-click button to clear all inputs.

3. TECHNICAL REQUIREMENTS
-------------------------
To run this project, you need Python 3.x installed along with 
the following libraries:
- pandas
- scikit-learn
- joblib

4. HOW TO RUN THE APPLICATION
-----------------------------
1. Open your terminal or command prompt.
2. Navigate to the project folder:
   cd C:\Users\admin\real-estate-ml-desktop
3. Install dependencies:
   pip install pipreqs
>> pipreqs . --force
4. Launch the application:
   python app/main.py

5. PROJECT STRUCTURE
--------------------
/app       -> Contains main.py (UI) and database.py (SQL logic).
/models    -> Contains the trained AI model (house_model.pkl).
/data      -> Contains the source dataset (CSV format).
/tests     -> Testing scripts (if applicable).

6. DATA SCIENCE WORKFLOW
------------------------
- Data Analysis: Performed using Pandas and Matplotlib.
- Feature Engineering: Categorical data was converted to binary.
- Model Choice: Trained using Linear Regression/Random Forest.
- Storage: SQLite was chosen for its lightweight integration.

============================================================
(C) 2026 - ISTA NTIC SYBA - OFPPT MOROCCO
============================================================