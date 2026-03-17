import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import pandas as pd
import os
from database import init_db, save_prediction 
class RealEstateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real Estate Price Predictor - ISTA NTIC SYBA")
        self.root.geometry("600x750")
        self.root.configure(padx=30, pady=20)
        self.create_label("Furnishing Status:", 11, 0)
        self.cb_furnishing = self.create_combobox(11, 1, ["furnished", "semi-furnished", "unfurnished"])
        self.cb_furnishing.set("furnished")
        # Main Title
        header = tk.Label(root, text="House Price Prediction Tool", font=("Helvetica", 18, "bold"), fg="#2c3e50")
        header.grid(row=0, column=0, columnspan=2, pady=20)

        # --- Input Fields Section ---
        # 1. Numerical Inputs
        self.create_label("Surface Area (sqft):", 1, 0)
        self.ent_area = self.create_entry(1, 1)

        self.create_label("Number of Bedrooms:", 2, 0)
        self.ent_bedrooms = self.create_entry(2, 1)

        self.create_label("Number of Bathrooms:", 3, 0)
        self.ent_bathrooms = self.create_entry(3, 1)

        self.create_label("Number of Stories:", 4, 0)
        self.ent_stories = self.create_entry(4, 1)

        self.create_label("Parking Spaces:", 5, 0)
        self.ent_parking = self.create_entry(5, 1)

        # 2. Categorical Inputs (Drop-downs)
        options = ["yes", "no"]
        
        self.create_label("Main Road Access:", 6, 0)
        self.cb_mainroad = self.create_combobox(6, 1, options)

        self.create_label("Guestroom Available:", 7, 0)
        self.cb_guestroom = self.create_combobox(7, 1, options)

        self.create_label("Basement:", 8, 0)
        self.cb_basement = self.create_combobox(8, 1, options)

        self.create_label("Air Conditioning:", 9, 0)
        self.cb_ac = self.create_combobox(9, 1, options)

        self.create_label("Preferred Area:", 10, 0)
        self.cb_prefarea = self.create_combobox(10, 1, options)

        # --- Predict Button ---
        self.btn_predict = tk.Button(root, text="PREDICT PRICE", command=self.make_prediction, 
                                    bg="#0099ff", fg="white", font=("Helvetica", 12, "bold"), 
                                    pady=10, cursor="hand2")
        self.btn_predict.grid(row=15, column=0, columnspan=2, sticky="ew", pady=30)

        # --- Output Section ---
        self.lbl_result = tk.Label(root, text="Estimated Price: --- $", font=("Helvetica", 16, "bold"), fg="#27ae60")
        self.lbl_result.grid(row=16, column=0, columnspan=2)

        self.lbl_info = tk.Label(root, text="Enter all features to get an accurate estimation.", font=("Helvetica", 9, "italic"))
        self.lbl_info.grid(row=17, column=0, columnspan=2, pady=5)

    # Helper functions to keep the code clean
    def create_label(self, text, row, col):
        lbl = tk.Label(self.root, text=text, font=("Helvetica", 10))
        lbl.grid(row=row, column=col, sticky="w", pady=8)

    def create_entry(self, row, col):
        entry = tk.Entry(self.root, font=("Helvetica", 10))
        entry.grid(row=row, column=col, sticky="ew", padx=10)
        return entry

    def create_combobox(self, row, col, values):
        cb = ttk.Combobox(self.root, values=values, state="readonly", font=("Helvetica", 10))
        cb.grid(row=row, column=col, sticky="ew", padx=10)
        cb.set("no")
        return cb

    def make_prediction(self):
            try:
                # 1 here we will gather all the inputs and prepare them for the model
                # Make sure to convert categorical inputs to the same format as the training data
                furnishing = self.cb_furnishing.get()
                input_dict = {
                    # Numerical features
                    'area': float(self.ent_area.get()),
                    'bedrooms': int(self.ent_bedrooms.get()),
                    'bathrooms': int(self.ent_bathrooms.get()),
                    'stories': int(self.ent_stories.get()),
                    'mainroad': 1 if self.cb_mainroad.get() == "yes" else 0,
                    'guestroom': 1 if self.cb_guestroom.get() == "yes" else 0,
                    'basement': 1 if self.cb_basement.get() == "yes" else 0,
                    'hotwaterheating': 0,  # Assuming this feature is not in the UI, set to 0 or adjust as needed
                    'airconditioning': 1 if self.cb_ac.get() == "yes" else 0,
                    'parking': int(self.ent_parking.get()),
                    # Binary features based on dropdowns                          
                    'prefarea': 1 if self.cb_prefarea.get() == "yes" else 0,
                    # Furnishing status (converted to binary features)
                    'furnishingstatus_semi-furnished': 1 if furnishing == "semi-furnished" else 0,
                    'furnishingstatus_unfurnished': 1 if furnishing == "unfurnished" else 0,
                    # Categorical features (converted to binary)
                    # ... (بعد سطر التنبؤ prediction = model.predict)
                }
# ... (بعد سطر التنبؤ prediction = model.predict)
            

                
               
                
                    # Adding the binary features based on the dropdowns
                input_df = pd.DataFrame([input_dict])

                # 3. Load the trained model (make sure the path is correct)
                # be sure to adjust the path if your model is stored in a different location
                model_path = os.path.join(os.path.dirname(__file__), '../models/house_model.pkl')
                model = joblib.load(model_path)

                # 4. التنبؤ بالعقد
                prediction = model.predict(input_df)[0]
                # 5. حفظ العملية في قاعدة البيانات
                save_prediction(input_dict['area'], input_dict['bedrooms'], float(prediction))
                # 6. عرض النتيجة
                self.lbl_result.config(text=f"Estimated Price: {prediction:,.0f} DH", fg="#27ae60")
                self.lbl_info.config(text="Prediction successful based on model analysis.")

            except ValueError as e:
                messagebox.showerror("Input Error", f"Specific Error: {e}")    
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
            except Exception as e:
             print(f"DEBUG ERROR: {e}") # This will print the error in the console for debugging purposes.
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
if __name__ == "__main__":
    init_db()  # Ensure the database is initialized before starting the app
    root = tk.Tk()
    app = RealEstateApp(root)
    root.mainloop()