import customtkinter as ctk
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd
import os
from database import init_db, save_prediction, get_all_history

# ── Theme & Palette ──────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLORS = {
    "bg":           "#0f1117",   # near-black canvas
    "card":         "#1a1d27",   # card surface
    "card_border":  "#2a2d3e",   # subtle card outline
    "accent":       "#3b82f6",   # electric blue
    "accent_hover": "#60a5fa",   # lighter blue on hover
    "success":      "#10b981",   # emerald green for result
    "danger":       "#ef4444",   # red for reset
    "text_primary": "#f1f5f9",
    "text_secondary":"#94a3b8",
    "input_bg":     "#242838",
    "row_odd":      "#1e2235",
    "row_even":     "#161929",
}

# ─────────────────────────────────────────────────────────────
class RealEstateApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        # ── Window ──────────────────────────────────────────
        self.title("Real Estate Price Predictor")
        self.geometry("820x780")
        self.resizable(False, False)
        self.configure(fg_color=COLORS["bg"])

        self._build_header()
        self._build_form()
        self._build_result_card()
        self._build_action_row()

    # ── Header ──────────────────────────────────────────────
    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=30, pady=(28, 0))

        # Decorative accent bar
        bar = ctk.CTkFrame(hdr, width=4, height=48,
                           fg_color=COLORS["accent"], corner_radius=2)
        bar.pack(side="left", padx=(0, 14))

        title_col = ctk.CTkFrame(hdr, fg_color="transparent")
        title_col.pack(side="left")

        ctk.CTkLabel(title_col, text="House Price Predictor",
                     font=ctk.CTkFont("Segoe UI", 26, "bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w")
        ctk.CTkLabel(title_col, text="Machine-learning powered valuation tool",
                     font=ctk.CTkFont("Segoe UI", 12),
                     text_color=COLORS["text_secondary"]).pack(anchor="w")

        # History button — top right
        ctk.CTkButton(hdr, text="  View History", width=130, height=34,
                      corner_radius=8,
                      fg_color=COLORS["card"], hover_color=COLORS["card_border"],
                      border_width=1, border_color=COLORS["card_border"],
                      font=ctk.CTkFont("Segoe UI", 12),
                      text_color=COLORS["text_secondary"],
                      command=self.show_history).pack(side="right", anchor="n")

    # ── Form (two cards side by side) ───────────────────────
    def _build_form(self):
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=30, pady=(22, 0))
        row.grid_columnconfigure((0, 1), weight=1, uniform="col")

        left  = self._card(row, "Basic Details", 0)
        right = self._card(row, "Property Features", 1)

        # ── Left card inputs ──
        self.ent_area      = self._input_row(left, "Area (sqft)",  "e.g. 7420",  0)
        self.ent_bedrooms  = self._input_row(left, "Bedrooms",     "e.g. 4",     1)
        self.ent_bathrooms = self._input_row(left, "Bathrooms",    "e.g. 2",     2)
        self.ent_stories   = self._input_row(left, "Stories",      "e.g. 3",     3)
        self.ent_parking   = self._input_row(left, "Parking Spots","e.g. 2",     4)

        # ── Right card dropdowns ──
        yn = ["yes", "no"]
        self.cb_mainroad  = self._dropdown_row(right, "Main Road",         yn, 0)
        self.cb_guestroom = self._dropdown_row(right, "Guest Room",        yn, 1)
        self.cb_basement  = self._dropdown_row(right, "Basement",          yn, 2)
        self.cb_ac        = self._dropdown_row(right, "Air Conditioning",  yn, 3)
        self.cb_prefarea  = self._dropdown_row(right, "Preferred Area",    yn, 4)
        self.cb_furnish   = self._dropdown_row(right, "Furnishing Status",
                                               ["furnished",
                                                "semi-furnished",
                                                "unfurnished"], 5)

    def _card(self, parent, title, col):
        frame = ctk.CTkFrame(parent, fg_color=COLORS["card"],
                             corner_radius=14,
                             border_width=1, border_color=COLORS["card_border"])
        frame.grid(row=0, column=col, sticky="nsew",
                   padx=(0, 10) if col == 0 else (10, 0))
        frame.grid_columnconfigure(1, weight=1)

        # Card title strip
        ctk.CTkLabel(frame, text=title,
                     font=ctk.CTkFont("Segoe UI", 12, "bold"),
                     text_color=COLORS["accent"],
                     fg_color="transparent").grid(
            row=0, column=0, columnspan=2, sticky="w",
            padx=18, pady=(14, 10))
        return frame

    def _input_row(self, card, label, placeholder, row):
        ctk.CTkLabel(card, text=label,
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=COLORS["text_secondary"]).grid(
            row=row + 1, column=0, sticky="w", padx=18, pady=5)
        entry = ctk.CTkEntry(card, placeholder_text=placeholder,
                             fg_color=COLORS["input_bg"],
                             border_color=COLORS["card_border"],
                             border_width=1,
                             text_color=COLORS["text_primary"],
                             height=34, corner_radius=7,
                             font=ctk.CTkFont("Segoe UI", 11))
        entry.grid(row=row + 1, column=1, sticky="ew", padx=(8, 18), pady=5)
        return entry

    def _dropdown_row(self, card, label, values, row):
        ctk.CTkLabel(card, text=label,
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=COLORS["text_secondary"]).grid(
            row=row + 1, column=0, sticky="w", padx=18, pady=5)
        combo = ctk.CTkComboBox(card, values=values,
                                fg_color=COLORS["input_bg"],
                                border_color=COLORS["card_border"],
                                border_width=1,
                                button_color=COLORS["accent"],
                                button_hover_color=COLORS["accent_hover"],
                                dropdown_fg_color=COLORS["card"],
                                text_color=COLORS["text_primary"],
                                height=34, corner_radius=7,
                                font=ctk.CTkFont("Segoe UI", 11),
                                state="readonly")
        combo.grid(row=row + 1, column=1, sticky="ew", padx=(8, 18), pady=5)
        combo.set(values[0])
        return combo

    # ── Result display card ──────────────────────────────────
    def _build_result_card(self):
        card = ctk.CTkFrame(self, fg_color=COLORS["card"],
                            corner_radius=14,
                            border_width=1, border_color=COLORS["card_border"])
        card.pack(fill="x", padx=30, pady=(20, 0))

        ctk.CTkLabel(card, text="Estimated Property Value",
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=COLORS["text_secondary"]).pack(pady=(14, 2))

        self.lbl_result = ctk.CTkLabel(
            card, text="— DH",
            font=ctk.CTkFont("Segoe UI", 36, "bold"),
            text_color=COLORS["success"])
        self.lbl_result.pack(pady=(0, 14))

    # ── Action buttons ───────────────────────────────────────
    def _build_action_row(self):
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(pady=20)

        ctk.CTkButton(row, text="  Calculate Price",
                      width=200, height=44,
                      corner_radius=10,
                      fg_color=COLORS["accent"],
                      hover_color=COLORS["accent_hover"],
                      font=ctk.CTkFont("Segoe UI", 13, "bold"),
                      command=self.make_prediction).grid(
            row=0, column=0, padx=10)

        ctk.CTkButton(row, text="  Reset",
                      width=120, height=44,
                      corner_radius=10,
                      fg_color=COLORS["card"],
                      hover_color=COLORS["card_border"],
                      border_width=1, border_color=COLORS["danger"],
                      text_color=COLORS["danger"],
                      font=ctk.CTkFont("Segoe UI", 13, "bold"),
                      command=self.reset_fields).grid(
            row=0, column=1, padx=10)

    # ── Logic ────────────────────────────────────────────────
    def make_prediction(self):
        try:
            area      = self.ent_area.get().strip()
            bedrooms  = self.ent_bedrooms.get().strip()
            bathrooms = self.ent_bathrooms.get().strip()
            stories   = self.ent_stories.get().strip()
            parking   = self.ent_parking.get().strip()

            if not all([area, bedrooms, bathrooms, stories, parking]):
                messagebox.showwarning("Missing Fields",
                    "Please fill in all numeric fields before calculating.")
                return

            furnishing = self.cb_furnish.get()
            input_dict = {
                "area":                          float(area),
                "bedrooms":                      int(bedrooms),
                "bathrooms":                     int(bathrooms),
                "stories":                       int(stories),
                "mainroad":                      1 if self.cb_mainroad.get()  == "yes" else 0,
                "guestroom":                     1 if self.cb_guestroom.get() == "yes" else 0,
                "basement":                      1 if self.cb_basement.get()  == "yes" else 0,
                "hotwaterheating":               0,
                "airconditioning":               1 if self.cb_ac.get()        == "yes" else 0,
                "parking":                       int(parking),
                "prefarea":                      1 if self.cb_prefarea.get()  == "yes" else 0,
                "furnishingstatus_semi-furnished": 1 if furnishing == "semi-furnished"  else 0,
                "furnishingstatus_unfurnished":    1 if furnishing == "unfurnished"      else 0,
            }

            model_path = os.path.join(os.path.dirname(__file__), "../models/house_model.pkl")
            model      = joblib.load(model_path)
            prediction = model.predict(pd.DataFrame([input_dict]))[0]

            save_prediction(input_dict["area"], input_dict["bedrooms"], float(prediction))
            self.lbl_result.configure(
                text=f"{prediction:,.0f} DH",
                text_color=COLORS["success"])

        except ValueError:
            messagebox.showerror("Input Error",
                "Please check your inputs — all numeric fields must contain valid numbers.")
        except FileNotFoundError:
            messagebox.showerror("Model Error",
                "house_model.pkl not found.\nExpected at: ../models/house_model.pkl")
        except Exception as e:
            messagebox.showerror("Unexpected Error", str(e))

    def reset_fields(self):
        for entry in (self.ent_area, self.ent_bedrooms, self.ent_bathrooms,
                      self.ent_stories, self.ent_parking):
            entry.delete(0, "end")
        self.cb_mainroad.set("no")
        self.cb_guestroom.set("no")
        self.cb_basement.set("no")
        self.cb_ac.set("no")
        self.cb_prefarea.set("no")
        self.cb_furnish.set("furnished")
        self.lbl_result.configure(text="— DH", text_color=COLORS["success"])

    # ── History window ───────────────────────────────────────
    def show_history(self):
        win = ctk.CTkToplevel(self)
        win.title("Prediction History")
        win.geometry("680x440")
        win.configure(fg_color=COLORS["bg"])
        win.grab_set()

        ctk.CTkLabel(win, text="Prediction History",
                     font=ctk.CTkFont("Segoe UI", 18, "bold"),
                     text_color=COLORS["text_primary"]).pack(
            anchor="w", padx=24, pady=(20, 2))
        ctk.CTkLabel(win, text="All saved predictions from this session and past runs.",
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=COLORS["text_secondary"]).pack(
            anchor="w", padx=24, pady=(0, 14))

        # ── Styled Treeview ──────────────────────────────────
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("History.Treeview",
                        background=COLORS["card"],
                        fieldbackground=COLORS["card"],
                        foreground=COLORS["text_primary"],
                        rowheight=32,
                        borderwidth=0,
                        font=("Segoe UI", 10))

        style.configure("History.Treeview.Heading",
                        background=COLORS["input_bg"],
                        foreground=COLORS["accent"],
                        relief="flat",
                        font=("Segoe UI", 10, "bold"))

        style.map("History.Treeview",
                  background=[("selected", COLORS["accent"])],
                  foreground=[("selected", "#ffffff")])

        columns = ("Date", "Area (sqft)", "Bedrooms", "Price (DH)")
        tree = ttk.Treeview(win, columns=columns, show="headings",
                            style="History.Treeview", selectmode="browse")

        col_widths = {"Date": 180, "Area (sqft)": 110, "Bedrooms": 90, "Price (DH)": 140}
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=col_widths[col], anchor="center")

        # Alternate row colours
        tree.tag_configure("odd",  background=COLORS["row_odd"])
        tree.tag_configure("even", background=COLORS["row_even"])

        rows = get_all_history()
        for i, row in enumerate(rows):
            date, area, rooms, price = row
            tree.insert("", "end",
                        values=(date,
                                f"{area:,.0f}",
                                int(rooms),
                                f"{price:,.0f}"),
                        tags=("odd" if i % 2 else "even",))

        # Scrollbar
        sb = ctk.CTkScrollbar(win, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y", padx=(0, 14), pady=14)
        tree.pack(fill="both", expand=True, padx=(24, 0), pady=(0, 14))

        # Row count footer
        ctk.CTkLabel(win, text=f"{len(rows)} record(s) found",
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=COLORS["text_secondary"]).pack(
            anchor="e", padx=24, pady=(0, 14))


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app = RealEstateApp()
    app.mainloop()