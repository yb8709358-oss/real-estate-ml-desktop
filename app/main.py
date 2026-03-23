# ============================================================
#  Real Estate Price Predictor
#  Features: Dark/Light mode, Feature Importance chart,
#            Compare 2 Houses, Live What-If Sliders
#  Requires: pip install customtkinter joblib pandas matplotlib
# ============================================================

import customtkinter as ctk
from tkinter import messagebox
from tkinter import ttk
import joblib
import pandas as pd
import os
import sys
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from database import init_db, save_prediction, get_all_history

# ── Two full palettes ────────────────────────────────────────
THEMES = {
    "dark": {
        "bg":             "#0f1117",
        "card":           "#1a1d27",
        "card_border":    "#2a2d3e",
        "accent":         "#3b82f6",
        "accent_hover":   "#60a5fa",
        "success":        "#10b981",
        "danger":         "#ef4444",
        "text_primary":   "#f1f5f9",
        "text_secondary": "#94a3b8",
        "input_bg":       "#242838",
        "row_odd":        "#1e2235",
        "row_even":       "#161929",
        "toggle_bg":      "#1a1d27",
        "toggle_border":  "#2a2d3e",
        "toggle_icon":    "☀️",
        "toggle_label":   "Light Mode",
        "plot_bg":        "#1a1d27",
        "plot_fg":        "#f1f5f9",
    },
    "light": {
        "bg":             "#f0f4f8",
        "card":           "#ffffff",
        "card_border":    "#dde3ed",
        "accent":         "#2563eb",
        "accent_hover":   "#3b82f6",
        "success":        "#059669",
        "danger":         "#dc2626",
        "text_primary":   "#1e293b",
        "text_secondary": "#64748b",
        "input_bg":       "#f8fafc",
        "row_odd":        "#f1f5f9",
        "row_even":       "#ffffff",
        "toggle_bg":      "#e2e8f0",
        "toggle_border":  "#cbd5e1",
        "toggle_icon":    "🌙",
        "toggle_label":   "Dark Mode",
        "plot_bg":        "#ffffff",
        "plot_fg":        "#1e293b",
    },
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
COLORS = dict(THEMES["dark"])

# ── Translations ─────────────────────────────────────────────
TRANSLATIONS = {
    "en": {
        # Header
        "app_title":        "House Price Predictor",
        "app_subtitle":     "Machine-learning powered valuation tool",
        "light_mode":       "Light Mode",
        "dark_mode":        "Dark Mode",
        "btn_history":      "📋  History",
        "btn_compare":      "↔️  Compare",
        # Form cards
        "card_basic":       "🏠  Basic Details",
        "card_features":    "✨  Property Features",
        # Basic fields
        "lbl_proptype":     "🏷️  Property Type",
        "lbl_area":         "📐  Area (sqft)",
        "lbl_bedrooms":     "🛏️  Bedrooms",
        "lbl_bathrooms":    "🚿  Bathrooms",
        "lbl_stories":      "🏢  Stories",
        "lbl_parking":      "🚗  Parking Spots",
        # Feature fields
        "lbl_mainroad":     "🛣️  Main Road",
        "lbl_guestroom":    "🛋️  Guest Room",
        "lbl_basement":     "🏚️  Basement",
        "lbl_ac":           "❄️  Air Conditioning",
        "lbl_prefarea":     "⭐  Preferred Area",
        "lbl_furnish":      "🪑  Furnishing Status",
        # Result card
        "lbl_estimate":     "💰  Estimated Property Value",
        "btn_importance":   "🎯  Feature\nImportance",
        # Sliders
        "sliders_title":    "🎚️  Live What-If Sliders",
        "sliders_sub":      "Adjust sliders to instantly recalculate the price",
        "sl_area":          "📐 Area (sqft)",
        "sl_bedrooms":      "🛏️ Bedrooms",
        "sl_parking":       "🚗 Parking",
        # Buttons
        "btn_calculate":    "  Calculate Price",
        "btn_reset":        "  Reset",
        # Property types
        "type_apartment":   "🏢  Apartment",
        "type_house":       "🏠  House",
        "type_villa":       "🏡  Villa",
        # Feature importance window
        "fi_title":         "🎯  Feature Importance",
        "fi_subtitle":      "Which inputs influence the price prediction most?",
        "fi_xlabel":        "Importance Score",
        # Compare window
        "cmp_title":        "↔️  Compare Two Houses",
        "cmp_subtitle":     "Fill in both panels below, then click Compare Now.",
        "cmp_house_a":      "🏠  House A",
        "cmp_house_b":      "🏡  House B",
        "cmp_lbl_a":        "🏠  — DH",
        "cmp_lbl_b":        "🏡  — DH",
        "cmp_btn":          "↔️  Compare Now",
        "cmp_higher":       "🏆 {name} is higher by {diff:,.0f} DH",
        "cmp_equal":        "🤝 Both houses have equal value!",
        "cmp_proptype":     "🏷️ Property Type",
        "cmp_area":         "📐 Area (sqft)",
        "cmp_bedrooms":     "🛏️ Bedrooms",
        "cmp_bathrooms":    "🚿 Bathrooms",
        "cmp_stories":      "🏢 Stories",
        "cmp_parking":      "🚗 Parking",
        "cmp_mainroad":     "🛣️ Main Road",
        "cmp_guestroom":    "🛋️ Guest Room",
        "cmp_basement":     "🏚️ Basement",
        "cmp_ac":           "❄️ Air Con",
        "cmp_prefarea":     "⭐ Pref. Area",
        "cmp_furnish":      "🪑 Furnishing",
        # History window
        "hist_title":       "📋  Prediction History",
        "hist_subtitle":    "All saved predictions from this session and past runs.",
        "hist_col_date":    "Date",
        "hist_col_area":    "Area (sqft)",
        "hist_col_rooms":   "Bedrooms",
        "hist_col_price":   "Price (DH)",
        "hist_records":     "{n} record(s) found",
        # Warnings / errors
        "warn_missing":     "Please fill in all numeric fields before calculating.",
        "warn_missing_cmp": "{label}: please fill in — {fields}",
        "err_model":        "house_model.pkl not found.\nExpected at: ../models/house_model.pkl",
        "err_input":        "Please check your inputs — all numeric fields must be valid numbers.",
        "err_input_cmp":    "{label}: numeric fields must contain valid numbers.",
        "no_importance":    "This model does not expose feature importances.\n(Works with tree-based models: RandomForest, XGBoost, etc.)",
    },
    "fr": {
        "app_title":        "Prédicteur de Prix Immobilier",
        "app_subtitle":     "Outil d'évaluation basé sur le machine learning",
        "light_mode":       "Mode Clair",
        "dark_mode":        "Mode Sombre",
        "btn_history":      "📋  Historique",
        "btn_compare":      "↔️  Comparer",
        "card_basic":       "🏠  Informations Générales",
        "card_features":    "✨  Caractéristiques",
        "lbl_proptype":     "🏷️  Type de Bien",
        "lbl_area":         "📐  Surface (m²)",
        "lbl_bedrooms":     "🛏️  Chambres",
        "lbl_bathrooms":    "🚿  Salles de bain",
        "lbl_stories":      "🏢  Étages",
        "lbl_parking":      "🚗  Places de parking",
        "lbl_mainroad":     "🛣️  Route Principale",
        "lbl_guestroom":    "🛋️  Chambre d'Amis",
        "lbl_basement":     "🏚️  Sous-sol",
        "lbl_ac":           "❄️  Climatisation",
        "lbl_prefarea":     "⭐  Zone Préférée",
        "lbl_furnish":      "🪑  Meublé",
        "lbl_estimate":     "💰  Valeur Estimée du Bien",
        "btn_importance":   "🎯  Importance\ndes Critères",
        "sliders_title":    "🎚️  Curseurs en Temps Réel",
        "sliders_sub":      "Ajustez les curseurs pour recalculer instantanément",
        "sl_area":          "📐 Surface (m²)",
        "sl_bedrooms":      "🛏️ Chambres",
        "sl_parking":       "🚗 Parking",
        "btn_calculate":    "  Calculer le Prix",
        "btn_reset":        "  Réinitialiser",
        "type_apartment":   "🏢  Appartement",
        "type_house":       "🏠  Maison",
        "type_villa":       "🏡  Villa",
        "fi_title":         "🎯  Importance des Critères",
        "fi_subtitle":      "Quels critères influencent le plus le prix prédit?",
        "fi_xlabel":        "Score d'importance",
        "cmp_title":        "↔️  Comparer Deux Biens",
        "cmp_subtitle":     "Remplissez les deux panneaux, puis cliquez sur Comparer.",
        "cmp_house_a":      "🏠  Bien A",
        "cmp_house_b":      "🏡  Bien B",
        "cmp_lbl_a":        "🏠  — DH",
        "cmp_lbl_b":        "🏡  — DH",
        "cmp_btn":          "↔️  Comparer",
        "cmp_higher":       "🏆 {name} est plus élevé de {diff:,.0f} DH",
        "cmp_equal":        "🤝 Les deux biens ont la même valeur!",
        "cmp_proptype":     "🏷️ Type de Bien",
        "cmp_area":         "📐 Surface (m²)",
        "cmp_bedrooms":     "🛏️ Chambres",
        "cmp_bathrooms":    "🚿 Salles de bain",
        "cmp_stories":      "🏢 Étages",
        "cmp_parking":      "🚗 Parking",
        "cmp_mainroad":     "🛣️ Route Principale",
        "cmp_guestroom":    "🛋️ Chambre d'Amis",
        "cmp_basement":     "🏚️ Sous-sol",
        "cmp_ac":           "❄️ Climatisation",
        "cmp_prefarea":     "⭐ Zone Préférée",
        "cmp_furnish":      "🪑 Meublé",
        "hist_title":       "📋  Historique des Prédictions",
        "hist_subtitle":    "Toutes les prédictions sauvegardées.",
        "hist_col_date":    "Date",
        "hist_col_area":    "Surface (m²)",
        "hist_col_rooms":   "Chambres",
        "hist_col_price":   "Prix (DH)",
        "hist_records":     "{n} enregistrement(s) trouvé(s)",
        "warn_missing":     "Veuillez remplir tous les champs numériques avant de calculer.",
        "warn_missing_cmp": "{label}: veuillez remplir — {fields}",
        "err_model":        "house_model.pkl introuvable.\nAttendu à: ../models/house_model.pkl",
        "err_input":        "Vérifiez vos saisies — tous les champs numériques doivent être valides.",
        "err_input_cmp":    "{label}: les champs numériques doivent contenir des nombres valides.",
        "no_importance":    "Ce modèle n'expose pas les importances de caractéristiques.\n(Fonctionne avec: RandomForest, XGBoost, etc.)",
    },
    "ar": {
        "app_title":        "أداة التنبؤ بأسعار العقارات",
        "app_subtitle":     "أداة تقييم مدعومة بالتعلم الآلي",
        "light_mode":       "الوضع الفاتح",
        "dark_mode":        "الوضع الداكن",
        "btn_history":      "📋  السجل",
        "btn_compare":      "↔️  مقارنة",
        "card_basic":       "🏠  البيانات الأساسية",
        "card_features":    "✨  مميزات العقار",
        "lbl_proptype":     "🏷️  نوع العقار",
        "lbl_area":         "📐  المساحة (م²)",
        "lbl_bedrooms":     "🛏️  غرف النوم",
        "lbl_bathrooms":    "🚿  الحمامات",
        "lbl_stories":      "🏢  الطوابق",
        "lbl_parking":      "🚗  مواقف السيارات",
        "lbl_mainroad":     "🛣️  الطريق الرئيسي",
        "lbl_guestroom":    "🛋️  غرفة الضيوف",
        "lbl_basement":     "🏚️  القبو",
        "lbl_ac":           "❄️  تكييف الهواء",
        "lbl_prefarea":     "⭐  المنطقة المفضلة",
        "lbl_furnish":      "🪑  حالة الأثاث",
        "lbl_estimate":     "💰  القيمة التقديرية للعقار",
        "btn_importance":   "🎯  أهمية\nالمعايير",
        "sliders_title":    "🎚️  المنزلقات التفاعلية",
        "sliders_sub":      "اضبط المنزلقات لإعادة الحساب فوراً",
        "sl_area":          "📐 المساحة",
        "sl_bedrooms":      "🛏️ غرف النوم",
        "sl_parking":       "🚗 مواقف",
        "btn_calculate":    "  حساب السعر",
        "btn_reset":        "  إعادة تعيين",
        "type_apartment":   "🏢  شقة",
        "type_house":       "🏠  منزل",
        "type_villa":       "🏡  فيلا",
        "fi_title":         "🎯  أهمية المعايير",
        "fi_subtitle":      "أي المدخلات تؤثر أكثر على التنبؤ بالسعر؟",
        "fi_xlabel":        "درجة الأهمية",
        "cmp_title":        "↔️  مقارنة عقارين",
        "cmp_subtitle":     "املأ العقارين ثم اضغط مقارنة الآن.",
        "cmp_house_a":      "🏠  العقار أ",
        "cmp_house_b":      "🏡  العقار ب",
        "cmp_lbl_a":        "🏠  — DH",
        "cmp_lbl_b":        "🏡  — DH",
        "cmp_btn":          "↔️  مقارنة الآن",
        "cmp_higher":       "🏆 {name} أعلى بـ {diff:,.0f} DH",
        "cmp_equal":        "🤝 العقاران متساويان في القيمة!",
        "cmp_proptype":     "🏷️ النوع",
        "cmp_area":         "📐 المساحة",
        "cmp_bedrooms":     "🛏️ غرف النوم",
        "cmp_bathrooms":    "🚿 الحمامات",
        "cmp_stories":      "🏢 الطوابق",
        "cmp_parking":      "🚗 مواقف",
        "cmp_mainroad":     "🛣️ طريق رئيسي",
        "cmp_guestroom":    "🛋️ غرفة ضيوف",
        "cmp_basement":     "🏚️ قبو",
        "cmp_ac":           "❄️ تكييف",
        "cmp_prefarea":     "⭐ منطقة مفضلة",
        "cmp_furnish":      "🪑 الأثاث",
        "hist_title":       "📋  سجل التنبؤات",
        "hist_subtitle":    "جميع التنبؤات المحفوظة من هذه الجلسة والجلسات السابقة.",
        "hist_col_date":    "التاريخ",
        "hist_col_area":    "المساحة",
        "hist_col_rooms":   "غرف النوم",
        "hist_col_price":   "السعر (DH)",
        "hist_records":     "{n} سجل",
        "warn_missing":     "يرجى ملء جميع الحقول الرقمية قبل الحساب.",
        "warn_missing_cmp": "{label}: يرجى ملء — {fields}",
        "err_model":        "house_model.pkl غير موجود.\nالمسار المتوقع: ../models/house_model.pkl",
        "err_input":        "تحقق من مدخلاتك — يجب أن تحتوي الحقول الرقمية على أرقام صحيحة.",
        "err_input_cmp":    "{label}: يجب أن تحتوي الحقول الرقمية على أرقام صحيحة.",
        "no_importance":    "هذا النموذج لا يوفر أهمية الميزات.\n(يعمل مع: RandomForest, XGBoost, إلخ)",
    },

}

LANG_OPTIONS = {
    "🇬🇧 English":  "en",
    "🇫🇷 Français": "fr",
    "🇸🇦 العربية":  "ar",
}
CURRENT_LANG = "en"

def T(key):
    """Return translated string for current language, fallback to English."""
    return TRANSLATIONS.get(CURRENT_LANG, TRANSLATIONS["en"]).get(
        key, TRANSLATIONS["en"].get(key, key))

# ── Feature labels (display names) ──────────────────────────
FEATURE_LABELS = {
    "area":                            "📐 Area",
    "bedrooms":                        "🛏️ Bedrooms",
    "bathrooms":                       "🚿 Bathrooms",
    "stories":                         "🏢 Stories",
    "mainroad":                        "🛣️ Main Road",
    "guestroom":                       "🛋️ Guest Room",
    "basement":                        "🏚️ Basement",
    "hotwaterheating":                 "🔥 Hot Water",
    "airconditioning":                 "❄️ Air Con",
    "parking":                         "🚗 Parking",
    "prefarea":                        "⭐ Pref. Area",
    "furnishingstatus_semi-furnished": "🪑 Semi-Furn.",
    "furnishingstatus_unfurnished":    "🪑 Unfurnished",
}

# ── Property type multipliers ────────────────────────────────
PROPERTY_MULTIPLIERS = {
    "🏢  Apartment": 1.00,
    "🏠  House":     1.20,
    "🏡  Villa":     1.50,
}
PROPERTY_TYPES = list(PROPERTY_MULTIPLIERS.keys())

# ── PyInstaller-compatible resource path ─────────────────────
def _resource_path(relative):
    """Works both in dev mode and when packaged as a .exe by PyInstaller."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)

def _find_model():
    """
    Find house_model.pkl in multiple locations:
    1. Bundled inside .exe  (sys._MEIPASS)
    2. Same folder as main.py  (dev)
    3. ../models/ relative to main.py  (dev project structure)
    4. Same folder as the .exe itself
    """
    candidates = [
        # 1 — PyInstaller bundle (_MEIPASS)
        _resource_path("house_model.pkl"),
        # 2 — same folder as main.py
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "house_model.pkl"),
        # 3 — ../models/ (original project layout)
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "house_model.pkl"),
        # 4 — next to the running .exe
        os.path.join(os.path.dirname(sys.executable), "house_model.pkl"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    # Return the most descriptive path for the error message
    return candidates[2]

MODEL_PATH = _find_model()


def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    return joblib.load(MODEL_PATH)


def build_input_dict(area, bedrooms, bathrooms, stories, parking,
                     mainroad, guestroom, basement, ac, prefarea, furnishing):
    return {
        "area":                            float(area),
        "bedrooms":                        int(bedrooms),
        "bathrooms":                       int(bathrooms),
        "stories":                         int(stories),
        "mainroad":                        1 if mainroad  == "yes" else 0,
        "guestroom":                       1 if guestroom == "yes" else 0,
        "basement":                        1 if basement  == "yes" else 0,
        "hotwaterheating":                 0,
        "airconditioning":                 1 if ac        == "yes" else 0,
        "parking":                         int(parking),
        "prefarea":                        1 if prefarea  == "yes" else 0,
        "furnishingstatus_semi-furnished": 1 if furnishing == "semi-furnished" else 0,
        "furnishingstatus_unfurnished":    1 if furnishing == "unfurnished"    else 0,
    }


# ─────────────────────────────────────────────────────────────
class RealEstateApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self._mode = "dark"
        self._cards:            list = []
        self._entries:          list = []
        self._combos:           list = []
        self._labels_secondary: list = []
        self._labels_primary:   list = []
        self._card_titles:      list = []
        self._slider_labels:    list = []
        # (widget, translation_key) pairs for live language switching
        self._t_widgets:        list = []

        self.title("Real Estate Price Predictor")
        self.geometry("860x920")
        self.resizable(True, True)
        self.minsize(820, 600)
        self.configure(fg_color=COLORS["bg"])

        # ── Scrollable main canvas ─────────────────────────────
        self._main_scroll = ctk.CTkScrollableFrame(self, fg_color=COLORS["bg"])
        self._main_scroll.pack(fill="both", expand=True)
        self._scroll_inner = self._main_scroll  # alias used by build methods

        self._build_header()
        self._build_form()
        self._build_result_card()
        self._build_whatif_card()
        self._build_action_row()

    # ══════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════
    def _build_header(self):
        hdr = ctk.CTkFrame(self._scroll_inner, fg_color="transparent")
        hdr.pack(fill="x", padx=30, pady=(28, 0))

        self._accent_bar = ctk.CTkFrame(hdr, width=4, height=48,
                                        fg_color=COLORS["accent"], corner_radius=2)
        self._accent_bar.pack(side="left", padx=(0, 14))

        title_col = ctk.CTkFrame(hdr, fg_color="transparent")
        title_col.pack(side="left")

        self._lbl_title = ctk.CTkLabel(
            title_col, text=T("app_title"),
            font=ctk.CTkFont("Segoe UI", 26, "bold"),
            text_color=COLORS["text_primary"])
        self._lbl_title.pack(anchor="w")
        self._labels_primary.append(self._lbl_title)
        self._t_widgets.append((self._lbl_title, "app_title"))

        self._lbl_subtitle = ctk.CTkLabel(
            title_col, text=T("app_subtitle"),
            font=ctk.CTkFont("Segoe UI", 12),
            text_color=COLORS["text_secondary"])
        self._lbl_subtitle.pack(anchor="w")
        self._labels_secondary.append(self._lbl_subtitle)
        self._t_widgets.append((self._lbl_subtitle, "app_subtitle"))

        btn_row = ctk.CTkFrame(hdr, fg_color="transparent")
        btn_row.pack(side="right", anchor="n")

        # ── Language selector ──
        self.lang_combo = ctk.CTkComboBox(
            btn_row, values=["🇬🇧 English", "🇫🇷 Français", "🇸🇦 العربية"],
            width=145, height=34, corner_radius=8,
            fg_color=COLORS["toggle_bg"],
            border_color=COLORS["toggle_border"], border_width=1,
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["card"],
            text_color=COLORS["text_secondary"],
            font=ctk.CTkFont("Segoe UI", 12),
            state="readonly",
            command=self.change_language)
        self.lang_combo.set("🇬🇧 English")
        self.lang_combo.pack(side="left", padx=(0, 8))

        self.theme_btn = ctk.CTkButton(
            btn_row, text=f"{COLORS['toggle_icon']}  {T('light_mode')}",
            width=145, height=34, corner_radius=8,
            fg_color=COLORS["toggle_bg"], hover_color=COLORS["card_border"],
            border_width=1, border_color=COLORS["toggle_border"],
            font=ctk.CTkFont("Segoe UI", 12), text_color=COLORS["text_secondary"],
            command=self.toggle_theme)
        self.theme_btn.pack(side="left", padx=(0, 8))

        self.history_btn = ctk.CTkButton(
            btn_row, text=T("btn_history"), width=120, height=34, corner_radius=8,
            fg_color=COLORS["toggle_bg"], hover_color=COLORS["card_border"],
            border_width=1, border_color=COLORS["toggle_border"],
            font=ctk.CTkFont("Segoe UI", 12), text_color=COLORS["text_secondary"],
            command=self.show_history)
        self.history_btn.pack(side="left", padx=(0, 8))
        self._t_widgets.append((self.history_btn, "btn_history"))

        self.compare_btn = ctk.CTkButton(
            btn_row, text=T("btn_compare"), width=120, height=34, corner_radius=8,
            fg_color=COLORS["toggle_bg"], hover_color=COLORS["card_border"],
            border_width=1, border_color=COLORS["toggle_border"],
            font=ctk.CTkFont("Segoe UI", 12), text_color=COLORS["text_secondary"],
            command=self.show_compare)
        self.compare_btn.pack(side="left")
        self._t_widgets.append((self.compare_btn, "btn_compare"))

    # ══════════════════════════════════════════════════════════
    # MAIN FORM
    # ══════════════════════════════════════════════════════════
    def _build_form(self):
        form_row = ctk.CTkFrame(self._scroll_inner, fg_color="transparent")
        form_row.pack(fill="x", padx=30, pady=(22, 0))
        form_row.grid_columnconfigure((0, 1), weight=1, uniform="col")

        left  = self._card(form_row, T("card_basic"),    0, t_key="card_basic")
        right = self._card(form_row, T("card_features"), 1, t_key="card_features")

        self.cb_proptype   = self._dropdown_row(left, T("lbl_proptype"), PROPERTY_TYPES, 0, t_key="lbl_proptype")
        self.ent_area      = self._input_row(left, T("lbl_area"),      "e.g. 7420", 1, t_key="lbl_area")
        self.ent_bedrooms  = self._input_row(left, T("lbl_bedrooms"),  "e.g. 4",    2, t_key="lbl_bedrooms")
        self.ent_bathrooms = self._input_row(left, T("lbl_bathrooms"), "e.g. 2",    3, t_key="lbl_bathrooms")
        self.ent_stories   = self._input_row(left, T("lbl_stories"),   "e.g. 3",    4, t_key="lbl_stories")
        self.ent_parking   = self._input_row(left, T("lbl_parking"),   "e.g. 2",    5, t_key="lbl_parking")

        yn = ["yes", "no"]
        self.cb_mainroad  = self._dropdown_row(right, T("lbl_mainroad"),  yn, 0, t_key="lbl_mainroad")
        self.cb_guestroom = self._dropdown_row(right, T("lbl_guestroom"), yn, 1, t_key="lbl_guestroom")
        self.cb_basement  = self._dropdown_row(right, T("lbl_basement"),  yn, 2, t_key="lbl_basement")
        self.cb_ac        = self._dropdown_row(right, T("lbl_ac"),        yn, 3, t_key="lbl_ac")
        self.cb_prefarea  = self._dropdown_row(right, T("lbl_prefarea"),  yn, 4, t_key="lbl_prefarea")
        self.cb_furnish   = self._dropdown_row(right, T("lbl_furnish"),
                                               ["furnished", "semi-furnished", "unfurnished"], 5, t_key="lbl_furnish")

        # ── Enter key navigation ──────────────────────────────
        # Each entry moves focus to the next; last entry triggers Calculate
        _order = [self.ent_area, self.ent_bedrooms,
                  self.ent_bathrooms, self.ent_stories, self.ent_parking]
        for i, w in enumerate(_order):
            nxt = _order[i + 1] if i + 1 < len(_order) else None
            if nxt:
                w.bind("<Return>", lambda e, n=nxt: n.focus_set())
            else:
                w.bind("<Return>", lambda e: self.make_prediction())

    # ══════════════════════════════════════════════════════════
    # RESULT CARD
    # ══════════════════════════════════════════════════════════
    def _build_result_card(self):
        self._result_card = ctk.CTkFrame(self._scroll_inner, fg_color=COLORS["card"],
                                         corner_radius=14, border_width=1,
                                         border_color=COLORS["card_border"])
        self._result_card.pack(fill="x", padx=30, pady=(20, 0))
        self._cards.append(self._result_card)

        inner = ctk.CTkFrame(self._result_card, fg_color="transparent")
        inner.pack(fill="x", padx=20, pady=14)

        # Left: price display
        left = ctk.CTkFrame(inner, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True)

        lbl_est = ctk.CTkLabel(left, text=T("lbl_estimate"),
                               font=ctk.CTkFont("Segoe UI", 11),
                               text_color=COLORS["text_secondary"])
        lbl_est.pack(anchor="w")
        self._labels_secondary.append(lbl_est)
        self._t_widgets.append((lbl_est, "lbl_estimate"))

        self.lbl_result = ctk.CTkLabel(
            left, text="— DH",
            font=ctk.CTkFont("Segoe UI", 36, "bold"),
            text_color=COLORS["success"])
        self.lbl_result.pack(anchor="w")

        # Right: Feature Importance button
        self.importance_btn = ctk.CTkButton(
            inner, text=T("btn_importance"),
            width=130, height=54, corner_radius=10,
            fg_color=COLORS["input_bg"], hover_color=COLORS["card_border"],
            border_width=1, border_color=COLORS["accent"],
            font=ctk.CTkFont("Segoe UI", 11, "bold"),
            text_color=COLORS["accent"],
            command=self.show_importance)
        self.importance_btn.pack(side="right")
        self._t_widgets.append((self.importance_btn, "btn_importance"))

    # ══════════════════════════════════════════════════════════
    # WHAT-IF SLIDERS CARD
    # ══════════════════════════════════════════════════════════
    def _build_whatif_card(self):
        self._whatif_card = ctk.CTkFrame(self._scroll_inner, fg_color=COLORS["card"],
                                          corner_radius=14, border_width=1,
                                          border_color=COLORS["card_border"])
        self._whatif_card.pack(fill="x", padx=30, pady=(16, 0))
        self._cards.append(self._whatif_card)

        title_lbl = ctk.CTkLabel(self._whatif_card,
                                  text=T("sliders_title"),
                                  font=ctk.CTkFont("Segoe UI", 12, "bold"),
                                  text_color=COLORS["accent"])
        title_lbl.pack(anchor="w", padx=18, pady=(14, 4))
        self._card_titles.append(title_lbl)
        self._t_widgets.append((title_lbl, "sliders_title"))

        sub_lbl = ctk.CTkLabel(self._whatif_card,
                                text=T("sliders_sub"),
                                font=ctk.CTkFont("Segoe UI", 10),
                                text_color=COLORS["text_secondary"])
        sub_lbl.pack(anchor="w", padx=18, pady=(0, 10))
        self._labels_secondary.append(sub_lbl)
        self._t_widgets.append((sub_lbl, "sliders_sub"))

        sliders_frame = ctk.CTkFrame(self._whatif_card, fg_color="transparent")
        sliders_frame.pack(fill="x", padx=18, pady=(0, 14))
        sliders_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="sl")

        # Area slider
        self._sl_area_lbl = self._slider_col(
            sliders_frame, T("sl_area"), 1000, 16200, 7420, 0,
            self._on_slider_change)

        # Bedrooms slider
        self._sl_bed_lbl = self._slider_col(
            sliders_frame, "🛏️ Bedrooms", 1, 6, 3, 1,
            self._on_slider_change)

        # Parking slider
        self._sl_park_lbl = self._slider_col(
            sliders_frame, "🚗 Parking", 0, 3, 1, 2,
            self._on_slider_change)

    def _slider_col(self, parent, label, from_, to, default, col, cmd, t_key=None):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=0, column=col, sticky="ew", padx=10)

        lbl_name = ctk.CTkLabel(frame, text=label,
                                 font=ctk.CTkFont("Segoe UI", 10),
                                 text_color=COLORS["text_secondary"])
        lbl_name.pack(anchor="w")
        self._labels_secondary.append(lbl_name)
        if t_key:
            self._t_widgets.append((lbl_name, t_key))

        # Value display
        val_var = ctk.StringVar(value=str(default))
        val_lbl = ctk.CTkLabel(frame, textvariable=val_var,
                                font=ctk.CTkFont("Segoe UI", 13, "bold"),
                                text_color=COLORS["text_primary"])
        val_lbl.pack(anchor="w")
        self._labels_primary.append(val_lbl)
        self._slider_labels.append(val_lbl)

        slider = ctk.CTkSlider(frame, from_=from_, to=to,
                                number_of_steps=int(to - from_),
                                button_color=COLORS["accent"],
                                button_hover_color=COLORS["accent_hover"],
                                progress_color=COLORS["accent"],
                                command=lambda v, vv=val_var, c=cmd: (
                                    vv.set(str(int(v))), c()))
        slider.set(default)
        slider.pack(fill="x", pady=(4, 0))

        # store reference
        if col == 0:
            self._sl_area   = slider
        elif col == 1:
            self._sl_bed    = slider
        else:
            self._sl_park   = slider

        return val_var

    def _on_slider_change(self):
        """Live recalculate when a slider moves — uses current form values."""
        try:
            area      = int(self._sl_area.get())
            bedrooms  = int(self._sl_bed.get())
            parking   = int(self._sl_park.get())

            # Try to grab remaining values from main form; fall back to defaults
            try:
                bathrooms = int(self.ent_bathrooms.get()) or 1
                stories   = int(self.ent_stories.get())   or 1
            except Exception:
                bathrooms, stories = 1, 1

            input_dict = build_input_dict(
                area, bedrooms, bathrooms, stories, parking,
                self.cb_mainroad.get(), self.cb_guestroom.get(),
                self.cb_basement.get(), self.cb_ac.get(),
                self.cb_prefarea.get(), self.cb_furnish.get())

            model        = load_model()
            raw          = model.predict(pd.DataFrame([input_dict]))[0]
            multiplier   = PROPERTY_MULTIPLIERS.get(self.cb_proptype.get(), 1.0)
            prediction   = raw * multiplier
            prop_label   = self.cb_proptype.get().split("  ")[1]
            self.lbl_result.configure(
                text=f"{prediction:,.0f} DH  ({prop_label})",
                text_color=COLORS["success"])
        except Exception:
            pass   # Silently ignore if form isn't ready yet

    # ══════════════════════════════════════════════════════════
    # ACTION BUTTONS
    # ══════════════════════════════════════════════════════════
    def _build_action_row(self):
        act = ctk.CTkFrame(self._scroll_inner, fg_color="transparent")
        act.pack(pady=20)

        self.calc_btn = ctk.CTkButton(
            act, text=T("btn_calculate"),
            width=200, height=44, corner_radius=10,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            command=self.make_prediction)
        self.calc_btn.grid(row=0, column=0, padx=10)
        self._t_widgets.append((self.calc_btn, "btn_calculate"))

        self.reset_btn = ctk.CTkButton(
            act, text=T("btn_reset"),
            width=120, height=44, corner_radius=10,
            fg_color=COLORS["toggle_bg"], hover_color=COLORS["card_border"],
            border_width=1, border_color=COLORS["danger"],
            text_color=COLORS["danger"],
            font=ctk.CTkFont("Segoe UI", 13, "bold"),
            command=self.reset_fields)
        self.reset_btn.grid(row=0, column=1, padx=10)
        self._t_widgets.append((self.reset_btn, "btn_reset"))

    # ══════════════════════════════════════════════════════════
    # CARD / ROW BUILDERS
    # ══════════════════════════════════════════════════════════
    def _card(self, parent, title, col, t_key=None):
        frame = ctk.CTkFrame(parent, fg_color=COLORS["card"],
                             corner_radius=14, border_width=1,
                             border_color=COLORS["card_border"])
        frame.grid(row=0, column=col, sticky="nsew",
                   padx=(0, 10) if col == 0 else (10, 0))
        frame.grid_columnconfigure(1, weight=1)
        self._cards.append(frame)

        lbl = ctk.CTkLabel(frame, text=title,
                           font=ctk.CTkFont("Segoe UI", 12, "bold"),
                           text_color=COLORS["accent"], fg_color="transparent")
        lbl.grid(row=0, column=0, columnspan=2, sticky="w",
                 padx=18, pady=(14, 10))
        self._card_titles.append(lbl)
        if t_key:
            self._t_widgets.append((lbl, t_key))
        return frame

    def _input_row(self, card, label, placeholder, row, t_key=None):
        lbl = ctk.CTkLabel(card, text=label, font=ctk.CTkFont("Segoe UI", 11),
                           text_color=COLORS["text_secondary"])
        lbl.grid(row=row + 1, column=0, sticky="w", padx=18, pady=5)
        self._labels_secondary.append(lbl)
        if t_key:
            self._t_widgets.append((lbl, t_key))

        entry = ctk.CTkEntry(card, placeholder_text=placeholder,
                             fg_color=COLORS["input_bg"],
                             border_color=COLORS["card_border"], border_width=1,
                             text_color=COLORS["text_primary"],
                             height=34, corner_radius=7,
                             font=ctk.CTkFont("Segoe UI", 11))
        entry.grid(row=row + 1, column=1, sticky="ew", padx=(8, 18), pady=5)
        self._entries.append(entry)
        return entry

    def _dropdown_row(self, card, label, values, row, t_key=None):
        lbl = ctk.CTkLabel(card, text=label, font=ctk.CTkFont("Segoe UI", 11),
                           text_color=COLORS["text_secondary"])
        lbl.grid(row=row + 1, column=0, sticky="w", padx=18, pady=5)
        self._labels_secondary.append(lbl)
        if t_key:
            self._t_widgets.append((lbl, t_key))

        combo = ctk.CTkComboBox(card, values=values,
                                fg_color=COLORS["input_bg"],
                                border_color=COLORS["card_border"], border_width=1,
                                button_color=COLORS["accent"],
                                button_hover_color=COLORS["accent_hover"],
                                dropdown_fg_color=COLORS["card"],
                                text_color=COLORS["text_primary"],
                                height=34, corner_radius=7,
                                font=ctk.CTkFont("Segoe UI", 11),
                                state="readonly")
        combo.grid(row=row + 1, column=1, sticky="ew", padx=(8, 18), pady=5)
        combo.set(values[0])
        self._combos.append(combo)
        return combo

    # ══════════════════════════════════════════════════════════
    # LANGUAGE SWITCH
    # ══════════════════════════════════════════════════════════
    def change_language(self, selection=None):
        global CURRENT_LANG
        sel = selection or self.lang_combo.get()
        CURRENT_LANG = LANG_OPTIONS.get(sel, "en")

        # Update all registered translatable widgets
        for widget, key in self._t_widgets:
            try:
                widget.configure(text=T(key))
            except Exception:
                pass

        # Theme button label also needs language update
        mode_key = "dark_mode" if self._mode == "light" else "light_mode"
        self.theme_btn.configure(
            text=f"{COLORS['toggle_icon']}  {T(mode_key)}")

        # Update property type options in main form
        new_types = [T("type_apartment"), T("type_house"), T("type_villa")]
        self.cb_proptype.configure(values=new_types)
        # Update PROPERTY_MULTIPLIERS keys to match new labels
        vals = list(PROPERTY_MULTIPLIERS.values())
        PROPERTY_MULTIPLIERS.clear()
        for k, v in zip(new_types, vals):
            PROPERTY_MULTIPLIERS[k] = v
        global PROPERTY_TYPES
        PROPERTY_TYPES = new_types
        self.cb_proptype.set(new_types[0])

        # RTL hint for Arabic
        if CURRENT_LANG == "ar":
            self._lbl_title.configure(justify="right")
            self._lbl_subtitle.configure(justify="right")
        else:
            self._lbl_title.configure(justify="left")
            self._lbl_subtitle.configure(justify="left")

    # THEME TOGGLE
    # ══════════════════════════════════════════════════════════
    def toggle_theme(self):
        self._mode = "light" if self._mode == "dark" else "dark"
        ctk.set_appearance_mode(self._mode)
        COLORS.update(THEMES[self._mode])

        self.configure(fg_color=COLORS["bg"])
        self._main_scroll.configure(fg_color=COLORS["bg"])
        self._accent_bar.configure(fg_color=COLORS["accent"])

        for card in self._cards:
            card.configure(fg_color=COLORS["card"], border_color=COLORS["card_border"])
        for lbl in self._card_titles:
            lbl.configure(text_color=COLORS["accent"])
        for lbl in self._labels_primary:
            lbl.configure(text_color=COLORS["text_primary"])
        for lbl in self._labels_secondary:
            lbl.configure(text_color=COLORS["text_secondary"])
        for entry in self._entries:
            entry.configure(fg_color=COLORS["input_bg"],
                            border_color=COLORS["card_border"],
                            text_color=COLORS["text_primary"])
        for combo in self._combos:
            combo.configure(fg_color=COLORS["input_bg"],
                            border_color=COLORS["card_border"],
                            button_color=COLORS["accent"],
                            button_hover_color=COLORS["accent_hover"],
                            dropdown_fg_color=COLORS["card"],
                            text_color=COLORS["text_primary"])

        self.lbl_result.configure(text_color=COLORS["success"])
        self.calc_btn.configure(fg_color=COLORS["accent"],
                                hover_color=COLORS["accent_hover"])
        self.reset_btn.configure(fg_color=COLORS["toggle_bg"],
                                 hover_color=COLORS["card_border"],
                                 border_color=COLORS["danger"],
                                 text_color=COLORS["danger"])

        for btn in (self.theme_btn, self.history_btn, self.compare_btn):
            btn.configure(fg_color=COLORS["toggle_bg"],
                          hover_color=COLORS["card_border"],
                          border_color=COLORS["toggle_border"],
                          text_color=COLORS["text_secondary"])

        self.importance_btn.configure(fg_color=COLORS["input_bg"],
                                      hover_color=COLORS["card_border"],
                                      border_color=COLORS["accent"],
                                      text_color=COLORS["accent"])

        # Sliders
        for sl in (self._sl_area, self._sl_bed, self._sl_park):
            sl.configure(button_color=COLORS["accent"],
                         button_hover_color=COLORS["accent_hover"],
                         progress_color=COLORS["accent"])

        self.theme_btn.configure(
            text=f"{COLORS['toggle_icon']}  {COLORS['toggle_label']}")

    # ══════════════════════════════════════════════════════════
    # PREDICTION
    # ══════════════════════════════════════════════════════════
    def make_prediction(self):
        try:
            area      = self.ent_area.get().strip()
            bedrooms  = self.ent_bedrooms.get().strip()
            bathrooms = self.ent_bathrooms.get().strip()
            stories   = self.ent_stories.get().strip()
            parking   = self.ent_parking.get().strip()

            if not all([area, bedrooms, bathrooms, stories, parking]):
                messagebox.showwarning("Missing Fields",
                    T("warn_missing"))
                return

            input_dict = build_input_dict(
                area, bedrooms, bathrooms, stories, parking,
                self.cb_mainroad.get(), self.cb_guestroom.get(),
                self.cb_basement.get(), self.cb_ac.get(),
                self.cb_prefarea.get(), self.cb_furnish.get())

            model        = load_model()
            raw          = model.predict(pd.DataFrame([input_dict]))[0]
            multiplier   = PROPERTY_MULTIPLIERS.get(self.cb_proptype.get(), 1.0)
            prediction   = raw * multiplier
            prop_label   = self.cb_proptype.get().split("  ")[1]  # strip emoji

            save_prediction(input_dict["area"], input_dict["bedrooms"],
                            float(prediction))
            self.lbl_result.configure(
                text=f"{prediction:,.0f} DH  ({prop_label})",
                text_color=COLORS["success"])

            # Sync sliders to form values
            self._sl_area.set(float(area))
            self._sl_area_lbl.set(str(int(float(area))))
            self._sl_bed.set(float(bedrooms))
            self._sl_bed_lbl.set(bedrooms)
            self._sl_park.set(float(parking))
            self._sl_park_lbl.set(parking)

        except ValueError:
            messagebox.showerror("Input Error",
                T("err_input"))
        except FileNotFoundError:
            messagebox.showerror("Model Error",
                T("err_model"))
        except Exception as e:
            messagebox.showerror("Unexpected Error", str(e))

    def reset_fields(self):
        for entry in (self.ent_area, self.ent_bedrooms, self.ent_bathrooms,
                      self.ent_stories, self.ent_parking):
            entry.delete(0, "end")
        self.cb_proptype.set(PROPERTY_TYPES[0])
        self.cb_mainroad.set("no");  self.cb_guestroom.set("no")
        self.cb_basement.set("no");  self.cb_ac.set("no")
        self.cb_prefarea.set("no");  self.cb_furnish.set("furnished")
        self.lbl_result.configure(text="— DH", text_color=COLORS["success"])
        self._sl_area.set(7420);  self._sl_area_lbl.set("7420")
        self._sl_bed.set(3);      self._sl_bed_lbl.set("3")
        self._sl_park.set(1);     self._sl_park_lbl.set("1")

    # ══════════════════════════════════════════════════════════
    # 🎯 FEATURE IMPORTANCE WINDOW
    # ══════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════
    # POPUP HELPER — stamps current theme onto any CTkToplevel
    # ══════════════════════════════════════════════════════════
    def _make_popup(self, title, geometry):
        """Create a themed CTkToplevel that always appears in front."""
        ctk.set_appearance_mode(self._mode)
        win = ctk.CTkToplevel(self)
        win.title(title)
        win.geometry(geometry)
        win.configure(fg_color=COLORS["bg"])
        # Force window to appear in front of the main window
        win.lift()
        win.attributes("-topmost", True)
        win.after(200, lambda: win.attributes("-topmost", False))
        win.focus_set()
        return win

    def _ttk_style(self):
        """Return a freshly configured ttk.Style for the current theme."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("History.Treeview",
                        background=COLORS["card"],
                        fieldbackground=COLORS["card"],
                        foreground=COLORS["text_primary"],
                        rowheight=32, borderwidth=0,
                        font=("Segoe UI", 10))
        style.configure("History.Treeview.Heading",
                        background=COLORS["input_bg"],
                        foreground=COLORS["accent"],
                        relief="flat",
                        font=("Segoe UI", 10, "bold"))
        style.map("History.Treeview",
                  background=[("selected", COLORS["accent"])],
                  foreground=[("selected", "#ffffff")])
        return style

    def show_importance(self):
        try:
            model = load_model()
        except FileNotFoundError:
            messagebox.showerror("Model Error",
                T("err_model"))
            return

        if not hasattr(model, "feature_importances_"):
            messagebox.showinfo("Not Supported", T("no_importance"))
            return

        importances = model.feature_importances_
        feature_names = list(FEATURE_LABELS.keys())
        labels = [FEATURE_LABELS.get(f, f) for f in feature_names]

        # Sort descending
        pairs = sorted(zip(importances, labels), reverse=True)
        vals, lbls = zip(*pairs)

        win = self._make_popup("🎯  Feature Importance", "640x500")

        ctk.CTkLabel(win, text="🎯  Feature Importance",
                     font=ctk.CTkFont("Segoe UI", 18, "bold"),
                     text_color=COLORS["text_primary"]).pack(
            anchor="w", padx=24, pady=(20, 2))
        ctk.CTkLabel(win, text="Which inputs influence the price prediction most?",
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=COLORS["text_secondary"]).pack(
            anchor="w", padx=24, pady=(0, 12))

        fig, ax = plt.subplots(figsize=(6.2, 4))
        fig.patch.set_facecolor(COLORS["plot_bg"])
        ax.set_facecolor(COLORS["plot_bg"])

        colors = [COLORS["accent"] if v == max(vals) else COLORS["card_border"]
                  for v in vals]
        bars = ax.barh(lbls[::-1], vals[::-1], color=colors[::-1],
                       height=0.6, edgecolor="none")

        # Value labels on bars
        for bar, val in zip(bars, vals[::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center",
                    color=COLORS["plot_fg"], fontsize=8)

        ax.set_xlabel("Importance Score", color=COLORS["text_secondary"], fontsize=9)
        ax.tick_params(colors=COLORS["plot_fg"], labelsize=9)
        ax.spines[:].set_color(COLORS["card_border"])
        ax.xaxis.label.set_color(COLORS["text_secondary"])
        fig.tight_layout(pad=1.5)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=(0, 16))

    # ══════════════════════════════════════════════════════════
    # ↔️ COMPARE 2 HOUSES WINDOW
    # ══════════════════════════════════════════════════════════
    def show_compare(self):
        win = self._make_popup("↔️  Compare Two Houses", "920x600")

        # ══ TOP BAR — always visible ══════════════════════════
        top = ctk.CTkFrame(win, fg_color=COLORS["card"],
                           corner_radius=12, border_width=1,
                           border_color=COLORS["card_border"])
        top.pack(fill="x", padx=24, pady=(18, 10))

        # Title
        title_left = ctk.CTkFrame(top, fg_color="transparent")
        title_left.pack(side="left", padx=18, pady=14)
        ctk.CTkLabel(title_left, text="↔️  Compare Two Houses",
                     font=ctk.CTkFont("Segoe UI", 16, "bold"),
                     text_color=COLORS["text_primary"]).pack(anchor="w")
        ctk.CTkLabel(title_left,
                     text="Fill in both panels below, then click Compare Now.",
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=COLORS["text_secondary"]).pack(anchor="w")

        # Results + button on the right
        right = ctk.CTkFrame(top, fg_color="transparent")
        right.pack(side="right", padx=18, pady=10)

        res_row = ctk.CTkFrame(right, fg_color="transparent")
        res_row.pack(anchor="e")

        lbl_a = ctk.CTkLabel(res_row, text="🏠  — DH",
                              font=ctk.CTkFont("Segoe UI", 16, "bold"),
                              text_color=COLORS["success"])
        lbl_a.pack(side="left", padx=(0, 24))

        lbl_b = ctk.CTkLabel(res_row, text="🏡  — DH",
                              font=ctk.CTkFont("Segoe UI", 16, "bold"),
                              text_color=COLORS["success"])
        lbl_b.pack(side="left")

        verdict_lbl = ctk.CTkLabel(right, text="",
                                    font=ctk.CTkFont("Segoe UI", 10, "bold"),
                                    text_color=COLORS["accent"])
        verdict_lbl.pack(anchor="e", pady=(4, 6))

        def do_compare():
            results = []
            for label, fields in (("House A", fields_a), ("House B", fields_b)):
                numeric_keys = ["area", "bedrooms", "bathrooms", "stories", "parking"]
                missing = [k for k in numeric_keys if not fields[k].get().strip()]
                if missing:
                    messagebox.showwarning("Missing Fields",
                        f"{label}: please fill in — {', '.join(missing)}")
                    return
                try:
                    d = build_input_dict(
                        fields["area"].get().strip(),
                        fields["bedrooms"].get().strip(),
                        fields["bathrooms"].get().strip(),
                        fields["stories"].get().strip(),
                        fields["parking"].get().strip(),
                        fields["mainroad"].get(), fields["guestroom"].get(),
                        fields["basement"].get(), fields["ac"].get(),
                        fields["prefarea"].get(), fields["furnish"].get())
                    m          = load_model()
                    raw        = m.predict(pd.DataFrame([d]))[0]
                    multiplier = PROPERTY_MULTIPLIERS.get(fields["proptype"].get(), 1.0)
                    results.append(raw * multiplier)
                except FileNotFoundError:
                    messagebox.showerror("Model Error",
                        T("err_model"))
                    return
                except ValueError:
                    messagebox.showerror("Input Error",
                        f"{label}: numeric fields must contain valid numbers.")
                    return
                except Exception as e:
                    messagebox.showerror("Error", f"{label}: {e}")
                    return

            if len(results) < 2:
                return

            pa, pb   = results
            type_a   = fields_a["proptype"].get().split("  ")[1]
            type_b   = fields_b["proptype"].get().split("  ")[1]
            lbl_a.configure(text=f"🏠  {pa:,.0f} DH  ({type_a})", text_color=COLORS["success"])
            lbl_b.configure(text=f"🏡  {pb:,.0f} DH  ({type_b})", text_color=COLORS["success"])

            diff = abs(pa - pb)
            if pa > pb:
                verdict_lbl.configure(
                    text=f"🏆 House A is higher by {diff:,.0f} DH",
                    text_color=COLORS["accent"])
                lbl_a.configure(text_color=COLORS["accent"])
                lbl_b.configure(text_color=COLORS["text_secondary"])
            elif pb > pa:
                verdict_lbl.configure(
                    text=f"🏆 House B is higher by {diff:,.0f} DH",
                    text_color=COLORS["accent"])
                lbl_b.configure(text_color=COLORS["accent"])
                lbl_a.configure(text_color=COLORS["text_secondary"])
            else:
                verdict_lbl.configure(
                    text="🤝 Both houses have equal value!",
                    text_color=COLORS["success"])

        ctk.CTkButton(right, text="↔️  Compare Now",
                      width=180, height=38, corner_radius=8,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                      font=ctk.CTkFont("Segoe UI", 12, "bold"),
                      command=do_compare).pack(anchor="e")

        # ══ SCROLLABLE PANELS ═════════════════════════════════
        scroll = ctk.CTkScrollableFrame(win, fg_color=COLORS["bg"])
        scroll.pack(fill="both", expand=True, padx=24, pady=(0, 14))
        scroll.grid_columnconfigure((0, 1), weight=1, uniform="cmp")

        fields_a = self._compare_panel(scroll, "🏠  House A", 0)
        fields_b = self._compare_panel(scroll, "🏡  House B", 1)

    def _compare_panel(self, parent, title, col):
        frame = ctk.CTkFrame(parent, fg_color=COLORS["card"],
                             corner_radius=14, border_width=1,
                             border_color=COLORS["card_border"])
        frame.grid(row=0, column=col, sticky="nsew",
                   padx=(0, 8) if col == 0 else (8, 0))
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text=title,
                     font=ctk.CTkFont("Segoe UI", 12, "bold"),
                     text_color=COLORS["accent"]).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(12, 8))

        fields = {}

        # ── Property type selector ──
        ctk.CTkLabel(frame, text="🏷️ Property Type", font=ctk.CTkFont("Segoe UI", 10),
                     text_color=COLORS["text_secondary"]).grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 3))
        cb_pt = ctk.CTkComboBox(frame, values=PROPERTY_TYPES,
                                fg_color=COLORS["input_bg"],
                                border_color=COLORS["card_border"], border_width=1,
                                button_color=COLORS["accent"],
                                button_hover_color=COLORS["accent_hover"],
                                dropdown_fg_color=COLORS["card"],
                                text_color=COLORS["text_primary"],
                                height=30, corner_radius=6,
                                font=ctk.CTkFont("Segoe UI", 10),
                                state="readonly")
        cb_pt.grid(row=0, column=1, sticky="ew", padx=(6, 14), pady=(12, 3))
        cb_pt.set(PROPERTY_TYPES[0])
        fields["proptype"] = cb_pt

        numeric = [("area",      T("sl_area"),  "7420"),
                   ("bedrooms",  "🛏️ Bedrooms",     "4"),
                   ("bathrooms", "🚿 Bathrooms",    "2"),
                   ("stories",   "🏢 Stories",      "3"),
                   ("parking",   "🚗 Parking",      "2")]

        entries_in_order = []
        for i, (key, lbl_text, ph) in enumerate(numeric, start=1):  # row 0 = proptype
            ctk.CTkLabel(frame, text=lbl_text, font=ctk.CTkFont("Segoe UI", 10),
                         text_color=COLORS["text_secondary"]).grid(
                row=i, column=0, sticky="w", padx=14, pady=3)
            e = ctk.CTkEntry(frame, placeholder_text=ph,
                             fg_color=COLORS["input_bg"],
                             border_color=COLORS["card_border"], border_width=1,
                             text_color=COLORS["text_primary"],
                             height=30, corner_radius=6,
                             font=ctk.CTkFont("Segoe UI", 10))
            e.grid(row=i, column=1, sticky="ew", padx=(6, 14), pady=3)
            fields[key] = e
            entries_in_order.append(e)

        # Enter key moves to next entry within this panel
        for i, w in enumerate(entries_in_order):
            nxt = entries_in_order[i + 1] if i + 1 < len(entries_in_order) else None
            if nxt:
                w.bind("<Return>", lambda e, n=nxt: n.focus_set())

        yn = ["yes", "no"]
        dropdowns = [("mainroad",  "🛣️ Main Road",  yn),
                     ("guestroom", "🛋️ Guest Room", yn),
                     ("basement",  "🏚️ Basement",   yn),
                     ("ac",        "❄️ Air Con",     yn),
                     ("prefarea",  "⭐ Pref. Area",  yn),
                     ("furnish",   "🪑 Furnishing",
                      ["furnished", "semi-furnished", "unfurnished"])]

        for i, (key, lbl_text, vals) in enumerate(dropdowns, start=6):
            ctk.CTkLabel(frame, text=lbl_text, font=ctk.CTkFont("Segoe UI", 10),
                         text_color=COLORS["text_secondary"]).grid(
                row=i, column=0, sticky="w", padx=14, pady=3)
            cb = ctk.CTkComboBox(frame, values=vals,
                                 fg_color=COLORS["input_bg"],
                                 border_color=COLORS["card_border"], border_width=1,
                                 button_color=COLORS["accent"],
                                 button_hover_color=COLORS["accent_hover"],
                                 dropdown_fg_color=COLORS["card"],
                                 text_color=COLORS["text_primary"],
                                 height=30, corner_radius=6,
                                 font=ctk.CTkFont("Segoe UI", 10),
                                 state="readonly")
            cb.grid(row=i, column=1, sticky="ew", padx=(6, 14), pady=3)
            cb.set(vals[0])
            fields[key] = cb

        return fields

    # ══════════════════════════════════════════════════════════
    # 📋 HISTORY WINDOW
    # ══════════════════════════════════════════════════════════
    def show_history(self):
        win = self._make_popup("📋  Prediction History", "700x480")

        ctk.CTkLabel(win, text="📋  Prediction History",
                     font=ctk.CTkFont("Segoe UI", 18, "bold"),
                     text_color=COLORS["text_primary"]).pack(
            anchor="w", padx=24, pady=(20, 2))
        ctk.CTkLabel(win, text="All saved predictions from this session and past runs.",
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=COLORS["text_secondary"]).pack(
            anchor="w", padx=24, pady=(0, 14))

        self._ttk_style()

        columns = ("Date", "Area (sqft)", "Bedrooms", "Price (DH)")
        tree = ttk.Treeview(win, columns=columns, show="headings",
                            style="History.Treeview", selectmode="browse")
        for col, w in zip(columns, [180, 110, 90, 140]):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center")

        tree.tag_configure("odd",  background=COLORS["row_odd"])
        tree.tag_configure("even", background=COLORS["row_even"])

        rows = get_all_history()
        for i, (date, area, rooms, price) in enumerate(rows):
            tree.insert("", "end",
                        values=(date, f"{area:,.0f}", int(rooms), f"{price:,.0f}"),
                        tags=("odd" if i % 2 else "even",))

        sb = ctk.CTkScrollbar(win, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y", padx=(0, 14), pady=14)
        tree.pack(fill="both", expand=True, padx=(24, 0), pady=(0, 14))

        ctk.CTkLabel(win, text=f"{len(rows)} record(s) found",
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=COLORS["text_secondary"]).pack(
            anchor="e", padx=24, pady=(0, 14))


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    app = RealEstateApp()
    app.mainloop()