import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            area REAL,
            rooms INTEGER,
            predicted_price REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(area, rooms, price):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO history (date, area, rooms, predicted_price)
        VALUES (?, ?, ?, ?)
    ''', (date_now, area, rooms, price))
    conn.commit()
    conn.close()

# هذه هي الدالة التي يشتكي بايثون من عدم وجودها
def get_all_history():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute("SELECT date, area, rooms, predicted_price FROM history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows