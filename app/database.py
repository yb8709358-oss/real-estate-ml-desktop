import sqlite3
from datetime import datetime

def init_db():
    # الاتصال بقاعدة البيانات (سيتم إنشاؤها تلقائياً إذا لم تكن موجودة)
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    # إنشاء جدول التاريخ
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
    # إدخال البيانات الجديدة في الجدول
    cursor.execute('''
        INSERT INTO history (date, area, rooms, predicted_price)
        VALUES (?, ?, ?, ?)
    ''', (date_now, area, rooms, price))
    conn.commit()
    conn.close()