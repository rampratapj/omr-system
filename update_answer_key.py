import sqlite3

# Path to your SQLite DB file
db_path = "omr_system.db"

# Your full correct answer key (replace with your real 60 answers)
answers = [
    0,1,2,3,0,1,2,3,1,2,0,1,2,3,0,1,2,3,0,1,
    2,3,0,1,2,3,1,2,0,1,2,3,0,1,2,3,0,1,2,3,
    1,2,0,1,2,3,0,1,2,3,0,1,2,3,1,2,0,1,2,3
]

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE answer_keys
        SET answers = ?, total_questions = ?
        WHERE key_name = 'OCT_ANSWER_KEY'
    """, [str(answers), len(answers)])

    conn.commit()
    conn.close()
    print(f"✅ Updated OCT_ANSWER_KEY with {len(answers)} answers successfully!")

except Exception as e:
    print(f"❌ Error updating key: {e}")
