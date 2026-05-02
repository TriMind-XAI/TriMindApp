import bcrypt
import sqlite3

DB = "users.db"

def init_db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT,
        google_id TEXT
    )""")
    con.commit()
    con.close()

def register_user(username, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    con = sqlite3.connect(DB)
    try:
        con.execute("INSERT INTO users VALUES (?,?,?)", (username, hashed, None))
        con.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # username taken
    finally:
        con.close()

def verify_user(username, password):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT password_hash FROM users WHERE username=?", (username,)).fetchone()
    con.close()
    if row:
        return bcrypt.checkpw(password.encode(), row[0])
    return False