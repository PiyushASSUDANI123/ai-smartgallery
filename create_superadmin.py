import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.database import get_db, hash_password

def update_superadmin():
    username = "piyush"
    password = "1234"
    hashed_pw = hash_password(password)

    with get_db() as wrapper:
        conn = wrapper.conn
        with conn.cursor() as cursor:
            cursor.execute("UPDATE users SET role = 'superadmin', password_hash = %s WHERE username = %s", (hashed_pw, username))
            conn.commit()
            print(f"Updated {username} to superadmin with new password.")

if __name__ == "__main__":
    update_superadmin()
