import sqlite3

def migrate():
    con = sqlite3.connect('instance/medora.db')
    cursor = con.cursor()
    
    # User Table changes
    try:
        cursor.execute("ALTER TABLE user ADD COLUMN working_hours_start VARCHAR(10) DEFAULT '09:00'")
        print("Added working_hours_start to user")
    except sqlite3.OperationalError as e:
        print("working_hours_start already exists or error:", e)

    try:
        cursor.execute("ALTER TABLE user ADD COLUMN working_hours_end VARCHAR(10) DEFAULT '17:00'")
        print("Added working_hours_end to user")
    except sqlite3.OperationalError as e:
        print("working_hours_end already exists or error:", e)

    try:
        cursor.execute("ALTER TABLE user ADD COLUMN is_approved BOOLEAN DEFAULT 1")
        print("Added is_approved to user")
    except sqlite3.OperationalError as e:
        print("is_approved already exists or error:", e)

    try:
        cursor.execute("ALTER TABLE user ADD COLUMN age INTEGER DEFAULT NULL")
        print("Added age to user")
    except sqlite3.OperationalError as e:
        print("age already exists or error:", e)

    try:
        cursor.execute("ALTER TABLE user ADD COLUMN gender VARCHAR(20) DEFAULT NULL")
        print("Added gender to user")
    except sqlite3.OperationalError as e:
        print("gender already exists or error:", e)

    # Appointment Table changes
    try:
        cursor.execute("ALTER TABLE appointment ADD COLUMN appointment_date DATE DEFAULT NULL")
        print("Added appointment_date to appointment")
    except sqlite3.OperationalError as e:
        print("appointment_date already exists or error:", e)

    try:
        cursor.execute("ALTER TABLE appointment ADD COLUMN appointment_time TIME DEFAULT NULL")
        print("Added appointment_time to appointment")
    except sqlite3.OperationalError as e:
        print("appointment_time already exists or error:", e)

    con.commit()
    con.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
