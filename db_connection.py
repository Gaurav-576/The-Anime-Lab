import mysql.connector

anime_db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="pass123"
)


anime_cursor = anime_db.cursor()
anime_cursor.execute("CREATE DATABASE IF NOT EXISTS anime_db")
print("Database created or already exists")
anime_cursor.execute("USE anime_db")
anime_cursor.execute("SHOW DATABASES")

print("Databases:")
for db in anime_cursor:
    print(db)