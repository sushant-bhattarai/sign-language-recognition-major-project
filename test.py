import sqlite3
import os

#
# def get_pred_text_from_db(pred_class):
# 	conn = sqlite3.connect("gesture_db.db")
# 	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
# 	cursor = conn.execute(cmd)
# 	print(cursor)
# 	for row in cursor:
# 		print (row)
#
# get_pred_text_from_db(17)


# var = chr(ord('@')+26)
# print((var))

def store_in_db(ges_id, ges_name):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (%s, \'%s\')" % (ges_id, ges_name)
    # try:
    conn.execute(cmd)
    # except sqlite3.IntegrityError:
    # 	choice = input("g_id already exists. Want to change the record? (y/n): ")
    # 	if choice.lower() == 'y':
    # 		cmd = "UPDATE gesture SET g_name = \'%s\' WHERE g_id = %s" % (g_name, g_id)
    # 		conn.execute(cmd)
    # 	else:
    # 		print("Doing nothing...")
    # 		return
    conn.commit()

if not os.path.exists("gesture_db.db"):
    conn = sqlite3.connect("gesture_db.db")
    create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT " \
                       "NOT NULL ) "
    conn.execute(create_table_cmd)
    conn.commit()

    for i in range(0, 26):
        j = chr(ord('@') + (i + 1))
        store_in_db(i, j)

    for i in range(26, 36):
        store_in_db(i, (i - 26))




