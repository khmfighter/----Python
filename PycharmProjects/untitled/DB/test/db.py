import MySQLdb
conn = MySQLdb.Connect(
    host = '127.0.0.1',
    port = 3306,
    user = 'root',
    passwd = '123456',
    db = 'test',
    charset = 'utf8')


cursor = conn.cursor()

def addUser(username, password):
    sql = "insert into username(username,password) values('%s','%s')"%(username,password)
    cursor.execute(sql)
    conn.commit()
    conn.close()


addUser('oumingwang','123456')