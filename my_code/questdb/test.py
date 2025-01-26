import psycopg2 as pg
import time

if __name__ == '__main__':
    conn_str = 'postgresql://admin:quest@127.0.0.1:8812/qdb'
    with pg.connect(conn_str) as connection:

        # Open a cursor to perform database operations

        with connection.cursor() as cur:
            # Query the database and obtain data as Python objects.

            cur.execute('SELECT * FROM btc_trades;')
            records = cur.fetchall()
            for row in records:
                print(row)