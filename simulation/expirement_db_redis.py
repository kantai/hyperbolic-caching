try:
    import psycopg2
except:
    try:
        import psycopg2cffi as psycopg2
    except:
        raise Exception("must install either psycopg2 or psycopg2cffi")
import time
from numpy import random
from instrument_redis import RedisSim

SQL_GET = """
SELECT value FROM %s WHERE key = '%%s';
"""

SQL_INS_CONDITIONAL = """
INSERT INTO %s
    (key, value)
SELECT %%s, %%s
WHERE
    NOT EXISTS (
        SELECT key FROM %s WHERE key = %%s
    );
"""

SQL_INS = """
INSERT INTO %s
    (key, value)
VALUES (%%s, %%s);
"""

SQL_CREATE_TABLE = """
CREATE TABLE %s ( key varchar(64) PRIMARY KEY, 
  value char(%s));
"""

class BackedRedisSim(RedisSim):
    def __init__(self, k, workload_driver, table_name = None,
                 pg_connection_str = "dbname=blanks user=blanks host=/tmp port=5435",
                 redis_args = None, measure_tput = True,
                 *args, **kwargs):
        if table_name == None:
            table_name = workload_driver.name
        self.table_name = table_name

        self.SQL_GET = SQL_GET % table_name
        self.SQL_INS = SQL_INS % table_name
        self.SQL_INS_CONDITIONAL = SQL_INS_CONDITIONAL % (table_name, table_name)

        self.blob_size_in_B = 32
        self.evaluate_requests = False
        self.R = random.RandomState(1337)

        self.conn = psycopg2.connect(pg_connection_str)
        self.cursor = self.conn.cursor()

        self.measure_tput = measure_tput
        if self.measure_tput:
            self.throughput = (0, 0)

        super(BackedRedisSim, self).__init__(k, workload_driver, redis_args = redis_args)

    def close(self):
        super(BackedRedisSim, self).close()
        self.cursor.close()
        self.conn.close()

    def _get_from_db(self, item):
        self.cursor.execute(self.SQL_GET, (item, ))
        return self.cursor.fetchone()

    def _insert(self, ins_q, item):
            blob = ''.join([chr(i) for i in self.R.random_integers(97, 122, int(self.blob_size_in_B)) ])
            if ins_q == self.SQL_INS:
                data = (str(item), blob)
            elif ins_q == self.SQL_INS_CONDITIONAL:
                data = (str(item), blob, str(item))
            self.cursor.execute(ins_q, data)

    def create_table(self):
        # what we do is run the whole driver, and insert items into the table.
        self.cursor.execute("DROP TABLE IF EXISTS %s ;" % self.table_name)
        self.cursor.execute(SQL_CREATE_TABLE % (self.table_name, self.blob_size_in_B))


        if self.evaluate_requests == False:
            print "inserting %d items into %s" % (self.driver.max_item, self.table_name)
            for item in range(0, self.driver.max_item):
                self._insert(self.SQL_INS, item)                
        else:
            print "inserting %e requests into %s" % (self.evaluate_requests, self.table_name)
            for i in range(0, self.evaluate_requests):
                if i % (self.evaluate_requests / 100) == 0:
                    print "%d" % i
                get__ = self.driver.sample_item_w_cost()
                if get__ == -1:
                    break
                item, cost = self.driver.sample_item_w_cost()[:2]
                self._insert(self.SQL_INS_CONDITIONAL, item)
        self.conn.commit()

    def do_cache(self, item, cost):
        blob = self._get_from_db(item)
        super(BackedRedisSim, self).do_cache(item, cost, blob = blob)

    def simulate_requests(self, num_requests, progress_updater = None):
        if self.measure_tput:
            start = time.clock()
            start_r = self.NUM_REQUESTS
        
        super(BackedRedisSim, self).simulate_requests(num_requests, progress_updater)
        
        if self.measure_tput:
            end = time.clock() 
            end_r = self.NUM_REQUESTS
            prev_t, prev_reqs = self.throughput
            self.throughput = (prev_t + end - start, prev_reqs + end_r - start_r)
        
