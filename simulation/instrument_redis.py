from inspect import getargspec

from numpy import random
from subprocess import Popen, STDOUT, call

REDIS_MAX_MEMORY = 10 * 10**6

class RedisSim(object):

    def __init__(self, k, workload_driver, redis_args = None, ident = 0):
        """
        type_dist_f : returns int corresponding to the sampled request type
        rid_dist_f  : returns int corresponding to the sampled request id
        eviction
        """
        
        port = 63790 + ident

        if redis_args != None:
            import redis
            # let's do this.
            call(["mkdir", "-p", "/tmp/redis_dir_%d" % port])
            redis_dir = "/disk/local/blanks/cost-cache.git/redis-hyper"
            self.redis_server = Popen([redis_dir + "/src/redis-server",
                                       redis_dir + "/redis.conf",
                                       "--port", "%d" % port,
                                       "--dir", "/tmp/redis_dir_%d/" % port],
                                      stderr = STDOUT,
                                      stdout = open("/tmp/redis_p_%d" % port, "w"))
            call(["sleep", "4"])
            self.redis = redis.StrictRedis(host='localhost', port=port, db=0)
            self.redis.flushall()
            self.redis.config_set("maxobjects", k)

        else:
            self.redis_server = False

        self.driver = workload_driver


        self.cur_cache = 0
        blob_size_in_B = 32

        R = random.RandomState(1337)

        self.strategy = object()
        self.k = k

        blob =  ''.join([chr(i) for i in R.random_integers(97, 122, int(blob_size_in_B)) ])

        self.blob = blob
        
        self.MISSES = 0.0
        self.NUM_REQUESTS = 0

        self.measuring = False

    def close(self):
        if self.redis_server:
            self.redis_server.terminate()
    
    def is_cached(self, val):
        return bool(self.redis.get("%d" % val))

    def evict(self):
        pass

    def do_cache(self, item, cost, blob = None):
        if blob == None:
            blob = self.blob
        self.redis.set("%d" % item, blob)

    def warmup(self):
        self.simulate_requests(55 * self.k)

    def simulate_requests(self, num_requests, progress_updater = None):
        thrown = False
        
        i = 0
        while num_requests == -1 or i < num_requests:
            if progress_updater is not None:
                progress_updater(i)
            i += 1

            get__ = self.driver.sample_item_w_cost()
            if get__ == -1:
                break
            item, cost = get__

            if cost > 1 and not thrown:
                print "E: %s" % self.driver.name
                thrown = True
                continue

            if self.measuring:
                self.NUM_REQUESTS += 1
            if not self.is_cached(item):
                self.do_cache(item, 1)
                if self.measuring:
                    self.MISSES += 1.0
                else:
                    if self.redis.info()['evicted_keys'] > 0:
                        self.measuring = True
