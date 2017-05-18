import gc

from inspect import getargspec
from numpy.random import RandomState

from workloads import cost_driver

from strategies import arc

START_NOW = -1
START_AFTER_WARMUP = 0
START_AFTER_FIRST_EVICT = 1

class Simulation(object):

    def __init__(self, k, workload_driver, eviction_strategy, start_counting = START_AFTER_FIRST_EVICT, **kwargs):
        """
        type_dist_f : returns int corresponding to the sampled request type
        rid_dist_f  : returns int corresponding to the sampled request id
        eviction
        """
        self.cache = {}
        self.driver = workload_driver

        self.cur_cache = 0
        self.k = k

        self.strategy = eviction_strategy

        # Special Strategies (combine evict and add):
        
        self.evict_and_add_same_time = isinstance(self.strategy, arc.EvictsAndAdds)

        self.pass_popularity = ((not self.evict_and_add_same_time) and
                                'popularity' in getargspec(self.strategy.add_node).args)

        # CACHE BY SIZE
        self.cache_by_size = False
        if hasattr(self.driver, "cache_by_size"):
            self.cache_by_size = self.driver.cache_by_size
        self.pass_size = False
        if self.cache_by_size:
            self.pass_size = ((not isinstance(self.strategy, arc.EvictsAndAdds)) and
                              ('size' in getargspec(self.strategy.add_node).args))
        self.handleSizeReturn = isinstance(self.driver, cost_driver.ReturnsSize)

        # EXPIRATION
        self.expiry_method = cost_driver.NO_EXPIRES
        if hasattr(self.driver, "expiring"):
            self.expiry_method = self.driver.expiring
        if self.expiry_method == cost_driver.EXPIRES_EVERY:
            self.expires_every = self.driver.expires_every
        elif self.expiry_method == cost_driver.GET_EXPIRY:
            self.get_expiry = self.driver.get_expiry
        
        # give strategy the cache size
        if 'cache_size' in self.strategy.__dict__:
            self.strategy.cache_size = self.k


        self.expiring = self.expiry_method != cost_driver.NO_EXPIRES

        self.pass_expiry = (self.expiring and 
                            (not isinstance(self.strategy, arc.EvictsAndAdds)) and
                            ('expires_at' in getargspec(self.strategy.add_node).args))
        self.pass_cost_class = ((not isinstance(self.strategy, arc.EvictsAndAdds)) and
                                ('cost_class' in getargspec(self.strategy.add_node).args))

        self.time = 0
        self.expire_cache = {}

        # Statistics
        if start_counting == START_AFTER_WARMUP or start_counting == START_AFTER_FIRST_EVICT:
            self.counting = False
            self.run_warmup = True
        else:
            self.counting = True
            self.run_warmup = False
        self.start_counting = start_counting

        self.first_eviction = True
        self.finished = False

        self.MISSES = 0
        self.N_MISSES = 0
        self.NUM_REQUESTS = 0

        if 'track_total_priority' in kwargs:
            self.track_total_priority = kwargs['track_total_priority']
        else:
            self.track_total_priority = 0
        self.priority_track = []

        if 'moving_window_missrates' in kwargs:
            self.moving_window_missrates = kwargs['moving_window_missrates']
        else:
            self.moving_window_missrates = False

        self.reset_moving_window()

        self.memory_profiling = True

    def get_cache_total_priority(self):
#        if len(self.cache) == 0:
#            return 0.0
#        sample = self.sampler.randint(0, len(self.cache), 100)

#        return sum([ self.strategy.objective_f( node ) for ix, node 
#                     in enumerate(self.cache.values()) if ix in sample ])
        return sum([ self.strategy.objective_f( node ) for ix, node 
                     in enumerate(self.cache.values())])


    def is_cached(self, val):
        if val in self.cache:
            node = self.cache[val]
            if self.cache_by_size:
                size = node.size
            rval = self.strategy.touch(node)
            if rval:
                if self.cache_by_size:
                    rval.size = size
                self.cache[val] = rval
            return True
        return False

    def is_expired(self, val):
        if self.expiring:
            expires_at = self.expire_cache[val]
            # already expired
            if expires_at == 0:
                return True
            if expires_at == -1:
                # never expires!
                return False
            # expires now
            if self.time >= expires_at:
                self.expire_cache[val] = 0
                return True
        else:
            return False

    def __inner_evict(self, evicted):
        if self.first_eviction:
            self.counting = True

        my_node = self.cache[evicted.value]
        del self.cache[evicted.value]
        if self.expiring:
            del self.expire_cache[evicted.value]

        if self.cache_by_size:
            self.cur_cache -= my_node.size
        else:
            self.cur_cache -= 1
        return evicted

    def evict(self):
        evicted = None
        while evicted == None:
            evicted = self.strategy.evict_node()
            evicted = self.__inner_evict(evicted)
        
        return evicted.value

    def do_cache(self, item, cost, expiry, size):
        if self.expiring:
            self.update_expiry(item, expiry)
        if self.cache_by_size and size >= self.k: # uncacheable!
            return

        if self.evict_and_add_same_time:
            if self.cache_by_size:
                raise NotImplementedError()
            evicted_node, new_node = self.strategy.evict_and_add(item, cost)
            if evicted_node:
                self.__inner_evict(evicted_node)
        else:
            if self.cache_by_size:
                while self.cur_cache + size >= self.k:
                    self.evict()
            else:
                if self.cur_cache >= self.k:
                    self.evict()

            new_node = self.strategy.add_node(item, cost, **self.get_add_node_args(item, size))
            

        if self.cache_by_size:
            self.cur_cache += size
            new_node.size = size
        else:
            self.cur_cache += 1
        
        self.cache[item] = new_node


    def handle_cache_expire(self, item, cost, expiry = None):
        self.update_expiry(item, expiry)
        if self.pass_expiry:
            self.strategy.set_expiry(self.cache[item], self.expire_cache[item])

    def update_expiry(self, item, expiry = None):
        if expiry == None:
            if self.expiry_method == cost_driver.EXPIRES_EVERY:
                expiry = self.time + self.expires_every
            elif self.expiry_method == cost_driver.GET_EXPIRY:
                expiry = self.get_expiry(item)
                if expiry != -1:
                    expiry += self.time 
            else:
                raise NotImplemented("expire setting failure.")
        self.expire_cache[item] = expiry

    def get_add_node_args(self, item, size):
        d = {}
        if self.pass_popularity:
            d["popularity"] = self.driver.get_item_pop(item)
        if self.pass_cost_class:
            d["cost_class"] = self.driver.get_cost_class(item)
        if self.pass_expiry:
            d["expires_at"] = self.expire_cache[item]
        if self.pass_size:
            d["size"] = size
        return d

    def warmup(self):
        self.simulate_requests(10**5)

        if self.start_counting == START_AFTER_WARMUP:
            self.counting = True


    def reset_moving_window(self):
        if self.moving_window_missrates:
            self.moving_window = []
            self.moving_window_total = 0.0
            self.moving_window_emit = []

    def close(self):
        pass
        
    def moving_window_handle(self, f_cost):
        window_sz = self.moving_window_missrates
        if len(self.moving_window) >= window_sz:
            if self.time % 10 == 0:
                self.moving_window_emit.append(self.moving_window_total / window_sz)
            out = self.moving_window.pop(0)
            self.moving_window_total -= out
        self.moving_window.append(f_cost)
        self.moving_window_total += f_cost

    def simulate_requests(self, num_requests, progress_updater = None):
        iter_index = 0
        if self.finished:
            return
        while num_requests == -1 or iter_index < num_requests:
            if iter_index > 1 and iter_index % (2*10**6) == 0:
                gc.collect()
            iter_index += 1

            if iter_index % 10000 == 0 and progress_updater is not None:
                progress_updater(iter_index, self.POLICY_NUMBER)

            self.time += 1
            req_tuple = self.driver.sample_item_w_cost()
            if req_tuple == -1:
                self.finished = True
                return
            item, cost = req_tuple[:2]
            remainder = req_tuple[2:]
            expiry = None
            if self.expiry_method == cost_driver.RETURNS_EXPIRY:
                expiry = remainder[0]
                remainder = remainder[1:]
            size = 1
            if self.handleSizeReturn:
                size = remainder[0]
                remainder = remainder[1:]

            if self.track_total_priority > 0 and i % self.track_total_priority == 0:
                self.priority_track.append(self.get_cache_total_priority())

            if self.counting:
                self.NUM_REQUESTS += 1
            
            is_cached = self.is_cached(item)

            if is_cached == False:
                self.do_cache(item, cost, expiry, size)
                if self.counting:
                    self.MISSES += cost
                    self.N_MISSES += 1
                if self.moving_window_missrates:
                    self.moving_window_handle(cost)
            elif self.is_expired(item):
                if self.counting:
                    self.MISSES += cost
                    self.N_MISSES += 1
                if self.moving_window_missrates:
                    self.moving_window_handle(cost)
                self.handle_cache_expire(item, cost, expiry)
            elif self.moving_window_missrates:
                self.moving_window_handle(0)
