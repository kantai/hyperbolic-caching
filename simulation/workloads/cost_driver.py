import numpy.random as random
from numpy.random import RandomState
#from ycsb_zipf import Deterministic_Zipfian as Zipfian_Generator
from ycsb_zipf import YCSB_Zipfian as Zipfian_Generator
#from ycsb_zipf import Deterministic_Zipfian_Lambda as Zipfian_Generator
#from geometric import GeoDistributionRand as Zipfian_Generator
from functools import wraps

# ENUMERATIONS 

NO_EXPIRES = 0
EXPIRES_EVERY = 1
GET_EXPIRY = 2
RETURNS_EXPIRY = 3

class ArbitraryDriver(object):
    def __init__(self, seed, item_range_max, permutation_seed = 100, 
                 name = None, zipf_param = 1.0001, d_second = -1):
        self.rand = RandomState(seed)
        self.max_item = item_range_max
        self.permute_seed = permutation_seed

        if name == None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    @classmethod
    def get_item(self, r_float):
        pass
    def get_cost(self, r_float, item_num):
        pass
    def permute_float(self, r_float):
        return r_float
    def sample_item_w_cost(self):
        r_float = self.rand.random_sample()        
        cost_float = self.permute_float(r_float)
        item = self.get_item(r_float)
        cost = self.get_cost(cost_float, item)

        return (item, cost)
        
class ReturnsExpiry(ArbitraryDriver):
    def __init__(self, *args, **kwargs):
        super(ReturnsExpiry, self).__init__(*args, **kwargs)

class ReturnsSize(ArbitraryDriver):
    def __init__(self, *args, **kwargs):
        super(ReturnsSize, self).__init__(*args, **kwargs)

class MixtureDriver(object):
    """ driver that mixes requests from two different drivers """
    def __init__(self, driver1, driver2, mix = 0.5, run_len = 10**5, name = "Mix(%s+%s; %.0e; %f)"):
        self.d1 = driver1
        self.d2 = driver2
        self.mix = mix
        self.run_len = run_len
        self.name = name % (driver1.name, driver2.name, run_len, mix)
        self.continuation = -1

    def sample_item_w_cost(self):
        self.continuation = ( self.continuation + 1 ) % self.run_len
        if self.continuation < self.run_len * self.mix:
            return self.d1.sample_item_w_cost()
        else:
            return self.d2.sample_item_w_cost()

class DoubleHitterDriver(object):
    def __init__(self, n_hitters_every_k, n_hits, driver):
        self.every_k = n_hitters_every_k
        self.n_hits = n_hits
        self.continuation = (0, 0)
        self.driver = driver
        self.name = "DH(%s; %f; %i)" % (self.driver.name, 1.0 / self.every_k, self.n_hits)

    def sample_item_w_cost(self):
        c = self.continuation
        if c[0] < 0:
            if c[0] == - (self.n_hits - 1):
                self.continuation = (0,0)
            else:
                self.continuation = (c[0] - 1, c[1])
            return c[1]
        else:
            r_val = self.driver.sample_item_w_cost()
            if c[0] == self.every_k - 1:
                self.continuation = (-1, r_val)
            else:
                self.continuation = (c[0] + 1, 0)
            return r_val

class ReverseShift(object):
    def __init__(self, T, permute = 113370, name = "ReversePops" ):
        self.name = name
        self.RNG = RandomState(permute)
        self.T = T
        self.t = 0
        
    def shift_pops(self, driver):
        driver.item_permute.reverse()

    def should_shift(self):
        return self.t in self.T

    def wrap_driver(self, driver):
        f = driver.get_item.__func__
        f_pop = driver.get_item_pop.__func__
        def get_item_wrapper(r_float):
            item = f(driver, r_float) # such hack. so wow.
            self.t += 1
            if self.should_shift():
                self.shift_pops(driver)
                driver.just_shifted = True
            return driver.item_permute[item]
        def item_pop_wrapper(item):
            item = driver.item_permute.index(item) # lol, so slow.
            return f_pop(driver, item)

        driver.just_shifted = False

        driver.item_permute = range(driver.max_item)
        driver.get_item = get_item_wrapper
        driver.get_item_pop = item_pop_wrapper
        
        driver.name += (".%s" % self.name) 
        return driver

class DynamicPromote(ReverseShift):
    def __init__(self, period = 1, permute = 113370, name = None):
        if name == None:
            name = "DynPromote.%d" % period
        super(DynamicPromote, self).__init__( [], permute = permute, 
                                              name = name)
        self.period = period
    def should_shift(self):
        return (self.t % self.period) == 0
    def shift_pops(self, driver):
        to_move = int(self.RNG.randint(driver.max_item))
        item = driver.item_permute.pop(to_move)
        driver.item_permute.insert(0, item)

class Introducing(DynamicPromote):
    def __init__(self, period = 1, permute = 113370, move_to = -1, name = "Intro.%d"):
        super(Introducing, self).__init__( period = period, permute = permute,
                                           name = (name % period) )
        self.move_to = move_to

    def shift_pops(self, driver):
        to_move = driver.max_item - 1
        item = driver.item_permute.pop(to_move)
        if self.move_to == -1:
            move_to = int(self.RNG.randint(driver.max_item)) 
        else:
            move_to = self.move_to
        driver.item_permute.insert(move_to, item)

class UniformDriver(ArbitraryDriver):
    def __init__(self, **kwargs):
        super(UniformDriver, self).__init__(name = "Uniform", **kwargs)
    def get_item(self, r_float):
        return int( r_float * self.max_item )
    def get_cost(self, r_float, item):
        return 1
    def get_item_pop(self, item):
        return 1.0 / self.max_item    

class UniformZipfDriver(ArbitraryDriver):
    def __init__(self, **kwargs):
        super(UniformZipfDriver, self).__init__(**kwargs)
        self.zipf_cost_gen = Zipfian_Generator(self.max_item, 1.0001)
    def get_item(self, r_float):
        return int( r_float * self.max_item )
    def get_cost(self, r_float, item):
        zipf_out, _ = self.zipf_cost_gen.get_next(r_float)
        return float(zipf_out)/self.max_item
    def get_item_pop(self, item):
        return 1.0 / self.max_item

class AlternateBadlyDriver(ArbitraryDriver):
    """
    Point of this workload is to force the LFU-like schemes to
    evict A, by requesting like:
       A, B, A, B, A, B
    where A and B constantly alternate between being evicted themselves.

    to do this, you need to send requests to the remaining elements of the cache.

      between accesses to A and B...

    
    """
    BUILDING_COUNTS = 0
    TARGET_REQUESTS = 1
    PHONY_REQUESTS = 2

    def __init__(self, build_counts_for = 10**5, item_range = 100, phony_requests_for = 1000, **kwargs):
        super(AlternateBadlyDriver, self).__init__(name = "ABD(%0e, %0e)" % (phony_requests_for, item_range), **kwargs)
        self.target = 1
        self.dummy = 2

        self.remaining_items = (3, self.max_item + 1)

        self.build_counts_for = build_counts_for
        self.phony_requests_for = phony_requests_for

        self.t = 0
        self.phase, self.phase_time = (AlternateBadlyDriver.BUILDING_COUNTS, self.t)
        self.phony_request_continuation = 0
        
        self.phony_item_range = item_range

    def sample_item_w_unit_cost(self):
        t_in_phase = self.t - self.phase_time
        item_range = (self.remaining_items[1] - self.remaining_items[0])
        if self.phase == AlternateBadlyDriver.BUILDING_COUNTS:            
            item = (t_in_phase % item_range) + self.remaining_items[0]
            if t_in_phase >= self.build_counts_for:
                self.phase = AlternateBadlyDriver.TARGET_REQUESTS
                self.phase_time = self.t
                self.phony_request_continuation = self.phase_time
        elif self.phase == AlternateBadlyDriver.TARGET_REQUESTS:
            if t_in_phase == 0:
                item = self.target
            elif t_in_phase == 1:
                item = self.dummy
                self.phase = AlternateBadlyDriver.PHONY_REQUESTS
                self.phase_time = self.t
        elif self.phase == AlternateBadlyDriver.PHONY_REQUESTS:
            item =  (self.phony_request_continuation % self.phony_item_range) + self.remaining_items[0]
            self.phony_request_continuation += 1
            if t_in_phase >= self.phony_requests_for:
                self.phase = AlternateBadlyDriver.TARGET_REQUESTS
                self.phase_time = self.t

        self.t += 1
        return (item, 1.0)

class AlternateCalibratedDriver(ArbitraryDriver):
    """
    Point of this workload is to force the LFU-like schemes to
    evict A, by requesting like:
       A, B, A, B, A, B
    where A and B constantly alternate between being evicted themselves.

    to do this, you need to send requests to the remaining elements of the cache.

      between accesses to A and B...
    """
    BUILDING_COUNTS = 0
    TARGET_REQUESTS = 1
    PHONY_REQUESTS = 2

    def __init__(self, count_to = 200, item_range = 1, cache_size = 5000, **kwargs):
        kwargs["item_range_max"] = cache_size + 2
        super(AlternateCalibratedDriver, self).__init__(name = "ACD(%.0e, %.0e)" % (count_to, cache_size), **kwargs)

        self.I_bounds = (0, cache_size) # I := [0, cache_size)
        self.target_A = cache_size
        self.target_B = cache_size + 1

        # at end of phase 1, item i will have priority = count_to / (count_to * max_item - i)
        #  the min(p_i | i \in I) = count_to / (count_to * max_item)

        self.min_pr_I_numer = float(count_to)

        self.build_counts_for = count_to * cache_size

        self.t = 0
        self.phase, self.phase_time = (AlternateCalibratedDriver.BUILDING_COUNTS, self.t)
        self.phony_request_continuation = 0
        self.alternation_continue = 0
        
        self.target_count = 0
        self.phony_item_range = item_range

    def sample_item_w_unit_cost(self):
        t_in_phase = self.t - self.phase_time
        if self.phase == AlternateCalibratedDriver.BUILDING_COUNTS:            
            item_range = (self.I_bounds[1] - self.I_bounds[0])
            item = (t_in_phase % item_range) + self.I_bounds[0]

            if t_in_phase >= self.build_counts_for:
                self.phase = AlternateCalibratedDriver.TARGET_REQUESTS
                self.phase_time = self.t

        elif self.phase == AlternateCalibratedDriver.TARGET_REQUESTS:
            if self.alternation_continue == 0:
                item = self.target_A
            else:
                item = self.target_B
            self.target_count += 1

            self.alternation_continue = (self.alternation_continue + 1) % 2
            self.phase = AlternateCalibratedDriver.PHONY_REQUESTS
            self.phase_time = self.t

        elif self.phase == AlternateCalibratedDriver.PHONY_REQUESTS:
            start_phony_items = (self.I_bounds[1] - self.I_bounds[0]) / 2
            item = (self.phony_request_continuation % self.phony_item_range) + start_phony_items
            self.phony_request_continuation += 1
            
            if t_in_phase > 1:                
                target_pr = 1.0 / (t_in_phase - 1)
                min_I_pr = self.min_pr_I_numer / self.t

                if target_pr < min_I_pr:
                    self.phase = AlternateCalibratedDriver.TARGET_REQUESTS
                    self.phase_time = self.t

        self.t += 1
        return (item, 1.0)

class WSetDriver_Centers(ArbitraryDriver):
    def __init__(self, working_set_reqs = 10, working_set_replay = 1, **kwargs):
        super(WSetDriver, self).__init__(name = "WSCentersUC(%d; %d)" % (working_set_reqs, working_set_replay), 
                                         **kwargs)
        self.X_continuation = (0, 0)
        self.X = working_set_reqs
        self.number_of_centers = self.max_item / self.X
        self.working_set_replay = working_set_replay
        self.replay_continuation = -1

    def get_item(self, r_float):
        cur_item, reqs_for = self.X_continuation 
        if reqs_for == 0:
            if self.replay_continuation == 0:
                cur_item = int(r_float * self.number_of_centers) * self.X
            self.X_continuation = (cur_item, 1)
            self.replay_continuation = (1 + self.replay_continuation) % self.working_set_replay
            return cur_item
        else:
            self.X_continuation = (cur_item, (reqs_for + 1) % self.X)
            return cur_item + reqs_for

    def get_cost(self, r_float, item_num):
        return 1.0

class WSetDriver(ArbitraryDriver):
    def __init__(self, working_set_reqs = 10, working_set_replay = 1, **kwargs):
        super(WSetDriver, self).__init__(name = "WorkingSetUC(%d; %d)" % (working_set_reqs, working_set_replay), 
                                         **kwargs)
        self.X_continuation = (0, 0)
        self.X = working_set_reqs
        self.working_set_replay = working_set_replay
        self.replay_continuation = -1

    def get_item(self, r_float):
        cur_item, reqs_for = self.X_continuation 
        if reqs_for <= 0:
            if self.replay_continuation == 0:
                cur_item = int(r_float * self.max_item)
            num_wset_hits = int(self.X * self.rand.random_sample())
            self.X_continuation = (cur_item, num_wset_hits)
            self.replay_continuation = (1 + self.replay_continuation) % self.working_set_replay
            return cur_item
        else:
            self.X_continuation = (cur_item, (reqs_for - 1))
            return (cur_item + reqs_for) % self.max_item

    def get_cost(self, r_float, item_num):
        return 1.0

class FIFODriver(ArbitraryDriver):
    def __init__(self, **kwargs):
        super(FIFODriver, self).__init__(name = "FIFO", **kwargs)
        self.cur_item = 0        
    def get_item(self, r_float):
        r = self.cur_item
        self.cur_item = (self.cur_item + 1) % self.max_item
        return r
    def get_cost(self, r_float, item_num):
        return 1.0

class ZipfUniformDriver(ArbitraryDriver):
    def __init__(self, zipf_param = 1.0001, **kwargs):
        super(ZipfUniformDriver, self).__init__(**kwargs)
        self.zipf_gen = Zipfian_Generator(self.max_item, zipf_param)
        self.costs = []

    def get_cost(self, r_float, item_num):
        if len(self.costs) == 0:
            r = RandomState(self.permute_seed)
            self.costs = r.random_sample(self.max_item)
        return float(self.costs[item_num])

    def get_item(self, r_float):
        zipf_out, _ = self.zipf_gen.get_next(r_float)
        return zipf_out

    def get_item_pop(self, item):
        return self.zipf_gen.get_popularity(item)


class UniformUniformDriver(ZipfUniformDriver):
    def __init__(self, **kwargs):
        super(UniformUniformDriver, self).__init__(**kwargs)
        self.zipf_gen = None

    def get_item(self, r_float):
        return int(r_float * self.max_item)

    def get_item_pop(self, item):
        return 1.0 / self.max_item

class ZipfFixedDriver(ArbitraryDriver):
    def __init__(self, costs = [1], generator = Zipfian_Generator, zipf_param = 1.001, **kwargs):
        super(ZipfFixedDriver, self).__init__(**kwargs)
        self.zipf_gen = generator(self.max_item, zipf_param)

        scale = max(costs)
        if scale > 1:
            self.costs = [float(c) / scale for c in costs]
        else:
            self.costs = [float(c) for c in costs]
    def get_cost(self, r_float, item_num):
        return self.costs[item_num % len(self.costs)]

    def get_item(self, r_float):
        zipf_out, _ = self.zipf_gen.get_next(r_float)
        return zipf_out

    def get_item_pop(self, item):
        return self.zipf_gen.get_popularity(item)

class ZPop_StochCostClass(ZipfFixedDriver):
    def __init__(self, cost_params, cost_dist = (lambda x : x.normal), 
                 name = None, **kwargs):
        if name == None:
            name = "StochCostClass(%s)" % (cost_params, )
        super(ZPop_StochCostClass, self).__init__(costs = [1], name = name,
                                                  **kwargs)
        self.num_classes = len(cost_params)
        self.sampler = cost_dist(self.rand)
        self.cost_params = cost_params
    def get_cost(self, r_float, item_num):
        cost_class = self.get_cost_class(item_num)
        cost = self.sampler(*self.cost_params[cost_class])
        if cost < 0:
            return self.get_cost(r_float, item_num)
        return cost
    def get_cost_class(self, item_num):
        return item_num % self.num_classes

def StochCostClass_Factory(d_second = [(1, 0.01)], **kwargs):
    return ZPop_StochCostClass(cost_params = d_second, **kwargs)

class ZPop_2Classes(ZipfFixedDriver):
    def __init__(self, **kwargs):
        super(ZPop_2Classes, self).__init__(name = "ZP2Class", **kwargs)

    def get_cost_class(self, item_num):
        return (item_num % 2)

class ZPop_HotCostClass(ZipfFixedDriver):
    def __init__(self, cold_costs = [1, 1, 1, 1], hot_costs = [100, 1, 1, 1],
                 hot_for = 10**7, hot_every = 2*10**5,
                 name = "HotCost(%.2e;%.2e)",
                 **kwargs):
        assert len(hot_costs) == len(cold_costs)

        super(ZPop_HotCostClass, self).__init__(name = name % (hot_every, hot_for), **kwargs)
        scale = max( hot_costs + cold_costs )
        if scale > 1:
            self.cold_costs = [float(c) / scale for c in cold_costs]
            self.hot_costs = [float(c) / scale for c in hot_costs]
        else:
            self.cold_costs = cold_costs
            self.hot_costs = hot_costs

        self.hot_time = 0
        self.hot_for = hot_for
        self.hot_every = hot_every

    def get_cost(self, r_float, item_num):
        if self.hot_time >= self.hot_every:
            costs = self.hot_costs
        else:
            costs = self.cold_costs

        # hot every n
        self.hot_time += 1
        if self.hot_time >= self.hot_for + self.hot_every:
            self.hot_time = 0

        return costs[item_num % len(costs)]

    def get_cost_class(self, item_num):
        return item_num % len(self.cold_costs)
        

class ZipfZipfDriver(ArbitraryDriver):
    def __init__(self, zipf_param = 1.0001, **kwargs):
        super(ZipfZipfDriver, self).__init__(**kwargs)
        self.zipf_cost_gen = Zipfian_Generator(self.max_item, 1.0001)
        self.zipf_item_gen = Zipfian_Generator(self.max_item, zipf_param)

    def get_item(self, r_float):
        zipf_out, _ = self.zipf_item_gen.get_next(r_float)
        return zipf_out

    def get_item_pop(self, item):
        return self.zipf_item_gen.get_popularity(item)

    def get_cost(self, r_float, item_num):
        zipf_out, _ = self.zipf_cost_gen.get_next(r_float)
        return float(zipf_out)/self.max_item

class ZipfEvenDriver(ArbitraryDriver):
    def __init__(self, severity = 2.0, perturbate = 0, zipf_param = 1.0001, **kwargs):
        super(ZipfEvenDriver, self).__init__(**kwargs)
        self.zipf_item_gen = Zipfian_Generator(self.max_item, zipf_param)

        self.cost_power = severity * self.zipf_item_gen.theta

        self.perturbate = perturbate
        self.perturbed = []
        self.inner_max_cost = self.max_item ** self.cost_power 

    def get_cost(self, r_float, item):
        cost = (item ** self.cost_power) / self.inner_max_cost

        if self.perturbate > 0: 
            if len(self.perturbed) == 0:
                r = RandomState(self.permute_seed)
                self.perturbed = (1.0 - (r.random_sample(self.max_item) * self.perturbate))
            
                #            item = float(self.perturbed[item] * item)
                cost *= float(self.perturbed[item])
        return cost

    def get_item(self, r_float):
        item, _ = self.zipf_item_gen.get_next(r_float)
        return item

    def get_item_pop(self, item):
        return self.zipf_item_gen.get_popularity(item)
