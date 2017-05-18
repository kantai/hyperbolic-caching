from collections import defaultdict, OrderedDict
from utils.sortedcollection import SortedCollection
from bisect import bisect_left
from math import log, exp
import sampling

from sortedcontainers import SortedListWithKey

measure_average = "Average_Restore_Old_Err"

class Sampling_Frequency(sampling.Sampling):
    def __init__(self, save_counts = False, 
                 name = None, **kwargs):
        super(Sampling_Frequency, self).__init__(name = name, **kwargs)
        if save_counts:
            self.name += ".keepCounts"
        self.saving_counts = save_counts
        self.historical_nodes = dict()
        self.average_restore = (0, 0)

    def touch(self, node):
        super(Sampling_Frequency, self).touch(node)
        node.count += 1

    def add_node(self, value, cost):
        if self.saving_counts and value in self.historical_nodes:
            new_node = self.restore_node(self.historical_nodes[value])
            del self.historical_nodes[value]
            self.Nodes[value] = new_node
        else:
            new_node = super(Sampling_Frequency, self).add_node(value, cost)
            new_node.count = 1 
            new_node.is_restored = False
            new_node.is_restoring = False
        return new_node

    def restore_node(self, old_node):
        old_node.is_restoring = True
        old_obj_f = self.objective_f(old_node)
        self.touch(old_node)
        old_node.is_restoring = False
        old_node.is_restored = True
        new_obj_f = self.objective_f(old_node)
        if measure_average == "Average_Restore_ObjF":
            update = new_obj_f
        elif measure_average == "Average_Restore_Err":
            if new_obj_f == 0:
                return old_node
            update = abs(self.last_obj_f - new_obj_f) / new_obj_f
        elif measure_average == "Average_Restore_Old_Err":
            if old_obj_f == 0:
                return old_node
            update = abs(self.last_obj_f - old_obj_f) / old_obj_f

        self.average_restore = (self.average_restore[0] + update,
                                self.average_restore[1] + 1)

        return old_node

    def evict_node(self):
        n = super(Sampling_Frequency, self).evict_node()

        if self.saving_counts:
            self.historical_nodes[n.value] = n

        return n

    def objective_f(self, node):
        return node.cost * node.count

def Velo_Sample_Hyperbolic(name = "SV_Hyper", **kwargs):
    strategy = Sample_Hyperbolic(name = name, **kwargs)
    strategy.retention_strategy = sampling.SAMPLING_VELOCITY_RETENTION
    return strategy

def Size_Aware_Hyperbolic(name = "S_Hyper_Sz", **kwargs):
    strategy = Sample_Hyperbolic(name = name, **kwargs)
    strategy.size_aware = True
    return strategy

class Sample_Hyperbolic(Sampling_Frequency):
    def __init__(self, leeway = 0.1, degrade = 1, 
                 name = "S_Hyper", **kwargs):
        name = "%s(%.0e; %.3f)" % (name, 1-degrade, leeway)
        super(Sample_Hyperbolic, self).__init__(name = name, **kwargs)
        self.degrade = degrade
        self.leeway = leeway
        self.last_miss = 0
        self.count_min = 10 ** -10
        self.size_aware = False

    def touch(self, node):
        if self.degrade != 1:
            count_minus_min = (node.count - self.count_min) * (self.degrade ** (self.time - node.last_degrade))
            node.count = self.count_min + count_minus_min
            node.last_degrade = self.time

        node.count += 1

        self.time += 1
        return node

    def add_node(self, value, cost, size = 1):
        # gdcost heuristic : cost = 1 / size
        if self.size_aware and size != 1:
            cost = float(cost) / size

        new_node = super(Sample_Hyperbolic, self).add_node(value, cost)
        # guess the count and delta for the object
        new_node.last_degrade = self.time
        new_node.entry_time = self.time - 1

        self.last_miss = self.time

        if not new_node.is_restored:
            if self.leeway == 1 or cost == 0 or self.last_obj_f == 0:
                return new_node
            
            
            new_node.count = self.leeway + (1.0 - self.leeway) * (self.last_obj_f / cost)

            # leeway =  1 - (proportion to weight guess)
            #            new_node.LA -= (time_guess * (1.0 - self.leeway))
            #            new_node.LA -= 1
            #            new_node.count = 1.0 - self.leeway


        return new_node

    def objective_f(self, node):
        return self.objective_f_tdelta(node, 0)

    def objective_f_tdelta(self, node, t_delta):
        time = self.time + t_delta
        if self.degrade != 1:
            degrade_by = self.degrade ** (time - node.last_degrade)
            return float(node.cost) * (self.count_min + ((node.count - self.count_min) * degrade_by)) / (time - node.entry_time)
        else:
            return float(node.cost * (node.count)) / (time - node.entry_time)

def moving_average(const = 0.97):
    return  (lambda a, b : ((const * a) + ((1.0 - const) * b)))
def replacement(a, b):
    return b

class Sample_Hyperbolic_ClassTrack(Sample_Hyperbolic):
    def __init__(self, update_cost = moving_average(), **kwargs):
        super(Sample_Hyperbolic_ClassTrack, self).__init__(
            name = "S_Hyper_ClassTrack", **kwargs)
        self.costs = {}
        self.update_cost_F = update_cost

    def update_cost(self, cost_class, cost):
        if cost_class in self.costs:
            self.costs[cost_class] = self.update_cost_F(
                self.costs[cost_class], cost)
        else:
            self.costs[cost_class] = cost

    def add_node(self, value, cost, cost_class):
        new_node = super(Sample_Hyperbolic_ClassTrack, self).add_node(value, cost)
        self.update_cost(cost_class, cost)
        new_node.cost_class = cost_class

        return new_node

    def objective_f_tdelta(self, node, t_delta):
        node.cost = self.costs[node.cost_class]
        return super(Sample_Hyperbolic_ClassTrack, self).objective_f_tdelta(node, t_delta)

class Sample_Hyperbolic_Pooled(Sample_Hyperbolic):
    def __init__(self, *args, **kwargs):
        super(Sample_Hyperbolic_Pooled, self).__init__(*args, name = "SP_Hyper", **kwargs)
        self.retention_strategy = sampling.SAMPLING_POOLED_RETENTION
        self.retain = 0
    def touch(self, node):
        super(Sample_Hyperbolic_Pooled, self).touch(node)
        self.pooled_retain_recheck(node)
    def add_node(self, *args, **kwargs):
        new_node = super(Sample_Hyperbolic_Pooled, self).add_node(*args, **kwargs)
        self.pooled_retain_recheck(new_node)
        return new_node

class Sampling_LNC_R_W3(sampling.Sampling):
    def __init__(self, k_access = 4, name = "S_LNC", **kwargs):
        name = "%s(%d)" % (name, k_access)
        super(Sampling_LNC_R_W3, self).__init__(name = name, **kwargs)
        assert k_access >= 1
        self.k_access = k_access
    def add_node(self, value, cost):
        new_node = super(Sampling_LNC_R_W3, self).add_node(value, cost)
        new_node.accesses = [new_node.LA]
        new_node.rate = (1.0 , new_node.LA)
        return new_node
    def touch(self, node):
        super(Sampling_LNC_R_W3, self).touch(node)
        if len(node.accesses) >= self.k_access:
            node.accesses.pop(0)
        node.accesses.append(node.LA)
        node.rate = (float(len(node.accesses)), node.accesses[0])
    def objective_f(self, node):
        return node.rate[0] * node.cost / (self.time - node.rate[1])

class Sampling_TSP(sampling.Sampling):
    def __init__(self, name = "S_TSP", **kwargs):
        super(Sampling_TSP, self).__init__(name = name, **kwargs)
        assert k_access >= 1
        self.k_access = k_access
    def add_node(self, value, cost):
        new_node = super(Sampling_TSP, self).add_node(value, cost)
        new_node.accesses = [new_node.LA]
        new_node.rate_of_access = 1.0
        return new_node
    def touch(self, node):
        super(Sampling_TSP, self).touch(node)
        if len(node.accesses) >= self.k_access:
            node.accesses.pop(0)
        node.accesses.append(node.LA)
        node.rate_of_access = float(len(node.accesses)) / node.accesses[0]
    def objective_f(self, node):
        return node.rate_of_access * node.cost

class Windowed_Hyper(Sample_Hyperbolic):
    def __init__(self, window_size = 10**4, name = "W(%0.e)DegF", **kwargs):
        super(Windowed_Hyper, self).__init__(degrade = 1.0, name = name % window_size, **kwargs)
        self.window_size = window_size
    
    def add_node(self, value, cost):
        new_node = super(Windowed_Hyper, self).add_node(value, cost)
        new_node.touches = [self.time - 1]
        new_node.cur_entry_time = new_node.touches[0]

        assert new_node.count >= 0.0 # should be set by parent!

        return new_node

    def touch(self, node):
        if node.count <= 0:
            node.cur_entry_time = self.time
        node.touches.append(self.time)
        node.count += 1.0
        self.time += 1
        return node

    def objective_f(self, node):
#        cutoff = bisect_left(node.touches, self.time - self.window_size)
#        del node.touches[:cutoff]
        window_start = self.time - self.window_size
        while node.count > 0 and node.cur_entry_time < window_start:
            node.touches.pop(0)
            node.count -= 1.0
            if node.count > 0:
                node.cur_entry_time = node.touches[0]
            else:
                node.count = 0.0 # min to 0, can drop lower if we have "init-guess"
            
        if node.count == 0:
            return 0

        return (node.cost * node.count) / (self.time - node.cur_entry_time)

class Windowed_Freq(Sampling_Frequency):
    def __init__(self, window_size = 10**4, name = "W(%0.e)LFU", **kwargs):
        super(Windowed_Freq, self).__init__(name = name % window_size, **kwargs)
        self.window_size = window_size
    
    def add_node(self, value, cost):
        new_node = super(Windowed_Freq, self).add_node(value, cost)
        new_node.touches = [self.time - 1]
        new_node.cur_entry_time = new_node.touches[0]

        assert new_node.count >= 0.0 # should be set by parent!

        return new_node

    def touch(self, node):
        if node.count <= 0:
            node.cur_entry_time = self.time
        node.touches.append(self.time)
        node.count += 1.0
        self.time += 1
        return node

    def objective_f(self, node):
#        cutoff = bisect_left(node.touches, self.time - self.window_size)
#        del node.touches[:cutoff]
        window_start = self.time - self.window_size
        while node.count > 0 and node.cur_entry_time < window_start:
            node.touches.pop(0)
            node.count -= 1.0
            if node.count > 0:
                node.cur_entry_time = node.touches[0]
            else:
                node.count = 0.0 # min to 0, can drop lower if we have "init-guess"
            
        if node.count == 0:
            return 0

        return (node.cost * node.count)

class Decrement_DegF(Sample_Hyperbolic):
    def __init__(self, window_size = 10**4, name = "Dec(%0.e)DegF", **kwargs):
        super(Decrement_DegF, self).__init__(S, degrade = 1.0, name = name % window_size, **kwargs)
        self.decr_window = window_size
    
    def add_node(self, value, cost):
        new_node = super(Sample_Hyperbolic, self).add_node(value, cost)
        new_node.count = 1.0
        new_node.cur_entry_time = self.time - 1
        new_node.last_decrement = float(self.time - 1)

        return new_node

    def decr_node(self, node):
        decr_by = int(( self.time - node.last_decrement ) / self.decr_window)
        if decr_by <= 0:
            return

        new_count = node.count - decr_by
        if new_count <= 0:
            node.count = 0
            node.cur_entry_time = self.time - 1
        else:
            new_entry_time = self.time - ((float(new_count) / node.count) * (self.time - node.cur_entry_time))
            node.count = new_count
            node.cur_entry_time = new_entry_time
        node.last_decrement = self.time - 1

    def touch(self, node):
        self.time += 1
        self.decr_node(node)

        if node.count <= 0:
            node.cur_entry_time = self.time - 1
        node.count += 1.0
        
        return node

    def objective_f(self, node):
        self.decr_node(node)
        if node.count == 0:
            return 0

        return (node.cost * node.count) / (self.time - node.cur_entry_time)


class AppWin_DegF(Sample_Hyperbolic):
    def __init__(self, window_size = 10**4, name = "AppWin(%0.e)DegF", **kwargs):
        super(AppWin_DegF, self).__init__(degrade = 1.0, name = name % window_size, **kwargs)
        self.decr_window = window_size
    
    def add_node(self, value, cost):
        new_node = super(Sample_Hyperbolic, self).add_node(value, cost)
        new_node.count = 1.0
        new_node.cur_entry_time = self.time - 1

        return new_node

    def decr_node(self, node, window_start):
        node.count *=  float(self.decr_window) / (self.time - node.cur_entry_time)
        node.cur_entry_time = window_start

    def touch(self, node):
        self.time += 1
        window_start = self.time - self.decr_window
        if node.cur_entry_time < window_start:
            self.decr_node(node, window_start)

        node.count += 1.0
        
        return node

    def objective_f(self, node):
        window_start = self.time - self.decr_window
        if node.cur_entry_time < window_start:
            self.decr_node(node, window_start)

        return (node.cost * node.count) / (self.time - node.cur_entry_time)


class GhostCache(object):
    def __init__(self, gc_prop):
        self.gc_prop = gc_prop
        self.cache_size = -1
        self.gc_sz = -1
        self.ghost_cache = OrderedDict()
    def touch(self, node):
        return self.s_policy.touch(node)
    def add_node(self, value, cost):
        if self.gc_sz == -1:
            self.gc_sz = int(self.gc_prop * self.cache_size)
            assert self.cache_size != -1
        if value in self.ghost_cache:
            new_node = self.ghost_cache.pop(value)
            self.touch(new_node) # should increment time.
            return self.s_policy.add_existing_node(new_node)
        return self.s_policy.add_node(value, cost)
    def evict_node(self):
        evicted = self.s_policy.evict_node()
        if len(self.ghost_cache) >= self.gc_sz:
            self.ghost_cache.popitem(last=False)
        self.ghost_cache[evicted.value] = evicted
        return evicted

class GC_RMDegF(GhostCache):
    def __init__(self, gc_prop = 0.01, name = "GC(%f)RM_DegF", **kwargs):
        super(GC_RMDegF, self).__init__(gc_prop)
        name = name % gc_prop
        self.s_policy =  RealMin_DegF(name = name, **kwargs)
        self.name = self.s_policy.name

class Sample_TimeAwareLRFU(Sampling_Frequency):
    def __init__(self, degrade = 0.9, leeway = 1, name = None, **kwargs):
        if name == None:
            name = "S_TA_LRFU(%f; %f)" % (degrade, leeway)
        super(Sample_TimeAwareLRFU, self).__init__(name = name, **kwargs)
        self.degrade = degrade
        self.leeway = leeway
        self.last_miss = 0

    def touch(self, node):
        time_delta   = (self.time - node.LA)

        node.count += 1

        node.denom_degrade_part *= (self.degrade ** time_delta)
        node.denom_degrade_part += time_delta

        node.numer_degrade_part *= (self.degrade ** time_delta)
        node.numer_degrade_part += 1
        
        node.LA = self.time

        self.time += 1

    def add_node(self, value, cost):
        new_node = super(Sample_TimeAwareLRFU, self).add_node(value, cost)
        # TODO: guess the count and delta for the object ?

        if not new_node.is_restored:
            new_node.numer_degrade_part = 0.0
            new_node.denom_degrade_part = 0.0

        self.last_miss = self.time
        return new_node

    def objective_f(self, node):
        time_delta = (self.time - node.LA)
        degrade_by = (self.degrade ** time_delta)
        pr_denom = (node.denom_degrade_part * degrade_by) + time_delta
        pr_numer = (node.numer_degrade_part * degrade_by) + 1
        
        return ((pr_numer / pr_denom) * node.cost)


class Sample_LRFU(Sampling_Frequency):
    """
    the degrade constant used here is a function of the \lambda
    defined in the LRFU '96 paper.
    degrade = (1/2)^(\lambda)
    """
    def __init__(self, degrade = 0.99999, name = "S_LRFU", **kwargs):
        name = "%s(%.0e)" % (name, 1-degrade)
        super(Sample_LRFU, self).__init__(name = name, **kwargs)
        self.degrade = degrade

    def touch(self, node):
        time_delta = (self.time - node.LA)
        
        old_counts = node.count
        new_counts = 1 + (old_counts * (self.degrade ** time_delta))
        
        node.count = new_counts
        
        node.LA = self.time

        self.time += 1
    
    def add_node(self, value, cost):
        new_node = super(Sample_LRFU, self).add_node(value, cost)
        new_node.insert_time = self.time
        
#        if self.last_obj_f > 0 and cost > 0:
#            self.counts[value] = self.last_obj_f / cost

        return new_node

    def objective_f(self, node):
        scale_counts = self.degrade ** (self.time - node.LA)
        return node.cost * node.count * scale_counts


class Sample_Frequency_Expiry(Sample_Hyperbolic):
    def __init__(self, degrade_expiry = 1.01, name = "S_HyperExpiry", 
                 **kwargs):
        super(Sample_Frequency_Expiry, self).__init__(
            name = "%s(%.3f)" % (name, degrade_expiry), **kwargs)
        self.degrade_expiry = degrade_expiry

    def add_node(self, value, cost, expires_at = -1):
        node = super(Sample_Frequency_Expiry, self).add_node(value, cost)
        node.expires_at = expires_at
        return node
    
    def set_expiry(self, node, expires_at):
        node.expires_at = expires_at

    def _expiry_weighted(self, node, time_expires):
        return (1 - exp(-self.degrade_expiry * time_expires))

    def expiry_weighted(self, node):
        expires_at = node.expires_at
        if expires_at == -1:
            return 1
        time_expires = expires_at - (self.time + 1)
        if time_expires <= 0:
            return 0        

        return self._expiry_weighted(node, time_expires)

    def _expiry_weighted_poisson(self, node, time_expires):
        lamb = float(node.count) / ((self.time + 1) - node.insert_time)
        lamb_te = lamb * time_expires
        rhs = exp(-1 * lamb_te) * (lamb_te + 1)
        return (1.0 - rhs)

    def _expiry_weighted_linear(self, node, time_expires):
        midpoint = 10000.0
        return min(midpoint, time_expires)

    def objective_f(self, node):
        base_f = super(Sample_Frequency_Expiry, self).objective_f(node)
        return base_f * self.expiry_weighted(node)

class DummyList:
    """ this class will pretend it's a list for you, but always disappoint."""
    def add(self, key, value):
        pass

class PQ_Frequency(Sampling_Frequency):
    def __init__(self):
        super(PQ_Frequency, self).__init__(sampling = 0)
        self.nodes = SortedListWithKey(key = self.objective_f)
        self.Nodes = DummyList()

    def add_node(self, item, cost):
        new_node = super(PQ_Frequency, self).add_node(item, cost)
        self.nodes.add(new_node)
        return new_node

    def touch(self, node):
        if not node.is_restoring:
            self.nodes.remove(node)
        super(PQ_Frequency, self).touch(node)
        if not node.is_restoring:
            self.nodes.add(node)
    
    def evict_node(self):
        to_evict = self.nodes.pop(0)

        if self.saving_counts:
            self.historical_nodes[to_evict.value] = to_evict
        self.last_obj_f = self.objective_f(to_evict)


        return to_evict

class RealMin_Frequency_Expiry(Sample_Frequency_Expiry):
    def __init__(self, name = "RM_HyperExpiry", **kwargs):
        super(RealMin_Frequency_Expiry, self).__init__(name = name, 
                                                       **kwargs)
        self.get_true_minimum = True

class RealMin_Hyper(Sample_Hyperbolic):
    def __init__(self, name = "RM_Hyper", **kwargs):
        super(RealMin_Hyper, self).__init__(name = name, **kwargs)
        self.get_true_minimum = True


class PerfectKnowledge_Expiry(RealMin_Frequency_Expiry):
    def __init__(self, degrade_expiry = 2.01):
        super(PerfectKnowledge_Expiry, self).__init__(degrade_expiry = degrade_expiry,
                                                      name = "PK_Expiry")

    def add_node(self, item, cost, popularity, expires_at):
        new_node = super(PerfectKnowledge_Expiry, self).add_node(item, cost, 
                                                                 expires_at = expires_at)
        new_node.popularity = popularity
        return new_node
    def objective_f(self, node):
        return node.popularity * node.cost * self.expiry_weighted(node)
        

class Bucket_Frequency(sampling.LastAccess):
    """
    bucket. perform frequency updates *per* bucket.
    """
    def __init__(self, bucket_bounds):
        super(CostBucket_Frequency, self).__init__(0)
        self.bucket_bounds = bucket_bounds
        self.bucket_queues = [ SortedCollection(key = self._get_count)
                               for _ in bucket_bounds ]
        self.bucket_values = [ (0, 0) for _ in bucket_bounds ]
        self.counts = defaultdict(lambda : 0)
        self.generator = None
        self.Nodes = DummyList()

    def _get_count(self, node):
        return self.counts[node.value]

    def touch(self, node):
        super(CostBucket_Frequency, self).touch(node)
        
        self.bucket_queues[node.bucket_ix].remove(node)
        self.counts[value] += 1
        self.bucket_queues[node.bucket_ix].insert(node)
        
    def evict_node(self):
        numerators = [ ((float(b_numer) * self.counts[q[0].value] / b_denom), ix)
                       for (ix, ((b_numer, b_denom), q)) in 
                       enumerate(zip(self.bucket_values, self.bucket_queues)) ]
        objective_f, evict_from = min(numerators)

        to_evict = self.bucket_queues[evict_from][0]
        del self.bucket_queues[evict_from][0]
        old_numer, old_denom = self.bucket_values[evict_from]
        self.bucket_values[evict_from] = (old_numer - to_evict.cost, old_denom - 1)
        
        return to_evict

    def add_node(self, value, cost):
        new_node = super(CostBucket_Frequency, self).add_node(value, cost)
        
        bucket_ix = bisect_left(self.bucket_bounds, cost)

        new_node.bucket_ix = bucket_ix

        self.counts[value] += 1
        self.bucket_queues[bucket_ix].insert(new_node)

        old_numer , old_denom = self.bucket_values[bucket_ix]
        self.bucket_values[bucket_ix] = (old_numer + cost, old_denom + 1)

        return new_node
