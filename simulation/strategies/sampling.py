import numpy.random as rand

from utils.sortedcollection import SortedCollection
from operator import attrgetter
from itertools import islice
from math import log

class StrategyNode:
    def __init__(self, value, cost):
        self.value = value
        self.cost = cost
    def __repr__(self):
        return "((%r))" % self.value

def LastAccessObjectiveF(LANode, time):
    return float(LANode.cost) / (time - LANode.LA)

class Marking(object):
    def __init__(self, unmark_every = 1000):
        self.generator = rand.RandomState( 500 )
        self.name = "Marking(%d)" % unmark_every 

        self.NODES_MARKED = {}
        self.NODES_UNMARKED = {}
        
        self.time = 1
        self.unmark_every = unmark_every
        
    def touch(self, node):
        if node.value in self.NODES_UNMARKED:
            del self.NODES_UNMARKED[node.value]
            self.NODES_MARKED[node.value] = node
        return node

    def _check_and_unmark(self):
        self.time += 1
        if self.time % self.unmark_every == 0:
            self.NODES_UNMARKED.update(self.NODES_MARKED)
            self.NODES_MARKED.clear()
        
    def evict_node(self):
        if len(self.NODES_UNMARKED) == 0:
            self.NODES_UNMARKED = self.NODES_MARKED
            self.NODES_MARKED = {}
        evicting_d = self.NODES_UNMARKED

        assert len(evicting_d) > 0
        evict_ix = self.generator.randint(len(evicting_d))

        evict_key = islice(evicting_d.iterkeys(), evict_ix, evict_ix + 1).next()
        evicted_n = evicting_d[evict_key]
        del evicting_d[evict_key]
        
        return evicted_n
    
    def add_node(self, value, cost):
        new_node = StrategyNode(value, cost)
        self.NODES_MARKED[value] = new_node
#        self._check_and_unmark()

        return new_node

SAMPLING_VELOCITY_RETENTION = 1
SAMPLING_STANDARD_RETENTION = 0
SAMPLING_POOLED_RETENTION = 2

class NodeSampler:
    def __init__(self):
        self.hmap = {}
        self.sampler = []

    def __len__(self):
        return len(self.sampler)
    
    def values(self):
        return self.sampler
    
    def add(self, key, value):
        self.hmap[key] = value
        self.sampler.append(value)
    
    def delete(self, index):
        val = self.sampler.pop(index)
        del self.hmap[val.value]

class NodeSamplerEmpty:
    def __init__(self):
        self.hmap = {}
        self.sampler = []

    def __len__(self):
        return 0
    
    def values(self):
        return []
    
    def add(self, key, value):
        pass
    def delete(self, index):
        pass
    

class Sampling(object):
    """ IMPLEMENTED """
    def __init__(self, sampling = 64, retain = 0, name = None, **kwargs):
        """
        S defines the LastAccess eviction sample size
        """
        if sampling < 0:
            raise Exception("Sampling Strategy initialized with bad S = %d" % sampling)

        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__

        self.generator = rand.RandomState( 500 )

        # okay - actual strategy data

        self.S    = sampling

        self.retention_strategy = 0

        self.retain = retain
        self.retention = []

        self.Nodes = NodeSampler()

        self.time = 0

        self.last_evict = -1
        self.last_obj_f = 0

        self.get_true_minimum = False

        self.measure_error_from_min = False
        self.error_to_minimum = (0, 0)
        self.error_to_minimum_record_ratio = False
        self.error_to_minimum_thresh = 0.05

        self.death_pool = {}
        self.pool_magnification = 100
        self.pool_threshhold = 1.01

    def sample(self):
        
        if len(self.retention) > 0:
            new_samples = self.S - len(self.retention)
        else:
            new_samples = self.S

        N = len(self.Nodes)

        if len(self.death_pool) > 0:
            # MAGNIFIER should be ~= pr of chosen while in Death Pool / pr of chosen while in main pool
            # pr of chosen while in main pool = 1 - (N-1/N)**S
            # pr of chosen while in death pool per d.p. pick = 1/D
            # 1 - (N-1/N)**S = 1/(M*D)
            # (N-1/N)**S = 1 - 1/(M*D)
            # S*log(N-1 / N) = log(1 - 1/(M*D))
            p1 = log(1 - 1.0/(self.pool_magnification * len(self.death_pool)))
            p2 = log((N-1.0)/N)
            s_mult = p1/p2
            death_samples = int(new_samples / (1.0 + s_mult))
            main_samples =  int((new_samples * s_mult) / (1.0 + s_mult))
            if death_samples <= 0:
                death_samples = 1
                main_samples -= 1
            sample = self.generator.randint(N, size = main_samples)
            sample = [vals[i] for i in sample]
            death_sample = self.generator.randint(len(self.death_pool), size = death_samples)
            dp_vals = self.death_pool.values()
            sample += [ dp_vals[i] for i in death_sample ]

        else:
            indexes = self.generator.randint(N, size = new_samples)
            values = self.Nodes.values()
            sample = [values[i] for i in indexes]

        if len(self.retention) > 0:
            retention_nodes = [values[i] for i in self.retention]
            return list(sample) + retention_nodes, list(indexes) + self.retention
        else:
            return sample, indexes

    def objective_f(self, node):
        raise NotImplemented()

    def objective_f_tdelta(self, node, t_delta):
        return self.objective_f(node)

    def get_minimum(self, sample, indexes):
        sampled_nodes = sorted([ (self.objective_f(n) , n, ix)
                                 for n, ix in zip(sample, indexes) ])

        evictee_objf , evictee, _ = sampled_nodes[0]

        self.last_obj_f = evictee_objf
        
        # this way of recording retentions *assumes*
        #  that no inserts will occur into lower indices than the retained items
        #  until after the next eviction (this is true here because we always append inserts!)
        if self.retention_strategy == SAMPLING_STANDARD_RETENTION and self.retain > 0:
            retention = []
            for _, n, n_ix in sampled_nodes[1 : self.retain + 1]:
                if not (n == evictee):
                    retention.append(n_ix)
            self.retention = retention

        # velocity retention
        if self.retention_strategy == SAMPLING_VELOCITY_RETENTION and self.last_evict >= 0:
            time_delta = self.time - self.last_evict
            retention = []
            for _, n, n_ix in sampled_nodes[1 :]:
                if not (n == evictee):
                    obj_f = self.objective_f_tdelta(n, time_delta)
                    if obj_f <= evictee_objf:
                        retention.append(n_ix)
                    if len(retention) >= self.S / 2:
                        break
            self.retention = retention

        if self.retention_strategy == SAMPLING_POOLED_RETENTION and self.last_evict >= 0:
            if evictee.value in self.death_pool:
                del self.death_pool[evictee.value]
            for _, n, n_ix in sampled_nodes[1 :]:
                if not (n == evictee):
                    self.pooled_retain_recheck(n)
                
        return evictee

    def evict_node(self):
        if self.get_true_minimum:
            self.last_obj_f, x, ix = min([(self.objective_f(n), n, ix) for ix, n in enumerate(self.Nodes.values())])
            if self.measure_error_from_min:
                self.error_to_minimum = (self.error_to_minimum[0] + self.last_obj_f, 0)
        else:
            # sample the minimum
            vals, indexes = self.sample()
            x = self.get_minimum( vals, indexes )
            if self.measure_error_from_min:
                #real_min = min([self.objective_f(n) for n in self.Nodes.values()])
                cur_num, cur_denom = self.error_to_minimum
                self.error_to_minimum = (cur_num + self.last_obj_f, 0)
            index_of = vals.index(x)
            ix = indexes[index_of]
        self.Nodes.delete(ix)
        if len(self.retention) > 0:
            fixed_retention = []
            for r_ix in self.retention:
                assert r_ix != ix
                if r_ix > ix:
                    fixed_retention.append(r_ix - 1)
                else:
                    fixed_retention.append(r_ix)
            self.retention = fixed_retention
        self.last_evict = self.time

        return x

    def touch(self, node):
        node.LA = self.time
        self.time += 1

    def pooled_retain_recheck(self, node):
        if self.retention_strategy == SAMPLING_POOLED_RETENTION:
            if node.in_death_pool:
                del self.death_pool[node.value]
                node.in_death_pool = False
            self.pooled_retain_check(node)

    def pooled_retain_check(self, node):
        thresh = self.pool_threshhold * self.last_obj_f
        time_delta = self.time - self.last_evict
        obj_f = self.objective_f_tdelta(node, time_delta)
        if obj_f < thresh:
            self.death_pool[node.value] = node
            node.in_death_pool = True

    def add_node(self, value, cost):
        new_node = StrategyNode(value, cost)
        new_node.LA = self.time
        new_node.in_death_pool = False
        self.time += 1
        return self.add_existing_node(new_node)

    def add_existing_node(self, new_node):
        self.Nodes.add(new_node.value, new_node)
        return new_node

class LastAccess(Sampling):
    def objective_f(self, node):
        return float(node.cost) / (self.time - node.LA)

    def objective_f_tdelta(self, node, t_delta):
        return float(node.cost) / ((self.time + t_delta) - node.LA)
        
class LastAccess_Bucketing(LastAccess):
    def __init__(self, S, bucket_bounds):
        super(LastAccess_Bucketing, self).__init__(S,)
        self.bucket_bounds = bucket_bounds
        self.bucket_costs = [float(a + b)/2 for a,b in zip([0] + bucket_bounds[:-1], bucket_bounds)]
        self.bucket_frac = [(0,0) for _ in bucket_bounds]

    def objective_f(self, node):
        return self.bucket_costs[node.cost] / (self.time - node.LA)

    def update_bucket(self, bucket_ix, new_cost):
        # default:: just set bucket to newest.
#        self.bucket_costs[bucket_ix] = float(new_cost)
        bucket_frac = self.bucket_frac[bucket_ix]
        new_frac = (bucket_frac[0] + new_cost, bucket_frac[1] + 1)

        self.bucket_frac[bucket_ix] = new_frac
        self.bucket_costs[bucket_ix] = float(new_frac[0])/new_frac[1]

        return self.bucket_costs[bucket_ix]

    def add_node(self, value, cost):
        bucket_ix = bisect_left(self.bucket_bounds, cost)
        self.update_bucket(bucket_ix, cost)

        new_node = StrategyNode(value, bucket_ix)
        new_node.LA = self.time
        new_node.real_c = cost
        self.time += 1
        self.Nodes.append(new_node)

        return new_node

    def evict_node(self):
        # grab the minimum
        to_evict = super(Sampling_PreBucketed, self).evict_node()

        bucket_ix = to_evict.cost
        bucket_frac = self.bucket_frac[bucket_ix]
        new_frac = (bucket_frac[0] - to_evict.real_c, bucket_frac[1] - 1)
        self.bucket_frac[bucket_ix] = new_frac
        if(new_frac[1] == 0):
            self.bucket_costs[bucket_ix] = 0
        else:
            self.bucket_costs[bucket_ix] = float(new_frac[0])/new_frac[1]
        
        return to_evict

class GD_PreBucketed(LastAccess):
    def __init__(self, S, bucket_bounds):
        super(GD_PreBucketed, self).__init__(S,)
        self.bucket_bounds = bucket_bounds
        self.bucket_costs = [0 for _ in bucket_bounds]
        self.L = 0

    def objective_f(self, node):
        node_L, bucket_ix = node.cost
        return (self.bucket_costs[bucket_ix] + node_L, node.LA) # such an absurd objf.

    def evict_node(self):
        # grab the minimum
        x = self.get_minimum( self.sample() )

        to_evict = self.Nodes[x]
        self.L = objective_f(to_evict)[0]

        del self.Nodes[x]

        return to_evict


    def update_bucket(self, bucket_ix, new_cost):
        # default:: just set bucket to newest.
        self.bucket_costs[bucket_ix] = float(new_cost)

        return new_cost

    def touch(self, node):
        node.cost = (self.L, node.cost[1])
        node.LA = self.time

        self.time += 1

    def add_node(self, value, cost):
        bucket_ix = bisect_left(self.bucket_bounds, cost)
        self.update_bucket(bucket_ix, cost)

        new_node = StrategyNode(value, (self.L, bucket_ix))
        new_node.LA = self.time
        self.time += 1
        self.Nodes.append(new_node)

        return new_node


class LastAccessNode:
    def __init__(self, x, LA):
        self.value = x
        self.LA = LA

class LRU_Ranked(LastAccess):
    def __init__(self, S, retain = 0):
        super(LRU_Ranked, self).__init__(S, retain = retain)
        self.Nodes = SortedCollection(key=attrgetter("LA"))
        # using error to track the average eviction rank
        self.error_denom = 0
        self.error_numer = 0

    def objective_f(self, node):
        return node.LA

    def evict_node(self):
        # grab the minimum
        x = self.get_minimum( self.sample() )

        to_evict = self.Nodes[x]
        
        self.error_denom += 1
        self.error_numer += x

        del self.Nodes[x]

        return to_evict

    def touch(self, node):
        self.Nodes.remove(node)
        node.LA = self.time
        self.time += 1
        self.Nodes.insert_right(node)
        
    def add_node(self, value, cost):
        new_node = LastAccessNode(value, self.time)
        self.time += 1

        self.Nodes.insert_right(new_node)

        return new_node

class GreedyDual(LastAccess):
    def __init__(self, S, retain = 0):
        super(GreedyDual, self).__init__(S, retain = 0)
        self.L = 0

    def objective_f(self, node):
        return node.H

    def evict_node(self):
        # grab the minimum
        to_evict = self.get_minimum( self.sample() )

        self.L = to_evict.H[0]

        del self.Nodes[to_evict.value]

        return to_evict

    def touch(self, node):
        node.H = (self.L + node.cost, self.time)
        self.time += 1

    def add_node(self, value, cost):
        new_node = StrategyNode(value, cost)
        new_node.H = (self.L + cost, self.time)

        self.time += 1
        self.Nodes[value] = new_node

        return new_node
