from operator import attrgetter
from bisect import bisect_left, bisect_right
from utils.sortedcollection import SortedCollection
from sortedcontainers import SortedListWithKey

class PQNode:
    def __init__(self, item, cost):
        self.value = item
        self.cost = cost
        self.priority = 0

class GD_PQ(object):
    def __init__(self, name = None, **kwargs):
        super(GD_PQ, self).__init__()
        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__


        self.H = 0
#        self.nodes = SortedCollection(key=attrgetter('priority'))
        self.error_numer = 0
        self.error_denom = 1

        self.size_aware = False

        self.nodes = SortedListWithKey(key = attrgetter('priority'))
        self.time = 0

    def add_node(self, item, cost, size = 1):
        if self.size_aware and size != 1:
            cost = float(cost) / size

        new_node = PQNode(item, cost)
        new_node.priority = self.H + new_node.cost
        
#        self.nodes.insert_right(new_node)
        self.nodes.add(new_node)

        self.time += 1
        return new_node

    def touch(self, node):
        self.nodes.remove(node)
        self.update_priority(node)
#        self.nodes.insert_right(node) 
        self.nodes.add(node) 
        self.time += 1

    def update_priority(self, node):
        node.priority = self.H + node.cost

    def evict_node(self):
#        to_evict = self.nodes[0]
#        del self.nodes[0]
        to_evict = self.nodes.pop(0)
        self.H = to_evict.priority
        return to_evict

def Size_Aware_GD_PQ(name = "GD_PQ_Sz"):
    strategy = GD_PQ( name )
    strategy.size_aware = True
    return strategy

class GD_TSP_PQ(GD_PQ):
    def __init__(self, name = "GD_TSP", **kwargs):
        super(GD_TSP_PQ, self).__init__(name = name, **kwargs)

    def add_node(self, item, cost):
        new_node = PQNode(item, cost)
        new_node.count = 1
        new_node.priority = float(new_node.count * new_node.cost) / 1.0
        new_node.accesses = (self.time, -1)
        self.time += 1
        self.nodes.insert_right(new_node)
        return new_node

    def update_priority(self, node):
        node.count += 1
        t_la1, t_la2 = node.accesses
        if t_la2 == -1:
            t_next_access = self.time - t_la1
        else:
            t_next_access = (5.0 * (self.time) - 4.0 * (t_la1) + t_la2) / 2.0 
        node.accesses = (self.time, t_la1)
        node.priority = float(node.count * node.cost) / t_next_access

class PerfectKnowledge_PQ(object):
    needs_driver = True
    def __init__(self, driver = None, name = "PK_Freq", **kwargs):
        self.name = name

        self.nodes = SortedCollection(key=attrgetter('priority'))
        self.error_numer = 0
        self.error_denom = 1
        self.driver_access = driver
        self.watch_for_shift = (driver_access and "just_shifted" in driver_access.__dict__)

    def handle_shift(self):
        for node in self.nodes:
            old_p = node.priority
            node.priority = self.driver_access.get_item_pop(node.value) * node.cost
                
        self.nodes.reorder()

    def add_node(self, item, cost, popularity):
        new_node = PQNode(item, cost)
        new_node.priority = popularity * new_node.cost

        self.nodes.insert_right(new_node)
        
        if self.watch_for_shift and self.driver_access.just_shifted:
            self.handle_shift()
            self.driver_access.just_shifted = False

        return new_node

    def touch(self, node):
        if self.watch_for_shift and self.driver_access.just_shifted:
            self.handle_shift()
            self.driver_access.just_shifted = False

        self.nodes.remove(node)
        self.nodes.insert_right(node)

    def evict_node(self):
        to_evict = self.nodes[0]

        del self.nodes[0]

        return to_evict

class GD_PQ_Bucketing_Old(object):
    def __init__(self, bucket_bounds, name = None, **kwargs):
        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__

        self.bucket_bounds = bucket_bounds
        self.bucket_costs = [float(a + b)/2 for a,b in zip([0] + bucket_bounds[:-1], bucket_bounds)]
        self.bucket_frac = [(0,0) for _ in bucket_bounds]
        self.nodes = SortedCollection(key=self.objective_f)
        self.H = 0
        self.time = 0

    def objective_f(self, node):
        node_L, bucket_ix = node.cost
        return (self.bucket_costs[bucket_ix] + node_L, node.priority) # such an absurd objf.

    def update_ordering(self):
#        self.nodes.key = self.objective_f # will resort the collection.
        pass

    def update_bucket(self, bucket_ix, new_cost):
        # default:: just set bucket to newest.
#        self.bucket_costs[bucket_ix] = float(new_cost)

        # do a moving window?
        bucket_frac = self.bucket_frac[bucket_ix]
        new_frac = (bucket_frac[0] + new_cost, bucket_frac[1] + 1)

        self.bucket_frac[bucket_ix] = new_frac
        self.bucket_costs[bucket_ix] = float(new_frac[0])/new_frac[1]

        return self.bucket_costs[bucket_ix]

    def touch(self, node):
        self.nodes.remove(node)

        node.cost = (self.H, node.cost[1])
        node.priority = self.time

        self.time += 1

        self.nodes.insert(node)

    def add_node(self, value, cost):
        bucket_ix = bisect_left(self.bucket_bounds, cost)
        self.update_bucket(bucket_ix, cost)
        self.update_ordering()

        new_node = PQNode(value, (self.H, bucket_ix))
        new_node.priority = self.time
        new_node.real_c = cost

        self.time += 1
        self.nodes.insert(new_node)

        return new_node

    def evict_node(self):
        to_evict = self.nodes[0]

        del self.nodes[0]
        self.H = self.objective_f(to_evict)[0]

        # update the bucket average
        bucket_ix = to_evict.cost[1]
        bucket_frac = self.bucket_frac[bucket_ix]
        new_frac = (bucket_frac[0] - to_evict.real_c, bucket_frac[1] - 1)
        self.bucket_frac[bucket_ix] = new_frac
        if(new_frac[1] == 0):
            self.bucket_costs[bucket_ix] = 0
        else:
            self.bucket_costs[bucket_ix] = float(new_frac[0])/new_frac[1]
        self.update_ordering()

        return to_evict


class GD_PQ_PCoarse(object):
    def __init__(self, n_buckets, name = None):
        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__

        self.nodes_cur = SortedCollection(key = attrgetter("priority"))
        self.nodes_next = SortedCollection(key = attrgetter("priority"))
        self.H = 0
        self.n_buckets = n_buckets
        self.error_numer = 0
        self.error_denom = 1

    def touch(self, node):
        if node in self.nodes_cur:
            self.nodes_cur.remove(node)
        else:
            self.nodes_next.remove(node)

        self._insert_with_p_(node)

    def add_node(self, value, cost):
        if cost >= self.n_buckets:
            print("cost(%f) > n_buckets(%d)") % (cost, self.n_buckets) 
            assert False
        new_node = PQNode(value, cost)
        self._insert_with_p_(new_node)

        return new_node

    def _insert_with_p_(self, node):
        p = int(self.H + node.cost + 0.5)

        if p >= self.n_buckets:
            p -= self.n_buckets
            target_q = self.nodes_next
        else:
            target_q = self.nodes_cur

        node.priority = p
        target_q.insert_right(node)
        
    def evict_node(self):
        if len(self.nodes_cur) == 0:
            t = self.nodes_cur
            self.nodes_cur = self.nodes_next
            self.nodes_next = t
        to_evict = self.nodes_cur[0]

        del self.nodes_cur[0]
        self.H = to_evict.priority

        return to_evict



class GD_PQ_Bucketing(object):
    def __init__(self, bucket_bounds, name = None):
        if name:
            self.name = name
        else:
            self.name = self.__class__.__name__

        self.bucket_bounds = bucket_bounds
        self.bucket_costs = [float(a + b)/2 for a,b in zip([0] + bucket_bounds[:-1], bucket_bounds)]
        self.bucket_frac = [(0,0) for _ in bucket_bounds]
        self.bucket_evictions = [0 for _ in self.bucket_bounds]
        self.bucket_misses = [0 for _ in self.bucket_bounds]

        self.nodes = SortedCollection(key=attrgetter("priority"))
        self.H = 0
        self.time = 0
        
        self.error_numer = 0
        self.error_denom = 1

    def update_bucket(self, bucket_ix, new_cost):
        # average of current contents
        bucket_frac = self.bucket_frac[bucket_ix]
        new_frac = (bucket_frac[0] + new_cost, bucket_frac[1] + 1)

        self.bucket_frac[bucket_ix] = new_frac
        self.bucket_costs[bucket_ix] = float(new_frac[0])/new_frac[1]

        return self.bucket_costs[bucket_ix]

    def touch(self, node):
        self.nodes.remove(node)
        
        b_cost = self.bucket_costs[node.bucket]

        node.cost = b_cost
        node.priority = (self.H + b_cost, self.time)

        self.time += 1

        self.error_numer += abs(b_cost - node.real_c)
        self.error_denom += 1

        self.nodes.insert(node)

    def add_node(self, value, cost):
        bucket_ix = bisect_left(self.bucket_bounds, cost)
        b_cost = self.update_bucket(bucket_ix, cost)

        new_node = PQNode(value, b_cost)
        new_node.priority = (self.H + b_cost, self.time)
        new_node.real_c = cost
        new_node.bucket = bucket_ix

        self.bucket_misses[bucket_ix] += cost

        self.time += 1
        self.nodes.insert(new_node)

        self.error_numer += abs(b_cost - cost)
        self.error_denom += 1

        return new_node

    def evict_node(self):
        to_evict = self.nodes[0]

        del self.nodes[0]
        self.H = to_evict.cost

        # update the bucket average
        bucket_ix = to_evict.bucket
        bucket_frac = self.bucket_frac[bucket_ix]
        new_frac = (bucket_frac[0] - to_evict.real_c, bucket_frac[1] - 1)
        self.bucket_frac[bucket_ix] = new_frac
        if(new_frac[1] == 0):
            self.bucket_costs[bucket_ix] = 0
        else:
            self.bucket_costs[bucket_ix] = float(new_frac[0])/new_frac[1]

        self.bucket_evictions[to_evict.bucket] += to_evict.real_c

        return to_evict

