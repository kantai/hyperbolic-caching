from bucketing import AveragingBucketUpkeep
from bisect import bisect_left
from llist import dllist
from arc import Wrapper

class LLNode:
    def __init__(self, value):
        self.next = None
        self.prev = None
        self.value = value
        self.cost = 0

    def remove(self):
        if self.next != None:
            self.next.prev = self.prev
        if self.prev != None:
            self.prev.next = self.next

        self.next = None
        self.prev = None

    def move_before(self, before):
        if self == before:
            return
        self.remove()
        if before == None:
            return

        self.prev = before.prev
        before.prev = self
        self.next = before
        if self.prev != None:
            self.prev.next = self

    def nextN(self, n):
        cur = self
        while n > 0 and cur.next:
            n -= 1
            cur = cur.next
        return cur

    def prevN(self, n):
        cur = self
        while n > 0 and cur.prev:
            n -= 1
            cur = cur.prev
        return cur

class LinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def remove(self, node):
        if node == None:
            return
        if node == self.head:
            self.head = node.next
        if node == self.tail:
            self.tail = node.prev
        node.remove()
        self.length -= 1

    def pushHead(self, node):
        assert node != None
        if self.head:
            node.move_before(self.head)
        else:
            self.tail = node
        self.head = node
        self.length += 1

    def popTail(self):
        if self.tail == None:
            if self.length != 0:
                print self.length
                assert self.length == 0
            return None

        r = self.tail
        if self.head == self.tail:
            self.head = None

        self.tail = r.prev

        r.remove()
        self.length -= 1
        return r

    def get_length(self):
        cur = self.head
        count = 0
        while cur != None:
            count += 1
            cur = cur.next
        return count

    def apply(self):
        cur = self.head
        while cur != None:
            yield cur
            cur = cur.next

#    def __repr__(self):
#        return "[" + " ".join(["%d" % i.value for i in self.apply()]) + "]"

    def __len__(self):
        if self.length != self.get_length():
            print "%r" % self
            print "%d != %d" % (self.length, self.get_length())
            assert False
        return self.length
            
class LRU_Strategy:
    """ IMPLEMENTED """
    def __init__(self, **kwargs):
        self.name = "LRU"
        self.q = dllist()
        self.cur_size = 0

    def evict_node(self):
        evictee = self.q.popleft()
        self.cur_size -= 1
        return Wrapper(evictee)

    def touch(self, node):
        value = self.q.remove(node)
        llnode = self.q.insert(value)
        return llnode

    def add_node(self, value, cost):
        llnode = self.q.insert(value)

        self.cur_size += 1

        return llnode


class MultiLRU_Strategy:
    def __init__(self, queue_boundaries, bucket_upkeep = AveragingBucketUpkeep):

        if type(queue_boundaries) is int:
            queue_boundaries = list(range(1, queue_boundaries + 1))

        scale = max(queue_boundaries)
        # so strange, but costs always use the full range of 2**32 - 1
        queue_boundaries = [float(int((2**32 - 1) * float(b) / scale)) for b in 
                            queue_boundaries]
        self.name = "MLRU_%d" % len(queue_boundaries)
        self.queue_boundaries = queue_boundaries
        self.queues = [LRU_Strategy() for _ in self.queue_boundaries]
        self.bucket_upkeeps = [bucket_upkeep() for _ in self.queue_boundaries]
        self.ideal_sizes = [ 1.0 / len(queue_boundaries) for i in queue_boundaries]

        self.DEGRADE_MISS = 1

    def update_ideal_sizes(self):
        priorities = [ q.frequency * q.repr_cost for q in self.queues ]
        scale = max(priorities)
        if scale != 0:
            self.ideal_sizes = [ float(p) / scale for p in priorities ]

    def touch(self, node):
        queue_ix = bisect_left(self.queue_boundaries, node.cost)
        queue = self.queues[queue_ix]

        queue.touch(node)
        queue.frequency += 1

#        queue.repr_cost = self.bucket_upkeeps[queue_ix].add_cost(node.cost)

        self.update_ideal_sizes()

    def register_miss(self, queue, cost):
        for q in self.queues:
            q.misses *= self.DEGRADE_MISS 
        queue.misses += cost

    def add_node(self, item, cost):
        queue_ix = bisect_left(self.queue_boundaries, cost)
        if queue_ix >= len(self.queues):
            print cost

        queue = self.queues[queue_ix]

        node = queue.add_node(item)
        node.cost = cost

        self.register_miss(queue, cost)
        
        queue.frequency += 1
        queue.repr_cost = self.bucket_upkeeps[queue_ix].add_cost(cost)

        self.update_ideal_sizes()

        return node
    
    def evict_node(self):
        queue_sizes = [ q.cur_size for q in self.queues ]
        scale = max(queue_sizes)
        #        ideal_distance = [ ((float(sz) / scale) - ideal_size, ix) for ix, (sz, ideal_size) in
        #                           enumerate(zip(queue_sizes, self.ideal_sizes)) ]
        #        candidate_qs = sorted(ideal_distance)
        candidate_qs = sorted([ (q.misses, q_ix) for q_ix, q in enumerate(self.queues) ])

        for _, evict_from in candidate_qs:
            q_evict_from = self.queues[evict_from]
            evicted = q_evict_from.evict_node()
            
            if evicted != False:
                q_evict_from.repr_cost = self.bucket_upkeeps[evict_from].rem_cost(evicted.cost)
                q_evict_from.evicted += 1
                return evicted

        return False

class Counting_LRU_Strategy:
    """ LRU which tracks WHERE a lookup suceeded.
        Used to generate miss rate vs. lru size graph model
    """
    def __init__(self, k):
        self.lru_head = None
        self.lru_tail = None
        self.cur_size = 0
        self.touch_data = []

    def evict_node(self):
        to_evict = self.lru_tail
        if to_evict:
            self.lru_tail = to_evict.prev
        if to_evict == self.lru_head:
            self.lru_head = None
        to_evict.remove()


        self.cur_size -= 1
        return to_evict

    def position(self, node):
        pos = 0
        while node != self.lru_head:
            node = node.prev
            pos += 1
        return pos

    def touch(self, node):
        self.touch_data.append(self.position(node))

        if node == self.lru_head:
            return

        if self.lru_tail == node:
            self.lru_tail = node.prev
            # node.prev cannot be null here, because then it 
            # would have been == lru_head. 
        node.move_before(self.lru_head)

        self.lru_head = node

    def add_node(self, value, evicted):
        new_lru = LLNode(value)

        new_lru.move_before(self.lru_head)
        self.lru_head = new_lru
        if self.lru_tail == None:
            self.lru_tail = self.lru_head

        self.cur_size += 1

        return new_lru


class Weighted_LRU_Strategy:
    """ IMPLEMENTED """
    def __init__(self, k, costs):
        self.lru_head = None
        self.lru_tail = None
        self.k = k
        self.max_cost = max(costs)
        self.cost = costs
        self.n = 0

    def evict_node(self):
        to_evict = self.lru_tail
        if to_evict:
            self.lru_tail = to_evict.prev
        if to_evict == self.lru_head:
            self.lru_head = None
        to_evict.remove()

        self.n -= 1

        return to_evict

    def touch(self, node):
        if node == self.lru_head:
            return

        if self.lru_tail == node:
            self.lru_tail = node.prev
            # node.prev cannot be null here, because then it 
            # would have been == lru_head. 

        cost = self.cost[node.value[0]]
        push = (self.k * cost) / self.max_cost

        move_to = node.prevN( push )

        node.move_before(move_to)

        if move_to == self.lru_head:
            self.lru_head = node

    def add_node(self, value, evicted):
        new_lru = LLNode(value)

        self.n += 1

        new_lru.move_before(self.lru_head)
        self.lru_head = new_lru
        if self.lru_tail == None:
            self.lru_tail = self.lru_head

        return new_lru

