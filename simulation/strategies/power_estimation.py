import heapq
from utils import zipfian_zeta
import strategies.lru as lru
 
REMOVED = -1

class Item:
    def __init__(self, value):
        self.value = value

class ClassEstimation:
    def __init__(self):
        self.freqs = dict()
        self.n = 0
    def witness(self, value):
        if value not in self.freqs:
            self.freqs[value] = 1
            self.n += 1
        else:
            self.freqs[value] += 1
    def pr_incidence(self, value):
        return self.freqs[value]

class ZipfianClassEstimator:
    """
    
    PDF:
                    1
    f(k) = -------------------
            k^rho * zeta(s,N)
    

    In terms of k:

    _
    k = (f * zeta)^(-1/rho)
    

    """

    def __init__(self, rho, N):
        self.rho  = rho
        self.rho_inv = 1.0 / rho
        self.N    = N
        self.zeta = zipfian_zeta(rho, N)

    def pr_incidence(self, k):
        """ Zipfian PDF for rank K """
        if k <= 0:
            return 100000
        denom = float(k ** self.rho) * self.zeta
        return 1.0/denom


class PerfectModel_CLRU:
    """
    This strategy has "perfect knowledge" of the underlying
    generative model for the requests.
    
    Each cached object type is kept in a separate LRU chain.
    Chains are chosen for eviction based on the incidence 
    probability of the object at the tail of the chain.
    This incidence probability is estimated by using the
    chain length as that item's estimated rank in the underlying
    distribution.
    """
    def __init__(self, k, costs, ratios, distribution_params):
        num_classes = len(costs)
        lru_chains = []
        for i in range(0, num_classes):
            lru_chains.append(lru.LRU_Strategy(k))
        self.ratios = ratios
        self.lru_chains = lru_chains
        self.num_classes = num_classes

        estimators = [ZipfianClassEstimator(rho, N) for (N, rho) in distribution_params]
        self.estimators = estimators
        self.costs = costs

    def evict_node(self):
        estimates = [(self.costs[i] * \
                      self.ratios[i] * \
                      self.estimators[i].pr_incidence(self.lru_chains[i].cur_size), \
                      self.lru_chains[i]) for i in range(0, self.num_classes)]
        _, lru_chain_to_evict = min(estimates) 

        return lru_chain_to_evict.evict_node()

    def add_node(self, value, evicted):
        req_class = value[0]
        return self.lru_chains[req_class].add_node(value, None)
        
    def touch(self, node):
        req_class = node.value[0]
        return self.lru_chains[req_class].touch(node)

class Full_CLFU:
    """ This strategy will estimate using ALL of the requests
    witnessed so far. This strategy is EXPENSIVE.

    Items in Q are only rank-computed on entry...
    that is obviously dumb.
    """
    def __init__(self, k, costs):
        self.Q = []
        self.k = k
        self.costs = costs
        self.estimator = ClassEstimation()

    def evict_node(self):
        _, to_evict = heapq.heappop(self.Q)
        while to_evict.value == REMOVED:
            _, to_evict = heapq.heappop(self.Q)
        return to_evict

    def add_node(self, value, evicted):
        cost = self.costs[value[0]]
        self.estimator.witness(value)
        pr = self.estimator.pr_incidence(value)
        priority = self.priority(pr, cost)
        
        node = Item(value)

        heapq.heappush(self.Q, (priority, node))

        return node

    def priority(self, pr, cost):
        return pr * cost

    def touch(self, node):
        value = node.value
        cost = self.costs[value[0]]
        self.estimator.witness(value)
        pr = self.estimator.pr_incidence(value)
        priority = self.priority(pr, cost)
 
        newNode = Item(value)
        node.value = REMOVED
        heapq.heappush(self.Q, (priority, newNode))
        return newNode


class Full_LFU(Full_CLFU):
    """ This strategy will estimate using ALL of the requests
    witnessed so far. This strategy is EXPENSIVE.

    Items in Q are only rank-computed on entry...
    that is obviously dumb.

    This is a cost-oblivioius strategy
    """

    def priority(self, pr, cost):
        return pr
