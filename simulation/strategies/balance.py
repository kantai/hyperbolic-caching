import heapq

REMOVED = -1

class BalanceItem:
    def __init__(self, slack, value):
        self.slack = slack
        self.value = value

class Balance:
    """ IMPLEMENTED """
    def __init__(self, k, costs):
        self.Q = []
        self.k = k
        self.costs = costs

    def evict_node(self):
        to_evict = heapq.heappop(self.Q)
        return to_evict[1]

    def add_node(self, value, evicted):
        slack = self.costs[value[0]]
        if evicted != None:
            slack += evicted.slack
        
        node = BalanceItem(slack, value)
        heapq.heappush(self.Q, (slack, node))

        return node

    def touch(self, node):
        """ BALANCE algorithm does nothing here ! """
        pass


class MBalance(Balance):
    """ IMPLEMENTED """
    def touch(self, node):
        newNode = BalanceItem(node.slack + self.costs[node.value[0]], node.value)
        node.value = REMOVED
        heapq.heappush(self.Q, (newNode.slack, newNode))
        return newNode

    def evict_node(self):
        _, to_evict = heapq.heappop(self.Q)
        while to_evict.value == REMOVED:
            _, to_evict = heapq.heappop(self.Q)
        return to_evict
