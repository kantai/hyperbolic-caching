#from lru import LinkedList, LLNode
from llist import dllist

class EvictsAndAdds(object):
    pass

class Wrapper:
    def __init__(self, value):
        self.value = value

class Arc(EvictsAndAdds):
    def __init__(self,  **kwargs):
        self.PLEASE_SET_k = -1

        self.T1 = dllist()
        self.T2 = dllist()
        self.B1 = dllist()
        self.B2 = dllist()
        self.p = 0
        self.shadow_cache = {}

        self.name = "ARC"

    def touch(self, node):
        LRU_list, llNode = node

        # CASE I
        if LRU_list == self.T1 or LRU_list == self.T2:
            value = LRU_list.remove(llNode)
            llNode = self.T2.insert(value)
        else:
            raise ValueError("hit on item not in T1 or T2!")
        
        return (self.T2, llNode)

    def evict_and_add(self, value, cost):
        K = self.PLEASE_SET_k
        if K == -1:
            raise ValueError("attempt to use cache before setting K!")

        if value in self.shadow_cache:
            LRU_list, llNode = self.shadow_cache.pop(value)
            if LRU_list == self.B1:
                # CASE II
                if len(self.B1) >= len(self.B2):
                    delta = 1
                else:
                    delta = float(len(self.B2)) / len(self.B1)
                self.p = min(self.p + delta, K)
                evict = Wrapper(self.REPLACE(value, False))
            elif LRU_list == self.B2:
                # CASE III
                if len(self.B2) >= len(self.B1):
                    delta = 1
                else:
                    delta = float(len(self.B1)) / len(self.B2)
                self.p = max(self.p - delta, 0)
                evict = Wrapper(self.REPLACE(value, True))
            else:
                raise ValueError("miss on item in shadow cache, but not B1 or B2!")
        
            removedValue = LRU_list.remove(llNode)

            llNode = self.T2.insert(removedValue)
            return (evict, (self.T2, llNode))
        else:
            # CASE IV
            evict = False
            if len(self.T1) + len(self.B1) == K:
                # CASE A
                if (len(self.T1) < K):
                    del self.shadow_cache[self.B1.popleft()]
                    evict = Wrapper(self.REPLACE(value, False))
                else:
                    evict = Wrapper(self.T1.popleft())
            elif len(self.T1) + len(self.B1) < K:
                # CASE B
                if len(self.T1) + len(self.T2) + len(self.B1) + len(self.B2) >= K:
                    if len(self.T1) + len(self.T2) + len(self.B1) + len(self.B2) >= 2*K:
                        b2removal = self.B2.popleft()
                        del self.shadow_cache[b2removal]
                    evict = Wrapper(self.REPLACE(value, False))
            new_node = self.T1.insert(value)
            return (evict, (self.T1, new_node))
        
    def REPLACE(self, value, is_value_in_B2):
        if len(self.T1) > 0 and (len(self.T1) > self.p or (is_value_in_B2 and len(self.T1) == self.p)):
            evict = self.T1.popleft()
            llNode = self.B1.insert(evict)
            self.shadow_cache[evict] = (self.B1, llNode)
        else:
            evict = self.T2.popleft()
            llNode = self.B2.insert(evict)
            self.shadow_cache[evict] = (self.B2, llNode)
        return evict
