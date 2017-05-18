from numpy.random import RandomState
from bisect import bisect_left
import cost_driver

####
#### These workloads are based on the workload descriptions 
#### found in the GDWheel paper.
#### 
#### Li, Conglong, and Alan L. Cox. "GD-Wheel: a cost-aware
#### replacement policy for key-value stores." Proceedings of the
#### Tenth European Conference on Computer Systems. ACM, 2015.
####

class GDWheelDriver(cost_driver.ZipfUniformDriver):
    def __init__(self, cost_groups, cost_group_prob, **kwargs):
        super(GDWheelDriver, self).__init__(**kwargs)

        self.cost_scalar = max([u for l,u in cost_groups])

        self.cost_groups = cost_groups
        cdfs = []
        cur = 0.0
        for p in cost_group_prob:
            cur += p
            cdfs.append(cur)
        self.cost_cdfs = cdfs
        self.cost_sample = []

    def get_cost(self, r_float, item_num):
        cost_float = super(GDWheelDriver, self).get_cost(r_float, item_num)
        
        cost_group = bisect_left(self.cost_cdfs, cost_float)
        cost_lower, cost_upper = self.cost_groups[cost_group]

        
        if cost_group == 0:
            unif2 = cost_float / self.cost_cdfs[0]
        else:
            cost_float -= self.cost_cdfs[cost_group - 1]
            unif2 = cost_float / (self.cost_cdfs[cost_group] - self.cost_cdfs[cost_group - 1])
        
        return ((unif2 * (cost_upper - cost_lower)) + cost_lower)/self.cost_scalar
        
def make_constructor(cost_groups, cost_group_prob, name):
    return (lambda **kwargs : GDWheelDriver(cost_groups = cost_groups,
                                            cost_group_prob = cost_group_prob,
                                            name = name,
                                            **kwargs))

COST_D1 = [(10, 30), (120, 180), (350, 450)]


GD1 = make_constructor(COST_D1, [.8, .15, .05], "GD1")
GD2 = make_constructor(COST_D1, [.2, .75, .05], "GD2")
GD3 = make_constructor(COST_D1, [.5, .25, .25], "GD3")
GD_NO_COST = make_constructor([(1,1)], [1], "GDNoCost")
