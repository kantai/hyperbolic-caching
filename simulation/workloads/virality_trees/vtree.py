###  Workload generator for Python simulator.
###  Reads json file and samples according to virality or popularity
###
###
###  Anyways -- the data that this driver depends on is not open. If
###  you're interested in obtaining it, you should contact Aaron and
###  he should be able to connect you to the right people (Microsoft)
###  to ask.

from workloads.cost_driver import ArbitraryDriver
from bisect import bisect_left
import json

def process_json_item_default(x, sample_by = 'virality'):
    cost = sum( [ (x[key][0] * 10**9) + x[key][1] for key in
                  ['elapsed_layout', 'elapsed_constr'] ])
    cost_us = float(cost) / (10**3)
    return (float(x[sample_by]), float(cost_us), x['url_id'])

class ViralityTreeDriver(ArbitraryDriver):
    def __init__(self, seed, 
                 item_range_max = -1,
                 json_processor = process_json_item_default,
                 filename = "workloads/virality_trees/output-40k.json", **kwargs):
        super(ViralityTreeDriver, self).__init__(seed, item_range_max, **kwargs)
        # load json
        jsonl = json.load(open(filename))
        items = [json_processor(x) for x in jsonl]

        # order by virality / popularity
        items.sort(reverse=True)

        # normalize and itemize
        prob_norm = sum([ x[0] for x in items ])
        cost_max = max([ x[1] for x in items ])

        cost_norm = ((2**32 - 2) / cost_max)
        self.cost_norm = cost_norm

        self.item_space = [ (ix, x[0] / prob_norm, int(x[1] * cost_norm) + 1) + x[2:] 
                            for ix,x in enumerate(items) ]

        self.max_item = len(self.item_space) - 1
        # construct probability lookup array
        accum = 0.0
        lookup = []
        for x in self.item_space:
            accum += x[1]
            lookup.append(accum)
        lookup[-1] = 1.0 # ensure that we don't have funky float-y flukes
        self.lookup = lookup

    def get_item(self, r_float):
        return bisect_left(self.lookup, r_float)
    
    def get_cost(self, r_float, item_num):
        return self.item_space[item_num][2]

    def print_stuff(self):
        for i in self.item_space:
            print ", ".join(["%s" % x for x in i[:3]])

if __name__ == "__main__":
    d = ViralityTreeDriver(0)
    print d.cost_norm
