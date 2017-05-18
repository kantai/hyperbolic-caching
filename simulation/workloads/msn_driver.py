import datetime as datetime
from dateutil.parser import parse as date_parser
from workloads.cost_driver import ArbitraryDriver, RETURNS_EXPIRY
from random import random

###  I've used this script to generate some access time data for
###  purposes _other_ than as a simulation workload driver. To get
###  those times, you can call this like:
###
###    Generating Access Times:
###      -> python -c "from workloads.msn_driver import MSNDriver; MSNDriver(access_timer = True)" > /tmp/foo
###
###  Anyways -- the data that this driver depends on is not open. If
###  you're interested in obtaining it, you should contact Aaron and
###  he should be able to connect you to the right people to ask.

class MSNDriver(ArbitraryDriver):
    def __init__(self, filename = "workloads/msn_events.txt",
                 compute_expiry = datetime.timedelta(minutes = 10),
                 access_timer = False,
                 seed = 0, item_range_max = 0,
                 **kwargs):
        super(MSNDriver, self).__init__(seed = seed, item_range_max = item_range_max, name = "MSN", **kwargs)

        self.expiring = RETURNS_EXPIRY
        if access_timer:
            self.first_access = {}

        self.max_cost = 1
        with open(filename) as f:
            if compute_expiry != None:
                item_stack = []
            reqs = []

            for ix, line in enumerate(f):
                (ts, obj_id, sz) = line.split()
                sz = int(sz)
                ts = date_parser(ts)

                if access_timer:
                    if obj_id in self.first_access:
                        already_merged, first_ts = self.first_access[obj_id]
                        if not already_merged:
                            self.first_access[obj_id] = (True, (ts - first_ts).total_seconds())
                    else:
                        self.first_access[obj_id] = (False, ts)
                if compute_expiry != None:
                    expiry = compute_expiry
#                    if random() > 0.5:
#                        expiry = datetime.timedelta(minutes = 5)
                    expires_at = ts + expiry
                    item_stack.append( (obj_id, sz, expires_at) )
                    if (sz > self.max_cost):
                        self.max_cost = sz
                    while len(item_stack) > 0 and item_stack[0][2] <= ts:
                        cur = item_stack.pop(0)
                        reqs.append( (cur[0], cur[1], ix) )
            infinity = -1
            for item in item_stack:
                reqs.append( (item[0], item[1], infinity) )
            self.iterate = 0
            self.reqs = reqs
            self.max_iter = len(reqs)

            if access_timer:
                a = [snd for fst, snd in self.first_access.values() if fst]
                for x in a:
                    print x

    def sample_item_w_cost(self):
        if self.iterate >= self.max_iter:
            return -1
        objid, cost, expires = self.reqs[self.iterate]
        cost /= self.max_cost
        self.iterate += 1
        return (objid, 1.0, expires)
