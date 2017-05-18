from workloads import cost_driver, memcachier
import csv, sys
from llist import dllist
from collections import defaultdict
import random as R

def trace_next_uses(driver, n, out = csv.writer(sys.stdout)):
    """
    Function creates a trace where "next request time" has been
    calculated.
    """
    handleExpirationReturn = isinstance(driver, cost_driver.ReturnsExpiry)
    handleSizeReturn = isinstance(driver, cost_driver.ReturnsSize)

    request_queue = dllist()

    last_use_cache = {}

    time = -1
    while time < n or n == -1:
        time += 1

        req_tuple = driver.sample_item_w_cost()
        if req_tuple == -1:
            break
        item, cost = req_tuple[:2]
        remainder = req_tuple[2:]
        expiry = -1
        if handleExpirationReturn:
            expiry = remainder[0]
            remainder = remainder[1:]
        size = 1
        if handleSizeReturn:
            size = remainder[0]
            remainder = remainder[1:]

        item_tuple = (time, item, cost, size, expiry, False)

        if item in last_use_cache:
            last_use = last_use_cache[item]
            last_tuple = last_use.value

            last_use.value = last_tuple[:-1] + (time,)
            

            # check if we can begin clearing the list...
            while request_queue.first.value[-1]:
                cur_tuple = request_queue.popleft()
                out.writerow(cur_tuple)
        
        ll_item = request_queue.insert(item_tuple)
        last_use_cache[item] = ll_item
    # flush the queue ?
    while request_queue.size > 0:
        cur_tuple = request_queue.popleft()
        out.writerow(cur_tuple[:-1] + ("-1",))        
    # blanks note : the way this gets counted may create a bunch of 
    # -1's towards the tail of the request stream that aren't really 
    # representative...


class LastKFeature:
    def __init__(self, k = 1):
        self.k = k
    def initialize_feature(self, time):
        return [time]
    def update_feature(self, val, time = -1, **kwargs):
        while len(val) >= self.k:
            val.pop()
        val.insert(0, time)
    def get_feature(self, val, time):
        return val

class Frequency:
    def initialize_feature(self, time):
        return [int(time) - 1, 0]
    
    def update_feature(self, val, **kwargs):
        val[1] += 1

    def get_feature(self, val, time):
        return [val[1], int(val[1]) / (int(time) - int(val[0]))]

class TimeOfEntry:
    def initialize_feature(self, time):
        return time
    def update_feature(self, val, **kwargs):
        pass
    def get_feature(self, val, time):
        return [val]

def featurize_trace(features, reader, writer):
    """
    Reads a trace produced by trace_next_uses and adds
    features to item requests
    """
    
    
    info_dict = defaultdict(lambda : [])
    
    for row in reader:        
        time, item, cost, size, expiry, next_use = row

        info = info_dict[item]

        if info == []:
            for f in features:
                info.append(f.initialize_feature(time))
        
        for f, feature_val in zip(features, info):
            f.update_feature(feature_val, time = time, cost = cost,
                             size = size, expiry = expiry)

        # when to output the features???
        output = row[:-1]
        for f, feature_val in zip(features, info):
            output += f.get_feature(feature_val, time)
        output += [row[-1]]
        
        writer.writerow(output)

def run_samplers(request_table, out_table, load_time = 5000, sample = 16, seed = None, requested_again = True):
    """
    requested_again := if true, only items which are requested 
    again will be tracked.
    """
    item_info = {}
    item_sampler = []

    for row in request_table:
        time, item = row[:2]
        item_requested_again = row[-1] != "-1"
        if requested_again and (not item_requested_again):
            if item in item_info:
                ix = item_sampler.index(item)
                del item_sampler[ix]
                del item_info[item]
        else:
            if item not in item_info:
                item_sampler.append(item)
            item_info[item] = row[2:]
        
        
        if int(time) > load_time and len(item_sampler) > 0:
            # emit some samples
            sample_size = min( len(item_sampler), sample )
            item_samples =  R.sample(item_sampler, sample_size)
            out_rows = [ [time, item] + item_info[item] for item in item_samples ]
            
            for out_row in out_rows:
                out_table.writerow(out_row)

def count_uses(fname):
    with open(fname, 'r') as f_in:
        csv_in = csv.reader(f_in)
        d = dict()
        for l in csv_in:
            key = l[0]
            sz = l[1]
            if key in d:
                cur = d[key]
                d[key] = (cur[0] + 1, cur[1])
            else:
                d[key] = (0, sz)
        output = [(b,c, a) for a,(b,c) in d.items()]
        output.sort(reverse = True)
        
        csv_out = csv.writer(sys.stdout)
        for out_row in output:
            csv_out.writerow(out_row)

def run_workload_memcachier(ix):
    out_file = '/tmp/memcachier_cat_featured_%d' % ix
    tmp_file = '/tmp/tmp_mc_cat_feats_%d' % ix
    
    workload = memcachier.Workload_lambdas_cat[ix]()

    with open(tmp_file, 'w') as csv_tmp_out:
        tmp_out = csv.writer(csv_tmp_out)
        trace_next_uses(workload, -1, tmp_out)

    features = [ LastKFeature(), Frequency(), TimeOfEntry() ]

    with open(tmp_file, 'r') as csv_tmp_in:
        tmp_in = csv.reader(csv_tmp_in)
        with open(out_file, 'w') as csv_out:
            out = csv.writer(csv_out)
            featurize_trace(features, tmp_in, out)


def create_data_set(ix):
    req_file = '/tmp/memcachier_cat_featured_%d' % ix
    out_file = '/tmp/memcachier_cat_sampleset_%d' % ix
    
    with open(req_file, 'r') as fd_in:
        csv_fd_in = csv.reader(fd_in)
        with open(out_file, 'w') as fd_out:
            output = csv.writer(fd_out)
            run_samplers(csv_fd_in, output)
