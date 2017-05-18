from workloads.cost_driver import ReturnsSize
import json, csv, os

class MemcachierWorkload_Lambda():
    def __init__(self, trace_file_name, trace_info, cache_by_size = False):
        self.loc = trace_info['trace_loc']
        self.trace_file_name = trace_file_name
        self.appid = str(trace_file_name[4:-7])
        self.uniqs = trace_info['uniqs'][trace_file_name]
        self.max_item_size = trace_info['max_item_szs'][trace_file_name]
        self.cache_by_size = cache_by_size
        self.app_allocation = int(trace_info['cache_sizes'][trace_file_name])
        self._reqs = False

    def reqs(self):
        if self._reqs:
            return self._reqs
        self._reqs = sum(1 for line in open(self.loc + self.trace_file_name))
        return self._reqs

    def __call__(self, *args, **kwargs):
        return MemcachierTraceWorkload(trace_file = self.loc + self.trace_file_name,
                                       name = self.trace_file_name, 
                                       cache_by_size = self.cache_by_size)


load_path = os.path.dirname(__file__)
if load_path == '':
    load_path = '.'
def try_load(fname):
    try:
        with open(load_path + "/" + fname, 'r') as f:
            return json.load(f)
    except IOError:
        return {'fnames' : []}


trace1_info = try_load("trace1_info.json")
trace1_info['trace_loc'] = '/disk/scratch1/blanks/memcachier_traces/split-sz/'
trace2_info = try_load("trace2_info.json")
trace2_info['trace_loc'] = '/disk/scratch2/blanks/memcachier_traces/split-sz/'
catenated_info = try_load("catenated_info.json")
catenated_info['trace_loc'] = '/disk/scratch1/blanks/memcachier_traces/cat/'

MIN_SIZE = 10*2

Workload_lambdas_t1 = [MemcachierWorkload_Lambda(f, trace1_info, 
                                                 cache_by_size = True) 
                       for f in trace1_info['fnames']]
Workload_lambdas_t2 = [MemcachierWorkload_Lambda(f, trace2_info,
                                                 cache_by_size = True)
                       for f in trace2_info['fnames']]
Workload_lambdas_cat = [MemcachierWorkload_Lambda(f, catenated_info,
                                                  cache_by_size = True)
                        for f in catenated_info['fnames']]

Workload_lambdas_t1 = [ l for l in Workload_lambdas_t1 if 
                        l.max_item_size * l.uniqs >= MIN_SIZE ] 
Workload_lambdas_t2 = [ l for l in Workload_lambdas_t2 if 
                        l.max_item_size * l.uniqs >= MIN_SIZE ] 
Workload_lambdas_cat = [ l for l in Workload_lambdas_cat if 
                         l.max_item_size * l.uniqs >= MIN_SIZE ] 


class MemcachierTraceWorkload(ReturnsSize):
    def __init__(self, seed = 0, name = None, trace_file = "/dev/null",
                 cache_by_size = False, **kwargs):
        if trace_file.endswith("bz2"):
            self.f = bz2.BZ2File(trace_file, "r")
        elif trace_file.endswith("gz"):
            self.f = gzip.open(trace_file, "r")
        else:
            self.f = open(trace_file, "r")
        
        self.csvreader = csv.reader(self.f)

        self.name = name
        self.cache_by_size = cache_by_size

    def get_next(self):
        try:
            next_row = self.csvreader.next()
        except StopIteration:
            self.f.close()
            self.f = None
            self.csvreader = None
            return -1
        key, size = next_row[0], next_row[1]
        key = key.strip()
        size = int(size.strip())
        assert size != 0
        return key, size
    def sample_item_w_cost(self):
        r_val = self.get_next()
        if r_val == -1:
            return -1
        if self.cache_by_size:
            return (r_val[0], 1.0, r_val[1])
        else:
            return (r_val[0], 1.0)
            
