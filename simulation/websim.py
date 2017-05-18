import pycurl
from StringIO import StringIO
import multiprocessing as mp
from workloads import cost_driver, spc_traces, gdwheel, developer_ubuntu
from workloads.virality_trees import vtree
import sys
import time
import numpy as np
import random

d_lambdas = [(lambda seed, nclients : 
              cost_driver.ZipfFixedDriver(seed = seed, 
                                          item_range_max = 10**5)),
             (lambda seed, nclients:
              spc_traces.SPCTraceDriver(trace_file = "workloads/spc1/WebSearch1.spc", 
                                        name = "SPCWebSearch", 
                                        column_map = spc_traces.SPCMap1, 
                                        seed = (seed, nclients))),
             (lambda seed, nclients : 
              cost_driver.ZipfFixedDriver(seed = seed, zipf_param = 0.75,
                                          item_range_max = 10**6)), # = 2
             (lambda seed, nclients: 
              cost_driver.DynamicPromote(period = 100).wrap_driver(cost_driver.ZipfFixedDriver(seed = seed, 
                                                                                               item_range_max = 10**5))), # = 3
             (lambda seed, nclients: 
              cost_driver.Introducing(period = 100, 
                                      move_to = 0, name = "IntHigh.%d").wrap_driver(cost_driver.ZipfFixedDriver(seed = seed, 
                                                                                                                item_range_max = 10**5))), # = 4
             (lambda seed, nclients:
              vtree.ViralityTreeDriver(seed = seed, name = "VTree(40k)", filename = "workloads/virality_trees/output-40k.json")),
             (lambda seed, nclients:
              gdwheel.GD2(seed = seed, max_cost = (2**32 - 2), item_range_max = 10**5)),
             (lambda seed, nclients:
              spc_traces.SPCTraceDriver(trace_file = "workloads/arc/S1.lis", 
                                        name = "Arc.S1", 
                                        column_map = spc_traces.ARCMap,)), #=7
             (lambda seed, nclients : 
              cost_driver.ZipfFixedDriver(seed = seed, 
                                          item_range_max = 10**6)), # = 8
             (lambda seed, nclients : 
              cost_driver.ZipfFixedDriver(seed = seed, zipf_param = 0.9,
                                          item_range_max = 10**6)), # = 9
             (lambda seed, nclients :
              developer_ubuntu.URLDriver(seed = seed)
              ), # = 10
             (lambda seed, nclients :
              developer_ubuntu.WikiZURLDriver(seed = seed, item_range_max = 10**6,)
              ), # = 11
             ]



PROGRESS_CTR_T = mp.Array('i', [0 for i in range(1024)], lock=False)
MEASURING = mp.Value('i', 0, lock = True)

REQ_ARRS = None
COST_ARRS = None

COST_DRIVERS = [5, 6]

SEED = random.randint(0, 10000)

def update_progress_bar(progress, total):
    col_size = 65
    
    count = (col_size * progress) / total

    sys.stderr.write("\r")
    sys.stderr.write("[" + "".join(["=" for i in range(count)]))
    sys.stderr.write(">")
    sys.stderr.write("".join([" " for i in range(col_size - count - 1)]))
    sys.stderr.write("] %.1f%%" % (float(100 * progress) / total))
    sys.stderr.flush()

def generate_req_arr(atuple):
    driver_lambda, seed, nreqs, nclients = atuple
    driver = d_lambdas[driver_lambda](seed+SEED, nclients)
    cost_out = None
    gen_costs = driver_lambda in COST_DRIVERS
    if gen_costs:
        cost_out = np.zeros(nreqs, np.uint32)

    using_np = True

    for i in range(nreqs):
        item, cost = driver.sample_item_w_cost()
        if i == 0:
            using_np = not isinstance(item, str)
            if using_np:
                out = np.zeros(nreqs, np.uint32)                
            else:
                out = []

        if using_np:
            out[i] = item
        else:
            out.append(item)

        if gen_costs:
            cost_out[i] = int(cost + 1)

    return (out, cost_out)

class WorkloadRunner(object):
    def __init__(self, driver_lambda, host ="sns49.cs.princeton.edu:3590", # "128.112.7.141:3590", 
                 nclients = 8, nreqs = 10**6):
        self.driver_lambda = driver_lambda

        self.cost = self.driver_lambda in COST_DRIVERS

        self.nclients = nclients
        if self.cost:
            self.host_str = "http://%s/%%s/%%s" % host
        else:
            self.host_str = "http://%s/%%s" % host

        self.workload = [(nreqs, self.host_str, seed, self.cost)
                         for seed in range(nclients)]
        self.nreqs = nreqs

    def bench_run(self):
        global PROGRESS_CTR_T, MEASURING 
        global REQ_ARRS
        global COST_ARRS


        if self.nclients == 1:
            results = map(generate_req_arr, 
                          [(self.driver_lambda, i, self.nreqs, self.nclients) 
                           for i in range(self.nclients)])
            REQ_ARRS = []
            COST_ARRS = []
            for (out_arr, cost_arr) in results:
                REQ_ARRS.append(out_arr)
                COST_ARRS.append(cost_arr)
            t_start = time.time()
            results = map(work_driver, self.workload)
            t_end = time.time()
            requests = sum([reqs for reqs, _ in results])
            misses = sum([misses for _, misses in results])

            throughput = float(requests) / (t_end - t_start)
            miss_rate = float(misses) / float(requests)
            
            return throughput, miss_rate, requests, (t_end - t_start)


        t_start = time.time()
        pool = mp.Pool(processes = 16)
        results = pool.map(generate_req_arr, 
                           [(self.driver_lambda, i, self.nreqs, self.nclients) 
                            for i in range(self.nclients)])
        
        REQ_ARRS = []
        COST_ARRS = []
        
        for (out_arr, cost_arr) in results:
            REQ_ARRS.append(out_arr)
            COST_ARRS.append(cost_arr)
        pool.close()
        t_end = time.time()
        sys.stderr.write("finished building request arrays in %f s" % ( t_end - t_start ))
        sys.stderr.write("\n")
        sys.stderr.flush()

        pool = mp.Pool(processes = self.nclients)        
        result_async = pool.map_async(work_driver, self.workload)
        t_start = time.time()

        update_time = 5
        start_window = 30 / update_time
        end_window = 1 * 30 / update_time
        end_window += update_time
        updates = 0

        while not result_async.ready():
            result_async.wait(5)
            updates += 1
            update_progress_bar(sum(PROGRESS_CTR_T), 
                                self.nclients * self.nreqs)
        
        t_end = time.time()

        sys.stderr.write("\n")
        sys.stderr.flush()
        results = result_async.get()

        requests = sum([reqs for reqs, _ in results])
        misses = sum([misses for _, misses in results])

        throughput = float(requests) / (t_end - t_start)
        miss_rate = float(misses) / float(requests)
        
        pool.close()
        return throughput, miss_rate, requests, (t_end - t_start)


def work_driver(arg_tup):
    global PROGRESS_CTR_T, MEASURING
    global REQ_ARRS
    global COST_ARRS
    nreqs, host_str, req_arr_ix, send_cost = arg_tup
    req_arr = REQ_ARRS[req_arr_ix]
    cost_arr = COST_ARRS[req_arr_ix]

    success = 0
    miss = 0

    NCURL = 128
    curl_hands = []
    for i in range(NCURL):
        c = pycurl.Curl()
        curl_hands.append(c)
    n = 0
    for ix, item in enumerate(req_arr):
        c = curl_hands[ix % NCURL]
        try:
            if send_cost:
                url = host_str % (item, cost_arr[ix])
            else:
                url = host_str % item
            c.setopt(c.URL, url)
            resp = StringIO()
            headers = StringIO()
            c.setopt(c.WRITEFUNCTION, resp.write)
            c.setopt(c.HEADERFUNCTION, headers.write)
            c.setopt(pycurl.CONNECTTIMEOUT, 20)
            c.setopt(pycurl.TIMEOUT, 20)
            c.perform()

            if c.getinfo(c.RESPONSE_CODE) == 200:
                success += 1
                is_hit = handle_response(resp, headers)
                if not is_hit:
                    miss += 1

        except Exception as e:
            pass

        PROGRESS_CTR_T[req_arr_ix] = ix

    return (success, miss)


HEADER_STRING = "X-CACHED-MIDDLEWARE: "
def handle_response(resp, headers):
    """
    return TRUE = HIT, FALSE = MISS
    """
    val = resp.getvalue()
    head_val = headers.getvalue()
    index = head_val.find("X-CACHED-MIDDLEWARE: ")
    if index == -1:
        return False
    response = head_val[index + len(HEADER_STRING)]
    return response == "H"
        

def unitc_zpop(d_lam, nr, nc, host = False):
    if host:
        WL = WorkloadRunner(d_lam, nclients = nc, nreqs = nr, host = host)
    else:
        WL = WorkloadRunner(d_lam, nclients = nc, nreqs = nr)
    return WL.bench_run()

if __name__ in ('main', '__main__'):
    if len(sys.argv) < 3:
        unitc_zpop(0, 10**3, 1)
    else:
        nclients = int(sys.argv[2])
        nreqs = int(float(sys.argv[3]))
        host = False
        if len(sys.argv) > 4:
            host = sys.argv[4]
        throughput, missrate, reqs, time = unitc_zpop(int(sys.argv[1]), nreqs, nclients, host)

        if len(sys.argv) > 3:
            print "%s, %s, %f, %f, %d, %s" % (sys.argv[5], sys.argv[6], throughput, missrate, reqs, time)
