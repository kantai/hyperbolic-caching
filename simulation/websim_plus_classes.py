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
              cost_driver.ZPop_HotCostClass(seed = seed, 
                                            hot_every = 2*10**3,
                                            item_range_max = 10**5)),
             (lambda seed, nclients :
              cost_driver.ZPop_2Classes(seed = seed,
                                        item_range_max = 10**5))]


PROGRESS_CTR_T = mp.Array('i', [0 for i in range(1024)], lock=False)
MEASURING = mp.Value('i', 0, lock = True)

REQ_ARRS = None
COST_ARRS = None
CLASS_ARRS = None

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
    cost_out = np.zeros(nreqs, np.uint32)
    class_out = np.zeros(nreqs, np.uint32)

    using_np = True

    for i in range(nreqs):
        item, cost = driver.sample_item_w_cost()
        objclass = 1+driver.get_cost_class(item)
        cost *= 100
        cost = max(int(cost), 1)
        assert cost <= 100

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

        cost_out[i] = int(cost + 1)
        class_out[i] = objclass

    return (out, cost_out, class_out)

class WorkloadRunner(object):
    def __init__(self, driver_lambda, host ="sns49.cs.princeton.edu:3590", # "128.112.7.141:3590", 
                 nclients = 8, nreqs = 10**6, name = "/tmp/"):
        self.driver_lambda = driver_lambda

        self.nclients = nclients
        self.host_str = "http://%s/%%s/%%s/%%s" % host

        self.workload = [(nreqs, self.host_str, seed)
                         for seed in range(nclients)]
        self.nreqs = nreqs
        self.name = name

    def bench_run(self):
        global PROGRESS_CTR_T, MEASURING 
        global REQ_ARRS
        global COST_ARRS
        global CLASS_ARRS

        t_start = time.time()
        if self.nclients == 1:
            results = map(generate_req_arr, 
                          [(self.driver_lambda, i, self.nreqs, self.nclients) 
                           for i in range(self.nclients)])
        else:
            pool = mp.Pool(processes = 16)
            results = pool.map(generate_req_arr, 
                           [(self.driver_lambda, i, self.nreqs, self.nclients) 
                            for i in range(self.nclients)])


        REQ_ARRS = []
        COST_ARRS = []
        CLASS_ARRS = []
        for (out_arr, cost_arr, class_arr) in results:
            REQ_ARRS.append(out_arr)
            COST_ARRS.append(cost_arr)
            CLASS_ARRS.append(class_arr)
        
        if self.nclients == 1:
            t_start = time.time()
            results = map(work_driver, self.workload)
            t_end = time.time()
            requests = sum([reqs for reqs, _, _ in results])
            misses = sum([misses for _, misses, _ in results])
            costs = sum([cost for _, _, cost in results])

            throughput = float(requests) / (t_end - t_start)
            miss_rate = float(misses) / float(requests)
            
            return throughput, miss_rate, requests, (t_end - t_start), costs


        pool.close()
        t_end = time.time()
        sys.stderr.write("finished building request arrays in %f s" % ( t_end - t_start ))
        sys.stderr.write("\n")
        sys.stderr.flush()

        pool = mp.Pool(processes = self.nclients)        
        result_async = pool.map_async(work_driver, self.workload)
        t_start = time.time()

        last_stats = 0
        time_wait_s = 15
        last_ts = 0
        
        time_total = 60 * 45
        
        fts = open(self.name + "tput_secs.csv", "w")

        while not result_async.ready():
            result_async.wait(time_wait_s)
            stats = sum(PROGRESS_CTR_T)
            last_ts += time_wait_s

            ts_tput = float(stats - last_stats) / time_wait_s
            fts.write("%d, %f\n" % (last_ts, ts_tput))

            last_stats = stats
            update_progress_bar(last_ts, time_total)

            update_progress_bar(sum(PROGRESS_CTR_T), 
                                self.nclients * self.nreqs)
#            if last_ts >= (time_total):
#                pool.terminate()
#                result_async = NullResult()
#                break
        
        t_end = time.time()
        fts.close()

 #       import gzip

#        freqs = gzip.open(self.name + "_reqs.csv.gz", "wb")
#        for i in range(self.nclients):
#            with open(TMP_FILE_NAME % i, "r") as f_in:
#                for l in f_in:
#                    freqs.write(l)
#        freqs.close()

        sys.stderr.write("\n")
        sys.stderr.flush()
        results = result_async.get()

        requests = sum([successes for successes, _, _ in results])
        misses = sum([misses for _, misses, _ in results])
        costs = sum([cost for _, _, cost in results])

        throughput = float(requests) / (t_end - t_start)
        miss_rate = float(misses) / max(float(requests), 1)
        
        pool.close()
        return throughput, miss_rate, requests, (t_end - t_start), costs

class NullResult:
    def get(self):
        return []

TMP_FILE_NAME = "/tmp/latency_grabs_temporary_file_%d"

def work_driver(arg_tup):
    global PROGRESS_CTR_T, MEASURING
    global REQ_ARRS, COST_ARRS, CLASS_ARRS
    nreqs, host_str, req_arr_ix = arg_tup
    req_arr = REQ_ARRS[req_arr_ix]
    cost_arr = COST_ARRS[req_arr_ix]
    class_arr = CLASS_ARRS[req_arr_ix]

    success = 0
    miss = 0
    costs = 0

    NCURL = 128
    curl_hands = []
    for i in range(NCURL):
        c = pycurl.Curl()
        curl_hands.append(c)
    n = 0

#    my_output = open(TMP_FILE_NAME % req_arr_ix, 'w')

    for ix, item in enumerate(req_arr):
        c = curl_hands[ix % NCURL]
        try:
            url = host_str % (item, cost_arr[ix], class_arr[ix])
            t_s = time.time()

            c.setopt(c.URL, url)
            resp = StringIO()
            headers = StringIO()
            c.setopt(c.WRITEFUNCTION, resp.write)
            c.setopt(c.HEADERFUNCTION, headers.write)
            c.setopt(pycurl.CONNECTTIMEOUT, 20)
            c.setopt(pycurl.TIMEOUT, 20)
            c.perform()
            t_end = time.time()

            if c.getinfo(c.RESPONSE_CODE) == 200:
                success += 1
                is_hit, cost = handle_response(resp, headers)
                if not is_hit:
                    miss += 1
                    costs += cost
 #               my_output.write("%f, %f\n" % (t_end, (t_end - t_s)))
 #               my_output.flush()

        except Exception as e:
            print e
            pass

        PROGRESS_CTR_T[req_arr_ix] = success

#    my_output.close()
    return (success, miss, costs)


HEADER_STRING = "X-CACHED-MIDDLEWARE: "
COST_HEADER = "X-CACHED-INCURRED-COST: "
def handle_response(resp, headers):
    """
    return TRUE = HIT, FALSE = MISS
    """
    val = resp.getvalue()
    head_val = headers.getvalue()
    index = head_val.find(HEADER_STRING)
    if index == -1:
        return False, 1
    response = head_val[index + len(HEADER_STRING)]
    is_hit = (response == "H")
    if is_hit:
        return True, 0
    else:
        index = head_val.find(COST_HEADER)
        if index == -1:
            return False, 1
        else:
            cost = float(head_val[(index + len(COST_HEADER)):].split()[0])
            return False, cost
        return False, 1

def unitc_zpop(d_lam, nr, nc, host = False, name = False):
    if host:
        WL = WorkloadRunner(d_lam, nclients = nc, nreqs = nr, host = host, name = name)
    else:
        WL = WorkloadRunner(d_lam, nclients = nc, nreqs = nr, name = name)
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
        name = "/home/blanks/" + sys.argv[5] + "/" + sys.argv[6] + "_"
        throughput, missrate, reqs, time, costs = unitc_zpop(int(sys.argv[1]), nreqs, nclients, host, name)

        if len(sys.argv) > 3:
            print "%s, %s, %f, %f, %d, %d, %s" % (sys.argv[5], sys.argv[6], throughput, missrate, reqs, costs, time)
