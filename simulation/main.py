from multiprocessing import Pool, Value, Array

from workloads import cost_driver, gdwheel, spc_traces, ycsb_zipf, binary_distr, msn_driver, memcachier
from workloads.virality_trees import vtree

import simulate_dist_costs as simulate

import strategies.pq_strategies as pq
import strategies.sampling as sampling
import strategies.arc as arc
import strategies.frequency as f
import strategies.lru as lru
from itertools import product

import traceback

import inspect
import subprocess
import ssd_simulation

from numpy import random as np_random
import numpy

import sys, os, time

SIM_GLOBAL = []

SIM_PROCESSES = 16
SIM_FUNCTION = None


RUN_SIM_ON_REDIS = False

PROGRESS_CTR = False
PROGRESS_CTR_T = False
PROGRESS_CTR_T_C = False
PROGRESS_CTR_POLICY = False

UNIVERSE = 10**5
REQUESTS = 10**7

default_sampling = 64
default_retain = 0

from main_globals import *

def make_policy(p, p_first, p_second, driver_inst = None):

    policy_lambda = policies[p]

    positional_args = []
    if p_first:
        positional_args.append(p_first)
        if p_second:
            positional_args.append(p_second)
    kwargs = {}
    if hasattr(policy_lambda, "needs_driver") and \
         policy_lambda.needs_driver:
        kwargs["driver"] = driver_inst

    return policy_lambda(*positional_args, **kwargs)
    

def instantiate_driver(seed, d, U, zipf_param = 1.0001, d_second = None, expiration = None):

    driver = drivers[d]

    driver_args = { "seed" : seed, "zipf_param" : zipf_param, 
                    "item_range_max" : U,
                    "permutation_seed" : seed * 5 + 100 }
    if d_second is not None:
        driver_args["d_second"] = d_second

    driver_inst = driver(**driver_args)
    
    if expiration != None:
        if callable(expiration):
            driver_inst.get_expiry = expiration
            driver_inst.name += "(e = call)"
        else:
            driver_inst.expires_every = expiration
            driver_inst.name += "(e = %d)" % expiration

    return driver_inst

def expire_power_law(base = 1.01, mult = 200):
    return (lambda item :
            int(mult * (base ** item)))

def expire_similar_but_different(expiries):
    l = len(expiries)
    return (lambda item :
            expiries[hash(item) % l])

def make_sim(seed, k, p, 
             d = 0, p_first = None, d_second = None, 
             p_second = None, expiration = None, track_total_priority = 0,
             zipf_param = 1.0001, U = UNIVERSE, R = REQUESTS, 
             start_counting = simulate.START_AFTER_FIRST_EVICT,
             **kwargs):

    driver_inst = instantiate_driver(
        seed = seed, d = d, U = U, zipf_param = zipf_param, 
        d_second = d_second, expiration = expiration)
    policy = make_policy(p, p_first, p_second, driver_inst)

    if "measure_error_from_min" in kwargs and kwargs["measure_error_from_min"]:
        policy.measure_error_from_min = True

    sim_kwargs = {}
    sim_kwargs['track_total_priority'] = track_total_priority

    sim_kwargs['start_counting'] = start_counting
    
    if "sim_moving_window_missrates" in kwargs:
        sim_kwargs['moving_window_missrates'] = kwargs["sim_moving_window_missrates"] 

    if 'PLEASE_SET_k' in policy.__dict__:
        policy.PLEASE_SET_k = k

    s = simulate.Simulation(k, driver_inst, policy, **sim_kwargs)

    s.DRIVER_NUMBER = d
    s.POLICY_NUMBER = p
    s.TOTAL_REQUESTS = R
    return s


def make_ssd_sim(seed, k, p, 
                 d = 0, p_first = None, d_second = None, 
                 p_second = None, expiration = None, 
                 track_total_priority = 0,
                 block_size = 1, p_threshold = 1.05,
                 zipf_param = 1.0001, U = UNIVERSE, R = REQUESTS, 
                 start_counting = simulate.START_AFTER_FIRST_EVICT,
**kwargs):

    driver_inst = instantiate_driver(
        seed = seed, d = d, U = U, zipf_param = zipf_param, 
        d_second = d_second, expiration = expiration)
    policy = make_policy(p, p_first, p_second, driver_inst)

    if "measure_error_from_min" in kwargs and kwargs["measure_error_from_min"]:
        policy.measure_error_from_min = True

    sim_kwargs = {}
    sim_kwargs['track_total_priority'] = track_total_priority

    sim_kwargs['start_counting'] = start_counting
    sim_kwargs['p_threshold'] = p_threshold
    
    if "sim_moving_window_missrates" in kwargs:
        sim_kwargs['moving_window_missrates'] = kwargs["sim_moving_window_missrates"] 

    if 'PLEASE_SET_k' in policy.__dict__:
        policy.PLEASE_SET_k = k

    if block_size is not None:
        sim_kwargs['block_size'] = block_size
    if "block_builders" in kwargs:
        sim_kwargs['block_builders'] = kwargs['block_builders']
    s = ssd_simulation.Simulation(k, driver_inst, policy, **sim_kwargs)

    s.DRIVER_NUMBER = d
    s.POLICY_NUMBER = p
    s.TOTAL_REQUESTS = R
    return s

def make_backed_redis_sim(seed, k, p, 
                          d = 0, p_first = None, d_second = None, 
                          p_second = None, expiration = None, track_total_priority = 0,
                          zipf_param = 1.0001, U = UNIVERSE, R = REQUESTS, cold_start = False,
                          measure_error_from_min = False, measure_throughput = False):
    from expirement_db_redis import BackedRedisSim

    driver_inst = instantiate_driver(seed = seed, d = d, U = U, zipf_param = zipf_param, 
                                     d_second = d_second, expiration = expiration)
    policy = make_policy(p, p_first, p_second, 
                         driver_inst)

    redis_args = {'redis_args' : {}}

    s = BackedRedisSim(k, driver_inst, redis_args = redis_args, measure_tput = measure_throughput)

    s.cold_start = cold_start
    s.TOTAL_REQUESTS = R
    return s

SPAWN_NEW_REDIS = True

def make_redis_sim(ident, seed, k, p, l, d = 0, 
                   p_first = None, d_second = None, p_second = None, 
                   measure_throughput = False):

    from instrument_redis import RedisSim

    driver = drivers[d] 
    if d_second == None:
        driver_inst = driver(seed)
    else:
        driver_inst = driver(seed, d_second)
    if SPAWN_NEW_REDIS:
        s = RedisSim(k, driver_inst, redis_args = {}, ident = ident)
    else:
        s = RedisSim(k, driver_inst, redis_args = None, ident = 0)
    return s
    

def run_sim_make(sim_tuple):
    global PROGRESS_CTR, PROGRESS_CTR_T, PROGRESS_CTR_T_C

    if "sim_function" in sim_tuple:
        f = sim_tuple["sim_function"]
    else:
        f = SIM_FUNCTION

    me_index = -1
    if PROGRESS_CTR:
        with PROGRESS_CTR_T_C.get_lock():
            for i in range(len(PROGRESS_CTR_T_C)):
                if PROGRESS_CTR_T_C[i] == 0:
                    PROGRESS_CTR_T_C[i] = 1
                    me_index = i
                    break

    try:
        sim = f(**sim_tuple)
        r_val = run_sim(sim, array_entry = me_index)
    except:
        traceback.print_exc()
        r_val = False
    
    if PROGRESS_CTR:
        with PROGRESS_CTR.get_lock():
            PROGRESS_CTR.value += 1
        with PROGRESS_CTR_T.get_lock():
            PROGRESS_CTR_T[me_index] = 0
        with PROGRESS_CTR_T_C.get_lock():
            PROGRESS_CTR_T_C[me_index] = 0

    return r_val

class SimResult:
    def __init__(self, B = 0, misses = 0, error_rate = 0,
                 bucket_evictions = 0, bucket_bounds = 0,
                 average_restore = (0,0), queue_freqs = [], 
                 policy_name = "", driver_name = "",
                 track_total_priority = [], **kwargs):
        self.B = B
        self.misses = misses
        self.queue_freqs = queue_freqs
        self.error_rate = error_rate
        self.bucket_evictions = bucket_evictions
        self.bucket_bounds = bucket_bounds
        self.average_restore = average_restore
        self.policy_name = policy_name
        self.driver_name = driver_name
        self.track_total_priority = track_total_priority
        self.throughput = -1
        self.__dict__.update(kwargs)

    def str_average_restore(self):
        if self.average_restore[1] != 0:
            return "%f" % (self.average_restore[0] / self.average_restore[1])
        else:
            return "NaN"

WORKLOAD_LEN = 0

class ARRAY_UPDATER:
    def __init__(self, i):
        self.i = i
    def __call__(self, x, s):
        global PROGRESS_CTR_T, PROGRESS_CTR_POLICY, WORKLOAD_LEN

        with PROGRESS_CTR_POLICY.get_lock():
            PROGRESS_CTR_POLICY[self.i] = s
        with PROGRESS_CTR_T.get_lock():
            PROGRESS_CTR_T[self.i] = x

        if x % 5000 == 0 and len(PROGRESS_CTR_T) == 1:
            update_progress_bar(PROGRESS_CTR.value, WORKLOAD_LEN, threads = PROGRESS_CTR_T, policies = PROGRESS_CTR_POLICY)


def run_sim(sim, array_entry = -1):
    if sim.run_warmup:
        sim.warmup()

    if array_entry != -1:
        sim.simulate_requests(sim.TOTAL_REQUESTS, ARRAY_UPDATER(array_entry))
    else:
        sim.simulate_requests(sim.TOTAL_REQUESTS)
    d = { "misses" : sim.MISSES }

    d["n_misses"] = sim.N_MISSES
    d["reqs"] = sim.NUM_REQUESTS

    d["cache_size"] = sim.k

    if hasattr(sim.strategy, "error_denom"):
        d["error_rate"] = float(sim.strategy.error_numer) / sim.strategy.error_denom

    if hasattr(sim.strategy, "bucket_bounds"):
        d["bucket_evictions"] = sim.strategy.bucket_evictions
        d["bucket_bounds"] = sim.strategy.bucket_bounds

    if hasattr(sim.strategy, "average_restore"):
        d["average_restore"] = sim.strategy.average_restore
    if hasattr(sim, "throughput"):
        d["throughput"] = float(sim.throughput[1]) / sim.throughput[0]

    if hasattr(sim.strategy, "error_to_minimum"):
        error_to_minimum = sim.strategy.error_to_minimum
        if error_to_minimum[1] > 0:
            d["error_to_minimum"] = (float(error_to_minimum[0]) / error_to_minimum[1])
        else:
            d["error_to_minimum"] = 0

    if hasattr(sim.strategy, "queues"):
        sizes = [(q.cur_size) for q in sim.strategy.queues]
        freqs = [(q.frequency * q.repr_cost) for q in sim.strategy.queues]
        #        d["queue_freqs"] = [(float(q.cur_size)/max_size, q.
    
    if hasattr(sim.driver, "target_count"):
        d["target_count"] = sim.driver.target_count

    d["driver_name"] = sim.driver.name
    if hasattr(sim.strategy, "name"):
        d["policy_name"] = sim.strategy.name
    else:
        d["policy_name"] = "???"

    
    if hasattr(sim, "priority_track"):
        d["track_total_priority"] = sim.priority_track

    if hasattr(sim, "rewrites"):
        d["rewrites"] = sim.rewrites
        d["inserts"] = sim.inserts
        d["p_threshhold"] = sim.p_threshhold
        d["block_size"] = sim.block_size
        d["block_builders"] = len(sim.cur_blocks) - 1
        d["sample_from_blocks"] = sim.n_samples_from_block
        d["blocks_to_sample"] = sim.blocks_to_sample
        d["evicts"] = sim.evicts
    else:
        d["rewrites"] = 0
        d["block_builders"] = 0
        d["inserts"] = 1
        d["p_threshhold"] = 0
        d["block_size"] = 0
        d["sample_from_blocks"] = 0
        d["blocks_to_sample"] = 0
        d["evicts"] = 0
        

    if hasattr(sim, "moving_window_missrates") and sim.moving_window_missrates:
        d["moving_window_missrates"] = sim.moving_window_emit
    
    d["driver_number"] = sim.DRIVER_NUMBER

    sim.close()

    del sim.driver
    del sim.strategy
    del sim

    return SimResult(**d)


def run_product(seed, k_all = [10**3], p_all = [0], d_all = [], 
                p_first_all = [None], d_second_all = [None], 
                expire_all = [None],
                print_restored_nodes = False, scale_results = 2,
                track_total_priority = 0, requests = [REQUESTS], 
                universe = [UNIVERSE],
                zipf_all = [1.0001], 
                start_counting = simulate.START_AFTER_FIRST_EVICT,
                measure_throughput = False, output = sys.stdout,
                **kwargs):

    if not isinstance(universe, list):
        universe = [universe]
    if not isinstance(requests, list):
        requests = [requests]

    product_all = list(product(k_all, universe, requests, zipf_all, p_all, 
                               p_first_all, d_all, d_second_all, expire_all))
    p_comb = list(product(p_all, p_first_all))
    d_comb = list(product(d_all, d_second_all, expire_all))

    top_labels = ("K", "U", "Reqs", "Zipf_A")
    top_levels = list(product(k_all, universe, requests, zipf_all))
    top_level_fmts = ("%.3e", "%.3e", "%.3e", "%.3f")

    workload = [{"seed" : seed, "U" : u, "R" : r, "k" : k, "p" : p, 
                 "d" : d, "p_first" : p_f, "d_second" : d_s, "p_second" : None, 
                 "expiration" : expire, "track_total_priority" : track_total_priority,
                 "zipf_param" : zipf_param, "start_counting" : start_counting,
                 "measure_throughput" : measure_throughput}
                for k, u, r, zipf_param, p, p_f, d, d_s, expire in product_all]

    if 'return_workload_only' in kwargs:
        return workload

    result = pool_exec(workload)

    result_d = dict({p : r for p, r in zip(product_all, result)})

    if track_total_priority > 0:
        for wload_def, result in result_d.items():
            fname = "/tmp/ttp%X.csv" % hash(wload_def)
            f = open(fname, 'w')
            for p in result.track_total_priority:
                f.write("%f\n" % p) 
            f.write("\n")
            f.close()
            graph = subprocess.check_output(["gnuplot", "-e", 
                                             "set terminal dumb; plot '%s'" % fname])

            output.write("-------------  %s  -------------\n" % (wload_def,))
            output.write(graph.decode("utf-8"))

    first = True
    for top_tuple in top_levels:
        if first:
            header_tail = ""
            if scale_results == 1:
                header_tail = ", Scaled_By"
            
            evict_policy_labels = [result_d[top_tuple + p + d_comb[0]].policy_name for p in p_comb]
            if measure_throughput:
                evict_policy_labels += ["%s-tput" % policy for policy in evict_policy_labels]

            output.write( ("driver, " + 
                           ", ".join( list(top_labels) + evict_policy_labels)
                           + header_tail + "\n") )
            first = False

        for d_spec in d_comb:
            tail = ""
            result_first = result_d[top_tuple + p_comb[0] + d_spec]

            if scale_results == 1:
                scale = result_first.misses
                scale_item = lambda x: x.misses / max(scale, 1)
                tail = ", %f" % scale;
            elif scale_results == 2:
                scale_item = lambda x: x.misses / max(x.reqs, 1)
            else:
                scale_item = lambda x: x.misses
                scale = 1.0

            name = result_first.driver_name

            headers = list(top_tuple)
            headers[2] = result_first.reqs

            top_tuple_string_l = [fmt % car for car, fmt in zip(headers, top_level_fmts)]


            row_vals = top_tuple_string_l + \
                       ["%f" % scale_item(result_d[top_tuple + p + d_spec]) for p in p_comb]

            if measure_throughput:
                row_vals += ["%f" % (result_d[top_tuple + p + d_spec].throughput) for p in p_comb]

            output.write("%s, " % name + ", ".join(row_vals) + tail)
            output.write("\n")

        if print_restored_nodes:
            output.write("-------- restored nodes? ---------")
            output.write("\n")

            for d_spec in d_comb:
                name = result_d[top_tuple + p_comb[0] + d_spec].driver_name

                output.write("%s" % name + 
                             ", ".join(["%f" % (result_d[top_tuple + p + d_spec].queue_freqs) for p in p_comb]) +
                             tail)
                output.write("\n")

    return result

def run_freq_motivation_perf(universe = 10**5, reqs = 10**7):
    d_all = [ZP1C_DRIVER] + GDWheel_1_3

    run_hitratecurve(100, p_all = [4, 0], d_all = d_all, 
                     granularity = 1000, 
                     universe = universe, reqs = reqs)

def run_hitratecurve(seed, p_all = [11, 12], d_all = [4], granularity = 200, universe = UNIVERSE, reqs = REQUESTS):
    step_by = int(universe / granularity)
    
    run_product(seed, k_all = [ i * step_by for i in range(1, granularity + 1) ],
                p_all = p_all, d_all = d_all, scale_results = 2, universe = universe, requests = reqs)

def permute_list(l):
    p = list(np_random.permutation(len(l)))
    out_l = [l[ix] for ix in p]
    return (out_l, p)

def unpermute_list(l, p):
    p_0 = list(enumerate(p))
    p_0.sort(key = lambda x : x[1])
    out_l = [l[ix] for ix,_ in p_0]
    return out_l

def pool_exec(workload, processes = -1):
    global PROGRESS_CTR, PROGRESS_CTR_T, PROGRESS_CTR_T_C, PROGRESS_CTR_POLICY, WORKLOAD_LEN
    
    if processes == -1:
        processes = SIM_PROCESSES

    if RUN_SIM_ON_REDIS:
        pool = Pool(processes=12)
        result = pool.map(run_sim_make, [ (ident,) + w for 
                                          (ident, w) in enumerate(workload) ])
        pool.close()
    else:
        sys.stderr.write("[ starting ]")
        sys.stderr.flush()

        PROGRESS_CTR = Value('i', 0)
        PROGRESS_CTR_T = Array('i', [0 for i in range(processes)])
        PROGRESS_CTR_POLICY = Array('i', [0 for i in range(processes)])
        PROGRESS_CTR_T_C = Array('i', [0 for i in range(processes)])
        WORKLOAD_LEN = len(workload)

        if (processes == 1):
            result = []
            for ix, w in enumerate(workload):
                result.append( run_sim_make( w ) )
                update_progress_bar(ix + 1, len(workload), threads = [])
            sys.stderr.write("\n")
            sys.stderr.flush()
            return result

        pool = Pool(processes = processes, maxtasksperchild = 1)
        workload_permuted, permutation = permute_list(workload)
        result_async = pool.map_async(run_sim_make, workload_permuted, 
                                      chunksize = 1)

        total_work = len(workload)

        while not result_async.ready():
            result_async.wait(15)
            value = PROGRESS_CTR.value
            update_progress_bar(PROGRESS_CTR.value, total_work, threads = PROGRESS_CTR_T, policies = PROGRESS_CTR_POLICY)
        
        sys.stderr.write("\n")
        sys.stderr.flush()
        result = unpermute_list(result_async.get(), permutation)
        pool.close()
    return result

def update_progress_bar(cur, total, width = 70, threads = [], policies = []):
    bar_width = int((cur * width) / total)
    percentage = float(cur * 100) / total

    sys.stderr.write("\r[%s>%s] %.2f%%" % ("=" * bar_width, " " * (width - bar_width - 1), percentage))

    if len(threads) > 0:
        sys.stderr.write("".join([" " for i in range(60)]))
        sys.stderr.write("      %s" % " ".join([" (%d => %.2f) " % ( policy, (100.0*t/10**7)) for policy,t in zip(policies, threads)]))

    sys.stderr.flush()


def run_compare_sampling(seed, k = 10**3, d = 0, policy = [6,7], S = range(1,30), retain = 2):
    workload = [ (seed, k, policy[0], 0, d) ]
    workload += [(seed, k, p, 1, d, s, None, retain) for p, s in product(policy, S)]
    result = pool_exec(workload)
    scale = float(result[0][1])
    out_tbl = { key : results for key, results in zip(product(policy, S), result[1:])}
    print "miss ratios (compared to full GD) by sample size"
    print ("sample size, " + ", ".join([policies[p][1](0).name  for p in policy]))
    for s in S:
        line = "%d, " % s
        line += ", ".join(["%f" % (float(out_tbl[(p, s)][1])/scale) for p in policy])
        print line

def run_measure_sampling_rank(seed, k = 10**3, d = 4, S = range(1,100), Retain = [0], measure_rank = True):
    policy = 5
                        
    workload = [ (seed, k, policy, 0, d) ]
    s_r_prod = [(s,r) for r, s in product(Retain, S) if s > r]

    workload += [(seed, k, policy, 1, d, s, None, r) for s,r in s_r_prod]

    result = pool_exec(workload)
    scale = float(result[0][1])

    out_tbl = { key : results for key, results in zip(s_r_prod, result[1:])}
    print ("miss ratio table ( cache size = %d )" % k)
    print ("sample size, " + ", ".join(["miss ratio (r=%d)" % retain for retain in Retain]) + ", "
           + ", ".join(["evict rank (r=%d)" % retain for retain in Retain]))
    for s in S:
        line = ("%d, " % s)
        result_stream = [out_tbl[(s, r)] for r in Retain if r < s]
        line += ", ".join(["%f" % (float(result_tup[1])/scale) for result_tup in result_stream])
        line += ", "
        line += ", ".join(["%f" % result_tup[2] for result_tup in result_stream])
        print line


def setup_pgsql_sim(workload_driver, U = UNIVERSE, table_name = None, d_second = None, pg_str = None, evaluate_requests = False):
    from expirement_db_redis import BackedRedisSim

    driver_inst = instantiate_driver(seed = 0, d = workload_driver, U = U, d_second = d_second)
    if pg_str:
        sim = BackedRedisSim(0, driver_inst, table_name, pg_connection_str = pg_str)
    else:
        sim = BackedRedisSim(0, driver_inst, table_name)
    sim.evaluate_requests = evaluate_requests
    sim.create_table()
    sim.close()

def redis_progress_bar(i):
    if i % 10**3 == 0:
        update_progress_bar(i, 7.991*10**6)

def run_redis_backed_spc_sim(workload_driver):
    from instrument_redis import RedisSim as brs

    driver_inst = instantiate_driver(seed = 0, d = workload_driver, U = 10**5, d_second = None)
    sim = brs(525000, driver_inst, redis_args = "hyper")
    sim.simulate_requests(-1, redis_progress_bar)
    
    print ""
    print ">>>>"
    print "(/ %d %d)" % (sim.MISSES, sim.NUM_REQUESTS)
    sim.close()

SIM_FUNCTION = make_sim

mrc_standard_K_ALL = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000,
                      30000, 40000, 50000, 75000, 100000, 150000]

mrc_standard_K_ALL_1M = [100, 500, 1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000, 
                         150000, 2.5*10**5, 5*10**5, 10**6]

def rename_if_exists(fname):
    try:
        os.rename(fname, fname + ".old.%s" % time.time())
    except Exception as e:
        pass

def get_object_size_stats_memcachier():
    fname = "results/memcachier_size_stats.csv" 
    rename_if_exists(fname)
    output = open(fname, "w")
    output.write("AppNumber, MinSize, Mean, Median, Max, Stdev\n")
    
    D_ALL = MEMCACHIER_CAT_WORKLOADS
    def filter2(d):
        x = drivers[d]
        return (x.reqs()) > 10**4
    def filter(d):
        x = drivers[d]
        return (x.uniqs * x.max_item_size) > x.app_allocation

    D_ALL = [x for x in D_ALL if filter(x)]
    D_ALL = [x for x in D_ALL if filter2(x)]

    pool = Pool(processes = 12)
    result_async = pool.map_async(object_size_statistics, 
                                  D_ALL,
                                  chunksize = 1)
    while not result_async.ready():
        result_async.wait(5)
        sys.stderr.write(".")
    sys.stderr.write("done\n")
    result = result_async.get()
    pool.close()

    for d, out in zip(D_ALL, result):
        (mymin, mean, median, mymax, std) = out
        appid = drivers[d].appid
        output.write("%s, %f, %f, %f, %f, %f\n" % 
                     (appid, mymin, mean, median, mymax, std))
    output.close()

def object_size_statistics( d ):
    driver_lambda = drivers[d]
    driver = driver_lambda()

    mymax = False
    mymin = False
    
    aggr = []
    try:
        cur = driver.get_next()
        while cur != -1:
            _, sz = cur
            if mymax is False or sz >= mymax:
                mymax = sz
            if mymin is False or sz >= mymin:
                mymin = sz
            aggr.append(int(sz))
            assert sz is not None
            cur = driver.get_next()
        mean = numpy.mean(aggr)
        median = numpy.median(aggr)
        std = numpy.std(aggr)
        return (mymin, mean, median, mymax, std)
    except Exception as e:
        import traceback as tb
        tb.print_exc()
        raise e

def msn_decision_mrc():
    fname = "results/msn_decision_mrc.csv"
    p_all = [P_HYPER, P_HYPER_EXPIRE]
    rename_if_exists(fname)
    output = open(fname, "w")
    k_all = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000,
             3*10**4, 4*10**4, 4.2*10**4]
    run_product(0, k_all, d_all = [MSR_MSN_DECISION],  p_all = p_all, 
                start_counting = simulate.START_AFTER_FIRST_EVICT, 
                scale_results = 2, requests = 10**6, output = output)
    output.close()

def msr_virality_mrc():
    fname = "results/msr_virality_mrc.csv"
    p_all = [P_HYPER]
    rename_if_exists(fname)
    output = open(fname, "w")
    k_all = [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000, 15000, 20000,
             3*10**4, 4*10**4]
    run_product(0, k_all, d_all = [MSR_VIRAL_TREE],  p_all = p_all, 
                start_counting = simulate.START_AFTER_FIRST_EVICT, 
                scale_results = 2, requests = 10**6, output = output)
    output.close()

def hyper_lfu_lru_mrc_introducing():
    fname = "results/hyper_lfu_lru_mrc_introducing.csv"
    p_all = [P_HYPER, P_LFU, P_LRU]
    universe = 10**5
    rename_if_exists(fname)
    output = open(fname, "w")
    run_product(100, mrc_standard_K_ALL, d_all = [DYN_INTRO],  p_all = p_all, start_counting = simulate.START_AFTER_FIRST_EVICT, scale_results = 2, requests = 10**6, output = output,
                universe = universe)
    output.close()

def lfu_v_hyper_mrc_introducing():
    fname = "results/lfu_v_hyper_mrc_introducing.csv"
    p_all = [P_LRU]
    universe = 10**5
    rename_if_exists(fname)
    output = open(fname, "a")
    run_product(100, mrc_standard_K_ALL, d_all = [DYN_INTRO],  p_all = p_all, 
                start_counting = simulate.START_AFTER_FIRST_EVICT, 
                scale_results = 2, requests = 10**7, universe = universe,
                output = output)
    output.close()

def dynamic_popularities_mrcs():
    fname = "results/dynamic_popularities_mrcs.csv"
    p_all = [P_LFU, P_HYPER, P_GD]
    d_all = [DYN_INTRO]
    universe = 10**5
    rename_if_exists(fname)
    output = open(fname, "a")
    run_product(100, mrc_standard_K_ALL, d_all = d_all, p_all = p_all, 
                start_counting = simulate.START_AFTER_FIRST_EVICT, 
                scale_results = 2, requests = 10**7, universe = universe,
                output = output)
    output.close()

def window_dynamic_pops_mrcs():
    fname = "results/window_dynamic_pops_mrcs.csv"
    p_all = [P_WINDOW_FREQ, P_WINDOW_HYPER, 2, P_S_FREQ, P_GD]
    d_all = [9]
    universe = 10**5
    rename_if_exists(fname)
    output = open(fname, "w")
    k_all =  mrc_standard_K_ALL
    run_product(100, k_all, d_all = d_all, p_all = p_all, 
                start_counting = simulate.START_AFTER_FIRST_EVICT, 
                scale_results = 2, requests = 10**6, universe = universe,
                output = output)
    output.close()

def hyper_inits_mrc_1cZp():
    fname = "results/hyper_inits_mrc_1cZp.csv"
    p_all = [P_HYPER]
    p_first = [1, .75, .5, .25, .10, 0]
    universe = 10**5
    rename_if_exists(fname)
    output = open(fname, "w")
    run_product(100, mrc_standard_K_ALL, d_all = [ZP1C_DRIVER],  p_all = p_all, 
                p_first_all = p_first, start_counting = simulate.START_AFTER_FIRST_EVICT, 
                scale_results = 2, 
                requests = 10**6, universe = universe, output = output)
    output.close()

def perf_v_lru_mrc_1cZp():
    fname = "results/perf_v_lru_mrc_1cZp.csv"
    p_all = [0, 4]
    universe = 10**5
    rename_if_exists(fname)
    output = open(fname, "w")
    run_product(100, mrc_standard_K_ALL, d_all = [4],  p_all = p_all, start_counting = simulate.START_AFTER_FIRST_EVICT, scale_results = 2, requests = 10**6, output = output,
                universe = universe)
    output.close()

def hyper_inits_mrc_heavy():
    fname = "results/hyper_inits_mrc.heavy.csv"
    p_all = [20]
    p_first = [1, .5, 0]
    universe = 10**6
    rename_if_exists(fname)
    output = open(fname, "w")
    run_product(100, mrc_standard_K_ALL_1M, d_all = [Z075P1C_DRIVER],  p_all = p_all, p_first_all = p_first, start_counting = simulate.START_AFTER_FIRST_EVICT, scale_results = 2, 
                requests = 10**7, universe = universe, output = output)
    output.close()

def perf_lru_lfu_mrc_1cZp_heavy():
    fname = "results/perf_v_lru_mrc_1cZp.heavy.csv"
    p_all = [0, 4, 9]
    universe = 10**6
    d_all = [Z075P1C_DRIVER]
    rename_if_exists(fname)
    output = open(fname, "w")
    run_product(100, mrc_standard_K_ALL_1M, d_all = d_all,  p_all = p_all, start_counting = simulate.START_AFTER_FIRST_EVICT, scale_results = 2, requests = 10**7, output = output,
                universe = universe)
    output.close()

def perf_lru_lfu_mrc_1cZp_big():
    fname = "results/perf_v_lru_mrc_1cZp.big.csv"
    p_all = [P_LRU, 4, 9]
    universe = 10**6
    d_all = [ZP1C_DRIVER]
    rename_if_exists(fname)
    output = open(fname, "w")
    run_product(100, mrc_standard_K_ALL_1M, d_all = d_all,  p_all = p_all, start_counting = simulate.START_AFTER_FIRST_EVICT, scale_results = 2, requests = 10**7, output = output,
                universe = universe)
    output.close()

def dynamic_popularities_mrcs_heavy():
    fname = "results/dynamic_popularities_mrcs.heavy.csv"
    p_all = [P_LFU, P_HYPER, P_GD]
    d_all = [52, 53]
    universe = 10**6
    rename_if_exists(fname)
    output = open(fname, "a")
    run_product(100, mrc_standard_K_ALL_1M, d_all = d_all, p_all = p_all, 
                start_counting = simulate.START_AFTER_FIRST_EVICT, 
                scale_results = 2, requests = 10**7, universe = universe,
                output = output)
    output.close()

def hyper_sampling_priority_mrc_heavy(sampling = [1, 2, 5, 10, 25, 500] , 
                                      seed = 100, ):
    universe = 10**6
    p_all = [P_HYPER_SAMPLE_PARAMETER]
    d_all = [Z075P1C_DRIVER]
    k_all = mrc_standard_K_ALL_1M
    workload = run_product(seed, k_all, d_all = d_all , 
                           p_all = p_all, p_first_all = sampling, 
                           return_workload_only = True, 
                           start_counting = simulate.START_AFTER_FIRST_EVICT, 
                           requests = 10**7, universe = universe)
    for setup_dict in workload:
        setup_dict['measure_error_from_min'] = True
    results = pool_exec(workload)
    fname = "results/hyper_sampling_measure_priority_mrc.heavy.csv"
    first = True
    rename_if_exists(fname)
    output = open(fname, "w")

    for d in d_all:
        d_results = [r for r in results if r.driver_number == d]        
        for k in k_all:
            row = [r for r in d_results if r.cache_size == k]
            if first:
                first = False
                output.write("k, ")
                output.write(", ".join(["%s" % r.policy_name for r in row]))
                output.write("\n")
            output.write("%d, " % k)
            output.write(", ".join(["%e" % r.error_to_minimum for r in row]))
            output.write("\n")
    output.close()

    fname = "results/hyper_sampling_performance_mrc.heavy.csv"
    first = True
    rename_if_exists(fname)
    output = open(fname, "w")

    for d in d_all:
        d_results = [r for r in results if r.driver_number == d]        
        for k in k_all:
            row = [r for r in d_results if r.cache_size == k]
            if first:
                first = False
                output.write("k, ")
                output.write(", ".join(["%s" % r.policy_name for r in row]))
                output.write("\n")
            output.write("%d, " % k)
            output.write(", ".join(["%f" % (float(r.misses) / max(1, r.reqs)) for r in row]))
            output.write("\n")
    
    output.close()


def hyper_ssd_tests(block_size_all = [1, 5, 10, 20, 40, 80] , 
                    seed = 100, ):
    universe = 10**6
    p_all = [P_HYPER_SAMPLE_PARAMETER]
    d_all = [ZP1C_DRIVER]
    k_all = mrc_standard_K_ALL_1M

    d_k_combos = []

    for d in d_all:
        k_all = [LRU_HIT_RATE_DATA[d],]
#                 LRU_HIT_RATE_DATA_70[d]]
        d_k_combos += run_product(seed, k_all, d_all = d_all , 
                                  p_all = p_all, 
                                  return_workload_only = True, 
                                  p_first_all = [1],
                                  start_counting = simulate.START_AFTER_FIRST_EVICT, 
                                  requests = 10**6, universe = universe)

    workload = []
    for setup_dict in d_k_combos:
        copy_first = dict(setup_dict)
        copy_first["p_first"] = 15
        workload.append(copy_first)
        copy_first = dict(setup_dict)
        copy_first["p_first"] = default_sampling
        workload.append(copy_first)
        setup_dict['sim_function'] = make_ssd_sim
        for z in block_size_all:
            copy_d = dict(setup_dict)
            copy_d['block_size'] = z
            workload.append(copy_d)

    results = pool_exec(workload)
    fname = "results/hyper.ssd.csv"
    first = True
    rename_if_exists(fname)
    output = open(fname, "w")

    output.write("driver, k, block_size, samples_in_block, blocks_to_sample," + 
                 " thresshold, reqs, rewrites, write_amplification, missrate\n")
    for d in d_all:
        d_results = [r for r in results if r.driver_number == d]        
        for k in k_all:
            row = [r for r in d_results if r.cache_size == k]
            for r in row:
                 line = [ 
                     r.driver_name,
                     "%d" % k,
                     "%d" % r.block_size,
                     "%d" % r.sample_from_blocks,
                     "%d" % r.blocks_to_sample,
                     "%f" % r.p_threshhold,
                     "%e" % r.reqs,
                     "%e" % r.rewrites,
                     "%f" % (float(r.rewrites + r.inserts) / r.inserts),
                     "%f" % (float(r.misses) / max(1, r.reqs))]
                 output.write(", ".join(line))
                 output.write("\n")
    output.close()

def hyper_ssd_thresh_tests(block_size = 100, 
                    threshold_all = [1.1, 1.5, 2, 3, 4, 5, 10],
                    seed = 100 ):
    universe = 10**6
    p_all = [P_HYPER_SAMPLE_PARAMETER]
    d_all = [ZP1C_DRIVER, Z075P1C_DRIVER]
    k_all = mrc_standard_K_ALL_1M

    d_k_combos = []

    for d in d_all:
        k_all = [LRU_HIT_RATE_DATA[d],]
#                 LRU_HIT_RATE_DATA_70[d]]
        d_k_combos += run_product(seed, k_all, d_all = d_all , 
                                  p_all = p_all, 
                                  return_workload_only = True, 
                                  p_first_all = [1],
                                  start_counting = simulate.START_AFTER_FIRST_EVICT, 
                                  requests = 5*10**6, universe = universe)

    workload = []
    for setup_dict in d_k_combos:
        if setup_dict["d"] == Z075P1C_DRIVER:
            setup_dict["d_second"] = 1.4
        copy_first = dict(setup_dict)
        copy_first["p_first"] = 15
        workload.append(copy_first)
        copy_first = dict(setup_dict)
        copy_first["p_first"] = default_sampling
        workload.append(copy_first)
        setup_dict['sim_function'] = make_ssd_sim
        setup_dict['block_size'] = block_size
        for z in threshold_all:
            copy_d = dict(setup_dict)
            copy_d['p_threshold'] = z
            workload.append(copy_d)
            copy_2 = dict(copy_d)
            copy_2['block_builders'] = 2
            workload.append(copy_2)

    results = pool_exec(workload)
    fname = "results/hyper.ssd.bythresh.csv"
    first = True
    rename_if_exists(fname)
    output = open(fname, "w")

    output.write("driver, k, block_size, num_builders, samples_in_block, blocks_to_sample," + 
                 " thresshold, reqs, rewrites, write_amplification, missrate\n")
    for d in d_all:
        d_results = [r for r in results if r.driver_number == d]        
        for k in k_all:
            row = [r for r in d_results if r.cache_size == k]
            for r in row:
                 line = [ 
                     r.driver_name,
                     "%d" % k,
                     "%d" % r.block_size,
                     "%d" % r.block_builders,
                     "%d" % r.sample_from_blocks,
                     "%d" % r.blocks_to_sample,
                     "%f" % r.p_threshhold,
                     "%e" % r.reqs,
                     "%e" % r.rewrites,
                     "%f" % (float(r.rewrites + r.inserts) / r.inserts),
                     "%f" % (float(r.misses) / max(1, r.reqs))]
                 output.write(", ".join(line))
                 output.write("\n")
    output.close()


def lfu_v_lru_mrc_1cZp():
    fname = "results/lfu_v_lru_mrc_1cZp.csv"
    p_all = [9, 0]
    universe = 10**5
    rename_if_exists(fname)
    output = open(fname, "w")
    run_product(100, mrc_standard_K_ALL, d_all = [4],  p_all = p_all, start_counting = simulate.START_AFTER_FIRST_EVICT, scale_results = 2, requests = 10**6, output = output,
                universe = universe)
    output.close()

def hyper_lfu_lru_mrc_1cZp():
    fname = "results/hyper_lfu_lru_mrc_1cZp.csv"
    p_all = [P_HYPER, P_LFU, P_LRU]
    universe = 10**5
    rename_if_exists(fname)
    output = open(fname, "w")
    run_product(100, mrc_standard_K_ALL, d_all = [4],  p_all = p_all, start_counting = simulate.START_AFTER_FIRST_EVICT, scale_results = 2, requests = 10**6, output = output,
                universe = universe)
    output.close()


def hyper_ARCPn(k_all, n_all = range(14)):
    p_all = [P_HYPER, P_LFU, P_ARC]
    d_all = [ARC_P1_P14[x] for x in n_all]
    run_product(10, k_all, d_all = d_all, p_all = p_all, start_counting = simulate.START_AFTER_FIRST_EVICT,
                scale_results = 2, requests = 10**7)

def hyper_compare_ARCSn(k_all = [10**5], requests = 10**7, n_all = [0,1,2] ):
    p_all = [P_HYPER, P_LFU, P_ARC, P_GD]
    d_all = [ARC_S1_S3[x] for x in n_all]
    fname = "results/hyper_compare_ARCSn.csv"
    rename_if_exists(fname)
    output = open(fname, "a")
    run_product(10, k_all, d_all = ARC_S1_S3, p_all = p_all, start_counting = simulate.START_AFTER_FIRST_EVICT,
                scale_results = 2, requests = requests, output = output)
    output.close()

EXPIRE_PARAM_10pp = [100] + [-1 for i in range(9)]
EXPIRE_PARAM_25pp = [100] + [-1 for i in range(3)]

def hyper_expiration_mrc_1cZp(expire_params = [EXPIRE_PARAM_25pp]):
    p_all = [11, 14]
    universe = 10**5

    expire_all = [expire_similar_but_different(i) for i in expire_params]
    run_product(100, mrc_standard_K_ALL, d_all = [2], expire_all = expire_all,  p_all = p_all, 
                start_counting = simulate.START_AFTER_WARMUP, 
                scale_results = 2, requests = 10**6, universe = universe)

def hyper_lru_mrc_memcachier(trace = "t1", workload_num = 1):
    import math
    if trace == "t1":
        D = MEMCACHIER_T1_WORKLOADS[workload_num]
    elif trace == "t2":
        D = MEMCACHIER_T2_WORKLOADS[workload_num]
    elif trace == "cat":
        D = MEMCACHIER_CAT_WORKLOADS[workload_num]
    
    K_SAMPLES = 24
    universe = drivers[D].uniqs * drivers[D].max_item_size
    
    log_universe = math.log(universe)
    log_min_cache = math.log(100 * drivers[D].max_item_size)

    log_k_all = [ log_min_cache + 
                  ((i + 1) * (log_universe - log_min_cache) / K_SAMPLES)
                  for i in range(K_SAMPLES) ]
    k_all = [int(math.exp(i)) for i in log_k_all]
    
    d_all = [ D ]
    p_all = [ P_HYPER, P_HYPER_SZ, P_LRU, P_GD_SZ ]

    sys.stderr.write("%d -> %e\n" % (workload_num, universe))

    driver_name = drivers[D].appid

    fname = "results/memcachier/%s/memcachier_%s.csv" % (trace, driver_name) 
    rename_if_exists(fname)
    output = open(fname, "w")

    run_product(10, k_all, d_all = d_all, p_all = p_all, 
                start_counting = simulate.START_NOW,
                scale_results = 2, requests = -1, output = output, 
                universe = universe)
    output.close()


def hyper_lru_eval_memcachier(trace = "cat", workload_nums = range(20)):
    if trace == "t1":
        D_ALL = MEMCACHIER_T1_WORKLOADS
    elif trace == "t2":
        D_ALL = MEMCACHIER_T2_WORKLOADS
    elif trace == "cat":
        D_ALL = MEMCACHIER_CAT_WORKLOADS

    def filter(d):
        x = drivers[d]
        return (x.uniqs * x.max_item_size) > x.app_allocation
    def filter2(d):
        x = drivers[d]
        return (x.reqs()) > 10**4
    def sort_key(d):
        x = drivers[d]
        return (x.uniqs * x.max_item_size)

    D_ALL = [x for x in D_ALL if filter(x)]
    print(len(D_ALL))
    D_ALL = [x for x in D_ALL if filter2(x)]
    print(len(D_ALL))
    D_ALL.sort(key=sort_key, reverse = True)


    if workload_nums:
        D_ALL = [D_ALL[i] for i in workload_nums]

    workload = []
    p_all = [ P_HYPER, P_HYPER_SZ, P_LRU, P_GD_SZ ]

    for d in D_ALL:
        driver = drivers[d]
        universe = driver.uniqs * driver.max_item_size
        k_all = [driver.app_allocation]
        workload += run_product(
            0, k_all, d_all = [ d ], 
            p_all = p_all, 
            return_workload_only = True, 
            start_counting = simulate.START_NOW, 
            requests = -1, universe = universe)

    results = pool_exec(workload)
    fname = "results/memcachier_app_allocations_%s.csv" % trace
    rename_if_exists(fname)
    output = open(fname, "w")
    
    row_size = len(p_all)
    table = [ [results[ix] for ix in 
               range(row_size * begin, (row_size * begin) + row_size)]
              for begin in range(len(results) / row_size) ]

    output.write("appid, reqs, ")
    policy_names = [r.policy_name for r in table[0]]
    output.write(", ".join(policy_names))
    output.write("\n")
    for row in table:
        appid = drivers[row[0].driver_number].appid
        output.write("%s, " % appid)
        output.write("%d, " % row[0].reqs)
        output.write(", ".join([ "%f" % (float(r.misses) / r.reqs)
                                 for r in row ]))
        output.write("\n")
        for r in row:
            assert drivers[r.driver_number].appid == appid

    output.close()

def hyper_sampling_mrc_1cZp(sampling = [1, 2, 5, 10, 25, 500, -1], seed = 100, p_all = [P_HYPER_SAMPLE_PARAMETER]):
    universe = 10**5
    fname = "results/hyper_sampling_mrc_1cZp.csv"
    rename_if_exists(fname)
    output = open(fname, "w")
    d_all = [ZP1C_DRIVER]
    run_product(seed, mrc_standard_K_ALL, d_all = d_all , p_all = p_all, p_first_all = sampling, output = output,
                start_counting = simulate.START_AFTER_FIRST_EVICT, scale_results = 2, requests = 10**6, universe = universe)
    output.close()

def hyper_sampling_lighttail_mrc(sampling = [1, 2, 5, 10, 64, -1], seed = 100, p_all = [P_HYPER_SAMPLE_PARAMETER]):
    universe = 10**5
    fname = "results/hyper_sampling_lighttail_mrc.csv"
    rename_if_exists(fname)
    output = open(fname, "w")
    d_all = [Z075P1C_DRIVER]
    d_second_all = [1.1, 1.4, 1.8, 2.0, 2.5]
    run_product(seed, mrc_standard_K_ALL, d_all = d_all , 
                d_second_all = d_second_all,
                p_all = p_all, p_first_all = sampling, output = output,
                start_counting = simulate.START_AFTER_FIRST_EVICT, scale_results = 2, requests = 10**6, universe = universe)
    output.close()

def hyper_sampling_lighttail_retention_mrc(retain = [-2, 1, 2, 5, 10, 20, -1], seed = 100, p_all = [P_HYPER_RETAIN_PARAMETER]):
    universe = 10**5
    fname = "results/hyper_sampling_lighttail_retain_mrc.csv"
    rename_if_exists(fname)
    output = open(fname, "w")
    d_all = [Z075P1C_DRIVER]
    d_second_all = [1.1, 1.4, 1.8, 2.0, 2.5]
    run_product(seed, mrc_standard_K_ALL, d_all = d_all , 
                d_second_all = d_second_all,
                p_all = p_all, p_first_all = retain, output = output,
                start_counting = simulate.START_AFTER_FIRST_EVICT, scale_results = 2, requests = 10**6, universe = universe)
    output.close()

def hyper_sampling_random_sampling_evals(seed = 100, retain_test = [-2, -1, 19], 
                                         sample_test = [1, 2, 5, 10, 64, -1],
                                         zipf_tests = [1.0001, 1.4]): 
    universe = 10**5
    d_all = [Z075P1C_DRIVER]
    d_second_all = zipf_tests
    p_all = [P_HYPER_RETAIN_PARAMETER]
    fname = "results/hyper_sampling_lighttail_retain_mrc.csv"
    rename_if_exists(fname)
    output = open(fname, "w")
    run_product(seed, mrc_standard_K_ALL, d_all = d_all , 
                d_second_all = d_second_all,
                p_all = p_all, p_first_all = retain_test, output = output,
                start_counting = simulate.START_AFTER_FIRST_EVICT, scale_results = 2, 
                requests = 10**6, universe = universe)
    output.close()
    
    p_all = [P_HYPER_SAMPLE_PARAMETER]
    fname = "results/hyper_sampling_lighttail_mrc.csv"
    rename_if_exists(fname)
    output = open(fname, "w")
    run_product(seed, mrc_standard_K_ALL, d_all = d_all , 
                d_second_all = d_second_all,
                p_all = p_all, p_first_all = sample_test, output = output,
                start_counting = simulate.START_AFTER_FIRST_EVICT, scale_results = 2, 
                requests = 10**6, universe = universe)
    output.close()
    

def hyper_sampling_priority_mrc_1cZp(sampling = [1, 2, 5, 10, 64] , seed = 100, p_all = [P_HYPER_SAMPLE_PARAMETER]):
    universe = 10**5
    d_all = [ZP1C_DRIVER, Z075P1C_DRIVER]
    k_all = mrc_standard_K_ALL
    workload = run_product(seed, k_all, d_all = d_all , 
                           p_all = p_all, p_first_all = sampling, 
                           return_workload_only = True, 
                           start_counting = simulate.START_AFTER_FIRST_EVICT, 
                           requests = 10**6, universe = universe)
    for setup_dict in workload:
        setup_dict['measure_error_from_min'] = True
    results = pool_exec(workload)
    fname = "results/hyper_sampling_measure_priority_mrc_1cZp.csv"
    first = True
    rename_if_exists(fname)
    output = open(fname, "w")

    for d in d_all:
        d_results = [r for r in results if r.driver_number == d]        
        output.write("Driver = %s\n" % d_results[0].driver_name)
        for k in k_all:
            row = [r for r in d_results if r.cache_size == k]
            if first:
                first = False
                output.write("k, ")
                output.write(", ".join(["%s" % r.policy_name for r in row]))
                output.write("\n")
            output.write("%d, " % k)
            output.write(", ".join(["%e" % r.error_to_minimum for r in row]))
            output.write("\n")
    
    output.close()

def hyper_v_hyper_class_mrc_hotclass(seed = 100):
    p_all = [P_HYPER, P_HYPER_CLASS_TRACK]
    d_all = [HOTCLASS]
    universe = 10**5
    fname = "results/hyper_v_hyper_class_mrc_hotclass.csv"
    rename_if_exists(fname)
    output = open(fname, "w")
    k_all = mrc_standard_K_ALL
    run_product(seed, k_all, d_all = d_all , p_all = p_all, output = output,
                start_counting = simulate.START_AFTER_FIRST_EVICT, 
                scale_results = 0, requests = 5*10**6, universe = universe)
    output.close()

def hyper_v_hyper_class_mrc_stochclass(seed = 100):
    p_all = [P_HYPER, P_HYPER_CLASS_TRACK]
    d_all = [STOCHASTIC_CLASS]
    d_second_all = [((1, 5),)] #[((1, 0.1), (7, 1.5)),
    
    universe = 10**5
    output = sys.stdout
    k_all = mrc_standard_K_ALL
    run_product(seed, k_all, d_all = d_all, d_second_all = d_second_all, 
                p_all = p_all, output = output,
                start_counting = simulate.START_AFTER_FIRST_EVICT, 
                scale_results = 2, requests = 10**7, universe = universe)


def lfu_v_perf_moving_window():
    k_all = [3000]
    p_all = [4, 9]
    universe = 10**5
    workload = []
    for seed in [100 * s for s in range(10)]:
        workload += run_product(seed, k_all, d_all = [ZP1C_DRIVER],  p_all = p_all, start_counting = simulate.START_AFTER_FIRST_EVICT, 
                                requests = 10**7, universe = universe, return_workload_only = True)
    for setup_dict in workload:
        setup_dict['sim_moving_window_missrates'] = 50000
    
    results = pool_exec(workload)
    for i in range(len(results) / 2):
        r_pair = results[i * 2: i * 2 + 2]
        results_emits = [r.moving_window_missrates for r in r_pair]
        print "t, perfect, lfu"
        for t, (perf, lfu) in enumerate(zip(*results_emits)):
            print "%d, %f, %f" % (t * 10, perf, lfu)
    
def hyper_v_perf_moving_window():
    k_all = [3000]
    p_all = [20]
    universe = 10**5
    workload = []
    for seed in [100 * s for s in range(10)]:
        workload += run_product(seed, k_all, d_all = [ZP1C_DRIVER],  p_all = p_all, start_counting = simulate.START_AFTER_FIRST_EVICT, 
                                requests = 10**7, universe = universe, return_workload_only = True)
    for setup_dict in workload:
        setup_dict['sim_moving_window_missrates'] = 50000
    
    results = pool_exec(workload)
    print "t, hyper"
    for r in results:
        results_emits = r.moving_window_missrates
        for t, hyper in enumerate(results_emits):
            print "%d, %f" % (t * 10, hyper)
    
def hyper_v_multi_mrc_WSet(seed = 100, d_second_all = [ (10, 1), (20, 1), (50,1) ]):
    universe = 10**5
    k_all = [100, 250, 1000, 2000, 3000, 
             5000, 10000, 15000, 20000,
             25000, 30000, 35000, 40000, 
             50000, 60000, 67500, 75000, 
             85000, 95000, 100000]

    p_all = [P_HYPER, P_LFU, P_ARC, P_GD]
    run_product(seed, k_all, d_all = [WSET_DRIVER] , p_all = p_all, 
                start_counting = simulate.START_AFTER_FIRST_EVICT, scale_results = 2, requests = 10**6, universe = universe)

def find_cache_size(seed, d, hit_rate, policy, universe):
    max_p = 16
    probes = [10] + [p * universe / (max_p - 1) for p in range(1, max_p)]
    # round 1
    results = run_product(seed, probes, d_all = [d], p_all = [policy],
                          start_counting = simulate.START_AFTER_FIRST_EVICT, requests = 10**6, 
                          universe = universe, 
                          output = open("/tmp/foo", "w"))
    hits_sz_pairs = [ ( float(r.reqs - r.n_misses)/max(r.reqs, 1), r.cache_size) for r in results ]
    hits_sz_pairs.sort()

    for (hr, sz) in hits_sz_pairs:
        if hr < hit_rate:
            cur_bottom = sz
        if hr == hit_rate:
            print (hr, sz)
            return
        else:
            cur_top = sz
    probes = [cur_bottom + ((cur_top - cur_bottom) * p) / max_p 
              for p in range(1, max_p + 1)]

    results = run_product(seed, probes, d_all = [d], p_all = [policy],
                          start_counting = simulate.START_AFTER_FIRST_EVICT, requests = 10**6, 
                          universe = universe, 
                          output = open("/tmp/foo", "w"))
    hits_sz_pairs = [ ( float(r.reqs - r.n_misses)/max(r.reqs, 1), r.cache_size) for r in results ]
    hits_sz_pairs.sort()

    cur_min = -1
    cur_min_sz = -1
    for (hr, sz) in hits_sz_pairs:
        if cur_min_sz == -1:
            cur_min_sz = sz
            cur_min_hr = abs(hr - hit_rate)
        else:
            if cur_min_hr > abs(hr - hit_rate):
                cur_min_sz = sz
                cur_min_hr = abs(hr - hit_rate)
    print (cur_min_hr, cur_min_sz)



LRU_HIT_RATE_DATA = {(ZP1C_DRIVER) : 39166,
                     (Z075P1C_DRIVER) : 74583,
                     (WSET_DRIVER): 89999,
                     (DYN_INTRO): 42500,
                     (GDWheel_1_3[0]): 42500,
                     (GDWheel_1_3[1]): 42500,
                     (GDWheel_1_3[2]): 42500,
                     (DYN_PROMOTE_GD3): 42500,
                     (DYN_PROMOTE): 42500,
                     MSR_MSN_DECISION : 1000,
                     MSR_VIRAL_TREE : 35666,
                     SPC_MERGE_S: 5*10**5,
                     SPC_OLTP : 2*10**4,
                     DOUBLE_HITTER : 2*10**4,
                     }

LRU_HIT_RATE_DATA_70 = {(ZP1C_DRIVER) : 3000,
                        (Z075P1C_DRIVER) : 37083,
                        (WSET_DRIVER) : 69999,
                        (DYN_INTRO): 5009,
                        (GDWheel_1_3[0]): 5009,
                        (GDWheel_1_3[1]): 5009,
                        (GDWheel_1_3[2]): 5009,
                        (DYN_PROMOTE_GD3): 5009,
                        (DYN_PROMOTE): 5009,
                    }

ARC_COMPARE_CACHE_SZ_S = 5.25*10**5
ARC_COMPARE_CACHE_SZ_P = 3.28*10**4

def count_single(d):
    driver = drivers[d]()
    count = 0
    while driver.get_next() != -1:
        count += 1
    return (driver.name, count)

def count_reqs(d_all = ARC_S1_S3 + ARC_P1_P14 + [SPC_OLTP, SPC_MERGE_S], processes = 16):
    pool = Pool(processes = processes)
    result = pool.map(count_single, d_all)
    pool.close()
    output = open("results/workload_req_counts.csv", "w")
    output.write("driver, count\n")
    for i in result:
        output.write("%s, %.3e\n" % i)
    output.close()

def run_p_compares(d_all = ARC_P1_P14[0:4] + ARC_P1_P14[8:1], requests = -1):
    compare_to_lru_on(d_all = d_all, requests = requests)

def compare_to_lru_on(seed = 0, d_all = [ZP1C_DRIVER, Z075P1C_DRIVER, WSET_DRIVER, DYN_INTRO] + GDWheel_1_3 + [DYN_PROMOTE_GD3], requests = 10**6,
                      p_all = [P_GD, P_HYPER, P_LFU, P_ARC], lookup_dict = LRU_HIT_RATE_DATA, processes = 12):
    if d_all == None:
        d_all = [ d for d in lookup_dict.keys() ]

    # first find size lru s.t. hit_rate is met.
    universe = 10**5
    
    workload = []
    for d in d_all:
        if d in lookup_dict:
            cache_size = lookup_dict[d]
        elif d in ARC_S1_S3:
            cache_size = ARC_COMPARE_CACHE_SZ_S
        elif d in ARC_P1_P14:
            cache_size = ARC_COMPARE_CACHE_SZ_P
        else:
            raise Exception("need to set the cache_size for driver = %d" % d )

        workload += run_product(seed, [cache_size], d_all = [d] , p_all = p_all, 
                                start_counting = simulate.START_AFTER_FIRST_EVICT, 
                                requests = requests, 
                                universe = universe, return_workload_only = True)
    results = pool_exec(workload, processes = processes)
    first = True
    fname = "results/fixed-hit-rate-90pp.csv"
    output = sys.stdout

    for d in d_all:
        row = [r for r in results if r.driver_number == d]        
        row = sorted(row, key = lambda r : r.policy_name)
        if first:
            first = False
            output.write("workload, K, scaled_by, ")
            output.write(", ".join(["%s" % r.policy_name for r in row]))
            output.write("\n")
        output.write("%s, " % row[0].driver_name)
        output.write("%s, " % row[0].cache_size)
        output.write("%s, " % row[0].reqs)
        output.write(", ".join(["%f" % (float(r.misses) / max(1, r.reqs)) for r in row]))
        output.write("\n")

    output.flush()
