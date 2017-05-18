"""
Moved bucketing related code from main.py to here, would require some 
refactoring to get all this code up and running again...
"""


def basic_bucket_generator(n_buckets):
    return sorted([float(i) / n_buckets for i in range(1, n_buckets+1)])


def run_coarse_p(s, k, buckets = [100], d = 0):
    
    workload_bucketed = [ (s, k, 3, b, d, b, b) for b in buckets ]
    workload_full_gd  = [ (s, k, 3, 0, d, 0, b) for b in buckets ]

    all_results = pool_exec(workload_bucketed + workload_full_gd)

    for r_b, r_gd in zip(all_results[:len(buckets)],
                         all_results[len(buckets):]):
        print "%d, %f, %f" % (r_b[0], r_b[1] / r_gd[1], r_gd[1])

def run_split_buckets(seed, k, start_buckets = 4, n_adds = 10, d = 0):
    pool = Pool(processes=16)

    results = go_split_buckets(seed, k, b_start=start_buckets,
                               n_adds = n_adds, pool=pool, d = d, split_strategy = 1)

    base_r = results[0][1]

    pool.close()
    print "BASE : %f" % base_r
    print "%d kmeans buckets : %f" % (start_buckets, 
                                      results[(start_buckets, "kmean")][1]/base_r)
    print "buckets, kmeans, e.split"

    for b in range(start_buckets + 1, start_buckets + n_adds + 1):
        print( ("%d, %%f, %%f" % b) % tuple([results[(b, k)][1] / base_r  
                                             for k in ("kmean", "split")]) )            


def run_split_aggressive(seed, k, buckets = [4]):
    pool = Pool(processes = 16)
    results = pool.map(lambda b : go_split_buckets(seed, k, b_start = b, 
                                                   n_adds = 1, split_strategy = 0),
                       buckets)

    pool.close()

    print "buckets, kmeans, with.1.split"

    for b, r_b in zip(buckets, results):
        base_r = r_b[0][1]
        print( ("%d, %%f, %%f" % b) % tuple([r_b[(b, k)][1] / base_r  
                                             for k in ("kmean", "split")]) )
    
def run_recursive_balance(s, k, buckets = [4], d = 0, n_recurs = 10):
    pool = Pool(processes = 16)

    r_buckets = [0] + buckets

    all_results = pool.map( (lambda b : go_recurse_balance(s, k, b, d, n_recurs)),
                            r_buckets )

    pool.close()

    base_r = all_results[0][1]
    
    balance_results = all_results[1:]

    print("buckets, nrecurs, ratio, delta.bounds")

    for bucket_r in balance_results:
        nrecurs, tcost, nbuckets  = max([(ix, r[1], len(r[3]))
                                         for ix, r in enumerate(bucket_r)])
        delta_bounds = sum([abs(a - b) for (a,b) in zip(bucket_r[-1][4],
                                                        bucket_r[-2][4])])
        print("%d, %d, %f, %f" % (nbuckets, nrecurs, tcost / base_r, delta_bounds))


def make_balanced_eviction_buckets(b_evictions, b_bounds, target_l):
    b_sizes = [u - l for (u, l) in zip(b_bounds, [0] + b_bounds[:-1])]
    b_densities = [float(e) / s for (e, s) in zip(b_evictions, b_sizes)]

    if target_l < len(b_bounds):
        print "target_l < len(b_bounds)"
        assert False

    target_evicts = float(sum(b_evictions)) / (target_l)

    def make_new_buckets(b_sizes, b_density, target_evicts):
        region = 0
        place = 0
        cur_evicts = 0

        while place < b_bounds[-1]:
            dist_w_cur_density = (target_evicts - cur_evicts) / b_density[region]
            if dist_w_cur_density + place <= b_bounds[region]:
                place += dist_w_cur_density
                cur_evicts = 0
                yield place
            else:
                dist_w_cur_density = b_bounds[region] - place
                cur_evicts += dist_w_cur_density * b_density[region]
                place = b_bounds[region]

            if place > b_bounds[region]:
                print "ASSERT FAILED: place <= b_bounds[region]"
                assert False

            if place == b_bounds[region]:
                region += 1
    
    r = list(make_new_buckets(b_sizes, b_densities, target_evicts))
    return r[:target_l-1] + b_bounds[-1:]

def go_recurse_balance(s, k, b, d = 0, n_recurs = 10):
    p = 0
    
    if b == 0:
        return run_sim_make((s,k,p,0,d))

    kmeans_results = run_sim_make((s,k,p,b,d))
    r = [kmeans_results]

    (bucket_balance, bucket_bounds) = (r[0].bucket_evictions, r[0].bucket_bounds)
    
    for i in range(n_recurs):
        buckets_new = make_balanced_eviction_buckets(bucket_balance,
                                                     bucket_bounds, b)
        results = run_sim_make(( s, k, p, b, d, buckets_new ))
        r.append(results)
        (bucket_balance, bucket_bounds) = (results.bucket_evictions, results.bucket_bounds)
    return r
        
def go_split_buckets(seed, k = 10**3, b_start = 4, d = 0, n_adds = 1, pool = None, split_strategy = 0):
    """ 
    split_strategy ::
    0 := full balance, 1 := split single bucket repeatedly, 
    2 := everything in one bucket, plus high and low 
    """

    p = 0


    sn = [(seed, k, p, 0,d) , (seed, k, p, b_start, d) ]
    if pool == None:
        base_results = map(run_sim_make, sn)
    else:
        base_results = pool.map(run_sim_make, sn)

    r = {}
    
    r[0] = base_results[0]
    r[(b_start, "kmean")] = base_results[1]

    (bucket_balance, bucket_bounds) = r[(b_start, "kmean")][3:5]

    _, s = max([(evictions, bucket_num) for (bucket_num, evictions) in 
                enumerate(bucket_balance)])
    
    sn = []

    b = b_start

    for i in range(n_adds):
        b += 1

        if split_strategy == 0:
            if n_adds == 1: # this is a pretty hacky check. for collecting data...
                b = b_start
            buckets_new = make_balanced_eviction_buckets(bucket_balance,
                                                         bucket_bounds, b)
        elif split_strategy == 1:
            buckets_new = list(bucket_bounds)
            upper = buckets_new[s]
            if s > 0:
                lower = buckets_new[s - 1]
            else:
                lower = 0
            delta = (upper - lower) / (i + 2)
            new_bounds = [ lower + (delta * c) for c in range(1, i + 2) ]
            for new_bound in reversed(new_bounds):
                buckets_new.insert(s, new_bound)
                
        elif split_strategy == 2:
            # we just do *one* focused split
            b = b_start
            # split sth bucket into b - 2 buckets,
            # everything else in 2 buckets
            if s == 0:
                buckets_new = [ bucket_bounds[0] / i for i in range(1, b) ]
                buckets_new.append(bucket_bounds[-1])
            else:
                l_bucket = bucket_bounds[s - 1]
                if s == len(bucket_bounds) - 1:
                    delta = 1.0 - l_bucket
                else:
                    delta = bucket_bounds[s] - l_bucket
                buckets_new = [l_bucket]
                buckets_new += [l_bucket + (delta / i) for i in range(1, b - 1) ]
                buckets_new += [bucket_bounds[s]]
        elif split_strategy == 3:
            b = b_start
            # split sth bucket into 2 buckets
            # delete the lowest eviction bucket
            _, lowest = min([(bal, bucket_num) for (bucket_num, bal) in 
                             enumerate(bucket_balance)])
            buckets_new = list(bucket_bounds)

            if lowest == len(bucket_bounds) - 1:
                # merge with lower bucket
                del buckets_new[lowest - 1]
            else:
                # otherwise, merge with higher bucket
                del buckets_new[lowest]

            if s > lowest:
                s -= 1  # ix of max bucket decreases b/c of above del
            if s == 0:
                l_bound = 0
            else:
                l_bound = buckets_new[s - 1]
            buckets_new.insert(s, float(bucket_bounds[s] + l_bound)/2)

        sn += [(seed, k, p, b, d), (seed, k, p, b, d, buckets_new)]

    if pool == None:
        map_results = map(run_sim_make, sn)
    else:
        map_results = pool.map(run_sim_make, sn)
        
    for ix, sim_result in enumerate(map_results):
        if ix % 2 == 0:
            key = "kmean"
        else:
            key = "split"
        sim = sn[ix]
        nbuckets = sim_result[0]
        r[(nbuckets, key)] = sim_result

    return r

def go(seed, k = 10**3, policy = 0, d = 0):
    workload = [(seed, k, policy, l, d) for l in range(0,17)]
    result = pool_exec(workload)

    scale = float(result[0][1])
    for a in ["%d, %f, %f, %f" % (b, m/scale, m, e) for b,m,e,_,_ in result]:
        print a
    return result
