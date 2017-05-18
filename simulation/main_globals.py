import strategies.pq_strategies as pq
import strategies.sampling as sampling
import strategies.arc as arc
import strategies.frequency as f
import strategies.lru as lru

from workloads import cost_driver, gdwheel, spc_traces, ycsb_zipf, binary_distr, msn_driver, memcachier
from workloads.virality_trees import vtree

policies = []

def register_policy(*p_to_add):
    global policies
    out = [ len(policies) + i for i in range(len(p_to_add)) ]
    policies += list(p_to_add)
    if len(out) == 1:
        return out[0]
    return out

drivers = []
 
def register_drivers(*d_to_add):
    global drivers
    out = [ len(drivers) + i for i in range(len(d_to_add)) ]
    drivers += list(d_to_add)
    if len(out) == 1:
        return out[0]
    return out

def construct_sampling_hyper(sampling, **kwargs):
    if sampling == -1:
        return f.RealMin_Hyper(**kwargs)
    else:
        return f.Sample_Hyperbolic(
            sampling = sampling, name = ("S(%d)_Hyper" % sampling), **kwargs)

def construct_retaining_hyper(retaining, **kwargs):
    if retaining == -2:
        retaining = 0

    if retaining == -1:
        return f.RealMin_Hyper(**kwargs)
    else:
        return f.Sample_Hyperbolic(
            retain = retaining, name = ("S(64; %d)_Hyper" % retaining), **kwargs)

#
#  REGISTERING POLICIES :
#

P_GD = register_policy(pq.GD_PQ)
P_LRU = register_policy(lru.LRU_Strategy)
P_PERF_KNOWLEDGE = register_policy(pq.PerfectKnowledge_PQ)
P_LFU = register_policy(f.PQ_Frequency)
P_S_FREQ = register_policy(f.Sampling_Frequency)
P_HYPER = register_policy(f.Sample_Hyperbolic)
P_HYPER_EXPIRE = register_policy(f.Sample_Frequency_Expiry)
P_ARC = register_policy(arc.Arc)
P_GD_TSP = register_policy(pq.GD_TSP_PQ)
P_HYPER_SAMPLE_PARAMETER = register_policy(construct_sampling_hyper)
P_HYPER_RETAIN_PARAMETER = register_policy(construct_retaining_hyper)
P_HYPER_CLASS_TRACK = register_policy(f.Sample_Hyperbolic_ClassTrack)
P_WINDOW_HYPER = register_policy(f.Windowed_Hyper)
P_WINDOW_FREQ = register_policy(f.Windowed_Freq)
P_HYPER_SZ = register_policy(f.Size_Aware_Hyperbolic)
P_GD_SZ = register_policy(pq.Size_Aware_GD_PQ)

#
#  REGISTERING WORKLOADS :
#

Z075P1C_DRIVER = register_drivers(
    (lambda zipf_param = -1, d_second = 0.75, **kwargs: 
     cost_driver.ZipfFixedDriver(name = "Z(%f)P1C" % (d_second),
                                zipf_param = d_second, **kwargs)) )
 
ZP1C_DRIVER = register_drivers( 
    (lambda **kwargs : 
     cost_driver.ZipfFixedDriver(name = "ZPop_UnitC", 
                                             **kwargs)) )
drivers += [
    (cost_driver.ZipfUniformDriver), ]


GDWheel_1_3 = register_drivers(gdwheel.GD1, gdwheel.GD2, gdwheel.GD3)

DYN_PROMOTE, DYN_PROMOTE_GD3, DYN_INTRO = register_drivers(
    (lambda d_second = 100, **kwargs: 
     cost_driver.DynamicPromote(period = d_second).wrap_driver(
         cost_driver.ZipfFixedDriver(**kwargs))),
    (lambda d_second = 100, **kwargs: 
      cost_driver.DynamicPromote(period = d_second).wrap_driver(
          gdwheel.GD3(**kwargs))),
    (lambda d_second = 100, **kwargs: 
      cost_driver.Introducing(period = d_second, 
                              move_to = 0, 
                              name = "IntHigh.%d").wrap_driver(
                                  cost_driver.ZipfFixedDriver(**kwargs))))

WSET_DRIVER = register_drivers(
    (lambda d_second = (10, 1), **kwargs: 
                 cost_driver.WSetDriver( working_set_reqs = d_second[0], 
                                         working_set_replay = d_second[1], 
                                         **kwargs)) )

HEAVY_DYNAMICS = register_drivers(
    (lambda d_second = 100, zipf_param = -1, **kwargs: 
     cost_driver.DynamicPromote(period = d_second).wrap_driver(
         cost_driver.ZipfFixedDriver(
            name = "Z(%f)P1C" % (0.75), zipf_param = 0.75, **kwargs))),
    (lambda d_second = 100, zipf_param = -1, **kwargs: 
     cost_driver.Introducing(
         period = d_second, move_to = 0, name = "IntHigh.%d").wrap_driver( 
             cost_driver.ZipfFixedDriver(name = "Z(%f)P1C" % (0.75), 
                                         zipf_param = 0.75, **kwargs))))


SPC_MERGE_S, SPC_OLTP = register_drivers(
    (lambda d_second = -1, **kwargs: 
      spc_traces.SPCTraceDriver(trace_file = "workloads/spc1/WebSearch1.spc",
                                name = "SPCWebSearch", 
                                column_map = spc_traces.SPCMap1, **kwargs)),
    (lambda d_second = -1, **kwargs: 
      spc_traces.SPCTraceDriver(trace_file = "workloads/spc1/Financial1.spc",
                                name = "SPCFinancial", 
                                column_map = spc_traces.SPCMap1, 
                                page_sz = 512, **kwargs)))

DOUBLE_HITTER = register_drivers(
    (lambda d_second = 20, **kwargs : 
     cost_driver.DoubleHitterDriver(
         driver = cost_driver.ZipfFixedDriver(name = "ZPop_1C", **kwargs),
         n_hitters_every_k = d_second,
         n_hits = 10)))

ARC_DRIVERS_ALL = register_drivers(*spc_traces.ARC_WORKLOADS)
ARC_P1_P14 = ARC_DRIVERS_ALL[5:(5+14)]
ARC_S1_S3  = ARC_DRIVERS_ALL[19:(19+3)]

MSR_VIRAL_TREE, MSR_MSN_DECISION = register_drivers(
    (lambda d_second = -1, **kwargs: 
     vtree.ViralityTreeDriver(
         name = "VTree(40k)", 
         filename = "workloads/virality_trees/output-40k.json", **kwargs)),
    msn_driver.MSNDriver)

HOTCLASS = register_drivers(cost_driver.ZPop_HotCostClass)

STOCHASTIC_CLASS = register_drivers(cost_driver.StochCostClass_Factory)

MEMCACHIER_T1_WORKLOADS = register_drivers(*memcachier.Workload_lambdas_t1)
MEMCACHIER_T2_WORKLOADS = register_drivers(*memcachier.Workload_lambdas_t2)
MEMCACHIER_CAT_WORKLOADS = register_drivers(*memcachier.Workload_lambdas_cat)
