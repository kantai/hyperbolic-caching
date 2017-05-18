import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cm as cm
import operator as o

import sys, csv

WORKING_DIR = "results/"
TMP_DIR = "/tmp"
OUT_DIR = "../paper/atc17/figures"

FOR_PAPER_OR_THESIS = "THESIS"

if FOR_PAPER_OR_THESIS == "THESIS":
    LABEL_SIZE = 10
    SINGLE_COL_WIDTH_INCHES = 7
else:
    LABEL_SIZE = 8
    SINGLE_COL_WIDTH_INCHES = 3.5

DEF_HEIGHT_INCHES = 0.75 * SINGLE_COL_WIDTH_INCHES

SINGLE_COL_DEF = (SINGLE_COL_WIDTH_INCHES, DEF_HEIGHT_INCHES)

STD_MRC_SETTINGS = { "x_logscale" : True, "x_lim" : (10**2, 10**5),
                     "fig_kw" : {"figsize" : (SINGLE_COL_WIDTH_INCHES, 0.65 * DEF_HEIGHT_INCHES)},
                     "plot_kwargs" : {"linestyle" : "solid", "marker" : "o"},
                     "x_title" :  "Cache Size (objects)", 
                     "y_title" : "Miss Rate Overhead",
                     "y_is_pdelta" : True, }

STD_MRC_SETTINGS_HALF = dict(STD_MRC_SETTINGS)
STD_MRC_SETTINGS_HALF["fig_kw"] = {"figsize" : (SINGLE_COL_WIDTH_INCHES, 0.5 * DEF_HEIGHT_INCHES)}
STD_MRC_SETTINGS_HALF["y_title"] = "Miss Rate\nOverhead"

LEGEND_ABOVE = {"loc" : "lower center", 
                "bbox_to_anchor" : (.5, 1.001), "ncol" : 2}
LEGEND_BELOW = {"loc" : "upper center", 
                "bbox_to_anchor" : (.5, -0.2), "ncol" : 2}

LEGEND_BEST = {"loc" : 0, "ncol" : 2}

POLICY_NAMES = {"ARC" : "ARC", 
                "GD_PQ" : "LRU/GD", 
                "PQ_Frequency" : "LFU", 
                'S_Hyper_Sz(0e+00; 0.100)' : "HC-Size",
                "GD_PQ_Sz" : "GD-Size",
                "S_Hyper(0e+00; 0.100)" : "HC"}

WLOAD_BAR_TITLES = {"GD1" : "GD1", 
                    "ZPop_UnitC" : "Zipf(1)", 
                    "GD3.DynPromote.100" : "DynPromote", 
                    "ZipfFixedDriver.DynPromote.100" : "DynPromote", 
                    "ZipfFixedDriver.IntHigh.100" : "DynIntro",
                    "GD2" : "GD2", "GD3" : "GD3",
                    "Z(0.750000)P1C" : "Zipf(0.75)", 
                    "WorkingSetUC(10; 1)" : "WorkingSet"}


import re

time_regx = re.compile("(\d+)m(.*)s")
def throughput_fmt(s):
    mins, secs = time_regx.match(s).groups()
    return (float(mins) * 60) + float(secs)


def load_input_file(input_file, select_x = "K", select_cols = None, select_rows = None, header = True, 
                    x_fmt = float, col_fmt = {}, merge = np.mean, merge_by = None, length = -1, 
                    return_dict = False):
    lines = 0
    with open(input_file, 'r') as f:
        first = True
        row_keys = {}


        data = {}

        if merge_by != None and merge_by in col_fmt:
            merge_by_fmt = col_fmt[merge_by]
        else:
            merge_by_fmt = float

        for line in f:
            if line == "":
                continue
            row_vals = [s.strip() for s in line.split(",")]

            if header and first:
                for ix, name in enumerate(row_vals):
                    row_keys[name] = ix
                x_ix = row_keys[select_x]
                if merge_by:
                    merge_ix = row_keys[merge_by]
                else:
                    merge_ix = None
                cols_ix = [row_keys[s] for s in select_cols]
                cols_ix_fmts = []
                for s in select_cols:
                    if s in col_fmt:
                        cols_ix_fmts.append(col_fmt[s])
                    else:
                        cols_ix_fmts.append(float)
                for s in cols_ix:
                    data[s] = []
                first = False
                continue

            if select_rows and not select_rows(row_vals):
                continue
                
            lines += 1
            if length > 0 and lines > length:
                break


            for s, col_fmt in zip(cols_ix, cols_ix_fmts):
                row = ( x_fmt(row_vals[x_ix]), col_fmt(row_vals[s]) )
                if merge_ix != None:
                    row = ( merge_by_fmt( row_vals[merge_ix] ), ) + row 
                data[s].append( row )

        if return_dict:
            out = dict()
            for s in select_cols:
                out[s] = data[row_keys[s]]
            return out
        plot_data = [ data[row_keys[s]] for s in select_cols ]

        if merge_ix != None:
            out_data = []
            for line in plot_data:
                line.sort()
                merged_line = []
                prev_m = None
                for i, (m, x, y) in enumerate(line):
                    if m == prev_m:
                        merge_x.append(x)
                        merge_y.append(y)
                    else:
                        if prev_m != None:
                            merged_line.append( (merge(merge_x), merge(merge_y)) )
                        prev_m = m
                        merge_x = [x]
                        merge_y = [y]
                    if i % 100 == 0:
                        sys.stderr.write("\r%d" % i)
                        sys.stderr.flush()
                merged_line.sort()
                out_data.append(merged_line)
            plot_data = out_data
        return plot_data, select_cols

def scale_atom(x, scale_by):
    if x == scale_by:
        return 1
    else:
        return float(x) / scale_by

def pdelta(y):
    return (y - 1.0)
    
def pdelta_str(y):
    if y < 1:
        return "%d%%" % (int (((y - 1.0) * 100) - 0.5))
    else:
        return "%d%%" % (int (((y - 1.0) * 100) + 0.5))

def pdelta_str_flt(y):
    if y < 1:
        return "%.1f%%" % (float (((y - 1.0) * 100) - 0.5))
    else:
        return "%.1f%%" % (float (((y - 1.0) * 100) + 0.5))
                    


def barplot(dpoints, subgrp_order = None, grp_order = None, fig_kw = {}, fname = "/tmp/foo.png",
            x_title = "", y_title = "", subgrp_titles = None, grp_titles = None, 
            legend_kw = {}, **kwargs):
    # plots grouped bar charts...
    # sort by default by means

    # legend = group = eviction strategy = car of datapoint
    # cluster = subgroup = workload = cadr of datapoint


    # Aggregate the conditions and the categories according to their
    # mean values
    grp = [(c, np.mean(dpoints[dpoints[:,0] == c][:,2].astype(float))) 
           for c in np.unique(dpoints[:,0])]
    subgrp = [(c, np.mean(dpoints[dpoints[:,1] == c][:,2].astype(float))) 
              for c in np.unique(dpoints[:,1])]
    
    # sort the conditions, categories and data so that the bars in
    # the plot will be ordered by category and condition
    grp = [c[0] for c in sorted(grp, key=o.itemgetter(1))]
    subgrp = [c[0] for c in sorted(subgrp, key=o.itemgetter(1))]
    if grp_order != None:
        for g in grp: 
            assert g in grp_order
        for g in grp_order: 
            assert g in grp
        grp = grp_order
    
    if subgrp_order != None:
        for g in subgrp: 
            assert g in subgrp_order
        for g in subgrp_order: 
            assert g in subgrp
        subgrp = subgrp_order
    
    
    dpoints = np.array(sorted(dpoints, key=lambda x: subgrp.index(x[1])))

    # the space between each set of bars
    space = 0.1
    n = len(grp)
    width = (1 - space) / (len(grp))
    
    fig, ax = plt.subplots( nrows=1, ncols=1 , **fig_kw) 

    bar_locs = []
    ax.grid(b=True, which='major', axis = 'y', color = 'grey')
    ax.axis(zorder = 4)


    patterns = (False, ) * 10
    if "no-hatch" not in kwargs and not FOR_PAPER_OR_THESIS == "THESIS": 
        patterns = (False, "/", '.', 'X', '\\')

    # Create a set of bars at each position
    for i,cur in enumerate(grp):
        indeces = range(1, len(subgrp)+1)
        vals = dpoints[dpoints[:,0] == cur][:,2].astype(np.float)
        pos = [j - (1 - space) / 2. + i * width for j in indeces]
        ax.bar(pos, vals, width=width, label=cur, 
               linewidth=1, hatch = patterns[i] * 5,
               color = ("C%d" % i), edgecolor = "black", zorder = 3)
        bar_locs += [(p, v, width) for p,v in zip(pos, vals)]
    
    # Set the x-axis tick labels to be equal to the subgrp
    ax.set_xticks([j - width/2. for j in indeces])
    if subgrp_titles:
        if callable(subgrp_titles):
            sg_t = [subgrp_titles(s) for s in subgrp]
        else:
            sg_t = [subgrp_titles[s] for s in subgrp]
        ax.set_xticklabels(sg_t)
    else:
        ax.set_xticklabels(subgrp)
    
    rotate_by = 25
    ha = 'right'
    if "rotation" in kwargs:
        rotate_by = kwargs["rotation"]
        ha = 'center'

    plt.setp(plt.xticks()[1], rotation=rotate_by, ha=ha)
    
    xl = ax.get_xlim()
    ax.set_xlim( (-space - width/2. + min(bar_locs)[0], max(bar_locs)[0] + width + space))

    ax.tick_params(labelsize = LABEL_SIZE)
    ax.tick_params(top = "off")

    ax.set_xlabel(x_title, fontsize = LABEL_SIZE)
    ax.set_ylabel(y_title, fontsize = LABEL_SIZE)
    
        
    if 'y_lim' in kwargs:
        ax.set_ylim(kwargs['y_lim'])
    if 'y_lim' in kwargs and not 'no_val_labels' in kwargs:
        y_top = kwargs['y_lim'][1]
        y_bot = kwargs['y_lim'][0]
        for i, (p, v, width) in enumerate(bar_locs):
            x_pos = p + width/2.0
            plus = ""
            tail = ""
            draw = False
            if v > (.97 * y_top):
                y_pos = y_top
                va = 'top'
                plus = ""
                draw = True
            elif v < 0:
                va = 'bottom'
                if v < (.97 * y_bot):
                    y_pos = y_bot + 0.005
                    draw = True
                elif abs(v) < (y_top - y_bot) / 6: 
                    y_pos = 0.01          
                else:
                    y_pos = v + 0.005
            elif v < (y_top - y_bot) / 5:
                y_pos = v
                va = 'bottom'
                plus = ""
            else:
                y_pos = v
                va = 'top'
                plus = ""
            if 'is_real' in kwargs:
                label = (plus + "%d" + tail) % (v + 0.5)
            else:
                label = (plus + "%d%%" + tail) % ((v)*100 + 0.5)
            if not draw:
                continue
            ax.text(x_pos, y_pos, label, 
                    verticalalignment = va, 
                    horizontalalignment = "right", rotation = 90, 
                    fontsize = LABEL_SIZE - 1, 
#                    bbox=dict(facecolor=cm.Pastel1(float((i % n) - 1) / n), 
#                              edgecolor =None,
#                              alpha=0.5,),
                    color = 'black', fontweight = 'bold')
                

    if 'is_real' in kwargs:
        ax.set_yticklabels(["%d" % int(round(y)) for y in ax.get_yticks()])
    else:
        ax.set_yticklabels(["%d%%" % int(round(100*y)) for y in ax.get_yticks()])

    handles, labels = ax.get_legend_handles_labels()
    if grp_titles:
        labels = [grp_titles[s] for s in labels]
    extra_artists = (ax.legend(handles, labels, fontsize = LABEL_SIZE, **legend_kw), )
    
    fig.savefig(fname, bbox_extra_artists=extra_artists, bbox_inches='tight')
    plt.close(fig)

def graph_miss_rate_curves(plot_data, y_title = "", x_title = "", col_titles = None, 
                           plot_kwargs = {}, fig_kw = {}, legend_kw = {},
                           scale_by = None, yticks_num = None, xticks = None,
                           second_x = None, draw_line_at = False, pdelta_str = pdelta_str,
                           fname = OUT_DIR + "/foo.png", **kwargs):
        fig, ax = plt.subplots( nrows=1, ncols=1 , **fig_kw) 

        if 'subtractive' in kwargs:
            for line in plot_data:
                line.sort()
            all_y = [ [y for x,y in line] for line in plot_data ]
            assert len(all_y) == 2
            out_y = [abs(y_0 - y_1) for y_0, y_1 in zip(*all_y)]
            out_x = [x for x, y in plot_data[0]]
            ax.plot(out_x, out_y)
            plot_data = []
        if scale_by != None:
            for line in plot_data:
                line.sort()
            scale_line = plot_data[scale_by]
            plot_data = plot_data[:scale_by] + plot_data[scale_by+1:]
            plot_data = [ [( x, scale_atom(y, scale)) for
                           ((x,y), (_, scale)) in zip(line, scale_line)]
                          for line in plot_data ]

        for col_ix, line in enumerate(plot_data):
            line.sort()
            x, y = zip(*line)
            
            if "lin_regression" in kwargs:
                z = np.polyfit(x,y,1,full=True)
                from scipy.stats.stats import linregress

                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                print (slope, intercept, r_value)

            plt_kwargs = dict(plot_kwargs)
            if col_titles != None:
                plt_kwargs["label"] = col_titles[col_ix]
            ax.plot(x,y, zorder = 3, **plt_kwargs)

        ax.grid(b=True, which='major', zorder = 1)
        ax.grid(b=True, which='minor', zorder = 1)
        
        ax.tick_params(labelsize = LABEL_SIZE)

        ax.set_xlabel(x_title, fontsize = LABEL_SIZE)
        ax.set_ylabel(y_title, fontsize = LABEL_SIZE)

        if 'x_logscale' in kwargs and kwargs['x_logscale']:
            ax.set_xscale('log')
        if 'y_logscale' in kwargs and kwargs['y_logscale']:
            ax.set_yscale('log')
        if 'x_lim' in kwargs:
            ax.set_xlim(kwargs['x_lim'])
        if 'y_lim' in kwargs:
            ax.set_ylim(kwargs['y_lim'])

        if yticks_num:
            ax.locator_params(axis='y',nbins=yticks_num)
        if xticks:
            ax.set_xticks(xticks)

        if 'y_is_pdelta' in kwargs and kwargs['y_is_pdelta']:
            ax.set_yticklabels([pdelta_str(y) for y in ax.get_yticks()])


        if draw_line_at:
            ax.vlines(*draw_line_at)


        if second_x:
            ax2 = ax.twiny()
            ax2.set_frame_on(True)
            ax2.patch.set_visible(False)
            ax2.xaxis.set_ticks_position('bottom')
            ax2.xaxis.set_label_position('bottom')
            ax2.spines['bottom'].set_position(('outward', 40))
 
            ax2.set_xscale(ax.get_xscale(), subsx = [])

            new_tick_locations = np.array(second_x)

            scale_dict = {}
            for x,y in scale_line:
                scale_dict[x] = y
            def tick_function(x):
                return "%.2f" % scale_dict[x]
    
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(second_x)
            ax2.set_xticklabels([tick_function(i) for i in new_tick_locations])

        if col_titles != None:
            extra_artists = (ax.legend(fontsize = LABEL_SIZE, numpoints = 1, **legend_kw), )
        else:
            extra_artists = ()
            
            #        fig.tight_layout(pad=0.2)

        fig.savefig(fname, bbox_extra_artists=extra_artists, bbox_inches='tight')
        plt.close(fig)

def graph_relative_miss_rate(select_cols, select_rows, scale_by, 
                             subgrp_titles, grp_titles, outname, 
                             select_x = "workload",
                             fname = "fixed-hit-rate-90pp", **kwargs):
    read_data = load_input_file( WORKING_DIR + fname + ".csv",
                                 select_cols = select_cols, select_rows = select_rows,
                                 select_x = select_x, x_fmt = str, return_dict = True)
    if scale_by:
        scale_dict = {}
        for workload, hitrate in read_data[scale_by]:
            scale_dict[workload] = hitrate

    flattened = []
    for c in select_cols:
        if scale_by:
            if c == scale_by:
                pass
#                flattened += [[c, workload, 1.0] for (workload, hitrate) in read_data[c] ]
            else:
                flattened += [[c, workload, pdelta(hitrate / scale_dict[workload])] for (workload, hitrate) in
                              read_data[c]]
        else:
            flattened += [ [c, workload, hitrate] for (workload, hitrate) in read_data[c] ]
    dpoints = np.array(flattened)

    
    
    SETTINGS = {}
    SETTINGS["y_title"] = STD_MRC_SETTINGS["y_title"]
    SETTINGS["legend_kw"] = dict(LEGEND_ABOVE)
    SETTINGS["legend_kw"]["ncol"] = 3
    
    SETTINGS["subgrp_titles"] = subgrp_titles
    SETTINGS["grp_titles"] = grp_titles

    SETTINGS.update(kwargs)
                     
    barplot(dpoints, fname = OUT_DIR + "/" + outname, **SETTINGS)

def graph_relative_miss_rate_ARC():
    workloads = [("Arc.P%d" % n, "P%d" % n) for n in range(1, 5)]
    workloads += [("Arc.S%d" % n, "S%d" % n) for n in range(1, 2)]
    workloads += [("SPCFinancial","Financial")]
    workloads += [("SPCWebSearch","WebSearch")]
    rows = [a for (a,b) in workloads]
    subgrp_titles = {}
    for (a,b) in workloads:
        subgrp_titles[a] = b
        
    select_cols = ["ARC", "GD_PQ", "PQ_Frequency", "S_Hyper(0e+00; 0.100)"]

    select_rows = (lambda row : row[0] in rows)
    scale_by = "S_Hyper(0e+00; 0.100)"

    legend_kw = dict(LEGEND_ABOVE)
    legend_kw["ncol"] = 3

    graph_relative_miss_rate(select_cols, select_rows, scale_by, subgrp_titles, POLICY_NAMES, 
                             "compare_arc.pdf", subgrp_order = rows, 
                             y_lim = (-0.1, .15),
                             fig_kw = {"figsize" : (SINGLE_COL_WIDTH_INCHES, 0.35 * DEF_HEIGHT_INCHES)},
                             grp_order = select_cols[:3],
                             legend_kw = legend_kw)


def graph_relative_miss_rate_synthetics_70pp():
    rows = ["ZPop_UnitC", "Z(0.750000)P1C",
            "GD1", "GD2", "GD3",
            "ZipfFixedDriver.IntHigh.100", "ZipfFixedDriver.DynPromote.100",
            ] #, "WorkingSetUC(10; 1)"]
    select_cols = ["ARC", "GD_PQ", "PQ_Frequency", "S_Hyper(0e+00; 0.100)"]
    select_rows = (lambda row : row[0] in rows)
    scale_by = "S_Hyper(0e+00; 0.100)"

    legend_kw = dict(LEGEND_ABOVE)
    legend_kw["ncol"] = 3
    legend_kw["columnspacing"] = 1

    graph_relative_miss_rate(select_cols, select_rows, scale_by, 
                             WLOAD_BAR_TITLES, POLICY_NAMES, 
                             "compare_synthetics_70pp.pdf", 
#                             legend_kw = legend_kw,
                             subgrp_order = rows, fname = "fixed-hit-rate-70pp",
                             grp_order = select_cols[:3],
                             y_lim = (-0.2, .5),
                             fig_kw = {"figsize" : (SINGLE_COL_WIDTH_INCHES, 
                                                    0.35 * DEF_HEIGHT_INCHES)},)

def graph_relative_miss_rate_synthetics():
    rows = ["ZPop_UnitC", "Z(0.750000)P1C",
            "GD1", "GD2", "GD3",
            "ZipfFixedDriver.IntHigh.100", "ZipfFixedDriver.DynPromote.100",
            ]#, "WorkingSetUC(10; 1)"]
    select_cols = ["ARC", "GD_PQ", "PQ_Frequency", "S_Hyper(0e+00; 0.100)"]
    select_rows = (lambda row : row[0] in rows)
    scale_by = "S_Hyper(0e+00; 0.100)"

    legend_kw = dict(LEGEND_ABOVE)
    legend_kw["ncol"] = 3
    legend_kw["columnspacing"] = 1.2

    graph_relative_miss_rate(select_cols, select_rows, scale_by, WLOAD_BAR_TITLES, POLICY_NAMES, 
                             "compare_synthetics.pdf", 
                             y_lim = (-0.2, .5),
                             fig_kw = {"figsize" : (SINGLE_COL_WIDTH_INCHES, 
                                                    0.35 * DEF_HEIGHT_INCHES)},
                             legend_kw = legend_kw,
                             subgrp_order = rows, 
                             grp_order = select_cols[:3])

def graph_relative_throughput_websim():

    measurements = {}
    outname = "throughput_node_zipf.pdf"
    with open( WORKING_DIR + "throughput_web_sim.csv" ) as flines:
        first = True
        for line in flines:
            if first:
                first = False
                continue
            if line == "":
                continue
            row_vals = [s.strip() for s in line.split(",")]
            cache_size, throughput, miss_rate = ( float(i) for i in row_vals[1:] )
            variant = row_vals[0]

            if (variant, cache_size) not in measurements:
                measurements[(variant, cache_size)] = []

            measurements[(variant, cache_size)].append((throughput, miss_rate))

    flattened = []
    for (variant, cache_size), l_measures in measurements.items():
        flattened.append( [ variant, "%d" % cache_size, np.mean([tput for tput, mr in l_measures]) ] )

    dpoints = np.array(flattened)
    
    legend_kw = dict(LEGEND_ABOVE)
    legend_kw["ncol"] = 3

    SETTINGS = {"fig_kw" : {"figsize" : (SINGLE_COL_WIDTH_INCHES, 0.65 * DEF_HEIGHT_INCHES)},
                "legend_kw" : legend_kw,
                "y_title" : "Throughput (reqs/s)"}

    SETTINGS["grp_titles"] = {"hyper" : "Hyperbolic", "default" : "Default"}
    SETTINGS["subgrp_titles"] = {"3000" : "Cache Size = 3k", "39166" : "Cache Size = 39k"}

    barplot(dpoints, fname = OUT_DIR + "/" + outname, **SETTINGS)

def load_file(f):
    with open(f) as fd:
        reader = csv.reader(fd, skipinitialspace = True)
        for line in reader:
            if len(line) > 2:
                continue
            time = float(line[0])
            results = int(line[1])
            yield (time, results)

def load_file_selective(f, x_title, y_title):
    with open(f) as fd:
        reader = csv.reader(fd, skipinitialspace = True)
        first = True
        for line in reader:
            if first:
                x_selector = line.index(x_title)
                y_selector = line.index(y_title)
                first = False
                continue
            x = float(line[x_selector])
            y = float(line[y_selector])
            yield (x,y)

def graph_wiki_windowed_tput():
    line_classes = sorted(load_file('results/wiki_partial_test_classes'))
    line_nocosts = sorted(load_file('results/wiki_partial_test_nocosts'))

    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    
    ax.grid(b=True, which='major', zorder = 1)
    ax.grid(b=True, which='minor', zorder = 1)
    
    for line in [line_classes, line_nocosts]:
        ax.plot( *zip(*line), zorder = 3)

    fig.savefig("/tmp/windowed_tput.pdf", bbox_inches='tight')
    plt.close(fig)


def graph_ssd_rewrites():
    ssd_rewrites = sorted(load_file_selective('results/hyper.ssd.csv', "block_size", "write_amplification"))

    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    
    ax.grid(b=True, which='major', zorder = 1)
    ax.grid(b=True, which='minor', zorder = 1)
    
    ax.plot( *zip(*ssd_rewrites),  zorder = 3)

    fig.savefig("/tmp/hyper_ssd_rewrites.pdf", bbox_inches='tight')
    plt.close(fig)

def graph_django_results():
    fname = 'results/hyper_django_results'
    outname = "hyper_django.pdf"
    dp_misses = []
    dp_tputs = []

    grp_titles = { "hyper-cost-size" : "HC-Costs" ,
                   "hyper-no-cost-size" : "HC",
                   "hyper-cost-class-size" : "HC-Classes" }

    subgrp_titles = {
        "wiki-1G" : "Django-Wiki",
        "5m-devportal" : "Django-CMS",
        "100m-markdown-1" : "Markup"
    }
    
    grp_order = ["hyper-no-cost-size", "hyper-cost-size" , "hyper-cost-class-size"]

    subgrp_order = [
        "wiki-1G",
        "5m-devportal",
        "100m-markdown-1"
    ]

    variants_allowed = ["default", "hyper-cost-size", "hyper-no-cost-size",
                        "hyper-cost-class-size"]
    
    scale_by = "default"

    scale_misses = {}
    scale_tputs = {}

    with open(fname) as fd:
        reader = csv.reader(fd, skipinitialspace = True)
        for line in reader:
            group, variant = line[:2]
            tput, missrate = line[2:4]
            
            if variant not in variants_allowed or \
               group not in subgrp_titles:
                continue
            
            if variant == scale_by:
                scale_misses[group] = float(missrate)
                scale_tputs[group] = float(tput)
                continue

            dp_misses.append( [ variant, group, float(missrate) ] )
            dp_tputs.append( [ variant, group, float(tput) ] )

    dp_misses = [ (v, g, (m / scale_misses[g]) - 1) for v,g,m in dp_misses ]
    dp_tputs = [ (v, g, (m / scale_tputs[g]) - 1) for v,g,m in dp_tputs ]

    legend_kw = dict(LEGEND_ABOVE)
    legend_kw["ncol"] = 3

    SETTINGS = {"fig_kw" : {"figsize" : (SINGLE_COL_WIDTH_INCHES, 
                                         0.45 * DEF_HEIGHT_INCHES)},
                "legend_kw" : legend_kw,
            #    "is_real" : True,
                "y_lim" : (0, .28),
                "grp_titles" : grp_titles,
                "grp_order" : grp_order,
                "subgrp_order" : subgrp_order,
                "subgrp_titles" : subgrp_titles,
                "y_title" : "$\Delta$ Throughput %"}

    barplot(np.array(dp_tputs), fname = OUT_DIR + "/" + outname, 
            rotation = 0, no_val_labels = True,
            **SETTINGS)

def graph_relative_miss_rate_memcachier():

    measurements = {}
    outname = "miss_rates_memcachier.pdf"

    select_cols = ["S_Hyper(0e+00; 0.100)", "S_Hyper_Sz(0e+00; 0.100)", "GD_PQ_Sz", "LRU"]
    select_x = "appid"

    scale_by = "LRU"

    legend_kw = dict(LEGEND_ABOVE)
    legend_kw["ncol"] = 3

    class titler:
        def __init__(self):
            self.count = 0
        def __call__(self, name):
            self.count += 1
            return "%s" % self.count

    graph_relative_miss_rate(select_cols, None, scale_by, 
                             titler(), POLICY_NAMES, 
                             outname,
                             select_x = select_x,
                             legend_kw = legend_kw,
                             subgrp_order = None, 
                             fname = "memcachier_app_allocations_cat",
                             grp_order = select_cols[:3],
                             y_lim = (-0.6, .15),
                             rotation = 80,
                             no_val_labels = True,
                             fig_kw = {"figsize" : 
                                       (SINGLE_COL_WIDTH_INCHES, 
                                        0.45 * DEF_HEIGHT_INCHES)},)


def graph_all_barplots():
    graph_relative_miss_rate_synthetics()
    graph_relative_miss_rate_synthetics_70pp()
    graph_relative_miss_rate_ARC()
    graph_relative_miss_rate_memcachier()

def graph_all_mrcs():
    graph_mrc_lfu_lru_perf_1cZp()
    graph_mrc_hyper_vs_lfu_introducing()
    graph_windowed_strategies()
    graph_hyper_v_lru_sweep_skews()
    graph_throughput_time_loaded()
    graph_zipf_basic_tput()
    graph_tail_latency_time_loaded()
    graph_hyper_sampling_mrc_lighttail_1cZp()
    graph_class_mrc_hotclass()
    graph_hyper_sampling_mrc_lighttail_retain_1cZp()
    graph_hyper_expiration_mrc_1cUp_expireN("25")

#    graph_dynamic_pops_mrc()
#    graph_hyper_sampling_accuracy_mrc_1cZp()
#    graph_hyper_sampling_mrc_1cZp()
    graph_mrc_hyper_inits_1cZp()

def graph_mrc_hyper_vs_lfu_introducing():
    fname = "dynamic_popularities_mrcs"
    select_cols = ["PQ_Frequency", "S_Hyper(0e+00; 0.100)"]                   
    plot_data, _ =  load_input_file(WORKING_DIR + fname + ".csv",
                                    select_rows = (lambda r : r[0] == "ZipfFixedDriver.IntHigh.100" and r[-1] != "0.000000"),
                                    select_cols = select_cols)

    settings = dict(STD_MRC_SETTINGS_HALF)
    settings["yticks_num"] = 5
    settings["fig_kw"] = {"figsize" : (SINGLE_COL_WIDTH_INCHES, 0.4 * DEF_HEIGHT_INCHES)}

    graph_miss_rate_curves(plot_data, 
                           scale_by = 1,
                           y_lim = (1,8),
                           fname = OUT_DIR + "/hyper_vs_lfu_introducing.pdf", 
                           **settings)

def graph_windowed_strategies():
    fname = WORKING_DIR + "window_dynamic_pops_mrcs.csv"
    select_cols_hyper = [ "W(1e+04)DegF(0e+00; 0.100)", "S_Hyper(0e+00; 1.000)",] 
    select_cols_lfu = ["W(1e+04)LFU", "Sampling_Frequency"]
    select_cols_0 = ["GD_PQ"]
    pd_lru = load_input_file(fname, select_cols = select_cols_0)[0]
    pd_hyper = load_input_file(fname, select_cols = select_cols_hyper)[0]
    pd_lfu = load_input_file(fname, select_cols = select_cols_lfu)[0]
    
    pd_hyper[0].sort()
    pd_hyper[1].sort()
    pd_lfu[0].sort()
    pd_lfu[1].sort()
    
    line_lfu = [ (x1, scale_atom(y1, y2))   for (x1, y1), (x2, y2) in zip(pd_lfu[0], pd_lfu[1]) ]
    line_hyper = [ (x1, scale_atom(y1, y2))   for (x1, y1), (x2, y2) in zip(pd_hyper[0], pd_hyper[1]) ]
    
    

    settings = dict(STD_MRC_SETTINGS_HALF)
    settings["fig_kw"] = {"figsize" : (SINGLE_COL_WIDTH_INCHES,
                                       0.4 * DEF_HEIGHT_INCHES)}

    settings["scale_by"] = 1
    
    col_titles_0 = ["Hyperbolic", "+Windowing"]
    col_titles_1 = ["HC-wnd", "LFU-wnd"]

    settings["legend_kw"] = dict(LEGEND_ABOVE)
    
    graph_miss_rate_curves(pd_lfu, 
                           #col_titles = col_titles_0,
                           fname = OUT_DIR + "/dyn_promote_lfu_window.pdf", 
                           y_lim = (0.5, 1.5),
                           **settings)
    graph_miss_rate_curves(pd_hyper, 
                           #col_titles = col_titles_1,
                           y_lim = (0.5, 1.5),
                           fname = OUT_DIR + "/dyn_promote_hyper_window.pdf", 
                           **settings)

    del settings["scale_by"]
    
    graph_miss_rate_curves([line_hyper, line_lfu], 
                           col_titles = col_titles_1,
                           y_lim = (0.5, 1.5),
                           fname = OUT_DIR + "/dyn_promote_both_window.pdf", 
                           **settings)
    

def graph_dynamic_pops_mrc():
    fname = "dynamic_popularities_mrcs"
    select_cols = ["PQ_Frequency",  "GD_PQ", "S_Hyper(0e+00; 0.100)"]                   
    col_titles = [POLICY_NAMES[s] for s in select_cols]

    pd_0, _ =  load_input_file(WORKING_DIR + fname + ".csv",
                               select_rows = (lambda r : r[0] == "GD3.DynPromote.100"),
                               select_cols = select_cols)
    pd_1, _ =  load_input_file(WORKING_DIR + fname + ".csv",
                               select_rows = (lambda r : r[0] == "ZipfFixedDriver.IntHigh.100"),
                               select_cols = select_cols)
    pd_2, _ =  load_input_file(WORKING_DIR + fname + ".csv",
                               select_rows = (lambda r : r[0] == "ZipfFixedDriver.DynPromote.100"),
                               select_cols = select_cols)

    settings = dict(STD_MRC_SETTINGS_HALF)

    graph_miss_rate_curves(pd_0, 
                           fname = OUT_DIR + "/" + fname + "_Promote.pdf",
                           scale_by = 2,
                           **settings)
    settings["legend_kw"] = dict(LEGEND_BELOW)
    settings["legend_kw"]["ncol"] = 3

    graph_miss_rate_curves(pd_1, 
                           scale_by = 2,
                           fname = OUT_DIR + "/" + fname + "_Intro.pdf",
                           **settings)
    del settings["x_title"] 

    graph_miss_rate_curves(pd_2, 
                           scale_by = 2,
                           col_titles = col_titles[0:2],
                           fname = OUT_DIR + "/" + fname + "_Promote_1c.pdf",
                           **settings)


def graph_zipf_basic_tput():
    plot_data, _ = load_input_file(WORKING_DIR + "throughput-node.csv", 
                                   select_x = "missrate", select_cols = ["throughput"],
                                   col_fmt = {"throughput" : (lambda x : float(x) / float(10**3))},
                                   x_fmt = (lambda x : float(x)),)
    plot_kwargs = {"linestyle" : "solid", "marker" : "o"}
    graph_miss_rate_curves(plot_data, x_title = "Miss Rate", y_title = "Throughput \n (kreqs/s)", 
                           fig_kw = {"figsize" : (SINGLE_COL_DEF[0], 0.35 * DEF_HEIGHT_INCHES)},
                           y_lim = (10, 30), x_lim = (0, 1.0), plot_kwargs = plot_kwargs,
                           yticks_num = 8,
                           fname = OUT_DIR + "/throughput_node.pdf")

def graph_mrc_expiry_msn():
    plot_data, _ = load_input_file(WORKING_DIR + "expiry_msn_sample64.csv", 
                                   select_cols = ["S_Hyper(0e+00; 0.100)", "S_HyperExpiry(1.010)(0e+00; 0.100)"])
    graph_miss_rate_curves(plot_data)
                        
def load_csv(f):
    import csv
    with open(f) as fd:
        reader = csv.reader(fd, skipinitialspace = True)
        for line in reader:
            yield (float(line[0]), float(line[1]))

def graph_throughput_time_wiki():
    pd_classes = list(load_csv(WORKING_DIR + "wiki_partial_test_classes"))
    pd_nocosts = list(load_csv(WORKING_DIR + "wiki_partial_test_nocosts"))
    
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    plot_data = [pd_classes, pd_nocosts]
    for col_ix, line in enumerate(plot_data):
        line.sort()
        x, y = zip(*line)
            
        ax.plot(x,y, zorder = 3)

        ax.grid(b=True, which='major', zorder = 1)
        ax.grid(b=True, which='minor', zorder = 1)

    fname = OUT_DIR + "wiki_partial.pdf"
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def average_next(y):
    for ix, y_1 in enumerate(y[:-1]):
        if ix % 2 == 0:
            yield (y_1 + y[ix + 1])/2

def graph_throughput_time_loaded(dir = WORKING_DIR + "100k_120s_10k-scan_30k-cache"):
    pd_classes = list(load_csv(dir +"/perclass_tput_secs.csv"))
    pd_costs = list(load_csv(dir + "/peritem_tput_secs.csv"))
    
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    col_titles = [ "Per-Item" , "Per-Class" ]
    plot_data = [pd_costs, pd_classes]
    for col_ix, line in enumerate(plot_data):
        line.sort()
        x, y = zip(*line)
        
        x = [x_v for (ix, x_v) in enumerate(x) if ix % 2 == 1]
        x = [x_v - x[0] for x_v in x]
        y = (list(average_next(y)))
        y = [y_v / 1000.0 for y_v in y]
        assert len(x) == len(y)
        plot_data[col_ix] = zip(x, y)

        #ax.plot(x,y, zorder = 3)

        #from scipy.interpolate import spline
        #xnew = np.linspace(min(x),max(x),300)
        #power_smooth = spline(x,y,xnew)
        #ax.plot(xnew,power_smooth, label = names[col_ix])

        #ax.grid(b=True, which='major', zorder = 1)
        #ax.grid(b=True, which='minor', zorder = 1)

#    ax.legend()

    settings = {}
    settings["legend_kw"] = dict(LEGEND_BEST)
    settings["legend_kw"]["columnspacing"] = 1.2

    settings["x_lim"] = (0,360)
    settings["y_lim"] = (20, 24)
    settings["xticks"] = [60 * i for i in range(7)]
    settings["yticks_num"] = 5
    settings["plot_kwargs"] = STD_MRC_SETTINGS_HALF["plot_kwargs"]
    settings["fig_kw"] = dict(STD_MRC_SETTINGS_HALF["fig_kw"])

    settings["fig_kw"]["figsize"]  = (SINGLE_COL_WIDTH_INCHES, 
                                      0.35 * DEF_HEIGHT_INCHES)

    settings["x_title"] = "Time (s)"
    settings["y_title"] = "Throughput (kreq/s)"

    draw_line_at = [120, 20, 24, "red"]
    
    graph_miss_rate_curves(plot_data, 
                           draw_line_at = draw_line_at,
                           col_titles = col_titles,
                           fname = OUT_DIR + "/tput_loaded.pdf",
                           **settings)

def graph_tail_latency_time_loaded(dir = WORKING_DIR + "100k_120s_10k-scan_30k-cache"):
    pd_classes = list(load_csv(dir +"/perclass__reqs.sorted.tails95.csv"))
    pd_costs = list(load_csv(dir + "/peritem__reqs.sorted.tails95.csv"))
    
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    col_titles = [ "Items" , "Classes" ]
    plot_data = [ pd_costs, pd_classes ]
    for col_ix, line in enumerate(plot_data):
        line.sort()
        x, y = zip(*line)
        
        x = [x_v - x[0] for x_v in x]
        y = [y_v * 1000.0 for y_v in y]
        
        assert len(x) == len(y)
        
        plot_data[col_ix] = zip(x, y)

    settings = {}
#    settings["legend_kw"] = dict(LEGEND_BEST)
    settings["x_lim"] = (0,360)
    settings["y_lim"] = (14,24)
    settings["xticks"] = [60 * i for i in range(7)]
    settings["plot_kwargs"] = STD_MRC_SETTINGS_HALF["plot_kwargs"]
    settings["fig_kw"] = STD_MRC_SETTINGS_HALF["fig_kw"]
    settings["fig_kw"]["figsize"]  = (SINGLE_COL_WIDTH_INCHES, 
                                      0.35 * DEF_HEIGHT_INCHES)
    settings["x_title"] = "Time (s)"
    settings["y_title"] = "95th Percentile\nLatency (ms)"
    
    draw_line_at = [120, 14, 24, "red"]

    graph_miss_rate_curves(plot_data, 
#                           col_titles = col_titles,
                           draw_line_at = draw_line_at,
                           fname = OUT_DIR + "/latencies_loaded.pdf",
                           **settings)


def graph_mrc_expiry_msn_rm():
    plot_data, _ =  load_input_file(WORKING_DIR + "expiry_msn_realmin.csv", 
                                    select_cols = ["RM_Hyper(0e+00; 0.100)", "RM_HyperExpiry(1.010)(0e+00; 0.100)"])
    graph_miss_rate_curves(plot_data)

def graph_mrc_multiple_memcachier( largest_n = 20 ):
    from workloads.memcachier import Workload_lambdas_cat as workloads
    w = [x for x in workloads if (x.uniqs * x.max_item_size) > int(x.app_allocation)]
    w.sort(key= lambda x : (x.uniqs * x.max_item_size), reverse = True)
    w = w[:largest_n]
    for w_cur in w:
        print "graphing %s :: alloc / universe = %d M / %d M" % (w_cur.appid, int(w_cur.app_allocation) / 10**6, (w_cur.uniqs * w_cur.max_item_size) / 10**6)
        graph_mrc_memcachier( w_cur.trace_file_name, "cat" , draw_line_at = int(w_cur.app_allocation))
    

def graph_mrc_memcachier( trace_fname , trace = "cat", draw_line_at = False):

    appid = ((trace_fname.split("/")[-1])[4:]).split(".")[0]

    fname = "memcachier/%s/memcachier_%s.csv" % (trace, appid)
    plot_data, _ =  load_input_file(WORKING_DIR + fname, 
                                    select_cols = 
                                    ["S_Hyper(0e+00; 0.100)", "S_Hyper_Sz(0e+00; 0.100)", "GD_PQ_Sz", "LRU"])

    settings = dict(STD_MRC_SETTINGS)
    #    settings["x_logscale"] = False

    settings["fig_kw"]["figsize"] =  (4,2.8)
    settings["x_title"] = "cache size (bytes)"
#    settings["y_is_pdelta"] = False
    del settings["x_lim"]
    settings["legend_kw"] = dict(LEGEND_BELOW)
    settings["legend_kw"]["ncol"] = 3

    graph_miss_rate_curves(plot_data,
                           scale_by = 3,
                           col_titles = ["H", "HS", "GDS"],
                           fname = OUT_DIR + ("/memcachier/memcachier_%s_%s_scaled.png" % (appid, trace)),
                           draw_line_at = draw_line_at,
                           **settings)

    
    settings["y_is_pdelta"] = False
    settings["legend_kw"]["ncol"] = 4
    settings["y_title"] = "miss rate"
    graph_miss_rate_curves(plot_data,
                           col_titles = ["H", "HS", "GDS", "LRU"],
                           fname = OUT_DIR + ("/memcachier/memcachier_%s_%s_noscale.png" % (appid, trace)),
                           draw_line_at = draw_line_at,
                           **settings)

def graph_class_mrc_hotclass():
    plot_data, _ =  load_input_file(WORKING_DIR + "hyper_v_hyper_class_mrc_hotclass.csv", 
                                    select_cols = ["S_Hyper(0e+00; 0.100)", "S_Hyper_ClassTrack(0e+00; 0.100)"])
    settings = dict(STD_MRC_SETTINGS_HALF)

    graph_miss_rate_curves(plot_data,
                           y_lim = (.95, 1.4),
                           scale_by = 1,
                           fname = OUT_DIR + "/hyper_v_hyper_class_mrc_hotclass.pdf", 
                           **settings)

def graph_mrc_lfu_lru_perf_1cZp():
    plot_data, _ =  load_input_file(WORKING_DIR + "lfu_v_lru_mrc_1cZp.csv", 
                                    select_cols = [ "GD_PQ", "PQ_Frequency" ])
    perf_data, _ =  load_input_file(WORKING_DIR + "perf_v_lru_mrc_1cZp.csv", 
                                    select_cols = [ "PK_Freq"])

    plot_data = perf_data + plot_data 
    settings = dict(STD_MRC_SETTINGS_HALF)
    settings["fig_kw"] = {"figsize" : (SINGLE_COL_WIDTH_INCHES,
                                       0.4 * DEF_HEIGHT_INCHES)}

    graph_miss_rate_curves(plot_data, 
                           col_titles = ["LRU", "LFU"],
                           scale_by = 0,
#                           second_x = [1500, 7500, 40000],
                           fname = OUT_DIR + "/lfu_vs_lru_over_perf_1cZp.pdf",
                           legend_kw = LEGEND_ABOVE,
                           **settings)

def graph_hyper_expiration_mrc_1cUp_expireN(pp = "33"):
    fname = "hyper_expiration_mrc_1cUp_expire%spp" % pp
    plot_data, _ =  load_input_file(WORKING_DIR + fname + ".csv", 
                                    select_cols = [ "S_Hyper(0e+00; 0.100)", "S_HyperExpiry(1.010)(0e+00; 0.100)" ])

    settings = dict(STD_MRC_SETTINGS)
#    settings["x_logscale"] = False


    graph_miss_rate_curves(plot_data,
                           scale_by = 1,
                           fname = OUT_DIR + "/" + fname + ".pdf",
                           **settings)

def graph_hyper_sampling_accuracy_mrc_1cZp():
    fname = "hyper_sampling_measure_priority_mrc_1cZp"
    array = [2,5, 10, 25, 500]
    select_cols = ["S(%d)_Hyper(0e+00; 0.100)" % i for i in array ]
    col_titles = ["S = %d" % i for i in array ]
    pd_0, _ =  load_input_file(WORKING_DIR + fname + ".csv",
                               select_x = "k",
                               select_cols = select_cols)

    settings = dict(STD_MRC_SETTINGS_HALF)
    settings["legend_kw"] = dict(LEGEND_ABOVE)
    settings["legend_kw"]["ncol"] = 5
#    settings["legend_kw"]["borderaxespad"] = 1
    settings["legend_kw"]["handlelength"] = 0
    settings["legend_kw"]["handletextpad"] = 1
    settings["legend_kw"]["borderpad"] = .6
    settings["legend_kw"]["columnspacing"] = 1.2
    settings["legend_kw"]["bbox_to_anchor"] = (.5, 1.03)
    del settings["y_title"]
    settings["plot_kwargs"]["linestyle"] = "solid"

    settings["y_is_pdelta"] = False

    graph_miss_rate_curves(pd_0, 
                           col_titles = col_titles,
                           y_title = "Avg. Evicted Priority \n / Avg. Min. Priority", 
                           fname = OUT_DIR + "/" + fname + ".pdf",
                           **settings)


def graph_hyper_sampling_mrc_1cZp():
    fname = "hyper_sampling_mrc_1cZp"
    array = [2,5, 10, 25, 64]
    select_cols = ["RM_Hyper(0e+00; 0.100)"]
    select_cols += ["S(%d)_Hyper(0e+00; 0.100)" % i for i in array ]
    col_titles = ["S = %d" % i for i in array ]
    pd_0, _ =  load_input_file(WORKING_DIR + fname + ".csv",
                               select_rows = (lambda r : r[0] == "ZPop_UnitC"),
                               select_cols = select_cols)

    settings = dict(STD_MRC_SETTINGS_HALF)

    graph_miss_rate_curves(pd_0, 
                           scale_by = 0,
                           fname = OUT_DIR + "/" + fname + "_0.pdf",
                           **settings)


def graph_hyper_sampling_mrc_lighttail_1cZp():
    in_fname = "hyper_sampling_lighttail_mrc"
    out_fname = in_fname + "_zipf_%d"
    array = [5, 10, 64]
    select_cols = ["RM_Hyper(0e+00; 0.100)"]
    select_cols += ["S(%d)_Hyper(0e+00; 0.100)" % i for i in array ]
    col_titles = ["S = %d" % i for i in array ]

    zipf_a_params = [ 1.0001, 1.4 ]
    for ix, zipf_a in enumerate(zipf_a_params):
        driver_name = "Z(%f)P1C" % (zipf_a)

        pd_0, _ =  load_input_file(WORKING_DIR + in_fname + ".csv",
                                   select_rows = (lambda r : r[0] == driver_name),
                                   select_cols = select_cols)

        settings = dict(STD_MRC_SETTINGS_HALF)
        settings["x_title"] = unicode(settings["x_title"]) + unicode("\n alpha = %.2f" % zipf_a)
        settings["fig_kw"]["figsize"]  = (SINGLE_COL_WIDTH_INCHES, 
                                          0.35 * DEF_HEIGHT_INCHES)
        if ix == 0:
            settings["legend_kw"] = dict(LEGEND_ABOVE)
#            settings["legend_kw"]["bbox_to_anchor"] = (.5, 1.07)
            settings["legend_kw"]["ncol"] = 3
            settings["col_titles"] = col_titles
            settings["fig_kw"]["figsize"]  = (SINGLE_COL_WIDTH_INCHES, 
                                              0.30 * DEF_HEIGHT_INCHES)

        settings["y_lim"] = (.95, 1.2)
        graph_miss_rate_curves(pd_0, 
                               scale_by = 0,
                               fname = OUT_DIR + "/lighttail/" + (out_fname % int(zipf_a*100)) + ".pdf",
                               **settings)

def graph_hyper_sampling_mrc_lighttail_retain_1cZp():
    in_fname = "hyper_sampling_lighttail_retain_mrc"
    out_fname = in_fname + "_zipf_%d"
    array = [0, 19]
    select_cols = ["RM_Hyper(0e+00; 0.100)"]
    select_cols += ["S(64; %d)_Hyper(0e+00; 0.100)" % i for i in array ]
    col_titles = ["M = %d" % i for i in array ]

    zipf_a_params = [ 1.0001, 1.4 ]
    for zipf_a in zipf_a_params:
        driver_name = "Z(%f)P1C" % (zipf_a)

        pd_0, _ =  load_input_file(WORKING_DIR + in_fname + ".csv",
                                   select_rows = (lambda r : r[0] == driver_name),
                                   select_cols = select_cols)

        settings = dict(STD_MRC_SETTINGS_HALF)
        settings["x_title"] += ("\n alpha = %.2f" % zipf_a)
        settings["legend_kw"] = dict(LEGEND_ABOVE)
        settings["legend_kw"]["ncol"] = 3
        settings["fig_kw"]["figsize"]  = (SINGLE_COL_WIDTH_INCHES, 
                                          0.35 * DEF_HEIGHT_INCHES)

        graph_miss_rate_curves(pd_0, 
                               scale_by = 0,
                               fname = OUT_DIR + "/lighttail/" + (out_fname % int(zipf_a*100)) + ".pdf",
                               pdelta_str = pdelta_str_flt,
                               col_titles = col_titles,
                               **settings)

def graph_hyper_v_lru_sweep_skews():
    in_fname = "hyper_v_lru_sweep_skew"
    out_fname = in_fname
    skew_array = [0.7, 0.9, 1.0001, 1.1, 1.4, 1.8, 2.0]
    skew_array.reverse()
    skew_lines = []
    coltitles = ["$\\alpha$ = %.1f" % s for s in skew_array]
    settings = dict(STD_MRC_SETTINGS_HALF)

    settings["legend_kw"] = {"loc" : "center left", "ncol" : 1,
                             "bbox_to_anchor" : (1.001, 0.5) }

    for skew_cur in skew_array:
        plot_data, _ = load_input_file(
            WORKING_DIR + in_fname + ".csv",
            select_cols = [ "S_Hyper(0e+00; 0.100)", "LRU" ],
            select_rows = (lambda r : r[0].startswith("Z(%.5f" % skew_cur)))
        plot_data[0].sort()
        plot_data[1].sort()
        skew_lines.append(
            [ ( x_1, (float(lru_y) + .000000000001) / (.000000000001 + float(hyper_y)) ) for
              ((x_1, hyper_y) , (_, lru_y)) in zip(plot_data[0], plot_data[1]) ])

    graph_miss_rate_curves(skew_lines, 
                           fname = OUT_DIR + "/" + out_fname + ".pdf",
                           col_titles = coltitles, **settings)

def graph_mrc_hyper_inits_1cZp():
    cols = ["S_Hyper(0e+00; 1.000)", "S_Hyper(0e+00; 0.100)"]
    col_titles = [r'$\beta = 1$', r'$\beta = 0.1$']

    settings = dict(STD_MRC_SETTINGS_HALF)
    settings["fig_kw"]["figsize"]  = (SINGLE_COL_WIDTH_INCHES, 0.40 * DEF_HEIGHT_INCHES)
    plot_data, _ =  load_input_file(WORKING_DIR + "hyper_inits_mrc_1cZp.csv", select_cols = cols)
    graph_miss_rate_curves(plot_data, 
                           fname = OUT_DIR + "/hyper_inits_1cZp.pdf",
                           scale_by = 0,
                           **settings)


def graph_perf_moving_window():
#    plot_data_0, m, _ = load_input_file(WORKING_DIR + "hyper_v_perf_moving_window.csv", 
#                                     select_x = "t",select_cols = [ "hyper"])
#    length = len(plot_data_0[0])
    plot_data, _ = load_input_file(WORKING_DIR + "lfu_v_perf_moving_window.1.csv", 
                                   merge_by = "t", x_fmt = (lambda x: (float(x) / 10**6)), 
                                   select_x = "t", select_cols = [ "lfu", "perfect" ])
    plot_data.reverse()

    SETTINGS = dict(STD_MRC_SETTINGS)
    SETTINGS["x_lim"] = (0, 5)
    SETTINGS["plot_kwargs"] = {"linestyle" : "solid", "linewidth" : 2}
    SETTINGS["x_logscale"] = False

    SETTINGS["x_title"] = "Time (million requests)"
    

    graph_miss_rate_curves(plot_data, 
#                           subtractive = True,
                           scale_by = 0,
                           fname = OUT_DIR + "/perf_vs_lfu_windows.pdf",
                           **SETTINGS)
