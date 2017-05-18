import sys
import numpy
import csv

def describe(a):
    mean = numpy.mean(a)
    median = numpy.median(a)
    std = numpy.std(a) / numpy.sqrt(len(a))
    return (mean, median, std, len(a))
    print "Mean: %f; Median: %f; STDErr: %f; N: %d" % (mean, median, std, len(a))

def load_node_file(fname, header_row = True):
    data_d = {}
    with open(fname) as fd:
        f = csv.reader(fd)
        for ix, row_data in enumerate(f):
            if header_row and ix == 0:
                assert row_data[0] == 'variant'
                continue
            # heading = (variant, size)
            heading = (row_data[0].strip(), row_data[1].strip())
            if heading not in data_d:
                data_d[heading] = ([] , [])
            tputs, missrates = data_d[heading]
            tputs.append(float(row_data[2]))
            missrates.append(float(row_data[3]))            
    return data_d

if __name__ == "__main__":
    fname = sys.argv[1]
    data_d = load_node_file(fname)
    print "workload: %s (note throughputs are kreqs/s)" % fname

    rows = {}

    for (_, size) in data_d.keys():
        if size not in rows:
            rows[size] = []

    for size, row in rows.items():
        row.append( "%dk" % (int(size) / 10**3) )

        tput_def = None
        tput_hyp = None

        for variant in ["default", "hyper"]:
            tputs, missrates = data_d[(variant, size)]

            tput_mean, _, tput_std, _ = describe([i / 10**3 for i in tputs])
            missrate_mean = describe(missrates)[0]
            
            if variant == "default":
                tput_def = tput_mean
            if variant == "hyper":
                tput_hyp = tput_mean

            row.append( "$%.1f \\pm %.2f$" % (tput_mean, tput_std) )
            row.append( "%.2f" % (missrate_mean) )
        
        tput_delta = (tput_hyp - tput_def) / tput_hyp
        row.append("%.1f\\%%" % (100*tput_delta))
    
    for size, row in rows.items():
        print " & ".join(row),
        print " \\\\"

