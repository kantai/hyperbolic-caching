import numpy as np
import gzip, sys, csv

## SORTING gzipped file:
##  zcat ./perclass__reqs.csv.gz | sort -n | gzip > ./perclass__reqs.sorted.csv.gz

def write_windowed_tail_latencies(fd_in, fd_out, window_sz, percentile):
    reader = csv.reader(fd_in, skipinitialspace = True)
    writer = csv.writer(fd_out)
    
    window_start = -1
    window_latencies = []

    ts_now = 0

    for line in reader:
        time, latency = float(line[0]), float(line[1])
        if window_start == -1:
            window_start = time
        while time > (window_sz + window_start):
            window_start += window_sz
            ts_now += window_sz
            writer.writerow( (ts_now, np.percentile(window_latencies, percentile)) )
            window_latencies = []

        window_latencies.append(latency)
    
    if len(window_latencies) > 0:
        ts_now += window_sz
        writer.writerow( (ts_now, np.percentile(window_latencies, percentile)) )
        window_latencies = []

def main(fname_in, window_sz, percentile):
    with gzip.open(fname_in, "rb") as fd_in:
        file_ext = ".tails%d.csv" % percentile
        with open(fname_in[:-7] + file_ext, "w") as fd_out:
            write_windowed_tail_latencies(fd_in, fd_out, window_sz, percentile)

if __name__ == "__main__":
    window_sz = int(sys.argv[1])
    percentile = float(sys.argv[2])
    fname_in = sys.argv[3]
    
    main(fname_in, window_sz, percentile)
