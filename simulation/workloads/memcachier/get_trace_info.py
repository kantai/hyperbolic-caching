import csv, json
from os import walk as os_walk
import sys

def describe_file(f):
    csvreader = csv.reader(f)
    uniq_keys = set()
    max_size = 0
    line_count = 0

    mean_size_numer = 0
    mean_size_denom = 0
    uniq_key_size = 0

    for l in csvreader:
        line_count += 1
        key, size = l[0], int(l[1])
        uniq_keys.add(key)
        if uniq_key_size != len(uniq_keys):
            uniq_key_size = len(uniq_keys)
            mean_size_numer += size
            mean_size_denom += 1
        if size > max_size:
            max_size = size

    if mean_size_denom == 0:
        assert len(uniq_keys) == 0
        mean_size_denom = 1

    return (line_count, len(uniq_keys), float(mean_size_numer) / mean_size_denom)


def main(directory, out_file, secondary = False):
    (_, _, fnames) = os_walk(directory).next()

    descriptions = {}
    sys.stderr.write("\n")
            
    for ix, fname in enumerate(fnames):
        with open(directory + "/" + fname, 'rU') as f:
            descriptions[fname] = describe_file(f)
            sys.stderr.write("\rProcessed %d" % (ix + 1))
            sys.stderr.flush()

    sys.stderr.write("\n")
        
    MIN_SIZE = 0
    file_scores = [ f for f, (line_count, uniqs, size) 
                    in descriptions.items() if (uniqs * size) >= MIN_SIZE ]
#    file_scores.sort(reverse = True)
#    top_files = [ f for (_, f) in file_scores[:20] ]

    uniqs = dict( [ (f, uniqs ) for f, (_, uniqs, _) in descriptions.items() ] )
    max_sizes = dict( [ (f, max_size ) for f, (_, _, max_size) in descriptions.items() ] )

    trace_files_infos = {}
    trace_files_infos['fnames'] = file_scores
    trace_files_infos['uniqs'] = uniqs
    trace_files_infos['max_item_szs'] = max_sizes

    cache_sizes = {}

    if secondary:
        with open(secondary, 'r') as read_allotments:
            for line in read_allotments:
                if line == "":
                    continue
                line = line.split(' --> ')
                name = "app_%s.traces" % line[0].strip()
                allotted = line[1].strip()
                if name in file_scores:
                    cache_sizes[name] = allotted

    trace_files_infos['cache_sizes'] = cache_sizes

    with open(out_file, 'w') as out:
        json.dump(trace_files_infos, out)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], secondary = sys.argv[3])
    else:
        main(sys.argv[1], sys.argv[2])
