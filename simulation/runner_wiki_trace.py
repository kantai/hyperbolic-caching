import pycurl, sys
from StringIO import StringIO
import multiprocessing as mp
import time, requests

from websim import handle_response

FILE_URL = "localhost:8000"
t_start = None
    
def wikipedia_urls(wiki_file = "workloads/wikipedia/wiki.1190153705.processed.articleids"):
    url_root = "http://%s/" % FILE_URL
    url_article = url_root + "%s/"
    with open(wiki_file, 'r') as in_file:
        for l in in_file:
            l = l.strip()
            if l == "-1":
                yield url_root, url_root
            else:
                yield url_article % l, l

def measure_object_sizes():
    uniqs = set(wikipedia_urls())
    total = len(uniqs)
    for ix, i in enumerate(uniqs):
        try:
            resp = requests.get(i[0])
            if resp.status_code == 200:
                print "%s, %d" % ( i[1], len(resp.text))
            update_progress_bar(ix, total)
        except:
            pass

def curl_read(url):
    try:
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        resp = StringIO()
        headers = StringIO()
        c.setopt(c.WRITEFUNCTION, resp.write)
        c.setopt(c.HEADERFUNCTION, headers.write)
        c.setopt(pycurl.CONNECTTIMEOUT, 20)
        c.setopt(pycurl.TIMEOUT, 20)
        c.perform()
        if c.getinfo(c.RESPONSE_CODE) == 200:
            c.close()
            is_hit = handle_response(resp, headers)
            size = len(resp)
            return True, is_hit, size
        
        return False, False, 0
    except:
        return False, False, 0

def update_progress_bar(progress, total):
    col_size = 65
    
    count = (col_size * progress) / total

    sys.stderr.write("\r")
    sys.stderr.write("[" + "".join(["=" for i in range(count)]))
    sys.stderr.write(">")
    sys.stderr.write("".join([" " for i in range(col_size - count - 1)]))
    sys.stderr.write("] %.1f%%" % (float(100 * progress) / total))
    sys.stderr.flush()


class CurlWorker(object):
    def __init__(self, q, pid = 0, partials = None, 
                 progress = False, results = None):
        self.q = q
        self.progress = progress
        self.out = results
        self.partials = partials
        self.pid = pid
    def __call__(self):
        ix = 0
        successes = 0
        misses = 1
        for item in iter( self.q.get, None ):
            url = item
            success, is_hit = curl_read(url)
            if success:
                successes += 1
            if not is_hit:
                misses += 1
            self.q.task_done()
            ix += 1
            if self.partials:
                self.partials[self.pid] = successes
                if self.progress and ix % 10 == 0:
                    t_status = time.time() - t_start
                    print "%f, %d" % (t_status, sum(self.partials))

            if self.progress and ix % 2 == 0:
                update_progress_bar(self.progress - self.q.qsize(), self.progress)
                
        self.out.put( (successes, misses) )
        self.q.task_done()

def multiprocess(X = -1, num_procs = 32, print_partials = False):
    global t_start
    q = mp.JoinableQueue()
    results = mp.Queue()
    procs = []
    if print_partials:
        partials = mp.Array('i', num_procs)
    else:
        partials = False
    t_start = time.time()
    load_urls(cb = lambda url : q.put(url), X = X)
    t_end = time.time()
    sys.stderr.write("finished building request arrays in %f s" % ( t_end - t_start ))
    sys.stderr.write("\n")
    sys.stderr.flush()

    total_reqs = q.qsize()
    t_start = time.time()
    for i in range(num_procs):
        if i == 0:
            progress = total_reqs
        else:
            progress = False
        worker = CurlWorker(q, pid = i, 
                            progress = progress, results = results,
                            partials = partials)
        procs.append( mp.Process(target=worker) )
        procs[-1].daemon = True
        procs[-1].start()
    q.join()
    for p in procs:
        q.put( None )
    q.join()
    t_end = time.time()
    sys.stderr.write("\nfinished requests in %f s" % ( t_end - t_start ))
    sys.stderr.write("throughput (req/s) = %f" % (float(total_reqs) / (t_end - t_start)))
    sys.stderr.write("\n")
    sys.stderr.flush()

    results_out = []
    for i in range(num_procs):
        results_out.append(results.get())
    misses = sum([misses for _, misses,_ in results_out])
    requests = sum([reqs for reqs, _,_ in results_out])
    time_tot = (t_end - t_start)
    tput = float(requests) / time_tot
    miss_rate = float(misses) / requests
    
    return (tput, miss_rate, requests, time_tot)
    

def load_urls(X = -1, cb = curl_read):
    for ix,url in enumerate(wikipedia_urls()):
        if X >= 0 and ix > X:
            break
        cb(url)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        FILE_URL = sys.argv[1].strip()
    if len(sys.argv) > 2:
        X = int(sys.argv[2])
    else:
        X = -1
    throughput, missrate, reqs, time = multiprocess(X = X)
    if len(sys.argv) > 3: ## for ''tagging'' the results
        print "%s, %s, " % (sys.argv[3], sys.argv[4])
    print "%f, %f, %d, %s s" % (throughput, missrate, reqs, time)
