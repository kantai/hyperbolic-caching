"""
This script will split a trace processed by process.py into different 
application traces. Also, when get_sizes = True, it will attempt to match GET 
requests to their responses (or subsequent SET reqs) to determine the object
size in bytes.
"""

import sys
import csv

def get_ignored(clients_data):
    return sum( [ d.ignored for d, _ in clients_data.values() if d ] )
def get_dropped(clients_data):
    return sum( [ d.dropped for _,d in clients_data.values() if d ] )
def get_written(clients_data):
    return sum( [ d.written for d, _ in clients_data.values() if d ] )

def process_file(f, outdir, get_sizes = False):
    clients_data = {}
    opaque_to_client_name = {}
    null_byte_generator = (line for line in f if '\0' not in line)
    csvreader = csv.reader(null_byte_generator, delimiter = '\t', quotechar = "'")

    error_lines = []

    for ix,line_cols in enumerate(csvreader):
        try:
            datetime, msg_type, command, key, val_size, \
                val, client_name, opaque, port = [s.strip() for s in line_cols]
            try:
                val_size = int(val_size)
            except:
                val_size = 0
        except:
            error_lines += (ix, line_cols)
            continue
        
        if ix % 10000 == 0:
            sys.stderr.write("\rprocessed %iK , errors: %i, no-size: %iK / %iK, written: %iK" % 
                             (ix / 1000, len(error_lines), 
                              get_ignored(clients_data) / 1000,
                              get_dropped(clients_data) / 1000,
                              get_written(clients_data) / 1000))
            sys.stderr.flush()

        if msg_type == 'Req' and command == 'Get':
            if client_name in clients_data:
                out, fill = clients_data[client_name]
            else:
                if not all(ord(c) < 128 for c in client_name):
                    out = None
                    fill = None
                else:
                    out = DatapointBuffer(outdir + ('/app_%s.traces' % client_name))
                    fill = DatapointFillQueue()
                clients_data[client_name] = out, fill
            if out == None:
                continue

            if get_sizes:
                # wait for response -- or a set
                fill.enqueue(opaque, key)
                opaque_to_client_name[(opaque)] = client_name
            else:
                size = 1
                out.write(key, size, datetime)

        elif get_sizes and msg_type == 'Req' and command == 'Set':
            if client_name in clients_data:
                out, fill = clients_data[client_name]
                if out == None:
                    continue
                _, buffer_place = fill.pop_by_key(key)
                if buffer_place != -1:
                    key_size = len(key) # one byte per char in redis?
                    # NOTE: there's a better way to find the size --
                    #  the actual command gives the key size!
                    size = val_size + key_size
                    out.place_in_buffer(buffer_place, key, size, datetime)
        elif get_sizes and msg_type == 'Res' and command == 'Get':
            if (opaque) in opaque_to_client_name:
                client_name = opaque_to_client_name[(opaque)]
            else:
                continue

            if client_name in clients_data:
                out, fill = clients_data[client_name]
                if out == None:
                    continue
                if val_size == 0:
                    continue

                key, buffer_place = fill.pop_by_opaque(opaque)
                del opaque_to_client_name[(opaque)]
                if buffer_place != -1:
                    key_size = len(key) # one byte per char in redis?
                    # NOTE: there's a better way to find the size --
                    #  the actual command gives the key size!
                    size = val_size + key_size
                    out.place_in_buffer(buffer_place, key, size, datetime)
                
            
            
    for out,_ in clients_data.values():
        if out is not None:
            out.close()

class DatapointBuffer:
    def __init__(self, fname):
        self.outfd = open(fname, 'w')
        self.writer = csv.writer(self.outfd)
        self.buffer = []
        self.cur_min = 0
        self.ignored = 0
        self.written = 0
        self.OLDNESS_MAX = 50

    def write(self, key, size, dt):
        self.written += 1
        self.writer.writerow((dt, key, size))

    def place_in_buffer(self, buffer_place, key, size, dt):
        translated_place = buffer_place - self.cur_min
            
        if translated_place < 0:
            print "Skipped by oldness of %d" % translated_place
            return

        if translated_place == 0:
            self.write(key, size, dt)
            self.cur_min += 1
            if len(self.buffer) > 0:
                self.buffer.pop()
            self.drain_check()
        else:
            if len(self.buffer) <= translated_place:
                # buffer cannot already hold this item, so need extend
                self.buffer = ([ None for i in 
                                 range(translated_place + 1 - len(self.buffer))] +
                               self.buffer)
            # place the item in the buffer
            try:
                self.buffer[-1 * (translated_place + 1)] = (key, size, dt)
            except IndexError:
                raise IndexError( "Buffer Length = %d ; Extend Len = %d, Index = %d" 
                                  % (len(self.buffer),
                                     translated_place + 1 - len(self.buffer),
                                     -1 * (translated_place + 1)) )

        if len(self.buffer) > self.OLDNESS_MAX:
            self.cur_min += 1
            self.buffer.pop()
            self.ignored += 1
            self.drain_check()

    def drain_check(self):
        while len(self.buffer) > 0 and self.buffer[-1] != None:
            key, size, dt = self.buffer.pop()
            self.write(key, size, dt)
            self.cur_min += 1

    def drain_buffer(self):
        while len(self.buffer) > 0:
            popped = self.buffer.pop()
            self.cur_min += 1
            if popped == None:
                self.ignored += 1
                continue
            self.write(*popped)
    
    def close(self):
        self.drain_buffer()
        self.outfd.close()

class DatapointFillQueue:
    def __init__(self):
        self.q_by_opaque = []
        self.q_by_key = []
        self.q_by_n = []
        self.n = 0
        self.dropped = 0
    def enqueue(self, opaque, key):
        self.q_by_opaque.append(opaque)
        self.q_by_key.append(key)
        self.q_by_n.append(self.n)
        self.n += 1
        while len(self.q_by_n) > 0 and (self.n - self.q_by_n[0]) > 50:
            self.q_by_n.pop()
            self.q_by_key.pop()
            self.q_by_opaque.pop()
            self.dropped += 1

    def pop_by_key(self, key):
        try:
            queue_index = self.q_by_key.index(key)
            return self.pop_queue_index(queue_index)
        except ValueError:
            return -1, -1
    def pop_by_opaque(self, opaque):
        try:
            queue_index = self.q_by_opaque.index(opaque)
            return self.pop_queue_index(queue_index)
        except ValueError:
            return -1, -1
    def pop_queue_index(self, queue_index):
        key = self.q_by_key.pop(queue_index)
        del self.q_by_opaque[queue_index]
        buffer_place = self.q_by_n.pop(queue_index)
        return key, buffer_place

if __name__ == "__main__":
    get_szs = False
    if len(sys.argv) > 2:
        in_f = open(sys.argv[1], 'rU')
        out = sys.argv[2]
        if len(sys.argv) > 3:
            get_szs = (sys.argv[3] == '--size')
    else:
        in_f = sys.stdin
        out = sys.argv[1]
    process_file(in_f, out, get_szs)

