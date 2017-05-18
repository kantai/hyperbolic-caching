from cost_driver import ArbitraryDriver
import gzip
import bz2

ALL_READS = "ALL_READS"
RW_CHAR = "char"
RW_INT = "int"

SPCMap1 = {"labels":["ix_asu", "ix_off", "ix_siz", "ix_rw", "ix_ts"],
           "rw" : RW_CHAR,
           "page_sz" : 4096}
SPCMap2 = {"labels":["ix_ts", "ix_rw", "ix_asu", "ix_siz", "ix_off"],
           "rw" : RW_INT,
           "page_sz" : 1}
ARCMap = {"labels" : ["ix_off", "ix_siz", "ignore", "ix_ts"],
          "rw" : ALL_READS,
          "page_sz" : 1,
          "field_sep" : " ",
          "scan_increment" : 1}

arc_fnames = [ "ConCat","DS1","MergeP","MergeS","OLTP","P1","P2","P3","P4",
               "P5","P6","P7","P8","P9","P10","P11","P12","P13","P14","S1",
               "S2","S3","spc1likeread" ]

class SPCTrace_Generator(object):
    def __init__(self, trace_file = "/dev/null",
                 name = None, column_map = SPCMap1, **kwargs):
        self.trace_file = trace_file
        self.name = name
        self.column_map = column_map
        self.call_kwargs = kwargs

    def __call__(self, d_second = -1, **kwargs):
        self.call_kwargs.update(kwargs)
        return SPCTraceDriver(trace_file = self.trace_file,
                              column_map = self.column_map,
                              name = self.name,
                              **self.call_kwargs)
                              

ARC_WORKLOADS = [ SPCTrace_Generator( trace_file = 
                                      ("workloads/arc/%s.lis" % fname),
                                      name = ("Arc.%s" % fname),
                                      column_map = ARCMap)
                  for fname in arc_fnames ]

class SPCTraceDriver(object):
    def __init__(self, seed = 0, name = None, trace_file = "/dev/null",
                 column_map = SPCMap1, **kwargs):
        if trace_file.endswith("bz2"):
            self.f = bz2.BZ2File(trace_file, "r")
        elif trace_file.endswith("gz"):
            self.f = gzip.open(trace_file, "r")
        else:
            self.f = open(trace_file, "r")

        self.rw_field = column_map["rw"]

        if "field_sep" in column_map:
            self.field_sep = column_map["field_sep"]
        else:
            self.field_sep = ","

        self.ix_asu = False
        for ix, label in enumerate(column_map["labels"]):
            self.__dict__[label] = ix
        
        self.block_size = column_map["page_sz"]

        self.scan_increment = 1
        if "scan_increment" in column_map:
            self.scan_increment = column_map["scan_increment"]

        if "page_sz" in kwargs:
            self.block_size = column_map["page_sz"]

        if name == None:
            self.name = "SPC." + trace_file
        else:
            self.name = name

        self.in_linear_scan = (None, 0)

        self.seed = seed
        self.FirstRequest = True

    def get_next(self):
        if self.seed == 0:
            r_val = self.get_next_inner()                
            self.FirstRequest = False
            return r_val
        else:
            my_id, n_readers = self.seed
            if self.FirstRequest:
                # scan forward to my_id
                for i in range(my_id):
                    r_val = self.get_next_inner()
                r_val = self.get_next_inner()                
                self.FirstRequest = False
                return r_val
            else:
                # scan forward to the next occurrence for me
                for i in range(n_readers - 1):
                    r_val = self.get_next_inner()
                return self.get_next_inner()

    def get_next_inner(self):
        item, remaining = self.in_linear_scan
        if remaining >= 1:
            remaining -= 1
            self.in_linear_scan = (item + self.scan_increment, remaining)
            return item

        if self.f == None:
            return -1
        next_line = self.f.readline()
        if next_line == "":
            self.f.close()
            self.f = None
            return -1
        csv = next_line.split(self.field_sep)
        if self.rw_field != ALL_READS:
            rw = csv[self.ix_rw] 
            if self.rw_field == RW_INT and rw == "1":
                return self.get_next()
            elif self.rw_field == RW_CHAR and rw not in ("R", "r"):
                return self.get_next()

            
        off = int(csv[self.ix_off])
        siz = int(int(csv[self.ix_siz]) / self.block_size) + 1
        
        if self.ix_asu != False:
            asu = int(csv[self.ix_asu])
            item = (asu, off)
        else:
            item = off
        self.in_linear_scan = (item + self.scan_increment, siz - 1)

        return item

    def sample_item_w_cost(self):
        r_val = (self.get_next(), 1)
        if r_val[0] == -1:
            return -1
        return r_val
