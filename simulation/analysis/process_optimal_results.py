import sys

def main(in_f):
    reader = open(in_f)
    ca_d = dict()
    co_d = dict()
    for l in reader:
        line = l.split(",")
        line = [li.strip() for li in line]
        key = "%s_%s" % (line[0], line[2])
        if line[1] == "cost-aware":
            d = ca_d
        else:
            d = co_d
        if key in d:
            d[key].append(int(line[4]))
        else:
            d[key] = [int(line[4])]
    for key in ca_d:
        ca_val = sum(ca_d[key])
        co_val = sum(co_d[key])
        out = float(co_val - ca_val) / float(co_val)
        print "%s : %s" % (key.ljust(25), out)

    reader.close()

def simulation_main(in_f):
    reader = open(in_f)
    out_d = {}
    for l in reader:
        line = l.split(",")
        line = [li.strip() for li in line]
        line_head = line[0].split("_")
        key = "%s_%s_%s" % (line_head[0], line_head[1], line_head[2])
        strategy = line_head[3]
        if strategy not in out_d:
            out_d[strategy] = {}
        d = out_d[strategy]

        entry = (float(line[2]) , float(line[3]))
        if key in d:
            d[key].append(entry)
        else:
            d[key] = [entry]

    for strategy, s_dict in out_d.items():
        for key, val in s_dict.items():
            lru_val = out_d["LRU"][key]

            lru_cost = sum(z[1] for z in lru_val) 

            cost = sum(z[1] for z in val)
            miss_rate = sum(z[0] for z in val) / float(len(val))
            cost_ratio = float(lru_cost - cost) / float(lru_cost)
            
            print "%s %s % .6f, % .6f" % ( ("%s, " % key).ljust(29) , 
                                     ("%s, " % strategy).ljust(7),
                                     miss_rate, cost_ratio)

    reader.close()


if __name__ == "__main__":
    simulation_main(sys.argv[1])
