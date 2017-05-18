"""hyper-store, 0.203475, 0.201781, pages-details-by-slug, views.decorators.cache.cache_page..GET.630d785810c92d779875401a3708919c.d41d8cd98f00b204e9800998ecf8427e.en.UTC"""

def load_file(f):
    for l in f:
        if not l.startswith("hyper-store"):
            continue
        data = l.split(",")
        key = data[4].strip()
        key_class = data[3].strip()
        rtime = float(data[1].strip())
        yield (key, key_class, rtime)

def summary(l):
    return sum(l)/len(l)

def main(fname = "django_dev_portal_timing"):
    with open(fname, 'r') as f:
        data = list(load_file(f))

    by_class = {}
    by_key = {}
    key_to_class = {}
    for key, key_class, rtime in data:
        if not key in by_key:
            by_key[key] = []
            key_to_class[key] = key_class
        if not key_class in  by_class:
            by_class[key_class] = []
        by_class[key_class].append(rtime)
        by_key[key].append(rtime)


    for key, rtimes in by_key.items():
        print "%s, %f, %f" % (key, summary(rtimes), 
                              summary(by_class[key_to_class[key]]))
    for clazz in by_class.keys():
        print "%s, %f" % (clazz, summary(by_class[clazz]))

if __name__ == "__main__":
    main()
