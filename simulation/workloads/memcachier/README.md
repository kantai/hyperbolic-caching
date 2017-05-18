Memcachier Traces
=================

I do not own or have the rights to these traces. Of course, if you're
really interested in obtaining them, e-mail Aaron, and he will tell you
who to e-mail. Depending on how far into the future from June 2017 you 
ask, Aaron may even be able to provide the dump (once you've obtained 
permission from the owner).

I *am* however able to provide the tools that I used to process the
data into a format usable by my simulator. The rest of this doc just
gives you the bash commands that I used.

### Translating gzipped traces to a fixed TCPDump

Straightforward removal of a header, then you need to use
[pcapfix](https://github.com/Rup0rt/pcapfix) to p

```
$ time gzip -d -c memcachier_traces/traces2.tar.gz | tail -c +513 > traces2.extracted
$ pcapfix-1.1.0/pcapfix traces2.extracted 
```

### TCPDump to the TCP Data

Get tshark and use it

```
$ tshark -r fixed_traces1.extracted -T fields -e data.len -e data -e tcp.port | head -c 200G > traces1.processed
```

### TCP Data to Readable TSVs

Now we need to use my scripts.

```
$ cat traces1.processed | python process.py > traces1.py.processed
$ cat traces2.processed | python process.py > traces2.py.processed
```

### Selecting only GETs

```
$ grep 'Get' traces1.py.processed | grep -v 'GetK' > traces1.py.processed.gets
```

### Splitting Traces

Split the traces up by app identifier.

```
python split_traces.py $MS2/traces2.py.processed split-sz --size
python split_traces.py $MS1/traces1.py.processed split-sz --size
```

### Outputting Trace Info JSONs (Used by workload execution)

```
python get_trace_info.py split-sz trace1_info.json
python get_trace_info.py split-sz trace2_info.json
```

### Concatenating traces

```
rm $MS1/cat/*
cd $MS1/split-sz
ls app_*.traces > /tmp/trace_apps
cd $MS2/split-sz
ls app_*.traces >> /tmp/trace_apps

for i in $(sort /tmp/trace_apps | uniq); do
    touch -a $MS1/split-sz/$i
    touch -a $MS2/split-sz/$i
    cat $MS1/split-sz/$i $MS2/split-sz/$i > $MS1/cat/$i
done
```