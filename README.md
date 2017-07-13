Hyperbolic Caching Simulations
==============================

Our modified version of Redis is available here:

[https://github.com/kantai/redis-hyperbolic-caching]


Setting up the Simulator
========================

I run the simulator in pypy and use NumPy for efficient memory and 
fast distribution sampling. Setting these up on Ubuntu LTS's can be
a real joy.

These are _roughly_ the commands I use to install these, setting up
a virtualenv to run the simulator in.

```
$ sudo apt-get install -y pypy pypy-dev python-virtualenv
$ virtualenv --python=pypy ve
$ ve/bin/pip install python-dateutil cffi llist sortedcontainers psycopg2cffi
$ git clone https://bitbucket.org/pypy/numpy.git
$ cd numpy/
$ git fetch && git checkout pypy-2.6.0
$ ../ve/bin/pypy setup.py install
```

And then clone into this repository
```
$ git clone git@bitbucket.org:kantai/cost-cache.git
```

Running the Simulator
=====================

The simulator's code is primarily accessed through
`simulation/main.py` The structure of this code is esoteric at best,
but no intentional obfuscation occurred. 

To facilitate easy testing and changing of parameters, I eschewed the
usage of a traditional CLI, opting instead to call functions directly
as in:

`$ python -c 'import main as m; m.hyper_v_hyper_class_mrc_hotclass()'`

The data-files that ultimately became figures and results in the paper
all have functions that will run their experiments directly.

If you'd like to 'peel back the covers' on these functions, what they
do is set up a `Simulation` object which interacts with a _workload
driver_ (usually referred to by `d` in the code) and a _caching
policy_ (useually referred to by `p` in the code).

The function `run_product` will perform some argument broadcasting to
construct a driver (passing it some parameters) and a policy (passing
it other parameters). The policies and drivers themselves are supplied
using names registered in `main_globals.py`

## Off-label Uses of the Simulator

At various points in time, I thought it was a good idea to use the
simulator to drive experiments on a real cache with a real database
backend. Towards that end, I implemented subclasses of the Simulation
class (implemented in `simulate.py`)-- `RedisSim` in
`instrument_redis.py` and `BackedRedisSim` in the conspicuously named
`expirement_db_redis.py`. When last I looked, these files worked, but
I didn't use them for any of the data collection in the paper, nor
have I used them particularly recently. However, I do use
`setup_pgsql_sim` in `main.py` to initialize the database that I used
in the Node simulations.

If you're interested in this, the function
`run_redis_backed_spc_sim()` is going to be your friend.

Node Simulation
===============

Setting up node:

```
cd node-sim
curl -sL https://deb.nodesource.com/setup_4.x | sudo -E bash -
sudo apt-get install -y nodejs libpq-dev
npm i
```

I used IPTables because I hate running test environments on open ports on the
SNS machines.

```
# IPTABLES configuration:
sudo iptables -I INPUT -p TCP --dport 3590 -j DROP
sudo iptables -I INPUT -p TCP --dport 3590 -s $REQUEST_MACHINE -j ACCEPT
sudo iptables -I INPUT -p TCP --dport 3590 -s localhost -j ACCEPT
```

You'll need to setup a postgres machine:

```
# IPTABLES configuration:
sudo iptables -I INPUT -p TCP --dport 5435 -j DROP
sudo iptables -I INPUT -p TCP --dport 5435 -s $NODE_MACHINE -j ACCEPT
sudo iptables -I INPUT -p TCP --dport 5435 -s localhost -j ACCEPT

sudo pg_createcluster -u $USER -p 5435 -d /disk/local/WHEREVER/ 9.3 $USER
pg_ctlcluster 9.3 $USER start
psql -p 5435 -h/tmp postgres -c "CREATE DATABASE $USER;"
echo "listen_addresses = '128.112.7.135'" >> /etc/postgresql/9.3/$USER/postgresql.conf
echo "host $USER $USER NODE_MACHINE_IP_ADDRESS/32 trust" >> /etc/postgresql/9.3/blanks/pg_hba.conf
pg_ctlcluster 9.3 $USER restart
```

And now this is the step where we fill the database using `main.py` -- you'll need to have
installed NumPY in this virtualenv, per the instructions above.
```
cd cost-cache/simulation/
../../ve/bin/python -c "from main import setup_pgsql_sim; from main_globals import ZP1C_DRIVER; setup_pgsql_sim(ZP1C_DRIVER, 10**6)"
```


Now let's set up the request machine to run the 
websim -- it needs numpy and pycurl -- but can run on
CPython, so just use Ubuntu's package manager:

`sudo apt-get install pycurl numpy`

And fire away from the request machine!

`./run_multiple 0 zipfian_1_100k.csv 39000 3000`

This script will attempt to spawn a node server, start redis, 
and then begin executing requests. The arguments specify the 
workload driver (numbered in websim.py), the output csv file,
the number of requests per client, and then the number of 
simultaneous clients.

Running Django Apps
===================

Getting the Django apps up and running is several degrees of _tricky_.
In particular, filling the wikipedia database was unpleasant to say the
least.

# Wikipedia Test

## Setting up the app and the test database

Let's start by setting up the app:

```
$ cd apps/django-wiki
$ tar -xzf django-wiki.tar.gz
$ virtualenv env && source env/bin/activate
$ cd wiki_project
$ pip install -r requirements.txt
$ pip install gunicorn
```

Before creating the app's tables and database, you should
edit `wiki_project/settings.py`, then you can sync the db.

```
$ python manage.py syncdb
```

## Loading Wikipedia Dump

In `simulation/workloads/wikipedia/`, there are a handful of scripts
used to load articles from a wikipedia XML dump.

The page dump we used was [enwiki-20080103-pages-articles.xml.bz2](https://dumps.wikimedia.org/archive/enwiki/20080103/)

Download that dump and link it into the directory of `extractor.py`
which will use `psycopg2` to fill your database. This script has a
bunch of hard-coded parameters for connecting to your database and
picking which table to use to fill the data -- you should change those
(hint, the tables get named when you run `python manage.py syncdb` to
install the wiki project.) You should be editing the connection
parameters and the SQL insert string.

```
$ python extractor.py
```

## Running the App

Once you've done the above, getting the app running is pretty
easy. Just call `gunicorn` with the app's WSGI module.

```
$ ../env/bin/gunicorn wiki_project.wsgi -b 0.0.0.0:8000 --workers 32
```

# Ubuntu Developer Portal

## Setting up the App and Database

Follow the instructions in the app dir.
(./django-middleware/apps/developer-ubuntu-com/README.md)

## Running the App

```
../env/bin/gunicorn developer_portal.wsgi -b 0.0.0.0:8213 --workers 24
```
