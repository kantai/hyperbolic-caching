var http = require('http');
var url = require('url');

var redis = require('redis');
var pgsql = require('pg');

pgsql.defaults.poolSize = 256;

var options = require('node-options');
var cluster = require('cluster');

var tables = ["zpop_unitc", "spcwebsearch", "arc_s1"];

var psqlConf = {
    user: 'blanks', 
    host: 'sns51', 
    port: '5435', 
    database: 'blanks'
}

var redisConf = {
    host: 'sns51',
    port: 63790,
    max_clients: 30 
}

var sleep = require('sleep');
var sleep_scalar = 1000*10;

var SELECT_Q = 'SELECT value FROM ' + tables[0] + ' WHERE key = $1;'

function reqHandler(req, resp){
    var pathname = url.parse(req.url).pathname.substr(1);
    var pathname = pathname.split("/");
    var objid = pathname[0];
    var cost = "";
    var objclass = "";
    if(pathname.length > 1)
	cost = pathname[1];
    if(pathname.length > 2)
	objclass = pathname[2];


    function writeResponse( reply, cache_hit, incurred_cost ){
	if (cache_hit){
	    var headers = {'X-CACHED-MIDDLEWARE' : 'Hit'}
	}else{
	    var headers = {'X-CACHED-MIDDLEWARE' : 'Miss',
			   "X-CACHED-INCURRED-COST" : incurred_cost}
	}
	resp.writeHead(200, headers);
	resp.write(reply + "\n");
	resp.end();
    }

    function fetchSQL( ){
	var queryConfig = {
	    text: SELECT_Q,
	    values: [objid]
	};

	pgsql.connect(psqlConf, function(err, p_client, done) {
	    if(err) {
		console.error('error fetching client from pool...');
		return fetchSQL();
	    }
	    p_client.query(queryConfig, function(err, result) {
		done();
		if(err) {
		    console.error('error running query', err);
		    writeResponse("e!!!", false, -1);
		}else{
		    var data = result.rows[0].value;
		    //console.log(data);
		    var setargs = [objid, data,];// "CX", parseInt(cost),]; //"CC", parseInt(objclass)];
		    r_pool.send_command('SET', setargs, function(err, reply){
			if (err){
			    console.log(err);
			}
			if (cost != ""){
			    var sleep_for = parseInt(parseInt(cost) * sleep_scalar);

			    sleep.usleep(sleep_for);

			    writeResponse("m:" + data, false, cost);
			}else{
			    writeResponse("m:" + data, false, 1);
			}
		    });
		}
	    });
	});

    }

    function afterRedis(err, reply){
	if(reply == null){
	    // miss!
	    //console.log("miss");
	    fetchSQL();
	}else{
	    // hit!
	    //console.log("hit");
	    writeResponse("h:" + reply, true, 0);
	}
    }

    r_pool.get(objid, afterRedis);
}


var opts = {
    "k" : -1
};

var result = options.parse(process.argv.slice(2), opts);

var r_client = redis.createClient(redisConf.port, redisConf.host);

if (opts.k != -1){
    r_client.config("SET", "maxobjects", parseInt(opts.k));
}

r_client.quit();

var r_pool = require('redis-connection-pool')('myRedisPool', redisConf);

//http.createServer(reqHandler).listen(3590);
var numCPUs = 16;

if (cluster.isMaster) {
  // Fork workers.
    for (var i = 0; i < numCPUs; i++) {
        cluster.fork(); 
    }
}else{
    http.createServer(reqHandler).listen(3590);
}
