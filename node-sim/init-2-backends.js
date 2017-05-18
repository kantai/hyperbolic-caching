var http = require('http');
var url = require('url');

var redis = require('redis');
var pg = require('pg');
var PgPool = require('pg-pool');

pg.defaults.poolSize = 256;

var options = require('node-options');
var cluster = require('cluster');

var tables = ["zpop_unitc", "spcwebsearch", "arc_s1"];

var psqlConf1 = {
    user: 'blanks', 
    host: 'sns45', 
    port: '5435', 
    database: 'blanks',
    max: 32, // max number of clients in the pool
    idleTimeoutMillis: 30000,
}

var psqlConf2 = {
    user: 'blanks', 
    host: 'sns47', 
    port: '5435', 
    database: 'blanks',
    max: 32, // max number of clients in the pool
    idleTimeoutMillis: 30000,
}

var redisConf = {
    host: 'sns51',
    port: 63790,
    max_clients: 30 
}

var sleep = require('sleep');
var sleep_scalar = 1000*10;

var SELECT_Q = 'select zp2.value from zpop_unitc as zp1 inner join zpop_unitc as zp2 on zp1.secondary = zp2.secondary WHERE zp1.key = $1::varchar';

function reqHandler(req, resp){
    var pathname = url.parse(req.url).pathname.substr(1);
    var pathname = pathname.split("/");
    var objid = pathname[0];
    var cost = "";
    var objclass = "";
    if(pathname.length > 2)
	objclass = pathname[2];
    else
	console.error('ruh oh guys.');


    function writeResponse( reply, cache_hit, incurred_cost ){
	if (cache_hit){
	    var headers = {'X-CACHED-MIDDLEWARE' : 'Hit'}
	}else{
	    var headers = {'X-CACHED-MIDDLEWARE' : 'Miss',
			   "X-CACHED-INCURRED-COST" : incurred_cost}
	}
	if (reply == "e!!!"){
	    resp.writeHead(400, headers);
	}else{
	    resp.writeHead(200, headers);
	}
	resp.write(reply + "\n");
	resp.end();
    }

    function fetchSQL( ){
	var queryConfig = {
	    text: SELECT_Q,
	    values: [parseInt(objid)]
	};

	if (objclass == "1"){
	    var pool = pg_pool1;
	}else if (objclass == "2"){
	    var pool = pg_pool2;
	}else{
	    console.error('unrecognized class: ' + objclass);
	    writeResponse("e!!!", false, -1);
	    return;
	}

	pool.connect(function(err, p_client, done) {
	    if(err) {
		console.error('error fetching client from pool' + objclass +  '...');
		done();
		writeResponse("e!!!", false, -1);
		return;
	    }
	    var start_time = process.hrtime();
	    p_client.query(queryConfig, function(err, result) {
		done();
		var time_of_q = process.hrtime(start_time);
		var ms_elapsed = time_of_q[0] * 1000 + (time_of_q[1] / 1000000);
		
		var cost = ms_elapsed * 100;

		if(err) {
		    console.error('error running query', err);
		    writeResponse("e!!!", false, -1);
		}else{
		    var data = result.rows[0].value;
		    //console.log(data);
		    var setargs = [objid, data, "CX", parseInt(cost), "CC", parseInt(objclass)];
		    r_pool.send_command('SET', setargs, function(err, reply){
			if (err){
			    console.log(err);
			}
			if (cost != ""){
			    writeResponse("m:" + data, false, parseInt(cost));
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

var pg_pool1 = new PgPool(psqlConf1);
var pg_pool2 = new PgPool(psqlConf2);

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
