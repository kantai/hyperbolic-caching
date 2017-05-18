from redis_cache import RedisCache
from redis_cache.backends.base import get_client

from django.core.cache.backends.filebased import FileBasedCache

from django.core.cache.backends.base import DEFAULT_TIMEOUT

class HyperbolicCache(RedisCache):
    def __init__(self, server, params):
        super(HyperbolicCache, self).__init__(server, params)

        self.use_costs = self.options.get('HYPERBOLIC_USE_COSTS', False)
        self.use_classes = self.options.get('HYPERBOLIC_USE_CLASSES', False)
        self.print_timing = self.options.get('HYPERBOLIC_PRINT_TIMING', False)
        if self.use_classes:
            self.urls_to_classes = {}
            for i,url in enumerate(self.use_classes):
                self.urls_to_classes[url] = i

    def _set(self, client, key, value, timeout, _add_only=False, 
             cost = False, cost_class = False):
        if timeout is None or timeout == 0:
            if _add_only:
                return client.set(key, value, nx = True, cx = cost, cc = cost_class)
            return client.set(key, value, cx = cost, cc = cost_class)
        elif timeout > 0:
            if _add_only:
                added = client.set(key, value, nx = True, ex = timeout,
                                   cx = cost, cc = cost_class)
                return added
            return client.set(key, value, ex = timeout,
                              cx = cost, cc = cost_class)
        else:
            return False

    @get_client(write=True)
    def set(self, client, key, response, timeout=DEFAULT_TIMEOUT):
        """Persist a value to the cache, and set an optional expiration time.
        """
        timeout = self.get_timeout(timeout)
        cost = False
        cost_class = False

        is_response = False

        if hasattr(response, "_cache_proctime"):
            data = (response._cache_proctime,
                    response._cache_realtime,
                    response._url_name,
                    key)
            if self.print_timing:
                print "hyper-store, %f, %f, %s, %s" % data 
            if self.use_costs:
                cost = response._cache_proctime
            if self.use_classes:
                cost_class = 1 + self.urls_to_classes[response._url_name]
            is_response = True
            response['X-CACHED-MIDDLEWARE'] = "Hit"

        result = self._set(client, key, self.prep_value(response), timeout, _add_only=False,
                           cost = int(100*cost), cost_class = cost_class)
        
        if is_response:
            response['X-CACHED-MIDDLEWARE'] = "Miss"

        return result


class HyperbolicCacheFS(FileBasedCache):
    def __init__(self, server, params):
        super(HyperbolicCacheFS, self).__init__(server, params)

    def set(self, key, response, timeout=None, version = None):
        is_response = False

        if hasattr(response, "_cache_proctime"):
            data = (response._cache_proctime,
                    response._cache_realtime,
                    response._url_name,
                    key)
            print "hyper-store, %f, %f, %s, %s" % data 
            is_response = True
            response['X-CACHED-MIDDLEWARE'] = "Hit"
            
        if timeout:
            result = super(HyperbolicCacheFS, self).set(key, response, timeout, version = version)
        else:
            result = super(HyperbolicCacheFS, self).set(key, response, version = version)

        if is_response:
            response['X-CACHED-MIDDLEWARE'] = "Miss"

        return result
