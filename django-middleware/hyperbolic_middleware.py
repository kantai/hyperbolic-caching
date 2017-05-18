import time

try:
    from django.utils.deprecation import MiddlewareMixin
except ImportError:
    MiddlewareMixin = object

from django.db import connection

from django.core.urlresolvers import resolve

def get_url_name(request):
    current_url = resolve(request.path_info).url_name
    return current_url

class CacheTimingMiddlewareTop(MiddlewareMixin):
    """
    Response middleware is evaluated backwards, so
    this needs to be placed immediately BELOW the 
    UpdateCacheMiddleware.
    """
    def process_response(self, request, response):
        if (not hasattr(request, '_cache_update_cache')) or \
           request._cache_update_cache == False:
            return response

        proctime_start = request._cache_proctime_start
        realtime_start = request._cache_realtime_start
        dbtime = sum([float(q['time']) for q in connection.queries])
        
        for q in connection.queries:
            if float(q['time']) > 1:
                print "SQL: %s" % q['sql']
                print "TIME: %s" % q['time']

        def stop_timer(r):
            realtime_stop = time.time()
            proctime_stop = time.clock()
            r._cache_proctime = dbtime + (proctime_stop - proctime_start)
            r._cache_realtime = (realtime_stop - realtime_start)
            r._url_name = get_url_name(request)

        if hasattr(response, 'render') and callable(response.render):
            response.add_post_render_callback( stop_timer )
        else:
            stop_timer(response)

        return response

class CacheTimingMiddlewareBottom(MiddlewareMixin):
    """
    Should be placed immediately BELOW the 
    FetchFromCacheMiddleware
    """
    def process_request(self, request):
        request._cache_proctime_start = time.clock()
        request._cache_realtime_start = time.time()
        
        connection.force_debug_cursor = True
