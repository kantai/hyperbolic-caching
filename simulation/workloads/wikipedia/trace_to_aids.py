import sys
import psycopg2 as pg2
from urllib2 import unquote as urldecode
from urllib2 import unquote as urldecode
from subprocess import PIPE, Popen

from cgi import escape
import sys

strip_out = "http://en.wikipedia.org/wiki/"

def url_to_lookup(url_in):
    name = url_in[len(strip_out):]
    find_name = urldecode(name)
    find_name = escape(find_name, quote = True)
    return fixup_find_name(find_name)

def fixup_find_name(find_name):
    find_name = find_name.replace("_", " ")

    if "#" in find_name:
        find_name = find_name.split("#")[0]

    if find_name == "":
        return "!Index"
    
        #    find_name = find_name[0].upper() + find_name[1:]

    try:
        return find_name.decode("utf-8")
    except UnicodeDecodeError:
        return False

SQL_FIND_REVISION = """
SELECT article_id, left(content, 64) FROM wiki_articlerevision
WHERE title = %s
"""

REDIRECT_STA = "#REDIRECT [["
REDIRECT_END = "]]"
def check_redirect(content):
    if not content.startswith(REDIRECT_STA):
        return None
    redirect = content[len(REDIRECT_STA):]
    redirect = redirect[:redirect.find(REDIRECT_END)]

    return fixup_find_name(redirect)

def process_lookup(lookup, cursor):
    if lookup == "!Index":
        return -1
    cursor.execute(SQL_FIND_REVISION, (lookup,))
    records = cursor.fetchall()
    if len(records) == 0:
        return -2
    elif len(records) == 1:
        article_id, content = records[0]
        redirect = check_redirect(content)
        if redirect:
            return process_lookup(redirect, cursor)
        else:
            return article_id
    elif len(records) == 2:
        sys.stderr.write("\n multiple records : '%s' \n" % lookup)
        return -2


def main(X = 10**5, out = sys.stdout):
    trace_fn = "/disk/scratch2/blanks/wikipedia_traces/wiki.1190153705.gz"

    load_str = "gunzip -c %s | grep en.wikipedia.org/wiki | grep -v 'wiki/Special:' | grep -v 'Image:' | grep -v 'Talk:' | grep -v 'User:' | grep -v 'User_talk:' | grep -v 'Category:'"
    proc = Popen(["/bin/sh", "-c", load_str % trace_fn], 
                 stdout = PIPE, stderr = PIPE)
    g = 0; b = 0

    CONNECT = {
        "database" : "wikipedia", 
        "user" : "blanks", 
        "host" : "/disk/local/blanks", 
        "port" : 5433
    }


    sys.stderr.write("starting...\n")
    with pg2.connect(**CONNECT) as connection:
        for l in proc.stdout:
            req = l.split(" ")[2].strip()
            lookup = url_to_lookup(req)
            if not lookup:
                continue
            with connection.cursor() as cursor:
                r_val = process_lookup(lookup, cursor)
            if r_val != -2:
                out.write("%d \n" % r_val)
                g += 1
            else:
                b += 1
            if X > 0 and (g+b) >= X:
                break
            if (g + b) % 1000 == 0:
                sys.stderr.write("\r%d processed..." % (g+b))
                sys.stderr.flush()
    proc.kill()
    sys.stderr.write("\n%.1f %% good.\n" % (100 * float(g) / (g + b)))


if __name__ == "__main__":
    with open("/tmp/trace_processed", 'w') as out:
        main(int(sys.argv[1]), out = out)
