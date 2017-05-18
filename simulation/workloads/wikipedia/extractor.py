import sys, re, fileinput
import multiprocessing

tagRE = re.compile(r'(.*?)<(/?\w+)[^>]*>(?:([^<]*)(<.*?>)?)?')
#                    1     2               3      4
def pages_from(input):
    """
    Scans input extracting pages.
    :return: (id, title, namespace, page), page is a list of lines.
    """
    # we collect individual lines, since str.join() is significantly faster
    # than concatenation
    page = []
    id = None
    ns = '0'
    last_id = None
    revid = None
    inText = False
    redirect = False
    title = None
    for line in input:
        line = line.decode('utf-8')
        if '<' not in line:  # faster than doing re.search()
            if inText:
                page.append(line)
            continue
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'page':
            page = []
            redirect = False
        elif tag == 'id' and not id:
            id = m.group(3)
        elif tag == 'id' and id:
            revid = m.group(3)
        elif tag == 'title':
            title = m.group(3)
        elif tag == 'ns':
            ns = m.group(3)
        elif tag == 'redirect':
            redirect = True
        elif tag == 'text':
            inText = True
            line = line[m.start(3):m.end(3)]
            page.append(line)
            if m.lastindex == 4:  # open-close
                inText = False
        elif tag == '/text':
            if m.group(1):
                page.append(m.group(1))
            inText = False
        elif inText:
            page.append(line)
        elif tag == '/page':
            if id != last_id and not redirect:
                yield (id, revid, title, ns, page)
                last_id = id
                ns = '0'
            id = None
            revid = None
            title = None
            page = []

def process(input_file, cb = None, lim = None):
    file = fileinput.FileInput(input_file, openhook=fileinput.hook_compressed)
    for ix, page_data in enumerate(pages_from(file)):
        if lim and lim < ix:
            break
        aid, revid, title, ns, page = page_data
        page_content = "\n".join(page)
        cb(aid, title, page_content)
    file.close()


def load_articles(worker, num_procs = 64):
    input_file = "enwiki-20080103-pages-articles.xml.bz2"

    q = multiprocessing.JoinableQueue(25000)
    procs = []
    for i in range(num_procs):         
        procs.append( multiprocessing.Process(
            target=worker(q, talker = (i == 0))))
        procs[-1].daemon = True
        procs[-1].start()
    def make_article_callback(aid, t, pc):
        q.put((aid,t,pc))
    sys.stderr.write("starting...\n")
    process(input_file, cb = make_article_callback, lim = None)
    q.join()
    for p in procs:
        q.put( None )
    q.join()
    sys.stderr.write("\n")

def load_into_django_wiki():
    from wiki import models
    from django.db import IntegrityError
    from django.utils.text import slugify
    models.URLPath.create_root(
        title='WikiRoot',
        content='WikiRoot'
    )
    from django import db
    db.connections.close_all()

    class django_wiki_worker:
        def __init__(self, q, talker = False):
            self.q = q
            self.talker = talker
        def __call__(self):
            q = self.q
            root = models.URLPath.root()
            ix = 0
            for item in iter( q.get, None ):
                ix += 1
                aid, t, pc = item
                slug = "id%s" % aid
                try:
                    models.URLPath.create_article(root, 
                                                  slug, 
                                                  title = t, 
                                                  content = pc)
                except IntegrityError:
                    pass
                if self.talker and int(aid) % 1000 == 0:
                    sys.stderr.write("\r %s of %d" % (aid, q.qsize()))
                    sys.stderr.flush()
                q.task_done()
            q.task_done()

    load_articles(django_wiki_worker)


insert_str = "INSERT INTO blanks_wiki_articles2 (content) VALUES %s;"
drop_str = "DROP TABLE IF EXISTS blanks_wiki_articles2;"
create_str = """CREATE TABLE blanks_wiki_articles2 
                (aid serial PRIMARY KEY,
                 content text);"""

def load_into_database():
    import psycopg2

    conn = psycopg2.connect("dbname=blanks user=blanks host=/tmp port=5435")
    cur = conn.cursor()
    cur.execute(drop_str)
    cur.execute(create_str)
    cur.close()
    conn.commit()

    class simple_db_worker:
        def __init__(self, q, talker = False):
            self.q = q
            self.talker = talker
            self.cursor = conn.cursor()

        def handle_batch(self, batch):
            inserter = insert_str % ", ".join(
                ["(%s)" for i in range(len(batch))])
            self.cursor.execute(inserter, tuple(batch))
            conn.commit()

        def __call__(self):
            q = self.q
            ix = 0

            batch = []

            for item in iter( q.get, None ):
                if len(batch) >= 1000:
                    self.handle_batch(batch)
                    batch = []
                ix += 1
                aid, t, pc = item
                #                if not pc.startswith("#"):
                batch.append(pc)

                if self.talker and int(aid) % 1000 == 0:
                    sys.stderr.write("\r                                ")
                    sys.stderr.write("\r %dk of 15070k" % (int(aid)/1000))
                    sys.stderr.flush()
                q.task_done()

            if len(batch) > 0:
                self.handle_batch(batch)
                batch = []

            q.task_done()


    load_articles(simple_db_worker, num_procs = 1)

def main(input_file, out = sys.stdout):
    def cb(a,b,c):
        if int(a) % 10000 == 0:
            sys.stderr.write("\r %s" % a)
            sys.stderr.flush()
    process(input_file, cb)
    sys.stderr.write("\n")

if __name__ == "__main__":
    load_into_database()
#    main("enwiki-20080103-pages-articles.xml.bz2")
