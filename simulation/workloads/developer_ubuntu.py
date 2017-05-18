from workloads.cost_driver import ZipfUniformDriver


class URLDriver(ZipfUniformDriver):
    def __init__(self, url_file = "workloads/ubuntu_developer_urls", item_range_max = 0, **kwargs):
        self.urls = self.fetch_urls(url_file)
        super(URLDriver, self).__init__(item_range_max = len(self.urls), 
                                        **kwargs)

    def fetch_urls(self, url_file):
        out = []
        with open(url_file, 'r') as file:
            for url in file:
                url = url.strip()
                if url.startswith("/"):
                    out.append(url[1:])
                else:
                    out.append(url)
        out.sort(cmp = lambda x, y: cmp(len(x), len(y)))
        return out

    def get_cost(self, r_float, item_num):
        return 1.0

    def get_item(self, r_float):
        item_number = super(URLDriver, self).get_item(r_float)
        return self.urls[item_number]


class WikiZURLDriver(ZipfUniformDriver):
    def __init__(self, item_range_max = 10**6, **kwargs):
        super(WikiZURLDriver, self).__init__(item_range_max = item_range_max, 
                                             **kwargs)

    def get_cost(self, r_float, item_num):
        return 1.0

    def get_item(self, r_float):
        if r_float > 0.5:
            url_first = "markdown"
            new_r_float = 2.0*(r_float - 0.5)
        else:
            url_first = "mediawiki"
            new_r_float = 2.0*(r_float)
        item_number = super(WikiZURLDriver, self).get_item(new_r_float)
        return "%s/%s" % (url_first, item_number+1)
