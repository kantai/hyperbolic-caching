import numpy
from utils.sortedcollection import SortedCollection
import bisect
from operator import itemgetter

def zipfian_zeta(s, N):
    """ compute zeta constant zipfian distribution.
        which is the generalized harmonic number N,s
    """

    out = 0.0
    for i in range(1, N + 1):
        out += 1.0 / (i ** s)

    return out

class YCSB_Zipfian(object):
# See Paper: Quickly Generating Billion-Record Synthetic Databases by Jim Gray

    def __init__(self, count, theta):

        zetan = zipfian_zeta(theta, count)
        zeta2theta = zipfian_zeta(theta, 2)

        self.alpha = 1.0/(1.0 - theta)
        self.zetan = zetan
        self.count = count
        self.theta = theta

        self.eta = (1 - (2.0/count)**(1 - theta))/(1 - zeta2theta/zetan)

        lambdas = [ float(i) ** -self.theta for i in range(1, count + 1) ]
        s = sum(lambdas)
        self.lambdas = [i / s for i in lambdas]

    def get_popularity(self, item):
        return self.lambdas[item]
        
    def get_next(self, random_float):
        u = random_float
        uz = u * self.zetan
        if (uz < 1.0):
            return (0, self.lambdas[0])
        if (uz < 1.0+(0.5**self.theta)):
            return (1, self.lambdas[1])
        eta = self.eta
        index = int(self.count * (eta*u - eta + 1)**(self.alpha))
        return (index, self.get_popularity(index))

class Deterministic_Zipfian(object):
# Take Accu_Zipf, and build request sequence s.t. we greedily request the object whose
# current req. frequency is furthest from the 'true' popularity
    def __init__(self, count, theta):
        zetan = zipfian_zeta(theta, count)
        zeta2theta = zipfian_zeta(theta, 2)

        self.count = count
        self.theta = theta
        
        self.freqs = numpy.array([float(i) ** -theta for i in range(1, count + 1)], numpy.float64)
        self.freqs = self.freqs / sum(self.freqs)
        self.goal_count = numpy.zeros(self.freqs.shape, numpy.float64)
        self.empirical_count = numpy.zeros(self.freqs.shape, numpy.int32)

    def get_popularity(self, item):
        return float(self.freqs[item])

    def get_next(self, random_float):
        self.goal_count += self.freqs
        to_req = int(numpy.argmax((self.goal_count - self.empirical_count)))
        self.empirical_count[to_req] += 1
        return (to_req, self.get_popularity(to_req))

class Deterministic_Zipfian_Lambda(object):
    def __init__(self, count, theta):
        zetan = zipfian_zeta(theta, count)
        zeta2theta = zipfian_zeta(theta, 2)

        self.count = count
        self.theta = theta
        
        freqs = [float(i) ** -theta for i in range(1, count + 1)]
        s = sum(freqs)
        self.lambdas = [ s / i for i in freqs]
        self.t = 0
        self.next_arrivals = SortedCollection(iterable = enumerate(self.lambdas), 
                                              key = itemgetter(1))

    def get_popularity(self, item):
        return ( 1.0 / self.lambdas[item] )

    def __get_next(self):
        self.t += 1
        item = self.next_arrivals[0]
        del self.next_arrivals[0]
        item_lambda = self.lambdas[item[0]]
        item_next = (item[0], item_lambda + self.t)
        self.next_arrivals.insert_right(item_next)
        return (item_next[0], 1.0 / item_lambda)

    def get_next(self, rand_float):
        return self.__get_next()

class AccurateDistributionGenerator(object):
    def __init__(self, freqs):
        cumulative_freqs = []
        accumulator = 0
        for i in freqs:
            accumulator += i
            cumulative_freqs.append(accumulator)
        self.cum_freqs = cumulative_freqs

    def get_popularity(self, item):
        if item == 0:
            prior = 0
        else:
            prior = self.cum_freqs[item - 1]
        
        return self.cum_freqs[item] - prior

    def get_next(self, random_float):
        index = bisect.bisect_left(self.cum_freqs, random_float)

        return (index, self.get_popularity(index))


class Accu_Zipfian(AccurateDistributionGenerator):
# this is an Aaron Blankstein direct implementation -- computes cdf of zipfian distribution
# and uses binary search to implement get_next
    
    def __init__(self, count, theta):

        zetan = zipfian_zeta(theta, count)
        zeta2theta = zipfian_zeta(theta, 2)

        self.count = count
        self.theta = theta

        cum_lambdas = []
        last = 0
        for i in range(1, count + 1):
            cur = (float(i) ** -self.theta) + last
            cum_lambdas.append( cur )
            last = cur

        total = last
        self.cum_freqs = [i / total for i in cum_lambdas]
        
def test_zipfian(N, theta, times):
    import numpy.random
    Z = YCSB_Zipfian(N, theta)
    counts = {}
    for i in range(0, N):
        counts[i + 1] = 0
    for i in range(0, times):
        next = 1 + Z.get_next(numpy.random.random())[0]
        counts[next] += 1

    for key, value in counts.items():
#        L = Z.get_lambda(key - 1)
        L = Z.lambdas[key - 1]
        print "%s, %f, %f" % (key, float(value) / times, L)
        

if __name__ == "__main__":
    import sys
    test_zipfian(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]))

