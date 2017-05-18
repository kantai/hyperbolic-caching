class AveragingBucketUpkeep:
    def __init__(self):
        self.numer = 0.0
        self.denom = 0

    def add_cost(self, cost):
        self.numer += cost
        self.denom += 1
        return self.numer / self.denom

    def rem_cost(self, cost):
        self.numer -= cost
        self.denom -= 1
        if self.denom == 0:
            return 0
        return self.numer / self.denom
