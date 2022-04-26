import collections
class Attrs:
    def __init__(self):
        self.attrs = collections.OrderedDict()
        self.op = None
    
    def get(self,key):
        return self.attrs[key]

    def set(self,key,value):
        self.attrs[key] = value