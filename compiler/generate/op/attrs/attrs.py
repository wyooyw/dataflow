import collections
import copy
class Attrs:
    def __init__(self):
        self.attrs = collections.OrderedDict()
        self.op = None
    
    def get(self,key):
        return self.attrs[key]

    def set(self,key,value):
        self.attrs[key] = value

    def __copy__(self):
        copy_self = type(self)()
        copy_self.attrs = copy.copy(self.attrs)
        copy_self.op = self.op
        return copy_self