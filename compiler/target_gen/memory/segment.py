from enum import Enum
class SegmentType(Enum):
    WEIGHT_STORAGE=0
    ACTIVATION_STORAGE=1
    GRAD_STORAGE=2
    TENSOR=3
    OPERATOR=4
    NET=5

class Segment:
    def __init__(self,type:SegmentType,base:int=0,size:int=0):
        self.type = type
        self.base = base
        self.size = size