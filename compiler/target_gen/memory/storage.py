from enum import Enum
from compiler.utils import unique_class_name

class StorageType:
    WEIGHT="WEIGHT"
    ACTIVATION="ACTIVATION"
    WEIGHT_GRAD="WEIGHT_GRAD"
    FEATURE_GRAD="FEATURE_GRAD"

class Storage:
    # def __init__(self,type:StorageType,addr:int,size:int,content:list):
    #     self.type = type
    def __init__(self,size:int,content:list,type:StorageType,addr:int=-1):
        self.addr = addr
        self.size = size
        self.type = type
        self.content = content
        self.name = unique_class_name(self)
        self.ref_tensors = set()
        self.data = None

    def same_as(self,storage):
        for tensor in self.ref_tensors:
            tensor.storage = storage
            storage.ref_tensors.add(tensor)
    
    def __str__(self):
        return f"[Storage] name={self.name}"

if __name__=="__main__":
    print(help(StorageType))