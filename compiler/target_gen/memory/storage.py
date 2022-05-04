from enum import Enum
from compiler.utils import unique_class_name
class StorageType(Enum):
    WEIGHT=0
    ACTIVATION=1
    GRAD=2

class Storage:
    # def __init__(self,type:StorageType,addr:int,size:int,content:list):
    #     self.type = type
    def __init__(self,size:int,content:list,addr:int=-1):
        self.addr = addr
        self.size = size
        self.content = content
        self.name = unique_class_name(self)
    
    def __str__(self):
        return f"[Storage] name={self.name}"

if __name__=="__main__":
    print(help(StorageType))