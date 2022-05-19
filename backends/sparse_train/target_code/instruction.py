import yaml
import collections
from bitarray import bitarray
class Instruction(object):
    def __init__(self,name="",config_path="backends/sparse_train/target_code/structure_info.yaml",init_data={},pad_to=None):
        self.name = name
        self.config = collections.OrderedDict()
        self.alias_to_value = {}
        self.value_to_alias = {}
        self.data = {}
        self.init_data = init_data
        self.pad_to = pad_to
        self.init_config(config_path)
        
    
    def set(self,key,value,use_bits=False):
        assert key in self.alias_to_value,f"Key is not find : {key}"
        if not self.is_instr_valid(key):
            assert "Key is not valid:{key}"
        if use_bits:
            assert len(value)==self.config[key]["long"],f"{key}.long={self.config[key]['long']},but got {len(value)}"
            self.data[key] = value
        else:
            assert value in self.alias_to_value[key],f"Alias is not find : {value}"
            self.data[key] = self.alias_to_value[key][value]
    
    def get(self,key,use_bits=False):
        assert key in self.data,f"Key is not find : {key}"
        value = self.data[key]
        if value:
            return self.value_to_alias[key][value]
        return value
    
    def is_instr_valid(self,instr_name):
        if instr_name not in self.config:
            return False
        if "condition" not in self.config[instr_name]:
            return True
        return eval(self.config[instr_name]["condition"])

    def iter(self,to_str=False):
        for name,instr in self.config.items():
            _long = instr["long"]
            value = self.data[instr['name']]
            alias = self.value_to_alias[name].get(value,value)
            if to_str:
                _long = str(_long)
                value = str(value)
                alias = str(alias)
            yield name,_long,value,alias
    
    def export(self):
        bits = bitarray()
        for name,_,value,_ in self.iter():
            assert value,"You can not export, beacause value of '{}' is None!".format(name)
            bits.extend(value)
        if self.pad_to and len(bits) < self.pad_to:
            bits.extend("0"*(self.pad_to-len(bits)))
        return bits

    def __str__(self):
        strs = ["\n"]
        title = f"Structure info of {self.name}"
        strs.append("{} {} {}".format("="*((80-len(title))//2),title,"="*((80-len(title))//2)))
        strs.append("{}{}{}{}".format("name".ljust(20),
                                        "long".ljust(20),
                                        "value".ljust(20),
                                        "value_alias".ljust(20)))
        strs.append("-"*80)
        total_length = 0
        for name,_long,value,alias in self.iter():
            total_length += _long
            strs.append("{}{}{}{}".format(name.ljust(20),
                                        str(_long).ljust(20),
                                        str(value).ljust(20),
                                        str(alias).ljust(20)))
        strs.append("-"*80)
        bits = self.export()
        strs.append("{}{}".format("length:".ljust(20),total_length))
        strs.append("{}{}".format("length after pad:".ljust(20),self.pad_to))
        strs.append(self.export().to01())
        strs.append("="*80)
        return "\n".join(strs)

    def init_config(self,path):
        config = self.load_config(path)
        for instr in config:
            #是否满足条件
            if "condition" in instr:
                if not eval(instr["condition"]):
                    continue
            #引用 或 指令
            if "ref" in instr:
                self.init_config(instr["ref"])
            else:
                name = instr["name"]
                assert not name in self.data,f"Instruction conflict: {name}"
                self.alias_to_value[name] = {}
                self.value_to_alias[name] = {}
                if "terms" in instr:
                    for value,alias in instr["terms"].items():
                        self.alias_to_value[name][alias] = value
                        self.value_to_alias[name][value] = alias
                self.data[name] = None
                self.config[name] = instr
                if name in self.init_data:
                    self.set(name,self.init_data[name])
                
    
    def load_config(self,path:str):
        with open(path, 'r', encoding="utf-8") as f:
            file_data = f.read()                 
        data = yaml.load(file_data,Loader=yaml.FullLoader)    
        return data

    @classmethod
    def load_from_bitarray(self,path:str,bits):
        #first load
        init_data = {}
        bits_start = 0
        bits_len = len(bits)
        instr = Instruction(path)
        for name,cfg in instr.config.items():
            bits_end = bits_start+cfg["long"]
            assert bits_end<=bits_len,"Bitarray is not long enough."
            value = bits[bits_start:bits_end].to01()
            alias = instr.value_to_alias[name][value]
            init_data[name] = alias
            bits_start = bits_end

        bits_start = 0
        bits_len = len(bits)
        instr = Instruction(path,init_data=init_data)
        for name,cfg in instr.config.items():
            bits_end = bits_start+cfg["long"]
            assert bits_end<=bits_len,"Bitarray is not long enough."
            value = bits[bits_start:bits_end].to01()
            instr.set(name,value,use_bits=True)
            bits_start = bits_end
        return instr

    
if __name__=="__main__":
    instr = Instruction("structure_info.yaml",{
        "net_type":"alexnet",
        "stage":"forward",
        "op_type":"conv",
        "stride":1,
        "padding":False,
        "relu": True,
        "maxpool": True,
        "kernel_size":3,
        "add":False,
        "bn":True,
        "part_sum":False,
        "softmax":False
    },pad_to=120)
    instr.set("in_feature","000000000",use_bits=True)
    instr.set("weight","000000001",use_bits=True)
    instr.set("output","000000010",use_bits=True)
    instr.set("relu_mask","000000010",use_bits=True)
    instr.set("pool_mask","000000010",use_bits=True)
    instr.set("bn_use","000000110",use_bits=True)
    print(instr)
    instr2 = Instruction.load_from_bitarray("structure_info.yaml",instr.export())
    instr2.pad_to=120
    print(instr2)
    # print(instr.export())