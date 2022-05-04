import yaml
import math
from bitarray import bitarray
from collections import OrderedDict
def int_to_bits(num,width,name="Num",endian="big"):
    assert num>=0 and num<(1<<width),f"{name} should in [{0},{(1<<width)-1}],but got {num}"
    bits = bitarray(endian=endian)
    bits.frombytes(int(num).to_bytes(math.ceil(width/8),byteorder=endian, signed=False))
    if endian=="big":
        bits = bits[-width:]
    else:
        bits = bits[0:width]
    return bits

class Instruction(object):
    def __init__(self):
        self.dict = OrderedDict()
    def set(self,key,value):
        self.dict[key] = value
    def get(self,key):
        return dict[key]
    def print(self):
        s = " ".join([str(item) for item in self.dict.values()])
        print(s)
    def export(self):
        bits = bitarray()
        for key,value in self.dict.items():
            bits.extend(value)
        return bits

class InstructionTool(object):
    def __init__(self,path):
        with open(path, 'r', encoding="utf-8") as f:
            file_data = f.read()                
        self.cfg = yaml.load(file_data,Loader=yaml.FullLoader)
        self.check_cfg()
        self.init_cfg()

        self.instructions = []
    
    def check_cfg(self):
        #Config is loaded
        assert self.cfg,"Load config file first!"

        #All needed attrs are exists
        assert "setting" in self.cfg,"Can not find 'setting'"
        assert "meta_instr" in self.cfg,"Can not find 'meta_instr'"
        assert "instr" in self.cfg,"Can not find 'instr'"

        assert "width" in self.cfg["setting"],"Can not find 'setting.width'"
        assert "operate_width" in self.cfg["setting"],"Can not find 'setting.operate_width'"

        for meta_instr in self.cfg["meta_instr"]:
            assert "name" in meta_instr,"Can not find 'meta_instr.name'"
            assert "layout" in meta_instr,"Can not find 'meta_instr.layout'"
            for layout in meta_instr["layout"]:
                assert "name" in layout,"Can not find meta_instr.layout.name"
                assert "width" in layout,"Can not find meta_instr.layout.width"
        
        for instr in self.cfg["instr"]:
            assert "name" in instr,"Can not find 'instr.name'"
            assert "type" in instr,"Can not find 'instr.type'"
            assert "code" in instr,"Can not find 'instr.code'"

        #width is correct
        width = self.cfg["setting"]["width"]
        operate_width = self.cfg["setting"]["operate_width"]
        for meta_instr in self.cfg["meta_instr"]:
            layout_width = operate_width
            for layout in meta_instr["layout"]:
                layout_width += layout["width"]
            assert layout_width == width,f"Width of \"{meta_instr['name']}\" is {layout_width}, not equals {width}."
        for instr in self.cfg["instr"]:
            code = instr["code"]
            assert len(code)==operate_width,f"Operate width of \"{instr['name']}\" is {code}, not equals {operate_width}"

    def init_cfg(self):
        self.cfg_setting = self.cfg["setting"]
        self.cfg_meta_instr = {meta["name"]:meta for meta in self.cfg["meta_instr"]}
        self.cfg_instr = {instr["name"]:instr for instr in self.cfg["instr"]}
        
    def add(self,**kwargs):
        operate = kwargs["operate"]
        assert operate in self.cfg_instr,f"Operate {operate} not exists!"
        cfg_instr = self.cfg_instr[operate]

        instruction = Instruction()
        instruction.set("operate",bitarray(cfg_instr["code"]))

        meta_instr = self.cfg_meta_instr[cfg_instr["type"]]
        layout = meta_instr["layout"]
        for item in layout:
            name = item["name"]
            width = item["width"]
            assert name in kwargs,f"Need {name}!"
            bits = int_to_bits(kwargs[name],name=name,width=width)
            instruction.set(name,bits)
        self.instructions.append(instruction)
    
    def export(self,path):
        bits = bitarray()
        for instruction in self.instructions:
            bits.extend(instruction.export())
        with open(path,'wb') as f:
            bits.tofile(f)
    
    def print(self):
        for instr in self.instructions:
            instr.print()


if __name__=="__main__":
    instr_tool = InstructionTool("backends/sparse_train/instruction.yaml")
    instr_tool.add(operate="send",tensor=128,port=15)
    instr_tool.add(operate="receive",tensor=64,port=1)
    instr_tool.add(operate="conv",tensor=64,port=1)
    instr_tool.print()
    # instr_tool.export("./export.bin")