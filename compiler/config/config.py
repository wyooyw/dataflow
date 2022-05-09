import yaml
import json
from compiler.utils import singleton
@singleton
class Config:
    def __init__(self):
        # self.replace_config = self.load("./backends/sparse_train/replace.yaml")
        # self.op_config = self.load("./compiler/config/op.yaml")
        self.convert_config = self.load("./convert.yaml")
        # self.deal_op_config()
        # print(json.dumps(self.op_config,indent=2))
        
    def load(self,path:str):
        with open(path, 'r', encoding="utf-8") as f:
            file_data = f.read()                 
        data = yaml.load(file_data,Loader=yaml.FullLoader)    
        return data

    def deal_op_config(self):
        """对op.yaml里的配置做预处理
        """
        for op in self.op_config["operators"]:
            for tensor in op["tensors"]:
                if tensor.get("grad",False)==True:
                    tensor_grad = tensor.copy()
                    tensor_grad["name"] += "_grad" 
                    del tensor_grad["grad"]
                    op["tensors"].append(tensor_grad)

            for direction in ["forward","backward"]:
                for section in ["tensors","input","output"]:
                    for idx,tensor in enumerate(op[direction][section]):
                        if tensor.endswith(".grad"):
                            tensor = tensor.replace(".grad","_grad")
                            op[direction][section][idx] = tensor
                