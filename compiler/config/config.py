import yaml
import json
class Config:
    def __init__(self):
        self.merge_config = self.load("./merge.yaml")
        
    def load(self,path:str):
        with open(path, 'r', encoding="utf-8") as f:
            file_data = f.read()                 
        data = yaml.load(f,Loader=yaml.FullLoader)    
        return data
