from compiler.config import Config
from jinja2 import Template, FileSystemLoader, Environment
class CodeGen:
    def __init__(self):
        self.config = Config()
    def generate_operator(self):
        config = self.config.op_config
        with open("./compiler/config/template/op.jinja",encoding="UTF-8") as f:
            content = f.read()
            j2_tmpl = Template(content)
            content = j2_tmpl.render(**config)
        with open(f"./compiler/config/template/operator.py", 'w',encoding="UTF-8") as df:
            df.write(content)
