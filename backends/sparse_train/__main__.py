if __name__=="__main__":
    net = Net.from_onnx("./resnet18.onnx")
    
    #算子替换
    replace_tool = ReplaceTool("backends/sparse_train/replacement.yaml")
    net = replace_tool.replace(net)
    net.to_onnx("./after_replacement.onnx")

    #把一些维度分割开
    net.execute("split")#对每个算子，如果有split方法，则执行
    net.to_onnx("./after_split.onnx")

    #生成目标代码
    net.generate("./instructions.txt")
    