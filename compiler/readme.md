# 编译

## 1.数据结构

每次创建一对算子，使用compiler.generate.op里面的类来生成



## 1.图变换(compiler.graph)

基于torch.fx对计算图进行变换，分为以下几个模块：

- all_in_op:计算全部换成“算子”
- back_graph_gen:backward生成反传的计算图
- op_merge:算子合并（不同算子合并）
- op_split:算子拆分（一个算子的拆分）


