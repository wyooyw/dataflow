import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import random

class Rectangle:
    def __init__(self,x_range,height,color,tensor=None):
        self.x_range = x_range
        self.height = height
        self.color = color
        self.tensor = tensor
        pass


class RectanglePainter:
    def __init__(self):
        self.color = {
            "red":[1,0,0],
            "green":[0,1,0],
            "blue":[0,0,1]
        }
        self.x_lim = [0,1]
        self.y_lim = [0,1]
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        pass

    def set_lim(self,x_lim,y_lim):
        self.x_lim = x_lim
        self.y_lim = y_lim

    def paint(self,range_x,range_y,color="green"):
        """绘制矩形
        range_x为矩形在横轴上的范围
        range_y为矩形在纵轴上的范围
        """
        x1,x2 = range_x
        y1,y2 = range_y
        assert color in self.color,f"Color is not exist:{color}"
        assert x1>=0 and x2>x1 and y1>=0 and y2>y1,f"Error! range_x={range_x},range_y={range_y}"
        plt.gca().add_patch(plt.Rectangle(xy=(x1,y1),
        width=x2-x1, 
        height=y2-y1,
        edgecolor=self.color[color],
        fill=False, linewidth=1))
    def show(self):
        plt.xlim(*self.x_lim)
        plt.ylim(*self.y_lim)
        plt.show()
    def save(self,path="test"):
        plt.xlim(*self.x_lim)
        plt.ylim(*self.y_lim)
        plt.savefig(f"{path}.png",dpi=300) 

class RectangleManager:
    def __init__(self,name="resnet-reverse"):
        self.rectangle_list = []
        self.rectangle_fix_list = []
        self.x_max = 0
        self.y_max = 0
        self.painter = RectanglePainter()
        self.name = name

    def add_rectangle(self,rectangle):
        self.rectangle_list.append(rectangle)
        if rectangle.x_range[1] > self.x_max:
            self.x_max = rectangle.x_range[1]

    def add_rectangle_fix(self,rectangle):
        self.rectangle_fix_list.append(rectangle)
        if rectangle.x_range[1] > self.x_max:
            self.x_max = rectangle.x_range[1]

    def layout(self):
        self.name="resnet-margin-5"
        ground = [0]*(self.x_max+1)

        def put_rec(rec):
            x_range = rec.x_range
            height = rec.height
            ground_max = self.max(ground,x_range[0],x_range[1])
            ground_new_height = ground_max+height
            for i in range(x_range[0],x_range[1]):
                ground[i] = ground_new_height
            rec.y_range = [ground_max,ground_new_height]
            self.y_max = max(self.y_max,ground_new_height)
        for rec1 in self.rectangle_fix_list:
            put_rec(rec1)
        # self.rectangle_list.reverse()
        # random.shuffle(self.rectangle_list)
        a = self.rectangle_list[0::5]
        b = self.rectangle_list[1::5]
        c = self.rectangle_list[2::5]
        d = self.rectangle_list[3::5]
        e = self.rectangle_list[4::5]
        a.extend(b)
        a.extend(c)
        a.extend(d)
        a.extend(e)
        assert len(a)==len(self.rectangle_list)
        self.rectangle_list = a
        for rec in self.rectangle_list:
            put_rec(rec)

        self.rectangle_list.extend(self.rectangle_fix_list)
        return self.y_max
        # self.y_max = [0,max_height]
    
    def max(self,ls,range_begin,range_stop):
        m = -999
        for i in range(range_begin,range_stop):
            m = max(m,ls[i])
        return m

    def title(self):
        B = self.y_max*2
        KB = int(B/1024*100)/100
        MB = int(KB/1024*100)/100
        plt.title(f'{self.name} ({B}B={KB}KB={MB}MB)', fontsize=10)
    
    def paint(self):
        for rec in self.rectangle_list:
            self.painter.paint(rec.x_range,rec.y_range,rec.color)
        self.title()
        self.y_max += 1
        self.x_max += 1
        self.painter.set_lim(x_lim=(0,self.x_max), y_lim=(0,self.y_max))
        self.painter.show()
    
    def save(self):
        for rec in self.rectangle_list:
            self.painter.paint(rec.x_range,rec.y_range,rec.color)
        self.title()
        self.y_max += 1
        self.x_max += 1
        self.painter.set_lim(x_lim=(0,self.x_max), y_lim=(0,self.y_max))
        self.painter.save(self.name)