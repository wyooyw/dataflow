from test_op.communicate.communicate import Message,Communicate
class FileMessage(Message):
    def __init__(self,filename,value):
        self.filename = filename
        self.value = value

    def setFilename(self,filename):
        self.filename = filename
    
    def getFilename(self):
        return self.filename
    
    def setValue(self,value):
        self.value = value
    
    def getValue(self):
        return self.value

class FileCommunicate(Communicate):
    def __init__(self,path="",send_end="send_end.txt",recv_end="receive_end.txt"):
        os.mkdir(path/send)
        os.mkdir(path/recv)
        self.path = path
        self.send_path = os.join(path,"send")
        self.recv_path = os.join(path,"recv")
        self.send_end = os.join(self.send_path,send_end)
        self.recv_end = os.join(self.recv_path,recv_end)
        pass

    def send(self,messages:List[FileMessage]):
        for message in messages:
            with open(os.join(path,message.getFilename()),"w") as f:
                f.write(message.getValue())
    
        with open(os.join(path,self.send_end),"w") as f:
            f.write("send_end")
        
    def receive(self):
        wait_time = 0
        wait_time_max = 20
        while wait_time < wait_time_max:
            if os.path.exists(self.recv_end):
                with open(self.recv_end,"r") as f:
                    end_msg = f.read()
                    break
            time.sleep(5)
            wait_time += 5
        messages = []
        for recv_file_name in self.recv_path:
             with open(recv_file_name,"r") as f:
                 value = f.read(
                 messages.append(FileMessage(recv_file_name,value))
        return messages
