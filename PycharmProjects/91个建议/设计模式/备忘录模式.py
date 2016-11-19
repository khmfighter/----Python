'''class Originator:
    def __init__(self):
        self.state = ""
    def Show(self):
        print self.state
    def CreateMemo(self):
        return Memo(self.state)
    def SetMemo(self,state):
        self.state = state

class Memo:
    state= ""
    def __init__(self,ts):
        self.state = ts

class Caretaker:
    memo = ""

if __name__ == "__main__":
    on = Originator()
    on.state = "on"
    on.Show()
    c = Memo(on.state)
    on.state="off"
    on.Show()
    on.SetMemo(c.state)
    on.Show()
'''
class Originator:
    def __init__(self):
        self.state = ""
    def Show(self):
        print self.state
    def CreateMemo(self):
        return Memo(self.state)
    def SetMemo(self,memo):
        self.state = memo.state

class Memo:
    state= ""
    def __init__(self,ts):
        self.state = ts

class Caretaker:
    memo = ""

if __name__ == "__main__":
    on = Originator()
    on.state = "on"
    on.Show()
    c = Caretaker()
    c.memo=on.CreateMemo()
    on.state="off"
    on.Show()
    on.SetMemo(c.memo)
    on.Show()