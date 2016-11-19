
class Stack():
    def __init__(st,size):
        st.size=size;
        st.stack=[];
        st.top=-1;
    def push(st,content):
        if st.Full():
            return "this is full"
        else:
            st.stack.append(content)
            st.top=st.top+1
    def Full(st):
        if st.top == st.size:
            return True
        else:
            return False

    def Pop(st):
        if st.Empty():
            return "this is empty"
        else:
            popdata = st.stack[st.top]
            st.top=st.top-1

    def Empty(st):
        if st.top == -1:
            return True
        else:
            return False