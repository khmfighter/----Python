from flask import Flask
from flask import request
from flask import render_template
from flask import redirect
from wtforms import Form,TextField,PasswordField,validators
app=Flask(__name__)
from db import *


class LoginForm(Form):
    username = TextField("username",[validators.Required()])
    password = PasswordField("password",[validators.Required()])

@app.route("/",methods=['GET','POST'])
def login():
    myform = LoginForm(request.form)
    if request.method == 'POST':
        if isExisted(myform.username.data,myform.password.data):
            return redirect("http://wwww.baidu.com")
        else:
            messages = "Login Failed"
            return render_template("index.html",message = messages,form = myform)
    return render_template("index.html" ,form = myform)



@app.route("/register",methods=['GET','POST'])
def register():
    myForm = LoginForm(request.form)
    if request.method == 'POST':
        addUser(myForm.username.data,myForm.password.data)
        return "Register Successfully"
    return render_template('index.html',form=myForm)


if __name__ == '__main__':
    app.run(port=8083)