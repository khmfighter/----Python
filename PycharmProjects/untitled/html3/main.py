from flask import Flask
from flask import request
from flask import render_template
from flask import redirect
from wtforms import Form,TextField,PasswordField,validators
app=Flask(__name__)

class LoginForm(Form):
    username = TextField("username",[validators.Required()])
    password = PasswordField("password",[validators.Required()])

@app.route("/",methods=['GET','POST'])
def login():
    myform = LoginForm(request.form)
    if request.method == 'POST':
        if myform.username.data =="oumingwang" and myform.password.data == "123456" and myform.validate():
            return redirect("http://wwww.baidu.com")
        else:
            messages = "Login Failed"
            return render_template("index.html",message = messages,form = myform)
    return render_template("index.html" ,form = myform)

if __name__ == '__main__':
    app.run(port=8083)