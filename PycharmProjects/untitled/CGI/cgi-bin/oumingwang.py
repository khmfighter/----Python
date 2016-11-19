def statr_response(resp="text/html"):
    return 'Content-type:'+ resp +'\n\n'

def img(src):
    return '<img src = "'+src+'">'

def  head(title):
    return '<head><title>'+title+'</title></head><body>'

def start_div(align,style):
    return '<div align = "'+align+'" style = "'+style+'">'

def end_div():
    return '</div>'

def start_form(the_url="",form_type="POST"):
    return '<form action="'+the_url+'" method = "'+form_type+'">'

def end_form(value="ubmit"):
    return '<p></p><input type="submit" value = "'+value+'" >'
S

def input_label(name,placeholder="",value="",readonly=None):
    if readonly is None:
        return '<input type="text" value = "'+value+'"placeholder="'+placeholder+'" name ="'+name+'">'
    else:
        return '<input type="text" value = "' + value + '" readonly = "'+readonly+'"placeholder="' + placeholder + '" name ="' + name + '">'