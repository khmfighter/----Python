def newfoo(arg1,arg2, *nkw, **nw):
    print 'arg1 is :',arg1
    print 'arg2 is :',arg2
    for eachNKW in nkw:
        print '*nkw is :',eachNKW
    for eachNW in nw:
        print 'nk is :',eachNW,':',nw[eachNW]


newfoo('wolf',3,'projects',freud=90,gamnle=96)
newfoo(10,20,30,40,foo=50,bar=60)
newfoo(2,4,*(6,8),**{'foo':10,'bar':12})
a = (1,2,3)
b={'hello':4,'nihao':5,'hehe':6}
newfoo(11,12,13,x=14,y=15,*a,**b)