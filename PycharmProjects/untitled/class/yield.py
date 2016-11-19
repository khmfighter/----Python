def testyield():
    print "please input your answer"
    while True:
        answer = (yield)
        if  answer is not None:
            if answer.endswith('?'):
                print 'Dont ask yourseflf more'
            elif 'good' in answer:
                print 'this is a good ,go on'
            elif 'bad' in answer:
                print 'this is a bad , stop it'

free = testyield()
free.send('i feel bad')

