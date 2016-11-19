

import pickle

lista=["mingyue","jishi","you"]
listb=pickle.dumps(lista)
print listb

listc=pickle.loads(listb)
print listc