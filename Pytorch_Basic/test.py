import numpy as np


# %%
def myfunc(a, b):
    "Return a-b if a>b, otherwise return a+b"
    if a > b:
        return a - b
    else:
        return a + b


vfunc = np.vectorize(myfunc, doc='Vectorized `myfunc`')
output1 = vfunc([1, 2, 3, 4], 2)
print(output1)


# %%
def mypolyval(p, x):
    _p = list(p)
    res = _p.pop(0)
    while _p:
        res = res * x + _p.pop(0)
    return res


# print(mypolyval([1, 2, 3, 4], 2))
vpolyval = np.vectorize(mypolyval, excluded=['p'])
# vpolyval = np.vectorize(mypolyval)

output2 = vpolyval(p=[1, 2, 3], x=[0, 1])
print(output2)