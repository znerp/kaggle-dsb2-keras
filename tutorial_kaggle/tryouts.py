def testfunction(a,b):
    """Unnecessary docstring."""
    if a>1:
        return a*b, a+b
    else:
        return a*b

ret1 = testfunction(1,3)
ret3,ret4 = testfunction(2,3)

#print(ret1, ret2)
#print(ret3)