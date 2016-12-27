import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

def test():
	input = np.loadtxt(open("data/"+stock_name+"intest.csv","rb"),delimiter=",")
	output = np.loadtxt(open("data/"+stock_name+"outtest.csv","rb"),delimiter=",")
	output= output.reshape((output.shape[0],1))
	error =[]
	for value1,value2 in zip(input,output):
	    temp =np.amax(value1)
	    value1/=temp
	    f = nonlin(np.dot(value1,syn0))
	    g = nonlin(np.dot(f,syn1))
	    error.append(abs(value2-g*temp))

	error_avg = float(sum(error) / float(len(error)))
	print "Average Error in predication is = $",error_avg



stock_name = "moto" #gg=google,a=apple,moto=motrola

X =np.loadtxt(open("data/"+stock_name+"in.csv","rb"),delimiter=",")
y = np.loadtxt(open("data/"+stock_name+"out.csv","rb"),delimiter=",")
y= y.reshape((y.shape[0],1))
x_max = np.amax(X)
y_max = np.amax(y)
X/= x_max
y/= y_max

np.random.seed(1)


syn0 = 2*np.random.random((X.shape[1],X.shape[0])) - 1
syn1 = 2*np.random.random((y.shape[0],1)) - 1
flag = True
print "Traning Start"
for j in xrange(5000):

	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))
	l2_error = y - l2

	# if (j% 1000) == 0:
	# 	print "Error:" + str(np.mean(np.abs(l2_error)))

	l2_delta = l2_error*nonlin(l2,deriv=True)

	l1_error = l2_delta.dot(syn1.T)

	l1_delta = l1_error * nonlin(l1,deriv=True)
	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)
print "Traning end"
test()
