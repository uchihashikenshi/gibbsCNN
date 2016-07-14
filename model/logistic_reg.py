import chainer 
import chainer.functions as F

class logistic_r(chainer.FunctionSet):

    """
    1 layer NN.  
    """

    def __init__(self,patchsize):
        super(logistic_r,self).__init__(
        fcf = F.Linear(patchsize**2, 2)
        )


    def forward(self, x_data, y_data, train = True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        h = self.fcf(x)
	pred = F.softmax(h)
        return F.softmax_cross_entropy(h,t), F.accuracy(h,t), pred
