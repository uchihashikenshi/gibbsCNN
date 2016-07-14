import chainer 
import chainer.functions as F
import numpy as np

class convnet_cdd(chainer.FunctionSet):
    
    """First trial of cnn on the images
    patchsize is assumed to be odd 
    Make the poolsize equal to stride
    """

    
    def __init__(self,patchsize):
        
        #patchsize = 15
        channel1 = 20.
        filtsize1 = 5. #3
        padsize1 = 0.
        poolsize1 = 1.
        stride1 = 1.

        channel2 = 50.
        filtsize2 = 5. #3
        padsize2 = 0.
        poolsize2= 1.
        stride2 = 1.

        num_categ = 2
        size1 = patchsize - (filtsize1 -1)+ padsize1 *2.
        size2 = np.ceil(size1/poolsize1)
        size3 = size2 - (filtsize2 -1)+ padsize2 *2.
        size4 = np.ceil(size3/poolsize2)
        sizef = size4**2*channel2            
        print 'final feature map dimension before the softmax %s' %str(size4)
     
        super(convnet_cdd,self).__init__(       
            conv1 = F.Convolution2D(2,int(channel1), int(filtsize1)),
            conv2 = F.Convolution2D(int(channel1),int(channel2),int(filtsize2),pad=0),
            fc3 = F.Linear(int(size4)**2*int(channel2),int(size4)**2*int(channel2)),
            fcf = F.Linear(int(size4)**2*int(channel2), num_categ)
        )

    def forward(self, x_data, y_data, train = True):
        
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)
        h = F.relu(self.conv1(x))
        h = F.dropout(F.relu(self.conv2(h)), train=train)
        h = F.dropout(F.relu(self.fc3(h)),train=train)      
        h = self.fcf(h)
        pred = F.softmax(h)
        
        return F.softmax_cross_entropy(h,t), F.accuracy(h,t), pred
