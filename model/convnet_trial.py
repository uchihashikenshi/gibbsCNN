import chainer 
import chainer.functions as F
import numpy as np

class convnet_trial(chainer.FunctionSet):
    
    """First trial of cnn on the images
    patchsize is assumed to be odd 
    Make the poolsize equal to stride
    """

    
    def __init__(self,patchsize):
        
        channel1 = 8.
        filtsize1 = 6. #3
        padsize1 = 0.
        poolsize1 = 2.
        stride1 = 2.

        channel2 = 16.
        filtsize2 = 5. #3
        padsize2 = 0.
        poolsize2= 2.
        stride2 = 2.

        num_categ = 2
        size1 = patchsize - (filtsize1 -1)+ padsize1 *2.
        size2 = np.ceil(size1/poolsize1)
        size3 = size2 - (filtsize2 -1)+ padsize2 *2.
        size4 = np.ceil(size3/poolsize2)
        sizef = size4**2*channel2            
        #print 'final feature map dimension before the softmax %s' %str(size4)
     
        super(convnet_trial,self).__init__(       
            conv1 = F.Convolution2D(1,int(channel1), int(filtsize1)),
            conv2 = F.Convolution2D(int(channel1),int(channel2),int(filtsize2),pad=0),
            fc3 = F.Linear(int(size4)**2*int(channel2),int(size4)**2*int(channel2)),
            fcf = F.Linear(int(size4)**2*int(channel2), num_categ)
        )

    def forward(self, x_data, y_data, train = True):
        

        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), (2,2), stride=2)
        h = F.dropout(F.max_pooling_2d(F.relu(self.conv2(h)), (2,2), stride=2),train=train)
        h = F.dropout(F.relu(self.fc3(h)),train=train)

        h = self.fcf(h)
        pred = F.softmax(h)
        return F.softmax_cross_entropy(h,t), F.accuracy(h,t), pred
