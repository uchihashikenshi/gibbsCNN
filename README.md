# gibbsCNN
All images have contextual information, which has been widely used as a rich source to segment multiple objects. A lot of models proposes its own model constructions for segmentation tasks, and gibbsCNN is one of these. gibbsCNN is a contextual model uses the relationships between the objects in a scene to facilitate object detection and segmentation. 

The contextual information is interpreted as intra-object configurations and inter-object relationships. For example, shape contexts are rich descriptions that have been used widely for solving high-level vision problems, and in the field of neuroscience, the existence of mitochondria strongly suggests that there is very likely cell membrane near it. To be precise, by contextual informations, we can refer to the probability map of the target object can be used as prior information together with the original image information to solve the MAP pixel-level classification problem.

The pixel-level classification is the problem of assigning an object label to each pixels. There have been many methods that employ context for solving vision problems such as image segmentation. This work proposes new pixel-level classification model which uses two convolutional neural networks and gibbs sampling. 

![modelsetups](https://github.com/uchihashikenshi/gibbsCNN/blob/master/images/modelsetups)


Pixel classification is the
problem of assigning an object label to each pixel.
There have been many methods that employ context for
solving vision problems such as image segmentation or
Manuscript received December 7, 2012; revised March 29, 2013; accepted
July 11, 2013. Date of publication July 23, 2013; date of current version
September 17, 2013. This work was supported in part by the NIH under Grant
1R01NS075314-01 and the NSF under Grant IIS-1149299. The associate
editor coordinating the review of this manuscript and approving it for
publication was Prof. Marios S. Pattichis.
The authors are with the Electrical and Computer Engineering Department
and the Scientific Computing and Imaging Institute, University of
Utah, Salt Lake City, UT 84112 USA (e-mail: mseyed@sci.utah.edu;
tolga@sci.utah.edu).
Color versions of one or more of the figures in this paper are available
online at http://ieeexplore.ieee.org.
Digital Object Identifier 10.1109/TIP.2013.2274388
image classification. Markov random fields (MRF) [7]
is one of the earliest and most widespread approaches.
Lafferty et al. [8] showed that better results for discrimination
problems can be obtained by modeling the conditional probability
of labels given an observation sequence directly. This
non-generative approach is called the conditional random field
(CRF). He et al. [9] generalized the CRF approach for the pixel
classification problem by learning features at different scales
of the image. Jain et al. [10] showed MRF and CRF algorithms
perform about the same as simple thresholding in pixel
classification for binary-like images. They proposed a new
single-scale version of the convolutional neural network [11]
strategy for restoring membranes in electron microscopic (EM)
images. Compared to other methods, convolutional networks
take advantage of context information from larger regions, but
need many hidden layers. In their model the back propagation
has to go over multiple hidden layers for the training, which
makes the training step computationally expensive. Tu and
Bai [2] proposed the auto-context algorithm which integrates
the original image features together with the contextual information
by learning a series of classifiers. Similar to CRF,
auto-context targets the posterior distribution directly without
splitting it to likelihood and prior distributions. The advantage
of auto-context over convolutional networks is its easier
training due to treating each classifier in the series one at
a time in sequential order. Although they used probabilistic
boosting tree as classifier (PBT), auto-context is not restricted
to any particular classifier and different type of classifiers can
be used. Jurrus et al. [12] employed artificial neural networks
(ANN) in a series classifier structure which learns a set of
convolutional filters from the data instead of applying large
filter banks to the input image.
Even though all the aforementioned approaches use contextual
information together with the input image information
to improve the accuracy of the achieved segmentation, they
do not take contextual information from multiple objects
into account and thus are not able to capture dependencies
between the objects. Torralba et al. [6] introduced boosted
random field (BRF) which uses boosting to learn the graph
structure of CRFs for multi-class object detection and region
labeling. Desai et al. [13] proposed a discriminative model
for multi-class object recognition that can learn intra-class
relationships between different categories. The cascaded classification
model [14] is a scene understanding framework that
combines object detection, multi-class segmentation, and 3D
reconstruction. Choi et al. [15] introduced a


— Contextual information has been widely used as a
rich source of information to segment multiple objects in an
image. A contextual model uses the relationships between the
objects in a scene to facilitate object detection and segmentation.
Using contextual information from different objects in an effective
way for object segmentation, however, remains a difficult
problem. In this paper, we introduce a novel framework, called
multiclass multiscale (MCMS) series contextual model, which
uses contextual information from multiple objects and at different
scales for learning discriminative models in a supervised setting.
The MCMS model incorporates cross-object and inter-object
information into one probabilistic framework and thus is able
to capture geometrical relationships and dependencies among
multiple objects in addition to local information from each single
object present in an image. We demonstrate that our MCMS
model improves object segmentation performance in electron
microscopy images and provides a coherent segmentation of
multiple objects. Through speeding up the segmentation process,
the proposed method will allow neurobiologists to move beyond
individual specimens and analyze populations paving the way
for understanding neurodegenerative diseases at the microscopic
level.
Index Terms— Image segme
