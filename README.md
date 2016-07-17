# gibbsCNN
All images have contextual information, which has been widely used as a rich source to segment multiple objects. A lot of models proposes its own model constructions for segmentation tasks, and gibbsCNN is one of these. gibbsCNN is a contextual model uses the relationships between the objects in a scene to facilitate object detection and segmentation. 

The contextual information is interpreted as intra-object configurations and inter-object relationships. For example, shape contexts are rich descriptions that have been used widely for solving high-level vision problems, and in the field of neuroscience, the existence of mitochondria strongly suggests that there is very likely cell membrane near it. To be precise, by contextual informations, we can refer to the probability map of the target object can be used as prior information together with the original image information to solve the MAP pixel-level classification problem.

The pixel-level classification is the problem of assigning an object label to each pixels. There have been many methods that employ context for solving vision problems such as image segmentation. This work proposes new pixel-level classification model which uses two convolutional neural networks and gibbs sampling. 

![modelsetups](https://github.com/uchihashikenshi/gibbsCNN/blob/master/images/modelsetups)
![modelsetups2](https://github.com/uchihashikenshi/gibbsCNN/blob/master/images/modelsetups2)
![pseude_codes](https://github.com/uchihashikenshi/gibbsCNN/blob/master/images/pseude_codes)
