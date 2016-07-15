#from utils import Preprocessing
from utils import preprocessing_past as prep

p = prep.Preprocessing_python2()

cropsize = input('Cropsize? Enter an odd integer\n')
p.patch_extract("preprocessed/training/pooled_training_dataset", "raw/train-labels", "256_", stride = 3, crop_size = cropsize) 
