# coding:utf-8
from PIL import Image
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt

from theano.tensor.signal import downsample
import theano.tensor as T
import theano

memCNN_home = os.getcwd() # projects/memCNNから走らせる予定

# ✕train ◯training

"""
Annotation
data_type: trainingとかtestとか
data_dir: データを入れてるディレクトリの名前。data/raw/(data_type)/まではprefix

ディレクトリ構造:
data---raw--------------------train-input
     |     |------------------test-input
     |     |------------------train-labels
     |-preprocessed_dataset---training------median_extract_training_dataset
     |                      |          |----pooled_training_dataset
     |                      |-test----------median_extract_test_dataset
     |                                 |----pooled_test_dataset
     |-training_dataset
     |-test-dataset
     |-lmdb
"""
class Preprocessing_python_rcstr(object):
    def __init__(self):
        pass

    def load_images(self, data_dir):
        os.chdir("data/%s" % data_dir)
        filelist = glob.glob('*.tif') # とりあえずtifのみ対応
        os.chdir("%s" % memCNN_home)
        return filelist

    def image_to_array(self, file):
        raw_image = Image.open(file)
        raw_matrix = np.array(list(raw_image.getdata())).reshape(1024, 1024)
        return raw_matrix

    def make_median_extracted_dataset(self, data_type, data_dir):
        filelist = self.load_images(data_dir)

        if os.path.exists("%s/data/preprocessed/%s/median_extract_%s_dataset" % (memCNN_home, data_type, data_type)) != True:
            os.mkdir("%s/data/preprocessed/%s/median_extract_%s_dataset" % (memCNN_home, data_type, data_type)) # データ置き場用意

        # スタック中全画像からmedianを求める(medianの平均値)
        # fixme: medianの平均でいいのか・・・？
        N, _sum = 0, 0
        for file in filelist:
            raw_matrix = self.image_to_array("data/%s/%s" % (data_dir, file))
            median = np.median(raw_matrix)
            _sum += median
            N += 1
        stack_median = _sum / N

        file_num = 1
        for file in filelist:
            raw_matrix = self.image_to_array("data/%s/%s" % (data_dir, file))
            median = np.median(raw_matrix) #中央値
            # スタックのmedianに各画像のmedianを合わせる
            median_extract_matrix = (raw_matrix - (median - stack_median))

            # 負の画素値を0に補正
            # fixme: こんな処理を入れずにスマートにやりたい
            for i in range(1024):
                for j in range(1024):
                    if median_extract_matrix[i][j] < 0:
                        median_extract_matrix[i][j] = 0

            median_extract_image = Image.fromarray(np.uint8(median_extract_matrix).reshape(1024, 1024))
            median_extract_image.save("%s/data/preprocessed/%s/median_extract_%s_dataset/median_extract_image_%03d.tif" % (memCNN_home, data_type, data_type, file_num))
            file_num += 1
            if file_num % 10 == 0:
                print "%s images ended" % file_num
        print "median_extract_%s_dataset is created." % data_type

    def make_average_pooled_dataset(self, data_type, data_dir):
        filelist = self.load_images(data_dir)

        if os.path.exists("%s/data/preprocessed/%s/pooled_%s_dataset" % (memCNN_home, data_type, data_type)) != True:
            os.mkdir("%s/data/preprocessed/%s/pooled_%s_dataset" % (memCNN_home, data_type, data_type)) # データ置き場用意

        file_num = 1
        for file in filelist:
            raw_matrix = self.image_to_array("data/%s/%s" % (data_dir, file))
            pooled_matrix = []
            for i in range(int(1024 / 4)):
                for j in range(int(1024 / 4)):
                    _sum = 0
                    for k in range(4):
                        for l in range(4):
                            _sum += raw_matrix[4 * i + k, 4 * j + l]
                    pooled_pixel = _sum / 16
                    pooled_matrix.append(pooled_pixel)
            pooled_image = Image.fromarray(np.uint8(pooled_matrix).reshape(256, 256))
            pooled_image.save("%s/data/preprocessed/%s/pooled_%s_dataset/pooled_image_%03d.tif" % (memCNN_home, data_type, data_type, file_num))
            file_num += 1
            if file_num % 10 == 0:
                print "%s images ended" % file_num
        print "pooled_%s_dataset is created." % data_type

    def patch_extract(self, data_dir, label_data_dir, prefix = "", image_size = 256, crop_size = 33, stride = 5):
        """
        1 stackをtraining 80枚、test20枚に分ける
        """
        filelist = self.load_images(data_dir)
        labellist = self.load_images(label_data_dir)

        center = (crop_size - 1) / 2
            
        reconstruction_path = "%s/data/test_dataset/reconstruction_All_%sdataset_crop%s/" % (memCNN_home, prefix, crop_size)
        label_reconstruction_path = "%s/data/test_dataset/label_reconstruction_All_%sdataset_crop%s/" % (memCNN_home, prefix, crop_size)
        
        
        # reconstruction test dataset作成
        if os.path.exists(reconstruction_path) != True:
            os.mkdir(reconstruction_path) # Reconstructionデータ置き場用意
        else:
            # 既にdatasetが存在すれば警告
            overwrite_rcst= input("The filepath\n" + reconstruction_path + "\n already exists. Ovewrite? type 'yes' or 'no' with  'sign. \n" )
            if overwrite_rcst != 'yes':
                print('preprocess of reconstuction data aborted')
                return
        if os.path.exists(label_reconstruction_path) != True:
            os.mkdir(reconstruction_path) # Reconstructionデータ置き場用意
        else:
            # 既にdatasetが存在すれば警告
            overwrite_rcst= input("The filepath\n" + label_reconstruction_path + "\n already exists. Ovewrite? type 'yes' or 'no' with  'sign. \n" )
            if overwrite_rcst != 'yes':
                print('preprocess of label reconsctution data aborted')
                return


        file_index = 1
        
        ans_0_number, ans_1_number = (0, 0)
        for file, label in zip(filelist, labellist):
            # trainig, testのデータベース作成用txt(名前は全てtest.txt)
            rcstr_txtfilepath = reconstruction_path + 'test%03d.txt' %file_index

            # reconstruction testのデータベース作成用txt(名前は全てtraining.txt, test.txt)
            reconstruction_test_f = open(rcstr_txtfilepath, 'w')           
            for h in xrange(int(image_size - crop_size)):
                for w in xrange(int(image_size - crop_size)):
                    # 画像のサイズを指定
                    patch_range = (w, h, w + crop_size, h + crop_size)
                    cropped_image = Image.open("%s/data/%s/%s" % (memCNN_home, data_dir, file)).crop(patch_range)
                    cropped_label = Image.open("%s/data/%s/%s" % (memCNN_home, label_data_dir, label)).crop(patch_range)
                    ans = int(np.array(list((cropped_label.getdata()))).reshape((crop_size, crop_size))[center][center] / 255)
                        
                    rcstr_filenameindex = "reconstruction_%stest_image_%03d%03d%03d.tif" %(prefix, file_index, h, w)
                    label_rcstr_filenameindex = "label_reconstruction_%stest_image_%03d%03d%03d.tif" %(prefix, file_index, h, w)
                    
                    rcstr_filepath = reconstruction_path + rcstr_filenameindex  
                    label_rcstr_filepath = label_reconstruction_path + label_rcstr_filenameindex  
                    cropped_image.save(label_rcstr_filepath)                        
                        
                        
                    rcstr_line2write = rcstr_filenameindex + " %s\n" % ans
                    reconstruction_test_f.write(rcstr_line2write)

            if file_index % 10 == 0:
                print "%s images ended" % file_index

            if file_index == 80:
                print "%straining_dataset is created." % prefix
                #debug:
                print ans_0_number
                print ans_1_number
            if file_index == 100:
                print "%stest_dataset is created." % prefix
            file_index += 1
