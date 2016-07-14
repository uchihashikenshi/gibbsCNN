import train_cnn_script as tc
import train_cnn_conditional as tc

cropsize = 15
resolution = 256
epochsize = 80
cooling_rate = 0.95
initial_temperature = 1.5
boringsize = 2 

mycnn_cdd = tc.train_cnn_cdd(cropsize=cropsize,
                             resolution=resolution)
mycnn_cdd.train(epochsize=epochsize,
                cooling_rate=cooling_rate,
                initial_temperature=initial_temperature,
                boringsize=boringsize)
