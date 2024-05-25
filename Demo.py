# Don't use now
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import serial
import time
import scipy as sp
import tkinter
from PIL import ImageTk, Image

if serial.Serial:
    serial.Serial().close()
# Open the serial port
s = serial.Serial("COM3", baudrate=57600)  # COMx in window or /dev/ttyACMx in Ubuntu with x is number of serial port.
# path = r"Data_Iso\Subject_1_15Hz.txt"
# file = open(path, "a")

x = 0  # iterator of sample
y = np.array([], dtype=int)  # value
time_rec = 20
k = 15 * 512  # window
window_size = k
noise = []
feature = []    

print("START!")
while x < (60 * 512):
    try:
        if x % 512 == 0:
            print(x // 512)
        x += 1
        data = s.readline().decode('utf-8').rstrip("\r\n")
        y = np.append(y, int(data))
        if x >= window_size:
            if x % (1 * 512) == 0:
                sliding_window_start = x - window_size
                sliding_window_end = x
                sliding_window = np.array(y[sliding_window_start:sliding_window_end])  # sliding_window ~ y
                print(len(sliding_window))
                # preprocess
                # slide = filter_data(slide)
                feature_window = FeatureExtract(sliding_window, plot=1)
                image = Image.open("test.png")
                image.show()
                image.close()
                # img = mpimg.imread('test.png')
                # imgplot = plt.imshow(img)
                # plt.pause(0.001)
                # plt.close()
                feature.append(feature_window)

            # plot
            # AI
            # file.write(data)
            # file.write('\n')
    except:
        pass

# Use to get data txt
# np.savetxt("Subject_1_momat.txt", y, fmt="%d")  # Save in int

# Close the serial port
print("DONE")
s.close()
# file.close()
