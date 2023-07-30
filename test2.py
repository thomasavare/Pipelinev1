#!/usr/bin/env python

import queue
import sys
from time import sleep

import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

q = queue.Queue()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


sr, device, blocksize, seconds = 16000, 1, 1600, 10
filename = "output.wav"

record = np.array([])
time, varray, signal = np.zeros(int(sr / blocksize * seconds)), np.zeros(int(sr / blocksize * seconds)), np.zeros(sr * seconds)

with sf.SoundFile(filename, mode='w', samplerate=sr, channels=1) as file:
    with sd.InputStream(samplerate=sr, device=device,
                        channels=1, callback=callback, blocksize=blocksize):
        print('#' * 80)
        print(f'record start in 1 sec and lasts {seconds}')
        print('#' * 80)
        sleep(1)
        print("starting")
        print('#' * 80)
        # TODO: modify stop condition for background noise
        # idea: 2 blocks per seconds, comparing the mean of the two blocks and the variance, maybe try another
        #       statistic indicator or apply fft to reduce noise if very fast
        # idea 2: computation of fft in another thread to not lose time before next block
        for i in range(int(sr / blocksize * seconds)):
            processed = q.get()
            file.write(processed)

            var = np.var(processed)
            print(var, blocksize / sr * i, sep=", ")
            varray[i] = var
            time[i] = blocksize / sr * i

            if not len(record) > 0:
                record = np.array(processed)
            record = np.vstack((record, processed))
        print("ending")
        print('#' * 80)

print("variance mean: ", np.mean(varray))
print("variance max:  ", np.max(varray))
print("variance min:  ", np.min(varray))

plt.subplot(2, 1, 1)
plt.plot(time, varray)
plt.subplot(2, 1, 2)
plt.plot(np.arange(0, seconds, 1/sr), record[:int(sr * seconds)])
plt.show()
