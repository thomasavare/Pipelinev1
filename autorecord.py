import queue
from time import sleep

import sounddevice as sd
import soundfile as sf
import numpy as np

from asr_whisper import asr_recording


q = queue.Queue()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


def autorecord(sr, device, blocksize):

    var = 1
    record = np.array([])
    with sd.InputStream(samplerate=sr, device=device,
                        channels=1, callback=callback, blocksize=blocksize):
        print('#' * 80)
        print('record start in 1 sec, Ctrl+C to stop manually, ')
        print('#' * 80)
        sleep(1)
        print("starting")
        print('#' * 80)
        # TODO: modify stop condition for background noise
        # idea: 2 blocks per seconds, comparing the mean of the two blocks and the variance, maybe try another
        #       statistic indicator or apply fft to reduce noise if very fast
        # idea 2: computation of fft in another thread to not lose time before next block
        while var >= 0.0005 and record.shape[0] < sr * 9:
            processed = q.get()
            var = np.var(processed)
            print(var)
            if not len(record) > 0:
                record = np.array(processed)
            record = np.vstack((record, processed))
        print("ending")
        print('#' * 80)

    return record.T[0]

def autorecord_save(sr, device, blocksize, filename):
    var = 1
    record = np.array([])
    with sf.SoundFile(filename, mode='w', samplerate=sr,
                      channels=1) as file:
        with sd.InputStream(samplerate=sr, device=device,
                            channels=1, callback=callback, blocksize=blocksize):
            print('#' * 80)
            print('record start in 1 sec, Ctrl+C to stop manually, ')
            print('#' * 80)
            sleep(1)
            print("starting")
            print('#' * 80)
            # TODO: modify stop condition for background noise
            # idea: 2 blocks per seconds, comparing the mean of the two blocks and the variance, maybe try another
            #       statistic indicator or apply fft to reduce noise if very fast
            # idea 2: computation of fft in another thread to not lose time before next block
            while var >= 0.001 and record.shape[0] <= sr * 9:
                processed = q.get()
                file.write(processed)
                var = np.var(processed)
                print(var)
                if not len(record) > 0:
                    record = np.array(processed)
                record = np.vstack((record, processed))
    print("ending")
    print('#' * 80)

    return record.T[0]