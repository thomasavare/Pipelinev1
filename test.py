#!/usr/bin/env python

# Does not work: sounddevice.PortAudioError: Error opening InputStream: Invalid number of channels [PaErrorCode -9998]
# Can't resolve it

import sounddevice as sd
from scipy.io.wavfile import write

if __name__ == "__main__":
    sr = 44100
    seconds = 3

    myrecording = sd.rec(int(seconds * sr), samplerate=sr, channels=3)
    sd.wait()  # Wait until recording is finished
    write('output.wav', sr, myrecording)  # Save as WAV file