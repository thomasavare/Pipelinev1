#!/usr/bin/env python3

import sounddevice as sd
import soundfile as sf


if __name__ == "__main__":
    sr = 16000
    seconds = 5
    # sd.default.device = 0
    # print(sd.query_devices())

    my_recording = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
    sd.wait()
    sf.write('output.wav', my_recording, sr)
