# -*- coding: utf-8 -*-
import numpy
import scipy.io.wavfile
import os
import sys
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import sounddevice as sd

# Audio Path configuration
VOICE_SAMPLES_TRAINING_PATH = "../../ELSDSR/train/"
VOICE_SAMPLES_TEST_PATH = "../../ELSDSR/test/"
# Emphasis configuration
USE_PRE_EMPHASIS = True
PRE_EMPHASIS = 0.97
# Silence configuration
REMOVE_SILENCE = True
THRESHOLD = 0;
# Framing configuration
FRAME_SIZE = 0.025
FRAME_STRIDE = 0.01
# FFT configuration
NFFT = 512
# Filterbank configuration
FILTER_NUMBER = 40
# MFCCs configuration
CEPSTRUM_NUMBER = 12

def read_audio_file(filename):
    sample_rate, signal = scipy.io.wavfile.read(filename)
    ### Apply pre emphasis
    if (USE_PRE_EMPHASIS):
        signal = numpy.append(signal[0], signal[1:] - PRE_EMPHASIS * signal[:-1])
    return sample_rate, signal

def framing(sample_rate, signal):
    frame_length, frame_step = FRAME_SIZE * sample_rate, FRAME_STRIDE * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    return frame_length, frames

def hamming(frame_length, frames):
    return frames * numpy.hamming(frame_length)

def FFT(frames):
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    return pow_frames

def mfcc(sample_rate, pow_frames):
    ### FilterBank
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, FILTER_NUMBER + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((FILTER_NUMBER, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, FILTER_NUMBER + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m): 
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB
    ### DCT
    mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (CEPSTRUM_NUMBER + 1)] # Keep 2-13
    ### Normalize
    mfccs -= (numpy.mean(mfccs, axis=0) + 1e-8) 
    return mfccs

def extract_features(filename):
    sample_rate,signal = read_audio_file(filename)
    frame_length,frames = framing(sample_rate, signal)
    frames = hamming(frame_length, frames)
    pow_frames = FFT(frames)
    mfccs = mfcc(sample_rate, pow_frames)
    return mfccs

for file in [doc for doc in os.listdir(VOICE_SAMPLES_TRAINING_PATH) 
if doc.endswith(".wav")]:
    mfccs = extract_features(VOICE_SAMPLES_TRAINING_PATH+file)
    file = file.split('.')
    filename_out = (VOICE_SAMPLES_TRAINING_PATH + "coefficients/" 
                    + file[0] + ".out")
    numpy.savetxt(filename_out,mfccs)

for file in [doc for doc in os.listdir(VOICE_SAMPLES_TEST_PATH) 
if doc.endswith(".wav")]:
    mfccs = extract_features(VOICE_SAMPLES_TEST_PATH+file)
    file = file.split('.')
    filename_out = (VOICE_SAMPLES_TEST_PATH + "coefficients/" 
                    + file[0] + ".out")    
    numpy.savetxt(filename_out,mfccs)

#mfccsfile = numpy.loadtxt('test.out')
### Plot audio
#plt.figure(1)
#plt.title('Signal Wave...')
#plt.plot(signal)
#plt.show()
    
### Play audio
#sd.play(signal,sample_rate)