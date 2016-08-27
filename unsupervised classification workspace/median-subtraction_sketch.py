"""noise reduction via spectral median subtraction
 
 median is calculated on a sliding window (10 sec) and 
 negative values are zeroed.

 this is just a sketch -- needs to be built into a function
 for general use

 todo
 * reduction (in db)
 * threshold
 * measure resulting signal to noise ratio
 * process stereo files

"""

import librosa
import scikits.audiolab

# params
FERP_audio_folder = "../../FERP/audio"
filename = '/selected/FERP_16-05-25_04_selection.wav' 
start, end = 0.0, 35.0

# feature param
feature_params = {'hop_length': 512, 'win_length':2048}

# load audio
y, sr, pcm = scikits.audiolab.wavread("{0}/{1}".format(FERP_audio_folder, filename))

# select region
y = y[start*sr: end*sr]

# sum to mono
y = np.sum(y, axis=1)

# mel spectrum
# X = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128) # Mel Spectrum
X = np.abs(librosa.stft(y, **feature_params))

# let's filter out below ~500hz (ONLY w/ FFT NOT MEL)
X[0:24,:] = np.zeros_like(X[0:24,:])

# params
win = 600 # 600 frames is ~7 seconds

# make space for processed data goes here
out = np.zeros_like(X)
noise = np.zeros_like(X)

# loop through frames
for frame in range(X.shape[1]):

    # select region sliding window
    region = X[:,max(frame-win/2,0): min(frame+win/2+1,X.shape[1])]

    # compute mean
    mean = np.mean(region, axis=1)

    # subtract mean
    out[:,frame] = X[:,frame] - mean 

    # store noise
    noise[:,frame] = mean

# zero negative values
out[out<0] = 0.0

# plot spectrum
plt.figure()
librosa.display.specshow(librosa.logamplitude(X), sr=sr, y_axis='linear')
plt.title('before')
plt.show()

# plot noise reduced spectrogram
plt.figure()
librosa.display.specshow(librosa.logamplitude(out), sr=sr, y_axis='linear')
plt.title('signal')
plt.show()

# plot mean / noise
plt.figure()
librosa.display.specshow(librosa.logamplitude(noise), sr=sr, y_axis='linear')
plt.title('noise')
plt.show()

# inverse synthesis (ONLY w/ FFT not MEL)
y_hat = librosa.istft(out, **feature_params)

# write to disk
scikits.audiolab.wavwrite(y_hat, 'letseee.wav', fs=sr)
