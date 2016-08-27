"""unsupervised classification:

  real quick and simple let's run a FERP clip through a k-means classifier

  next up
  -> regression analysis for mixed sources
  -> variable / thresholded n_clusters
  -> segment width and hop rather than frame-to-frame
  -> object function cosine distance on mel frequency spectrum
  -> add panogram data (need stereo input)

 """

import numpy as np
import librosa
import sklearn
import matplotlib.pyplot as plt
import scikits.audiolab

# params
FERP_audio_folder = "../../FERP/audio"
filename = '/selected/FERP_08_clip.wav'
start, end = 0.0, 35.0
n_clusters = 5

# load audio
y, sr, pcm = scikits.audiolab.wavread("{0}/{1}".format(FERP_audio_folder, filename))

# select region
y = y[start*sr: end*sr]

# features
X = librosa.feature.mfcc(y=y, sr=sr) # MFCCs
# X = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128) # Mel Spectrum

# cluster
est = sklearn.cluster.KMeans(n_clusters=n_clusters)
est.fit(X.T)

# plot
plt.figure()
plt.subplot(2,1,1)
plt.scatter(np.arange(est.labels_.shape[0]), est.labels_)
plt.xlim([0, est.labels_.shape[0]])
plt.subplot(2,1,2)
plt.plot(y, c='grey')
plt.xlim([0, y.shape[0]])
plt.show()
