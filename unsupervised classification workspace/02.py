"""now let's take an FFT of it

  whichever component captured the repeating sound
  should show energy at ~4Hz and harmonics of ~4Hz

  next up
  -> cepstrum, and other fundamental frequency methods

"""

# spectrum of the entire scene (all n sources)
ALL = np.fft.rfft(est.labels_, n=4096) # FFT
ALL = np.abs(ALL) # magnitude spectrum

# separate array for each source 1 when active 0 everywhere else
sources = np.zeros((n_clusters, est.labels_.shape[0]))
for i,source in enumerate(sources): source[est.labels_==i] = 1.0

# spectrum source by source 
SOURCES = np.array([np.fft.rfft(source, n=4096) for source in sources]) # FFT
SOURCES = np.abs(SOURCES) # magnitude spectrum

# plot combined spectrum
fig = plt.figure()
ax=plt.subplot(n_clusters+1,1,1)
ax.plot(ALL)

# plot individual source spectrums
for i,SOURCE in enumerate(SOURCES):
    ax=plt.subplot(n_clusters+1,1,2+i)
    ax.plot(SOURCE)

# formatting
titles = ['All'] + ["{0}".format(x) for x in range(n_clusters)]
for i,ax in enumerate(fig.get_axes()):
    ax.set_yticks([ax.get_ylim()[1]/2])
    ax.tick_params(axis='y', which='both',length=0)
    ax.set_yticklabels(titles[i])
    # ax.set_xlim([0,40]) # plot the first 40 bins
    if ax != fig.get_axes()[-1]: ax.set_xticklabels([])
plt.show()
