import mne
import numpy as np
from scipy.io import wavfile
import pandas as pd
from sklearn.cluster import KMeans

def filter_small_changes(sig, N=20):
    sig[int( N /2):-int( N /2)] = np.stack([sig[i:-( N -i)] for i in np.arange(N)]).sum(axis=0) > N/ 2
    return sig


def nb_outliers(data, m=2.):
    return np.sum(abs(data - np.mean(data)) > m * np.std(data))


def blocked_stft(data, wsize, nb_wsize_max=50000, verbose=True):
  tfs = []
  for offset in np.arange(0, len(data), nb_wsize_max*wsize):
    if verbose:
      print(f"Offset {offset} samples")

    tf = mne.time_frequency.stft(data[offset:offset+nb_wsize_max*wsize], wsize=wsize)
    tfs.append(np.abs(tf.squeeze()))

  return np.concatenate(tfs, axis=1)


def get_tone_times(file_name, tone_freq=2000, wsize=2500, filter_N=20, threshold=None, verbose=True):
    sfreq, data = wavfile.read(file_name)

    dt = wsize / sfreq / 2
    tf = blocked_stft(data, wsize, verbose=verbose)
    freqs = mne.time_frequency.stftfreq(wsize=wsize, sfreq=sfreq)
    #time = np.arange(0, dt * tf.shape[1], dt)

    if threshold is None:
      ecg = tf[freqs == tone_freq].squeeze()
      mask = KMeans(n_clusters=2).fit_transform(ecg[:, None]).argmax(1) == 0
      threshold = ecg[mask].min() if ecg[mask].mean() > ecg[~mask].mean() else ecg[~mask].min()

    thresholded_sig = tf[freqs == tone_freq].squeeze() > threshold

    filtered_sig = filter_small_changes(thresholded_sig, N=filter_N)

    up = np.where(np.diff(filtered_sig.astype(int)) == 1)[0]
    down = np.where(np.diff(filtered_sig.astype(int)) == -1)[0]
    assert (len(up) == len(down))
    tone_durations = (down - up) * dt
    trial_durations = (up[1:] - up[:-1]) * dt
    if verbose:
        print(f"Number of tones: {len(up)}")
        print(
            f"Tone duration: {np.round(tone_durations.mean(), 2)} +/- {np.round(tone_durations.std(), 2)} [{nb_outliers(tone_durations)} outliers]")
        print(
            f"Trial duration: {np.round(trial_durations.mean(), 2)} +/- {np.round(trial_durations.std(), 2)} [{nb_outliers(trial_durations)} outliers]")
        print("Tone timing:")
        for u, d in zip(up, down):
            print(f"[{np.round(u * dt, 2)}, {np.round(d * dt, 2)}] s")

    return pd.DataFrame({'onset': up * dt, 'offset': down * dt})
