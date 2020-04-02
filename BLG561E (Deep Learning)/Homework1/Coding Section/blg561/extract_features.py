#!/usr/bin/env python3

# FMA: A Dataset For Music Analysis
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.

# All features are extracted using [librosa](https://github.com/librosa/librosa).
# Alternatives:
# * [MARSYAS](https://github.com/marsyas/marsyas) (C++ with Python bindings)
# * [RP extract](http://www.ifs.tuwien.ac.at/mir/downloads.html) (Matlab, Java, Python)
# * [jMIR jAudio](http://jmir.sourceforge.net) (Java)
# * [MIRtoolbox](https://www.jyu.fi/hum/laitokset/musiikki/en/research/coe/materials/mirtoolbox) (Matlab)

from scipy import stats
import pandas as pd
import numpy as np
import librosa

def extract_features(x,sr):
    def columns():
        feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                             tonnetz=6, mfcc=20, zcr=1,
                             spectral_centroid=1, spectral_bandwidth=1,
                             spectral_contrast=7, spectral_rolloff=1)
        moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

        columns = []
        for name, size in feature_sizes.items():
            for moment in moments:
                it = ((name, moment, '{:02d}'.format(i + 1)) for i in range(size))
                columns.extend(it)

        names = ('feature', 'statistics', 'number')
        columns = pd.MultiIndex.from_tuples(columns, names=names)

        # More efficient to slice if indexes are sorted.
        return columns.sort_values()

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    features = pd.Series(index=columns(), dtype=np.float32, name=1)

    f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
    feature_stats('zcr', f)

    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                             n_bins=7 * 12, tuning=None))
    assert cqt.shape[0] == 7 * 12
    assert np.ceil(len(x) / 512) <= cqt.shape[1] <= np.ceil(len(x) / 512) + 1

    f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cqt', f)
    f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cens', f)
    f = librosa.feature.tonnetz(chroma=f)
    feature_stats('tonnetz', f)

    del cqt
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
    del x

    f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)
    feature_stats('chroma_stft', f)

    # f = librosa.feature.rmse(S=stft)
    # feature_stats('rmse', f)

    f = librosa.feature.spectral_centroid(S=stft)
    feature_stats('spectral_centroid', f)
    f = librosa.feature.spectral_bandwidth(S=stft)
    feature_stats('spectral_bandwidth', f)
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    feature_stats('spectral_contrast', f)
    f = librosa.feature.spectral_rolloff(S=stft)
    feature_stats('spectral_rolloff', f)

    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    del stft
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    feature_stats('mfcc', f)

    return features
