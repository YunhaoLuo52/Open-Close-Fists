# -*- coding: utf-8 -*-

from glob import glob
import os
from collections import OrderedDict

from mne import create_info, concatenate_raws
from mne.io import RawArray
from mne.channels import make_standard_montage
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


sns.set_context('talk')
sns.set_style('white')


import pandas as pd
import mne
from mne import create_info, concatenate_raws
from mne.io import RawArray
from mne.channels import make_standard_montage

import pandas as pd
import mne
from mne import create_info, concatenate_raws
from mne.io import RawArray
from mne.channels import make_standard_montage

def load_muse_csv_as_raw(filename, sfreq=256., ch_ind=[1, 2, 3, 4], stim_ind=6, replace_ch_names=None):
    """Load CSV files into a Raw object.

    Args:
        filename (str or list): path or paths to CSV files to load

    Keyword Args:
        sfreq (float): EEG sampling frequency
        ch_ind (list): indices of the EEG channels to keep (default: TP9, AF7, AF8, TP10)
        stim_ind (int): index of the stim channel (default: Marker0)
        replace_ch_names (dict or None): dictionary containing a mapping to rename channels.

    Returns:
        (mne.io.array.array.RawArray): loaded EEG
    """
    raw = []
    if isinstance(filename, (str, os.PathLike)):
        filename = [filename]
    for fname in filename:
        data = pd.read_csv(fname)

        if 'timestamps' in data.columns:
            data = data.drop(columns=['timestamps']) 

        print(f"Loaded {fname} with shape {data.shape}")
        
        # filter out 999 marker
        data['Marker0'] = data['Marker0'].replace(999, 0)


        if max(ch_ind + [stim_ind]) >= data.shape[1]:
            raise ValueError(f"ch_ind or stim_ind exceeds ({data.shape[1]} columns)")

        selected_data = data.iloc[:, ch_ind + [stim_ind]].T

        ch_names = [data.columns[i] for i in ch_ind] + ['Stim']

        if replace_ch_names:
            ch_names = [replace_ch_names.get(c, c) for c in ch_names]
        ch_types = ['eeg'] * len(ch_ind) + ['stim']

        # 
        selected_data[:-1] *= 1e-6


        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        raw.append(RawArray(data=selected_data.values, info=info))
    raws = concatenate_raws(raw) if len(raw) > 1 else raw[0]

    return raws



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict



def plot_conditions(epochs, conditions=OrderedDict(), ci=97.5, n_boot=1000,
                    title='', palette=None, ylim=(-6, 6),
                    diff_waveform=(1, 2)):
    """Plot ERP conditions.

    Args:
        epochs (mne.epochs): EEG epochs

    Keyword Args:
        conditions (OrderedDict): dictionary that contains the names of the
            conditions to plot as keys, and the list of corresponding marker
            numbers as value. E.g.,

                conditions = {'Non-target': [0, 1],
                               'Target': [2, 3, 4]}

        ci (float): confidence interval in range [0, 100]
        n_boot (int): number of bootstrap samples
        title (str): title of the figure
        palette (list): color palette to use for conditions
        ylim (tuple): (ymin, ymax)
        diff_waveform (tuple or None): tuple of ints indicating which
            conditions to subtract for producing the difference waveform.
            If None, do not plot a difference waveform

    Returns:
        (matplotlib.figure.Figure): figure object
        (list of matplotlib.axes._subplots.AxesSubplot): list of axes
    """
    if isinstance(conditions, dict):
        conditions = OrderedDict(conditions)

    if palette is None:
        palette = sns.color_palette("hls", len(conditions) + 1)

    X = epochs.get_data() * 1e6
    times = epochs.times
    y = pd.Series(epochs.events[:, -1])

    fig, axes = plt.subplots(2, 2, figsize=[12, 6],
                             sharex=True, sharey=True)
    axes = [axes[1, 0], axes[0, 0], axes[0, 1], axes[1, 1]]

    for ch in range(4):
        for cond, color in zip(conditions.values(), palette):
           
            data = np.nanmean(X[y.isin(cond), ch], axis=0)
            sns.lineplot(x=times, y=data, color=color, ci=ci, ax=axes[ch])
            
        if diff_waveform:
            diff = (np.nanmean(X[y == diff_waveform[1], ch], axis=0) -
                    np.nanmean(X[y == diff_waveform[0], ch], axis=0))
            axes[ch].plot(times, diff, color='k', lw=1)

        axes[ch].set_title(epochs.ch_names[ch])
        axes[ch].set_ylim(ylim)
        axes[ch].axvline(x=0, ymin=ylim[0], ymax=ylim[1], color='k',
                         lw=1, label='_nolegend_')

    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (uV)')
    axes[-1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (uV)')

    if diff_waveform:
        legend = (['{} - {}'.format(diff_waveform[1], diff_waveform[0])] +
                  list(conditions.keys()))
    else:
        legend = conditions.keys()
    axes[-1].legend(legend)
    sns.despine()
    plt.tight_layout()

    if title:
        fig.suptitle(title, fontsize=20)

    return fig, axes


def plot_highlight_regions(x, y, hue, hue_thresh=0, xlabel='', ylabel='',
                           legend_str=()):
    """Plot a line with highlighted regions based on additional value.

    Plot a line and highlight ranges of x for which an additional value
    is lower than a threshold. For example, the additional value might be
    pvalues, and the threshold might be 0.05.

    Args:
        x (array_like): x coordinates
        y (array_like): y values of same shape as x

    Keyword Args:
        hue (array_like): values to be plotted as hue based on hue_thresh.
            Must be of the same shape as x and y.
        hue_thresh (float): threshold to be applied to hue. Regions for which
            hue is lower than hue_thresh will be highlighted.
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        legend_str (tuple): legend for the line and the highlighted regions

    Returns:
        (matplotlib.figure.Figure): figure object
        (list of matplotlib.axes._subplots.AxesSubplot): list of axes
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharey=True)

    axes.plot(x, y, lw=2, c='k')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    kk = 0
    a = []
    while kk < len(hue):
        if hue[kk] < hue_thresh:
            b = kk
            kk += 1
            while kk < len(hue):
                if hue[kk] > hue_thresh:
                    break
                else:
                    kk += 1
            a.append([b, kk - 1])
        else:
            kk += 1

    st = (x[1] - x[0]) / 2.0
    for p in a:
        axes.axvspan(x[p[0]]-st, x[p[1]]+st, facecolor='g', alpha=0.5)
    plt.legend(legend_str)
    sns.despine()

    return fig, axes

