from matplotlib import pyplot as plt
import numpy as np
from fastdtw import fastdtw
from tslearn.metrics import dtw_path, dtw
from scipy.spatial.distance import euclidean
from collections import defaultdict
def plot_compare_subepochs(X_sub_target, X_sub_nontarget, sfreq=256, title="Compare Target vs. Non-Target"):
    """
    Plot the average epochs for the 2 second window
    """
    sub_avg_t = X_sub_target.mean(axis=0) * 1e6
    sub_avg_nt = X_sub_nontarget.mean(axis=0) * 1e6
    n_times = sub_avg_t.shape[1]
    times = np.linspace(0, n_times / sfreq, n_times, endpoint=False)
    plt.figure(figsize=(8, 4))
    n_channels = sub_avg_t.shape[0]
    for ch_idx in range(n_channels):
        plt.plot(times, sub_avg_t[ch_idx], label='Target - avg')
        plt.plot(times, sub_avg_nt[ch_idx], label='Non-Target - avg')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_erp(epochs, ch_names, ylim):
    """
    Plot the average ERP for each channel.

    Args:
        epochs (mne.Epochs): Epochs object containing segmented data.
        ch_names (list): list of channel names to plot.
        ylim (tuple): y-axis limits for the plot as (ymin, ymax) in µV.

    Returns:
        None: displays ERP plot for specified conditions.
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    colors = ['b', 'g', 'r', 'c']

    for i, ch in enumerate(ch_names):
        evoked_non_target = epochs['Non-Target'].copy().average().pick([ch]) 
        evoked_target = epochs['Target'].copy().average().pick([ch]) 

        # rescale back to microvolt from the converted mne raw object
        evoked_non_target.data *= 1e6
        evoked_target.data *= 1e6

        times = evoked_non_target.times
        data_non_target = evoked_non_target.data[0]
        data_target = evoked_target.data[0]
        data_diff = data_target - data_non_target
        axes[i].plot(times, data_non_target, label='Non-target', color='r')
        axes[i].plot(times, data_target, label='Target', color='g')
        axes[i].plot(times, data_diff, label='Difference', color='k', linestyle='--')

        axes[i].set_title(f'Average ERP for {ch}')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude (µV)')
        axes[i].set_ylim(ylim)
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


def plot_real_time_classified_eeg(segments, labels,
                                  align_fn=None,          # e.g. align_with_dtw
                                  mode="both",            # "avg" | "both"
                                  channel_names=("TP9", "TP10"),
                                  alpha=0.5):
    """
    Quick glance at how classified segments look, with or without alignment.

    Parameters
    ----------
    segments : list[np.ndarray]    # (L, 2) each
    labels   : list[int]           # 1 = target / blink, 0 = non-target
    align_fn : callable or None    # None -> raw ; function(seg) -> aligned
    mode     : "avg" plots one trace (mean of 2 channels)
               "both" plots two sub-plots (one per channel)
    alpha    : line transparency
    """

    # Optionally align every segment
    if align_fn is not None:
        segments = [align_fn(seg) for seg in segments]

    # Decide window length from first segment
    win_len = segments[0].shape[0]
    x = np.arange(win_len)

    # Prepare figure
    n_sub = 1 if mode == "avg" else 2
    fig, axes = plt.subplots(n_sub, 1, figsize=(12, 4 * n_sub), sharex=True)
    if n_sub == 1:
        axes = [axes]  # make iterable

    # Plot green (1) first, red (0) second so green is not hidden
    for lbl in (1, 0):
        z = 2 if lbl == 1 else 3          # green below red, but both visible
        color = "tab:green" if lbl else "tab:red"
        mask = (labels == lbl) if isinstance(labels, np.ndarray) else \
               [l == lbl for l in labels]

        for seg in np.asarray(segments)[mask]:
            if mode == "avg":
                axes[0].plot(x, seg.mean(axis=1),
                             color=color, alpha=alpha, zorder=z)
            else:
                for ch, ax in enumerate(axes):
                    ax.plot(x, seg[:, ch],
                            color=color, alpha=alpha, zorder=z)

    # Cosmetics
    axes[-1].set_xlabel("Sample index")
    if mode == "avg":
        axes[0].set_ylabel("Average amplitude")
    else:
        for ch, ax in enumerate(axes):
            ax.set_ylabel(f"{channel_names[ch]} amplitude")

    title = "RT Classified EEG (aligned)" if align_fn else "RT Classified EEG (raw)"
    plt.suptitle(title, fontsize=14)
    for ax in axes:
        ax.grid(True)

    # Custom legend (one entry per class)
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], color="tab:green", lw=2, label="Class 1"),
                    Line2D([0], [0], color="tab:red",   lw=2, label="Class 0")]
    axes[0].legend(handles=legend_elems, loc="upper right")
    plt.tight_layout()
    plt.show()

# def _resample(arr, target_len):
#     """Resample (T, C) → (target_len, C) using 1-D linear interpolation."""
#     if arr.shape[0] == target_len:
#         return arr
#     x_old = np.linspace(0, 1, arr.shape[0])
#     x_new = np.linspace(0, 1, target_len)
#     return np.stack(
#         [np.interp(x_new, x_old, arr[:, c]) for c in range(arr.shape[1])],
#         axis=1
#     )

def align_with_dtw(
        seg, template,
        use_fast=True,          # fastdtw or exact dtw_path
        w=None,                 # Sakoe-Chiba radius (only use_fast=False)
        resample=False,
        z_score=False):        # True→512 点；False→len(path) 点
    """
    返回一条 (N,2) 轨迹
        • resample=False : N == len(path)       —— 完全展开
        • resample=True  : N == 512            —— 每 j 取最短距离那点
    """

    # -------- shape (N,2) --------
    if seg.shape[0] == 2 and seg.shape[1] != 2:
        seg = seg.T
    if template.shape[0] == 2 and template.shape[1] != 2:
        template = template.T

    # -------- z-score --------
    z = lambda x: (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-12) if z_score else x
    seg_z, tmp_z = z(seg), z(template)           # tmp_z 

    # -------- DTW path --------
    if use_fast:
        _, path = fastdtw(seg_z, tmp_z, dist=euclidean)
    else:
        kw = {"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": w} if w else {}
        path, _ = dtw_path(seg_z, tmp_z, **kw)
    if not resample:
        return np.asarray([seg_z[i] for i, _ in path])   # (len(path), 2)

    tmpl2eeg = defaultdict(list)
    for i, j in path:           
        tmpl2eeg[j].append(i)

    aligned = np.empty_like(tmp_z)       # (512,2)
    for j in range(tmp_z.shape[0]):     
        idxs = tmpl2eeg[j]
        # closest point to tmp_z[j] 
        i_best = min(idxs, key=lambda i: np.linalg.norm(seg_z[i]-tmp_z[j]))
        aligned[j] = seg_z[i_best]

    return aligned                       # (512,2)


# ──────────────── visualization ────────────────
def plot_aligned_eeg(segments, labels, template,
                     w=None, use_fast=True,
                     channel_names=("TP9","TP10"),
                     resample=False):
    colors = {0:"red", 1:"green"}
    n_ch = 2
    fig, axes = plt.subplots(n_ch,1,figsize=(13,5),sharex=True)

    for seg, lbl in zip(segments, labels):
        aligned = align_with_dtw(seg, template, use_fast, w, resample=resample)
        x = np.arange(aligned.shape[0])    
        for ch in range(n_ch):
            axes[ch].plot(x, aligned[:,ch],
                           color=colors.get(lbl,"blue"),
                           alpha=.4 if lbl else .2)
            axes[ch].set_title(f"{channel_names[ch]} (aligned)")
            axes[ch].set_ylabel("z-amp"); axes[ch].grid(True)

    axes[-1].set_xlabel("Aligned index (0-511)")
    plt.suptitle("DTW-aligned EEG segments", fontsize=14)
    plt.tight_layout(); plt.show()

def plot_mean_curves(segments, labels, template,
                     use_fast=True, w=None,
                     channel_names=("TP9", "TP10"),
                     colors=("tab:green", "tab:red"),
                     resample=False):
    """
    Plot mean ± std curves for blink (label 1) vs non-blink (label 0),
    after DTW alignment.
    """

    # Align all segments
    aligned_list = [align_with_dtw(s, template,
                                   use_fast=use_fast,
                                   w=w,
                                   resample=resample)
                    for s in segments]

    # Pad with NaN if trajectories have different lengths
    if resample:
        aligned = np.stack(aligned_list)                  # (N, 512, 2)
    else:
        lengths = [a.shape[0] for a in aligned_list]
        max_len = max(lengths)
        aligned = np.full((len(aligned_list), max_len, 2), np.nan)
        for k, a in enumerate(aligned_list):
            aligned[k, :a.shape[0], :] = a

    labels = np.asarray(labels)
    blink, non = aligned[labels == 1], aligned[labels == 0]

    mean_b, std_b = np.nanmean(blink, 0), np.nanstd(blink, 0)
    mean_n, std_n = np.nanmean(non,   0), np.nanstd(non,   0)

    x = np.arange(mean_b.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    for ch, ax in enumerate(axes):
        # Blink
        ax.fill_between(x, mean_b[:, ch] - std_b[:, ch],
                           mean_b[:, ch] + std_b[:, ch],
                           color=colors[0], alpha=0.25)
        ax.plot(x, mean_b[:, ch], color=colors[0], lw=2, label="blink")

        # Non-blink
        ax.fill_between(x, mean_n[:, ch] - std_n[:, ch],
                           mean_n[:, ch] + std_n[:, ch],
                           color=colors[1], alpha=0.20, linestyle="--")
        ax.plot(x, mean_n[:, ch], color=colors[1], lw=2,
                linestyle="--", label="non-blink")

        ax.set_title(f"Channel {channel_names[ch]}")
        ax.set_ylabel("z-amplitude")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Aligned index")
    plt.suptitle("Blink vs Non-blink — mean ± 1 SD after DTW alignment",
                 fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_all_segments(model, X_lists, y_lists,
                           exp_indices=None, sub_indices=None,
                           use_fast=None):
    """

    Parameters
    ----------
    model        : trained BlinkTemplateMatcher
    X_lists      : same结构 as all_X  (exp → sub → session → segments)
    y_lists      : same结构 as all_y
    exp_indices  : iterable[int]  
    sub_indices  : iterable[int]  
    use_fast     : bool or None  
    """
    if exp_indices is None:
        exp_indices = range(len(X_lists))
    if sub_indices is None:
        sub_indices = range(len(X_lists[0]))

    segments, labels = [], []
    for ei in exp_indices:
        for si in sub_indices:
            for sess in (0, 1):
                segments.extend(X_lists[ei][si][sess])
                labels.extend(y_lists[ei][si][sess])

    use_fast = ("fast" in model.mode) if use_fast is None else use_fast
    tmpl     = model.template_
    w        = model.w

    plot_aligned_eeg(segments, labels, tmpl,
                     w=w, use_fast=use_fast)
    plot_mean_curves(segments, labels, tmpl,
                     use_fast=use_fast, w=w)


# ------------------------------------------------------------------
def _ensure_L2(seg):
    """
    Make sure an EEG segment is (L, 2).  Most loaders give (2, L).
    """
    if seg.ndim == 2 and seg.shape[0] == 2 and seg.shape[1] != 2:
        return seg.T
    return seg


def _plot_rt_segment(ax, seg, lbl, alpha=0.5):
    """
    Plot one raw segment (single channel) into the supplied Axes.
    """
    seg = _ensure_L2(seg)
    color = "tab:green" if lbl else "tab:red"
    ax.plot(np.arange(seg.shape[0]), seg, color=color, alpha=alpha)
# ------------------------------------------------------------------


def _to_L2(arr):
    """
    Ensure segment is (L, 2).  Many loaders give (2, L).
    """
    return arr.T if arr.shape[0] == 2 and arr.shape[1] != 2 else arr


import numpy as np
import matplotlib.pyplot as plt

def visualize_by_subject(model, X_lists, y_lists,
                         exp_indices=None, sub_indices=None,
                         use_fast=None,
                         resample=True, show_raw=True,
                         channel_names=("TP9", "TP10")):
    """
    Row = one subject
    Col 0/1 : aligned TP9 / TP10
    Col 2/3 : raw      TP9 / TP10   (if show_raw=True)
    Col 4/5 : raw-norm TP9 / TP10   (if show_raw=True)
    """

    # choose indices -------------------------------------------------
    exp_indices = range(len(X_lists))    if exp_indices is None else exp_indices
    sub_indices = range(len(X_lists[0])) if sub_indices is None else sub_indices

    # DTW engine -----------------------------------------------------
    use_fast = ("fast" in model.mode) if use_fast is None else use_fast
    template, w = model.template_, model.w

    # figure layout --------------------------------------------------
    base_cols = 2                      # aligned
    extra_cols = 4 if show_raw else 0  # raw + norm
    n_cols = base_cols + extra_cols
    n_sub  = len(sub_indices)
    share_opt = {"sharex": True} if (resample and not show_raw) else {}
    fig, axes = plt.subplots(n_sub, n_cols,
                             figsize=(4 * n_cols, 2.8 * n_sub),
                             **share_opt)
    if n_sub == 1:
        axes = axes.reshape(1, n_cols)

    # iterate subjects ----------------------------------------------
    for row, s in enumerate(sub_indices):
        seg_s, lab_s = [], []
        for ei in exp_indices:
            for sess in (0, 1):
                seg_s.extend(X_lists[ei][s][sess])
                lab_s.extend(y_lists[ei][s][sess])

        for seg, lbl in zip(seg_s, lab_s):
            # aligned ------------------------------------------------
            seg_l2 = _to_L2(seg)
            aligned = align_with_dtw(seg_l2, template,
                                     use_fast=use_fast, w=w,
                                     resample=resample)

            x_aln = np.arange(aligned.shape[0])
            c = "tab:green" if lbl else "tab:red"
            a = 0.4 if lbl else 0.15
            axes[row, 0].plot(x_aln, aligned[:, 0], color=c, alpha=a)
            axes[row, 1].plot(x_aln, aligned[:, 1], color=c, alpha=a)

            if show_raw:
                # raw ------------------------------------------------
                x_raw = np.arange(seg_l2.shape[0])
                axes[row, 2].plot(x_raw, seg_l2[:, 0], color=c, alpha=a)
                axes[row, 3].plot(x_raw, seg_l2[:, 1], color=c, alpha=a)

                # normalized raw ------------------------------------
                seg_norm = (seg_l2 - seg_l2.mean(axis=0)) / (seg_l2.std(axis=0) + 1e-8)
                axes[row, 4].plot(x_raw, seg_norm[:, 0], color=c, alpha=a)
                axes[row, 5].plot(x_raw, seg_norm[:, 1], color=c, alpha=a)

        axes[row, 0].set_ylabel(f"Sub-{s}  amp (z)")
        for col in range(n_cols):
            axes[row, col].grid(True)

    # titles ---------------------------------------------------------
    axes[0, 0].set_title(f"{channel_names[0]}  aligned")
    axes[0, 1].set_title(f"{channel_names[1]}  aligned")
    if show_raw:
        axes[0, 2].set_title(f"{channel_names[0]}  raw")
        axes[0, 3].set_title(f"{channel_names[1]}  raw")
        axes[0, 4].set_title(f"{channel_names[0]}  raw-norm")
        axes[0, 5].set_title(f"{channel_names[1]}  raw-norm")

    xlabel_aln = "Aligned index 0–511" if resample else "DTW path index"
    axes[-1, 0].set_xlabel(xlabel_aln)
    axes[-1, 1].set_xlabel(xlabel_aln)
    if show_raw:
        for col in (2, 3, 4, 5):
            axes[-1, col].set_xlabel("Sample index")

    plt.suptitle("EEG per subject — aligned, raw, normalized", fontsize=13)
    plt.tight_layout()
    plt.show()
