"""
This file origianlly was used to classify eyeblinks but is not 
used to test the open-close fist and open close feet experiment. 
"""
import os 
import mne
import numpy as np
import pandas as pd
from mne import Epochs, find_events
from matplotlib import pyplot as plt
from model_utils import model
import utils
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from plot import plot_compare_subepochs, plot_erp
from datetime import datetime
from sklearn.metrics import accuracy_score

def refine_events(raw, code:dict, tmin:float=0.0, tmax:float=5.0, picks=[0, 1, 2, 3], 
                  filter:bool=False,l_freq:float=1., h_freq:float=30.,
                  method:str='iir'):
    """
    change the experiment onset to 11 and 13, 
    where target maps to 11
    and non-target maps to 13.
    And split the event into epochs based on the interval defined
    """
    target = code['target']
    non_target = code['non-target']
    all_events = find_events(raw, shortest_event=1)

    new_events_list = []
    for ev in all_events:
        sample = ev[0]
        code   = ev[2]
        # Blink start
        if code == target:
            new_events_list.append([sample, 0, 11])
        # Open start
        elif code == non_target:
            new_events_list.append([sample, 0, 13])
       
    new_events = np.array(new_events_list, dtype=int)
    event_id = {
        'Target': 11,       
        'Non-Target': 13    
    }

    # bandpass filter 
    if filter:
        raw.filter(l_freq=l_freq, h_freq=h_freq, method=method)
    
    # epoch the data
    epochs = Epochs(
        raw, 
        events=new_events,
        event_id=event_id,
        tmin=tmin,       
        tmax=tmax,        
        baseline=None,
        preload=True,
        picks=picks  
    )
    epochs
    return epochs


def epoch_all_subject_events(data_list:list, code:dict, exp_num:int, tmin:float=0.0, tmax:float=4.0, picks=[0, 1, 2, 3], 
                             filter:bool=False, l_freq:float=1., h_freq:float=30.):
    """
    Rename all the event labels and split
    raw into epochs for all data
    return a list of list, where each sublist represents the subject, 
    and each sublist holds two epochs (two runs per subject)
    """
    epoch_data_by_sub = []
    
    for sub_data in data_list:
        sub_epochs = []
        for trail in sub_data:
            sub_epochs.append(refine_events(trail, code, tmin=tmin, tmax=tmax, picks=picks, filter=filter, l_freq=l_freq, h_freq=h_freq))
                
        epoch_data_by_sub.append(sub_epochs)
    return epoch_data_by_sub


def load_data(data_folder:str):
    """
    Read and return both experiments data with two lists.
    Sorted based upon subject number and session 
    """
    experiment_1_list = []
    experiment_2_list = []
    #data_folder = "EB-Data"
    for folder in os.listdir(data_folder):
        sub_folder = os.path.join(data_folder, folder, "Muse_data")
        if folder.endswith("1") or folder.endswith("2"):
            for file in os.listdir(sub_folder):
                if "EEG" in file:
                    fp = os.path.join(sub_folder, file)
                    experiment_1_list.append([fp])
        elif folder.endswith("3") or folder.endswith("4") or folder.endswith("5"): #subject 1, 14 was interrupted, 15 was redo
            for file in os.listdir(sub_folder):
                if "EEG" in file:
                    fp = os.path.join(sub_folder, file)
                    experiment_2_list.append([fp])
    
    def get_subject_id(elem):
        if isinstance(elem, list):
            path = elem[0]
        else:
            path = elem
        match = re.search(r"sub-EB-(\d+)", path)
        if match:
            return int(match.group(1))
        return float('inf')
    experiment_1_list.sort(key=get_subject_id)
    experiment_2_list.sort(key=get_subject_id)
    print(f"Experiment 1 list contains file paths:{experiment_1_list}")
    print(f"Experiment 2 list contains file paths:{experiment_2_list}")
    return experiment_1_list, experiment_2_list

def load_oc_data(data_folder:str):
    """
    Load the open-close fist/feet data, one subject only.
    """
    def extract_dt(path):
        # sub-EB-43_EEG_recording_2025-06-08-19.57.52.csv
        ts = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2})", path).group(1)
        return datetime.strptime(ts, "%Y-%m-%d-%H.%M.%S")
    eeg_paths = []
    for fname in os.listdir(data_folder):
        if "EEG" in fname and ".csv" in fname:
            eeg_paths.append(os.path.join(data_folder, fname))
    eeg_paths.sort(key=extract_dt)
    exp1, exp2 = [], []
    for idx, fp in enumerate(eeg_paths): #0,1,2,3
        (exp1 if idx % 2 == 0 else exp2).append([fp]) #even number is for open/close fists

    print("Experiment-1 (fists) files:", exp1) #open/close fists
    print("Experiment-2 (feet)  files:", exp2) #open/close feet
    return exp1, exp2


def split_data_by_subject(experiment_data_list: list):
    """
    Split data by subject, and return the split list
    """
    split_data = []
    for i in range(0, len(experiment_data_list), 2):
        sub_data = []
        raw_data_s1 = utils.load_muse_csv_as_raw(experiment_data_list[i], sfreq=256., ch_ind=[0, 1, 2, 3], stim_ind=5, replace_ch_names=None)
        raw_data_s2 = utils.load_muse_csv_as_raw(experiment_data_list[i+1], sfreq=256., ch_ind=[0, 1, 2, 3], stim_ind=5, replace_ch_names=None)
        sub_data.append(raw_data_s1)
        sub_data.append(raw_data_s2)
        split_data.append(sub_data)
    return split_data

def split_into_subepochs(data, label, sfreq=256, subepoch_sec=4.0):
    """nonverlapping sampling of window"""
    n_epochs, n_channels, n_times = data.shape
    window_size = int(subepoch_sec * sfreq)  
    
    X_sub = []
    y_sub = []

    for i in range(n_epochs):
        epoch_data = data[i]  
        start = 0
        while (start + window_size) <= n_times:
            sub_data = epoch_data[:, start:start + window_size]
            X_sub.append(sub_data)
            y_sub.append(label)
            start += window_size 

    return np.array(X_sub), np.array(y_sub)


def remove_outliers_with_mean(data, threshold):
    """
    Quality control of outliers, remove abnormally big data
    """
    data_clean = data.copy()
    replaced_count = 0

    n_epochs, n_channels, n_times = data_clean.shape
    for i in range(n_epochs):
        for c in range(n_channels):
            channel_data = data_clean[i, c, :]
            outliers = np.abs(channel_data) > threshold
            inliers = ~outliers
            if np.any(inliers):
                mean_inliers = channel_data[inliers].mean()
            replaced_count += np.sum(outliers)

            channel_data[outliers] = mean_inliers
    
    print(f"Total replaced points: {replaced_count}")
    return data_clean

def process_epochs(ep, picks=['TP9', 'AF7', 'AF8', 'TP10'], sfreq=256, subepoch_sec=4.0,
                       target_label=1, nontarget_label=0,
                       skip_plotting=True,
                       qc = False,
                       threshold=100e-6):
    """
    ep: Epoch object
    """
    # plot_image function documentation: https://mne.tools/1.8/auto_tutorials/epochs/20_visualize_epochs.html
    if not skip_plotting:
        ep['Target'].plot_image(picks=picks)
        ep['Non-Target'].plot_image(picks=picks)
       
        plot_erp(ep, picks,(-50, 50))
    # ep_t and ep_nt contains both TP9 and TP10 channels
    ep_t = ep['Target'].copy().pick(picks).get_data()
    ep_nt = ep['Non-Target'].copy().pick(picks).get_data()
    

    # extra quality control
    if qc == True:
        ep_nt_clean = remove_outliers_with_mean(ep_nt, threshold=threshold)
        ep_nontarget = ep['Non-Target'].copy().pick(picks)
        ep_nt_clean_obj = mne.EpochsArray(
        data=ep_nt_clean,
        info=ep_nontarget.info.copy(),
        events=ep_nontarget.events,
        event_id={'Non-Target': ep.event_id['Non-Target']},
        tmin=ep_nontarget.tmin,
        metadata=ep_nontarget.metadata
        )
        ep_nt = ep_nt_clean
        if not skip_plotting:
            ep_nt_clean_obj.plot_image(picks=picks, combine="mean", show=True, title="Non-Target - after QC")

    X_sub_t, y_sub_t = split_into_subepochs(ep_t, label=target_label, sfreq=sfreq, subepoch_sec=subepoch_sec)
    X_sub_nt, y_sub_nt = split_into_subepochs(ep_nt, label=nontarget_label, sfreq=sfreq, subepoch_sec=subepoch_sec)

    # plot_compare_subepochs(X_sub_t, X_sub_nt, sfreq=256, title="Target vs Non-Target")
    X_sub = np.concatenate([X_sub_t, X_sub_nt], axis=0)
    y_sub = np.concatenate([y_sub_t, y_sub_nt], axis=0)
    print("X_sub shape:", X_sub.shape)
    print("y_sub shape:", y_sub.shape)
    return X_sub, y_sub

def process_epochs_TP9_10(ep, picks=['TP9', 'TP10'], sfreq=256, subepoch_sec=2.0,
                       target_label=1, nontarget_label=0, 
                       skip_plotting=True,
                       qc = False,
                       threshold=100e-6):
    if not skip_plotting:
        # plot to see average of TP9 and TP10
        ep['Target'].plot_image(picks=picks, combine="mean")
        ep['Non-Target'].plot_image(picks=picks, combine="mean")

    # ep_t and ep_nt contains both TP9 and TP10 channels
    ep_t = ep['Target'].copy().pick(picks).get_data()
    ep_nt = ep['Non-Target'].copy().pick(picks).get_data()
    blink_template = np.mean(ep_t, axis=0)
    print("shape for blink_template is", blink_template.shape)
    # extra quality control
    if qc == True:
        ep_nt_clean = remove_outliers_with_mean(ep_nt, threshold=threshold)
        ep_nontarget = ep['Non-Target'].copy().pick(picks)
        ep_nt_clean_obj = mne.EpochsArray(
        data=ep_nt_clean,
        info=ep_nontarget.info.copy(),
        events=ep_nontarget.events, 
        event_id={'Non-Target': ep.event_id['Non-Target']},
        tmin=ep_nontarget.tmin,
        metadata=ep_nontarget.metadata
        )
        ep_nt = ep_nt_clean
        if not skip_plotting:
            ep_nt_clean_obj.plot_image(picks=picks, combine="mean", show=True, title="Non-Target - after QC")

    X_sub_t, y_sub_t = split_into_subepochs(ep_t, label=target_label, sfreq=sfreq, subepoch_sec=subepoch_sec)
    X_sub_nt, y_sub_nt = split_into_subepochs(ep_nt, label=nontarget_label, sfreq=sfreq, subepoch_sec=subepoch_sec)

    # plot_compare_subepochs(X_sub_t, X_sub_nt, sfreq=256, title="Target vs Non-Target")
    X_sub = np.concatenate([X_sub_t, X_sub_nt], axis=0)
    y_sub = np.concatenate([y_sub_t, y_sub_nt], axis=0)
    print("X_sub shape:", X_sub.shape)  
    print("y_sub shape:", y_sub.shape) 
    return X_sub, y_sub, blink_template


def preprocess_data(data_folder:str, exp_num:int, merge:bool=False, 
                    skip_plotting:bool=True, qc:bool=False, filter:bool=False, l_freq:float=1., h_freq:float=30.):
    """
    Preprocess the data from the folder, used for off-line model training and evaluation.
    """
    # retrieve the data list
    experiment_1_list, experiment_2_list = load_oc_data(data_folder)
    X_list = []
    y_list = []
    # exp 1: open close fist
    if exp_num == 1:
        # split data by subject 
        code = {"target": 101, "non-target": 201}
        sub_data_list = split_data_by_subject(experiment_1_list)
        # refine data label and split into epochs
        epochs_list = epoch_all_subject_events(sub_data_list, code, exp_num, tmax=4, filter=filter, h_freq=h_freq, l_freq=l_freq)

    # exp 2: open close feet
    elif exp_num == 2:
        code = {"target":102, "non-target": 202}
        sub_data_list = split_data_by_subject(experiment_2_list)
        epochs_list = epoch_all_subject_events(sub_data_list, code, exp_num, tmax=4, filter=filter, h_freq=h_freq, l_freq=l_freq)
    
    # epoch the experiment and concatenate data based on subject,
    # and return the list for all the data
    for epoch_by_sub in epochs_list:
        X_temp_list = []
        y_temp_list = []
        for epoch in epoch_by_sub:
            X_temp, y_temp = process_epochs(epoch, qc=qc, skip_plotting=skip_plotting)
            X_temp_list.append(X_temp)    
            y_temp_list.append(y_temp)  
        if merge: 
            X_sub = np.concatenate(X_temp_list, axis=0)
            y_sub = np.concatenate(y_temp_list, axis=0)
            X_list.append(X_sub)
            y_list.append(y_sub)
        else:
            X_list.append(X_temp_list)
            y_list.append(y_temp_list)
    print("length of X_list is ", len(X_list))          # ➜ 1
    print([x.shape for x in X_list[0]])  # ➜ [ (n_epochs_run1, 4, 1024), (n_epochs_run2, 4, 1024) ]
    for sub_y_list in y_list[0]:
        print("number of targets:")       
        print(sum([1 if y == 1 else 0 for y in sub_y_list]))
        print("number of non-targets")
        print(sum([1 if y == 0 else 0 for y in sub_y_list]))
    return X_list, y_list
    
def decoder_train(X:np.array, y:np.array, sub:str, exp:int, randomized:bool=False, test_size:float=0.2, random_state:int=42,
                   scoring:str='accuracy'):
    """
    train the decoder.
    """
    # random
    if randomized:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        # # not random, fixed one
        # X_train=X[0] #first run for training
        # X_test=X[1]  #second run for testing
        # y_train=y[0]
        # y_test=y[1] 
        # # use second run for training and first one for testing because open/close fists' first run has only 32 trials instead of 50 trials
        X_train=X[1] #first run for training
        X_test=X[0]  #second run for testing
        y_train=y[1]
        y_test=y[0] 
        print(f'y_test shape is {y_test.shape}')

    my_model = model()
    #method1: 10 iterations, for each iteration, randomly select 75% for training and 25% for testing
    best_model_name = my_model.model_selection(X_train, y_train, scoring=scoring,
                                      n_splits=10, test_size=0.25, random_state=random_state)
    # #method2: using leave one trial out CV inside model_selection() , so n_splits and test_size are useless
    # best_model_name = my_model.model_selection(X_train, y_train, scoring=scoring,
    #                                         n_splits=10, test_size=0.25, random_state=random_state)
    
    my_model.train(X_train, y_train) #once select the best model from above, then retrain the model using all samples
    # check both score for the off-line model on the test data 
    accuracy=my_model.evaluate(X_test, y_test, metric='accuracy')
    auc_score, y_proba = my_model.evaluate(X_test, y_test, metric='roc_auc')
    cm=my_model.evaluate(X_test, y_test, metric='confusion_matrix')

    # caculate recall and precision
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  
    specificity = tn / (tn + fp) 

    # roc curve
    fpr, tpr, _ = roc_curve(y_test, y_proba, drop_intermediate=False)  # Compute ROC curve
    roc_auc = auc(fpr, tpr)  # Compute AUC score
    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random chance line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    # save model configuration
    #my_model.save_model(f"my_model_exp_{exp}_sub_{sub}.pkl")
    my_model.save_model(f"openCloseFistsFeet_model_exp_{exp}_sub_{sub}.pkl") #e.g., my_model_exp_1_sub_1.pkl is for fists, my_model_exp_1my_model_exp_2_sub_1.pkl is for feet
    print("-"*30)

    return my_model, best_model_name, sensitivity, specificity, accuracy, auc_score

def bandpass_filter_ndarray(data: np.ndarray, sfreq:float=256, l_freq:float=1., h_freq:float=30.,method='iir'):
    """
    Convert a np ndarray first to raw then call the bandpassfilter
    """
    n_channels, n_times = data.shape
    ch_names = [f'ch{i}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels 
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.filter(l_freq=l_freq, h_freq=h_freq, method=method)
    data_filt = raw.get_data()
    return data_filt

def decoder_predict(X_input:np.ndarray, trained_model: model, qc:bool=False, threshold:float=100e-6, filter:bool=False):
    """
    Predict using real-time data.
    X_input shape (2, 512) where 2 is the num_channels and 512 is the time.
    """
    # perform qc and filter if requested
    if X_input.ndim !=2:
        raise ValueError("X_input must be 2D: (2, 512).")

    if qc:
        X_input = remove_outliers_with_mean(X_input, threshold=threshold)
    if filter:
        X_input = bandpass_filter_ndarray(X_input)

    # expand 1 dimension to use sklearn structure
    X_input = X_input[np.newaxis, ...]
    print(X_input.shape)
    prob = trained_model.best_model_.predict_proba(X_input)[0] # （1, n_classes)
    prob_class1 = prob[1] # select the probability for class 1
    pred_label = trained_model.best_model_.predict(X_input)[0]
    return prob_class1, pred_label

def process_and_plot_eeg(fp, loaded_model, window_size=1024, step_size=0.1):
    """
    method_1: loading existing csv and predict using loaded model with overlapping windows
    """
    df = pd.read_csv(fp)
    total_rows = len(df)
    fs = 256  # sampling rate in Hz
    step_size = int(step_size * fs)  # sliding window
    idx = 0
    segment_count = 0

    while idx + window_size <= total_rows:
        start_row = idx
        end_row = idx + window_size - 1
        x_data = df[['TP9','TP10']].iloc[start_row : end_row + 1].to_numpy()  # shape (512, 2)
        
        similarity, label = decoder_predict(x_data, loaded_model)

        print(f"Segment {segment_count}: similarity={similarity:.4f}, label={label}")
        plt.figure(figsize=(8,4))
        plt.plot(x_data[:,0], label='TP9')
        plt.plot(x_data[:,1], label='TP10')
        plt.title(f"EEG snippet (rows {idx}~{idx+window_size}), label={label}")
        plt.xlabel("Sample index (0~511)")
        plt.ylabel("Amplitude (raw units)")
        plt.legend()
        plt.grid(True)
        plt.show()
        idx += step_size  # Move window by step_size
        segment_count += 1

def predictionCSVFile(loaded_model, csvFilename, exp_num, 
                      sfreq=256, channels=("TP9", "AF7", "AF8", "TP10"),
                      stim_ind=5,
                      t_min=0.0,
                      t_max=4.0,
                      l_freq=1,
                      h_freq=30,
                      do_filter=False):
    #predit the csvFilanme giving model
    CODEMAP = {
    1: {"target": 101, "non-target": 201},   # exp 1: open-close fist
    2: {"target": 102, "non-target": 202},   # exp 2: open-close feet
}
    raw = utils.load_muse_csv_as_raw(
        csvFilename,
        sfreq=sfreq,
        ch_ind=[0, 1, 2, 3],    
        stim_ind=stim_ind,     
        replace_ch_names=None
    )
    raw.load_data()
    if do_filter:
        raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir')

    code = CODEMAP[exp_num]
    events = find_events(raw, shortest_event=1)
    event_id = {"Target": code["target"],
                "Non-Target": code["non-target"]}
    epochs = Epochs(raw, events, event_id,
                    tmin=t_min, tmax=t_max,
                    picks=list(channels),
                    baseline=None, preload=True)
    
    X = epochs.get_data() *1e6                                  # (n_ev, 4, 1025)
    X = X[:, :, :-1]  
    print(X.shape)
    y_true = (epochs.events[:, 2] == event_id['Target']).astype(int)
    y_pred = loaded_model.predict(X)
    # probability for target
    prob   = loaded_model.best_model_.predict_proba(X)[:, 1] 
    print(events[:10])
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy = {acc:.3f}   ( {y_pred.sum()} / {len(y_pred)} positive )")
    return y_pred, y_true, prob, acc

if __name__ == "__main__":
    #step1. build the decoder
    # exp_num can only be 1 or 2
    # exp_1 is the 5s blinking and 5s eyes-opening
    # exp_2 is the 2s blinking and 2-5s random eyes-opening
    data_folder = "/Users/hiro/Desktop/fun-website/decoder/EB-Data/sub-EB-43/Muse_data_OpenCloseFeet 2"
    # data_folder = r"D:\Faculty\ColumbiaUniversity\dataprocess\EEG\dataProcessing\Interaxon\museS\LSL\Python\muse-lsl-python\Data\sub-EB-43\Muse_data_OpenCloseFeet"
    #exp_nums = [1, 2]
    exp_nums = [1,2] #1 is for open/close fists, 2 is for open/close feet
    # skip_plotting=False #plot it
    skip_plotting = True  # skipping plot it
    qc = False
    filter=True
    #filter=False
    h_freq=30.0 #30 hz
    l_freq= 1. #1 hz
    # merge parameter control how the data is preprocessed
        # if using merge==False, randomized has to be set False,
        # otherwise merge=True, randomized should be True as well
    merge=False
    randomized=False

    methods = ['fir', 'iir']

    all_X = []
    all_y = []
    for exp_num in exp_nums:
        X_list, y_list = preprocess_data(data_folder, exp_num, merge, skip_plotting, qc,filter,l_freq, h_freq)
        all_X.append(X_list)
        all_y.append(y_list)


    # perform the training for all off-line model, decoder training is determined by randomized
    # if randomized = True, concatenate subject's data and do random tran test split
    # if randomized = False, train 
   
    all_results = [] 
    for i, X_list in enumerate(all_X): #all_X: 1st D is experiment, 2nd D is subject, 3rd D is run, 4th D is sample, e.g., all_X[0][0][0].shape=(63,4,1024), all_X[1][0][0].shape=(100, 4, 1024)
        exp = str(i + 1)
        for idx, sub in enumerate(X_list):
            X = X_list[idx]
            y = all_y[i][idx]
            print(type(X))
            if isinstance(X, list):
                X = [run * 1e6 for run in X]
            else:
                X *= 1e6
            # trian all models off-line
            
            # print(X)
            sub = str(idx + 1)
            
            _, _, sens, spec, acc, auc_score = decoder_train(X, y, sub=sub, exp=exp, randomized=randomized)
            
            all_results.append({
                'exp': exp,
                'sub': sub,
                'sensitivity': sens,
                'specificity': spec,
                'accuracy': acc,
                'AUC': auc_score
            })


    df = pd.DataFrame(all_results)
    print(df)

    #step2. prediction/testing
    import pickle
    # sample usage: decoder_predict
    # model_file = "my_model_exp_1_sub_1.pkl"
    model_file = "openCloseFistsFeet_model_exp_1_sub_1.pkl" #for fists
    # model_file = "openCloseFistsFeet_model_exp_2_sub_1.pkl" #for feet
    with open(model_file, 'rb') as f:
        loaded_model = pickle.load(f)
    
    # 
    #method1: use .csv file of real-time data as input
    #e.g., use offline data D:\Faculty\ColumbiaUniversity\dataprocess\EEG\dataProcessing\Interaxon\museS\LSL\Python\muse-lsl-python\decoder\EB-Data\sub-EB-33\Muse_data\sub-EB-33_EEG_recording.csv
    #use real-time data 
    #fp = "EB-Data/sub-EB-41/Muse_data/sub-EB-41_EEG_recording_session2.csv"
    #process_and_plot_eeg(fp, loaded_model, window_size=1024, step_size=0.5)
    #method2: use .csv file of off-line data as input
    # csvFilename=r"D:\Faculty\ColumbiaUniversity\dataprocess\EEG\dataProcessing\Interaxon\museS\LSL\Python\muse-lsl-python\Data\sub-EB-43\Muse_data_OpenCloseFeet\sub-EB-43_EEG_recording_2025-06-08-19.46.10.csv"
    csvFilename="/Users/hiro/Desktop/fun-website/decoder/EB-Data/sub-EB-43/Muse_data_OpenCloseFeet 2/sub-EB-43_EEG_recording_2025-06-08-20.11.07.csv"
    predictionCSVFile(loaded_model,csvFilename,1)


        # #method2: using existing loaded data
        # # all_X outmost index refers to the experiment number
        # # second index refers every subjects
        # # third layer refers to each session by subject
        # # test data point loading method changes as we change how the training data was processed
        # example_input = all_X[0][0][1][2] #first 0 means experiment, 2nd 0 is subject#, 3rd 1 is run#, 4th 2 is sample#
        # label_input=all_y[0][0][1][2]
        # print(example_input.shape)
        # print(type(example_input))
        # y_prob, y_pred = decoder_predict(X_input=example_input, trained_model=loaded_model)
        # print(f"ground truth label is {label_input}")
        # print(f'prob for target class_1 is {y_prob}, predicted_label is {y_pred}')
        # #class_0: non-target, class_1 is target


    
   



    



 



    
