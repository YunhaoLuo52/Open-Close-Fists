#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Eight‐Module PsychoPy Experiment with Module Selection Dialog
and Integrated Camera Recording

Each module consists of 50 trials. On each trial:
  - Show a 2 s instruction (“task”) screen (or longer for modules 1 & 2)
  - Send a “start” marker via LSL
  - Show a small black cross for 2 s ± 0.2 s (or 4 s ± 0.5 s for modules 1 & 2)
  - Send an “end” marker via LSL

At launch, a small dialog lets the user pick exactly one of the eight modules.
If “Cancel” is clicked, the experiment quits. Otherwise the chosen module’s trials run.
After finishing—or if ESC is pressed—a goodbye screen appears, a final 999 marker is sent,
the camera thread is stopped, and the recording is saved to the same folder as the EEG data.

This script expects to be called as:
    python eye_multiTask.py -p <participant_id> -s <start_time_unix>

It will reconstruct the data path in “../Data/sub-<participant>/Muse_data_eyeMotion”
and save the camera file there as “camera_recording_<timestamp>.avi”.

Requires:
  • PsychoPy 2023.2.3 (or newer)
  • pylsl (for LSL markers)
  • OpenCV (cv2) for camera capture
  • threading (standard library)
  • argparse (standard library)
"""
import os
import time
import threading
import cv2
import random
import argparse

from psychopy import core, visual, event, gui
from pylsl import StreamInfo, StreamOutlet

# ------------------------------------------------------------------------
# Global variables for camera thread control
# ------------------------------------------------------------------------
stop_event = None
cam_thread = None

# ------------------------------------------------------------------------
# Argument parsing (so we know where to save EEG + camera) from eye_multiTask_start.py
# ------------------------------------------------------------------------
parser = argparse.ArgumentParser()
#method1: 
parser.add_argument('-p', '--participant', required=True,
                    help='Participant ID (used to construct data folder)')
parser.add_argument('-s', '--starttime', required=True, type=int,
                    help='Unix timestamp for session start (passed from launcher)')
args = parser.parse_args()
participant = args.participant

# Build the data folder path:
cwd = os.getcwd()
data_path_relative = os.path.join('..', 'Data', f"sub-{participant}", 'Muse_data_eyeMotion')
data_path = os.path.join(cwd, data_path_relative)
os.makedirs(data_path, exist_ok=True)
output_filename=os.path.join(data_path,"%s_%s_recording_%s.avi" %
                            (''.join(['sub-', participant]),
                             'video',
                             time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime())))
#video_filename = video_filename.replace('\\', '/')  # Replace backslashes with slashes

# timestamp = time.strftime("%Y%m%d_%H%M%S")
# output_filename = os.path.join(data_path, f"camera_recording_{timestamp}.avi")

# #method2: use output_filename which is passed from *_start.py bud didn't work!
# parser.add_argument('-f', '--video_filename', required=True,
#                     help='video filename)')
# args = parser.parse_args()
# #output_filename = args.video_filename
# # print(output_filename)
# # Normalize path (optional)
# output_filename = os.path.normpath(args.video_filename)

#output_filename=r"D:\Faculty\ColumbiaUniversity\dataprocess\EEG\dataProcessing\Interaxon\museS\LSL\Python\muse-lsl-python\Data\sub-EB-41\Muse_data_eyeMotion\video.avi"
#print(output_filename)

# ------------------------------------------------------------------------
# LSL Setup
# ------------------------------------------------------------------------
info = StreamInfo('Markers', 'Markers', 1, 0, 'int32', 'myuidw43536')
outlet = StreamOutlet(info)

# ------------------------------------------------------------------------
# Experiment Configuration
# ------------------------------------------------------------------------
modules = [
    (1, "Open and close your both fists"),
    (2, "Open and close your both feet"),
    (3, "Keep your eyes open"),
    (4, "Blink your eyes once"),
    (5, "Move your eyeballs horizontally"),
    (6, "Move your eyeballs vertically"),
    (7, "Chewing"),
    (8, "Open and close your mouth")
]

# ------------------------------------------------------------------------
# Helper to ensure camera stops before quitting
# ------------------------------------------------------------------------
def safe_quit():
    """Stops camera recording if it's running, then sends final marker 999 and quits PsychoPy."""
    global stop_event, cam_thread
    # Send 999 marker before quitting
    try:
        outlet.push_sample([999], time.time())
    except:
        pass
    if stop_event is not None:
        stop_event.set()
    if cam_thread is not None:
        cam_thread.join()
    core.quit()

# ------------------------------------------------------------------------
# Shared helper functions
# ------------------------------------------------------------------------
def wait_for_enter_or_escape():
    """Wait until ENTER is pressed or ESC is detected (to quit)."""
    while True:
        keys = event.getKeys()
        if 'return' in keys:
            return
        if 'escape' in keys:
            safe_quit()
        core.wait(0.01)

def show_stim_for_duration(win, stim, duration):
    """
    Draw ‘stim’ in window ‘win’ for exactly ‘duration’ seconds,
    quitting immediately if ESC is pressed.
    """
    clock = core.Clock()
    while clock.getTime() < duration:
        stim.draw()
        win.flip()
        if 'escape' in event.getKeys():
            safe_quit()
        core.wait(0.01)

# ------------------------------------------------------------------------
# Camera recording thread function
# ------------------------------------------------------------------------
def camera_record(output_path, stop_event_local):
    """
    Opens the default camera (index 0) and records frames to `output_path`
    until `stop_event_local` is set. Saves as AVI at 30 FPS.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while not stop_event_local.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()

# ------------------------------------------------------------------------
# –– Before Experiment: Module Selection Dialog ––
# ------------------------------------------------------------------------
module_labels = [f"{idx}: {label}" for (idx, label) in modules]
selected_module = None

while selected_module is None:
    dlg = gui.Dlg(title="Select Module to Run")
    dlg.addField('Module', choices=module_labels, initial=module_labels[0])
    dlg.show()
    if dlg.OK:
        selected_module = dlg.data['Module']
    else:
        safe_quit()

# Extract numeric index (e.g. "3: Keep your eyes open" → 3)
module_index = int(selected_module.split(':')[0])

# ------------------------------------------------------------------------
# Main Experiment Function
# ------------------------------------------------------------------------
def run_task_experiment():
    global stop_event, cam_thread

    # — Setup Window (fullscreen) —
    win = visual.Window(fullscr=True, color='grey', units='norm')

    # — Prepare stimuli —
    welcome_text = visual.TextStim(
        win=win,
        text="Welcome!\n\nPress ENTER to begin.",
        color='black',
        height=0.1,
        wrapWidth=1.5
    )
    goodbye_text = visual.TextStim(
        win=win,
        text="Goodbye!\n\nPress ESC to exit.",
        color='black',
        height=0.1,
        wrapWidth=1.5
    )
    cross_stim = visual.TextStim(
        win=win,
        text='+',
        color='black',
        height=0.2,
        bold=True
    )

    # Determine which module was chosen
    mod_idx, mod_label = next(item for item in modules if item[0] == module_index)

    # Default durations (modules 3–8)
    stimuDur = 2.0    # “task” instruction duration
    interLow = 2.0    # cross interval lower bound
    interHigh = 2.5   # cross interval upper bound

    # Override for modules 1 & 2
    if mod_idx in (1, 2):
        stimuDur = 4.0
        interLow = 4.0
        interHigh = 4.5

    # Hard‐code start/end markers
    if mod_idx == 1:
        marker_start = 101; marker_end = 201
    elif mod_idx == 2:
        marker_start = 102; marker_end = 202
    elif mod_idx == 3:
        marker_start = 103; marker_end = 203
    elif mod_idx == 4:
        marker_start = 104; marker_end = 204
    elif mod_idx == 5:
        marker_start = 105; marker_end = 205
    elif mod_idx == 6:
        marker_start = 106; marker_end = 206
    elif mod_idx == 7:
        marker_start = 107; marker_end = 207
    elif mod_idx == 8:
        marker_start = 108; marker_end = 208
    else:
        marker_start = 0;   marker_end = 0

    task_text = visual.TextStim(
        win=win,
        text=mod_label,
        color='black',
        height=0.1,
        wrapWidth=1.5
    )

    # — Welcome screen —
    welcome_text.draw()
    win.flip()
    wait_for_enter_or_escape()

    # — Transition to chosen module —
    transition = visual.TextStim(
        win=win,
        text=f"Next task: {mod_label}",
        color='black',
        height=0.1,
        wrapWidth=1.5
    )
    show_stim_for_duration(win, transition, 2.0)

    # — Start camera recording in a separate thread —
    # #method1:
    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    # output_filename = os.path.join(data_path, f"camera_recording_{timestamp}.avi")
    #method2: hxf 6/7/2025 use para passed from *_start.py

    stop_event = threading.Event()
    cam_thread = threading.Thread(target=camera_record, args=(output_filename, stop_event))
    cam_thread.start()

    # — Main trials for the selected module —
    num_trials = 50
    #num_trials = 2 #testing purpose
    for trial in range(num_trials):
        # Begin‐of‐trial marker
        outlet.push_sample([marker_start], time.time())

        # Show the “task” instruction
        show_stim_for_duration(win, task_text, stimuDur)

        # End‐of‐trial marker
        outlet.push_sample([marker_end], time.time())

        # Show black cross for inter‐trial interval
        interval = random.uniform(interLow, interHigh)
        show_stim_for_duration(win, cross_stim, interval)

    # — Goodbye screen —
    goodbye_text.draw()
    win.flip()
    while True:
        if 'escape' in event.getKeys():
            break
        core.wait(0.01)

    # Send final marker 999
    outlet.push_sample([999], time.time())

    # — Stop camera recording and wait for thread to finish —
    stop_event.set()
    cam_thread.join()

    # Cleanup
    win.close()
    core.quit()

# ------------------------------------------------------------------------
# Run the experiment
# ------------------------------------------------------------------------
if __name__ == "__main__":
    run_task_experiment()
