#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Eight‐Module PsychoPy Experiment with Module Selection Dialog

Each module consists of 50 trials. On each trial:
  - Show a 2 s instruction (“task”) screen
  - Send a “start” marker via LSL
  - Show a small black cross for 2 s ± 0.2 s
  - Send an “end” marker via LSL

At launch, a small dialog lets the user pick exactly one of the eight modules.
If “Cancel” is clicked, the experiment quits. Otherwise the chosen module’s trials run; all other modules are skipped.
After finishing, a goodbye screen appears until the participant presses ESC.

Markers (via pylsl) are hard‐coded so that Module 1 → 101/201, Module 2 → 102/202, …, Module 8 → 108/208.
A final marker 999 is sent once the module is complete.
"""
from psychopy import core, visual, event, gui
import random
import time
from pylsl import StreamInfo, StreamOutlet

#added 6/6/2025
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

def create_marker_outlet():
    """Set up an LSL outlet for sending integer markers."""
    info = StreamInfo('Markers', 'Markers', 1, 0, 'int32', 'myuidw43536')
    return StreamOutlet(info)


# ------------------------------------------------------------------------
# Shared helper functions
# ------------------------------------------------------------------------
def wait_for_enter_or_escape():
    """Wait until ENTER is pressed or ESC is detected (to quit)."""
    while True:
        keys = event.getKeys()
        if 'return' in keys:
            break
        if 'escape' in keys:
            core.quit()
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
            core.quit()
        core.wait(0.01)


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
        # Access by field name "Module" instead of dlg.data[0]
        selected_module = dlg.data['Module']
    else:
        # If the user clicks “Cancel,” quit immediately.
        core.quit()

# Extract the numeric index from the label string (e.g. "3: Open your eyes" → 3)
module_index = int(selected_module.split(':')[0])


# ------------------------------------------------------------------------
# Main Experiment Function
# ------------------------------------------------------------------------
def run_task_experiment():
    # — Setup Window (fullscreen) and LSL outlet —
    win = visual.Window(fullscr=True, color='grey', units='norm')
    #outlet = create_marker_outlet() #no need here because defined in the begining, 6/6/2025

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

    #hxf 6/6/2025
    # Hard‐code start/end markers for the chosen module
    stimuDur=2.0 #Show the “task” instruction for exactly 2 s
    # #Show black cross for 2 s ± 0.2 s
    # interTrialInterval_low=1.8 #1.8s
    # interTrialInterval_up=2.2 #2.2s
    #Show black cross for 2 s + 0.5 s
    interTrialInterval_low=2.0 #2.0s
    interTrialInterval_up=2.5 #2.5s

    if mod_idx == 1:
        marker_start = 101
        marker_end = 201
        stimuDur=4.0 #Show the “task” instruction for exactly 4 s
        #Show black cross for 4 s ~4.5 s
        interTrialInterval_low=4.0 #4 s
        interTrialInterval_up=4.5 #4.5s
    elif mod_idx == 2:
        marker_start = 102
        marker_end = 202
        stimuDur=4.0 #Show the “task” instruction for exactly 4 s
        #Show black cross for 4 s ~4.5 s
        interTrialInterval_low=4.0 #4 s
        interTrialInterval_up=4.5 #4.5s
    elif mod_idx == 3:
        marker_start = 103
        marker_end = 203
    elif mod_idx == 4:
        marker_start = 104
        marker_end = 204
    elif mod_idx == 5:
        marker_start = 105
        marker_end = 205
    elif mod_idx == 6:
        marker_start = 106
        marker_end = 206
    elif mod_idx == 7:
        marker_start = 107
        marker_end = 207
    elif mod_idx == 8:
        marker_start = 108
        marker_end = 208
    else:
        marker_start = 0
        marker_end = 0

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

    # — Main trials for the selected module —
    num_trials = 50
    for trial in range(num_trials):
        # Begin‐of‐trial marker
        outlet.push_sample([marker_start], time.time())

        # Show the “task” instruction for exactly 2 s
        show_stim_for_duration(win, task_text, stimuDur)

        # End‐of‐trial marker
        outlet.push_sample([marker_end], time.time())

        # Show black cross for 2 s ± 0.2 s
        interval = random.uniform(interTrialInterval_low, interTrialInterval_up)
        show_stim_for_duration(win, cross_stim, interval)

    # — Goodbye screen —
    goodbye_text.draw()
    win.flip()
    # Wait for ESC to quit
    while True:
        if 'escape' in event.getKeys():
            break
        core.wait(0.01)

    # Final marker to indicate the module has ended
    outlet.push_sample([999], time.time())

    # Cleanup
    win.close()
    core.quit()


# ------------------------------------------------------------------------
# Run the experiment if this script is executed directly
# ------------------------------------------------------------------------
if __name__ == "__main__":
    run_task_experiment()
