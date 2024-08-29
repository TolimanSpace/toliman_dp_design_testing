"""
    Find the most still period in data acquisition and take frame from there
"""

import numpy as np
import matplotlib.pyplot as plt

fname =  "/Volumes/Morgana2/gpir9156/toliman/glued/15_08_red_149us_19.5gain_img_stack_batch_0.npy"
data = np.load(fname)
max_px_idx = np.unravel_index(np.argmax(data[0,:], axis=None), data[0,:].shape) # follow the brightest pixel

vals = data[:,max_px_idx[0],max_px_idx[1]]/100 # to avoid bit precision errors when calc diff, make smaller (data is uint16)
abs_diff = np.abs(np.diff(vals))
thresh_diff = abs_diff.min() + 0.5*np.std(abs_diff)

diff_frames = np.arange(len(vals)-1)
diff_groups = abs_diff[abs_diff < thresh_diff]
group_frames = diff_frames[abs_diff < thresh_diff]

# Count the size of low-variance frame bunches. Pick largest bunch as containing the most still frame
branches = np.abs(np.diff(group_frames))
branch_lengths = []
branch_end_idx = []
count = 0
for i in range(len(branches)):
    # bunch defined by consecutive grouped-frame idxs (i.e where branches = 1)
    if branches[i] == 1:
        count += 1
    else:
        branch_lengths.append(count)
        branch_end_idx.append(group_frames[i])
        count = 0

branch_lengths = np.asarray(branch_lengths)
branch_end_idx = np.asarray(branch_end_idx)
longest_branch = branch_lengths.max()
longest_branch_idx = int(branch_end_idx[np.argmax(branch_lengths)] - longest_branch/2) # take middle
pt_w_longest_branch = longest_branch_idx + 1 # middle (since we've performed diff twice to vals)

print("Most still frame idx: {}".format(pt_w_longest_branch))

plt.figure(figsize=(15,6))
plt.subplot(2,1,1)
plt.plot(vals, '--o', label="Bright Pixel Instensity")
plt.plot(pt_w_longest_branch, vals[pt_w_longest_branch], 'ro', label="Most still frame")
plt.title("Single Bright Px Intensity")
plt.grid(axis='y')
plt.ylabel("Intensity")
plt.subplot(2,1,2)
plt.plot(abs_diff, '--o', label="Abs Difference")
plt.plot(group_frames, diff_groups, 'o', label= "Diff < min + 0.5sigma")
plt.plot(longest_branch_idx, abs_diff[longest_branch_idx], 'ro', label="Most still frame")
plt.legend()
plt.xlabel("Frame")
plt.grid(axis='y')
plt.show()