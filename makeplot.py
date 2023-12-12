# This script generates Fig 2 in the paper. 
# Xiaoyu Yang, Nov 2023

import numpy as np
import matplotlib.pyplot as plt
# rewrite scipy loadmat/savemat function to make it compatible with Matlab
import scipy.io 
def savemat(file_name, mdict):
    data = scipy.io.savemat(file_name, mdict, do_compression=True)
    return data 
def loadmat(file_name): 
    data = scipy.io.loadmat(file_name, squeeze_me=True)
    return data

# load the sensory stimuli
stimuli = loadmat('stimuli.mat')
Q, eta = stimuli['Q'], stimuli['map'] # number of stimuli and sensory map

# --------------------------------------------------------------------------
# Compute the overlap between network activity and sensory stimuli
if 1:
    Ne = 800
    tbins = np.arange(0, 60001, 50)
    overlap = np.zeros((40, Q, len(tbins)-1))
    for k in range(overlap.shape[0]):
        spkdata = loadmat('spkdata/spkdata_%d.mat'%(k+1))['spkdata']
        fr = np.zeros((Ne, len(tbins)-1))
        for i in range(Ne):
            spktime, edges = np.histogram(spkdata[spkdata[:,1]==i,0], bins=tbins)
            fr[i,:] = spktime/np.diff(edges)*1000
        overlap[k,:,:] = (eta.T@fr)/np.sum(fr, axis=0) # See Eq.(8)
    savemat('overlap.mat', {'overlap': overlap})

# --------------------------------------------------------------------------
# Raster plot (Fig. 2a)
spkdata = loadmat('spkdata/spkdata_40.mat')['spkdata'] # 30 mins post-training
spkdata = spkdata[spkdata[:,0]<10000, :] 
overlap = loadmat('overlap.mat')['overlap'][39,:,:]
tbins = np.arange(0,60001,50)

plt.subplots(figsize=(4,4))
for k in range(Q):
    neurons = np.argwhere(eta[:,k]).flatten() # indices of neurons in the k-th cluster
    idx = np.argwhere(np.outer(np.ones(spkdata.shape[0]), neurons) == np.outer(spkdata[:,1], np.ones(neurons.size)))
    plt.scatter(spkdata[idx[:,0],0]/1000, idx[:,1]/neurons.size+k, s=1, edgecolors='none') # raster
    plt.stairs(overlap[k,:]+k, tbins/1000, color='k', linewidth=0.5, baseline=k) # overlap
    plt.xlim([0,10])
    plt.ylim([0,Q])
plt.xlabel('Time (s)')
plt.ylabel('Neurons')
plt.yticks([])
plt.box(False)
plt.tight_layout()

# --------------------------------------------------------------------------
# Synaptic matrix (Fig. 2b)
plt.subplots(figsize=(4,4))
net = loadmat('block_40.mat') # 30 mins post-training
S = net['S'] # synaptic matrix
neurons = [np.argwhere(eta[:,k]).flatten() for k in np.arange(Q)] # re-arrange neurons according to cluster membership
if 0: # show background population
    neurons.append(np.argwhere(np.all(eta==0, axis=1)).flatten())
neurons = np.hstack(neurons)
plt.imshow(S[np.ix_(neurons, neurons)], cmap='hot', vmax=0.5, extent=[0,1,0,1])
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.tight_layout()

# --------------------------------------------------------------------------
# Distribution of durations of cluster activations (Fig. 2c)
# Samples were taken from 20 mins PT to 30 mins PT
overlap = np.hstack([loadmat('overlap')['overlap'][i] for i in range(30, 40)]) 
max_overlap = np.amax(overlap, axis=0)
overlap = np.insert(overlap, 0, 1-max_overlap, axis=0)
# get index of active cluster (0 if none of them is active)
active_cluster = np.argmax(overlap, axis=0)
active_cluster = np.pad(active_cluster, (1,1), mode='constant', constant_values=0) # add zeros to the ends
# find durations of active states
states, durations = [], []
count = 0
current_state = 0
tbin = 50
for k in active_cluster:
    if current_state == 0:
        if k > 0:
            current_state = k
            count += 1
        continue 
    if k == current_state:
        count += 1
        continue 
    durations.append(count*tbin)
    states.append(current_state)
    current_state = k 
    count = 1*(k>0)
durations = np.array(durations)
states = np.array(states)

plt.subplots(figsize=(4,4))
weights = np.ones_like(durations)/len(durations)
plt.hist(durations, bins=np.arange(0,1501,100), weights=weights, edgecolor='k')
plt.xlabel('Duration (ms)')
plt.ylabel('Probability')
plt.tight_layout()

# --------------------------------------------------------------------------
# Time course of synaptic weights w1/w0 (Fig. 2d)
w0 = np.hstack([loadmat('spkdata/spkdata_%d.mat'%(Q+1+k))['w0'] for k in range(30)])
w1 = np.hstack([loadmat('spkdata/spkdata_%d.mat'%(Q+1+k))['w1'] for k in range(30)])

plt.subplots(figsize=(4,4))
plt.plot(np.arange(w1.shape[0])/6, w1, 'k', linewidth=1, label='$w_1$')
plt.plot(np.arange(w1.shape[0])/6, w0, 'k--', linewidth=1, label='$w_0$')
plt.xlabel('Post-training time (min)')
plt.ylabel('$w_{EE}$')
plt.xlim([0,30])
plt.legend(loc='center left')
plt.tight_layout()
plt.show()