# This script creates a spiking network model of EIF neurons and simulates it in the presence of ongoing synaptic plasiticty. 

# Network model and synaptic plasticity
# The network contains N=1000 EIF neurons, Ne=800 of them are excitatory and Ni=200 are inhibitory. 
# Only synapses between excitatory neurons are plastic and they are modified according to a voltage-dependent STDP rule. 
# The other synaptic weights (E->I, I->E, I->I) are non-plastic and constant. 

# Simulation
# We simulate the network for 40 mins, which can be divided into two parts:
# Training (10 mins): during the first 10 mins, a set of 10 pre-defined stimuli were presented to the network in random order. 
# Post-training (30 mins): Stimuli are removed but the synaptic weights are continuously being modified by the ongoing activity. 
# (same plasticity rule as during training)

# Scaling
# One could edit the section "Training protocol" to test a different network size or use a different training protocol. 

# Data
# The spking data of the k-th block are saved in the file "spkdata/spkdata_k.mat". 
# The snapshots of the network (e.g. synaptic matrix, vector of membrane potential, etc.) are saved in the file "block_k.mat"
# Xiaoyu Yang, Nov 2023. 
# -------------------------------------------------------------------------- 
# set number of CPUs to run on (set this before importing numpy)
import os
# ncore = "1"
# os.environ["OMP_NUM_THREADS"] = ncore
# os.environ["OPENBLAS_NUM_THREADS"] = ncore
# os.environ["MKL_NUM_THREADS"] = ncore
# os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
# os.environ["NUMEXPR_NUM_THREADS"] = ncore
# -------------------------------------------------------------------------- 
import numpy as np 
from time import time
# rewrite scipy loadmat/savemat function to make it compatible with Matlab
import scipy.io 
def savemat(file_name, mdict):
    data = scipy.io.savemat(file_name, mdict, do_compression=True)
    return data 

def loadmat(file_name): 
    data = scipy.io.loadmat(file_name, squeeze_me=True)
    return data
# --------------------------------------------------------------------------    
# network initiation
def initNet(N):
    # neural and network parameters
    net = {
        'Vr': 0, # reset potential, mV
        'VT': 20, # soft spiking theshold, mV
        'Vpeak': 25, # peak potential, mV
        'tauE': 15, # ms, exc membrane time constant
        'tauI': 10, # ms, inh membrane time constant
        'tausE': 3, # ms, exc synaptic time constant
        'tausI': 2, # ms, inh synaptic time constant
        'Ne': int(0.8*N), # number of exc neurons
        'Ni': int(0.2*N), # number of inh neurons
        'pee': 0.2, # exc->exc connectivity
        'pei': 0.5, # inh->exc connectivity
        'pie': 0.5, # exc->inh connectivity
        'pii': 0.5, # inh->inh connectivity
        'wee': 0.005, # initial exc->exc synaptic efficacy, mV
        'wei': 0.34, 
        'wie': 0.54,
        'wii': 0.46,
        'heext': 1.50, # external current to exc neurons, mV/ms
        'hiext': 2.19, # external current to inh neurons, mV/ms
        'hst': 0.5, # stimulus current, mV/ms
    }
    # synaptic matrix
    See = net['wee']*(np.random.rand(net['Ne'],net['Ne'])<net['pee'])
    Sei = net['wei']*(np.random.rand(net['Ne'],net['Ni'])<net['pei'])
    Sie = net['wie']*(np.random.rand(net['Ni'],net['Ne'])<net['pie'])
    Sii = net['wii']*(np.random.rand(net['Ni'],net['Ni'])<net['pii'])
    S = np.block([[See, Sei], [Sie, Sii]])
    np.fill_diagonal(S, 0) # remove self-connections

    # network initiation
    net['S'] = S # synaptic matrix
    net['P'] = 1*(S!=0) # connectivity matrix
    net['V'] = np.random.normal(0,5,N) # initial membrane potential 
    net['hsynE'] = 0*net['V'] # initial exc synaptic current
    net['hsynI'] = 0*net['V'] # initial inh synaptic current

    # learning-related variables (post-synaptic)
    net['sE'] = np.zeros(net['Ne']) # tilde_s
    net['vE'] = np.zeros(net['Ne']) # tilde_v
    net['uE'] = np.zeros(net['Ne']) # same as theta (dynamical threshold)
    return net
# -----------------------------------------------------------------------------
# learning parameters (default)
lr = {
    'taus': 1000, # time constant of tilde_s, ms
    'tauv': 50, # time constant of tilde_v, ms
    'tauu': 500, # time constant of theta, ms
    'g': 5, # gain factor
    'gamma': 50, 
    'beta': 0.1,
    'Altp': 0.005, # LTPe rate
    'Altd': 0.015, # LTDe rate
}
# -----------------------------------------------------------------------------
# Training protocol (this is the only section one should edit)
N = 1000 # total number of neurons 
Q = int(0.01*N) # number of sensory stimuli 
f = 1./Q # coding level 

# The spiking data was saved every 60 seconds (called one block). 
blocks = []
# warmup
blocks.append({'lr_on': 0, 'stim_on': 0, 'savenet': 0})
# training (10 mins)
[blocks.append(
    {'lr_on':1, 'stim_on':1, 'savenet': np.mod(k+1,Q)==0, 'stim_idx': np.random.randint(0,Q,30)}
    ) for k in np.arange(Q)] # 10 mins t
# post-training (30 mins)
[blocks.append(
    {'lr_on':1, 'stim_on':0, 'savenet': np.mod(k+1,30)==0}
    ) for k in np.arange(30)] 
# --------------------------------------------------------------------------    
# Simulation
block_number = 0 # starting block 
if not os.path.exists('spkdata'):
    os.makedirs('spkdata')  # create a folder called "spkdata"
# network initialization (if start with block 0)
if block_number == 0:
    net = initNet(N)
    stimuli = {
        'Q': Q, 
        'f': f, 
        'map': 1*(np.random.rand(net['Ne'],Q)<f) # define the membership of each exc neuron
    }
    savemat('stimuli.mat', stimuli)
# continue a previous simulation (if not start with block 0)
if block_number > 0:
    net = loadmat('block_%d.mat'%(block_number-1))
    stimuli = loadmat('stimuli.mat')
    
S = net['S']
P = net['P']
See = S[:net['Ne'],:net['Ne']] 
Pee = P[:net['Ne'],:net['Ne']] 
V = net['V']
hsynE = net['hsynE']
hsynI = net['hsynI']
sE = net['sE']
vE = net['vE']
uE = net['uE']
tauv = np.r_[net['tauE']+np.zeros(net['Ne']), net['tauI']+np.zeros(net['Ni'])]
hext = np.r_[net['heext']+np.zeros(net['Ne']), net['hiext']+np.zeros(net['Ni'])]

for block in blocks[block_number:]: 
    t_start = time()
    dt = 0.1 # time step
    totalT = 60000 # ms
    timev = np.arange(0, totalT, dt)
    stim_start = np.arange(500, totalT, 2000) # starting time of stimuli
    stim_end = stim_start+500 # end time of stimuli
    fired, firedE, firedI = [], [], [] # list of indices of fired neurons
    # recording
    spkdata = [np.empty((0,2))]
    sharedStimuli = stimuli['map'] @ stimuli['map'].T # number of shared stimuli by neuron (i,j)
    w0, w1 = [], [] # average weights
    for t in timev:
        # -----------------------------------------------------------------------------
        # stimulus presentation
        hstim = 0*V
        if block['stim_on']:       
            k = np.argwhere(np.logical_and(t>=stim_start, t<stim_end)).flatten() # k-th presentation
            if k.size > 0: 
                stim_idx = block['stim_idx'][k[0]]
                hstim[0:net['Ne']] = net['hst']*stimuli['map'][:,stim_idx]
        # -----------------------------------------------------------------------------
        # recurrent synaptic current 
        hsynE = hsynE * (1-dt/net['tausE'])
        hsynI = hsynI * (1-dt/net['tausI'])
        for idx in firedE:
            hsynE += S[:,idx]/net['tausE']
        for idx in firedI:
            hsynI -= S[:,idx]/net['tausI']
        # -----------------------------------------------------------------------------
        # membrane potential dynamics
        htotal = hsynE + hsynI + hext + hstim
        expV = np.exp(V-net['VT']) # exponential term
        V += -(V-expV)*dt/tauv+htotal*dt
        # -----------------------------------------------------------------------------
        # detect spikes
        fired = np.argwhere(V>=net['Vpeak']).flatten()
        firedE = [idx for idx in fired if idx < net['Ne']]
        firedI = [idx for idx in fired if idx >= net['Ne']]
        # -----------------------------------------------------------------------------
        # synaptic plasticity
        sE = sE - sE*dt/lr['taus']
        sE[firedE] += 1/lr['taus']
        vE = vE + (-vE+expV[:net['Ne']])*dt/lr['tauv'] # tilde_v
        uE = uE + np.tanh(lr['g']*(-uE+vE+lr['gamma']*sE))*dt/lr['tauu'] # theta 
        xE = vE - uE # v-theta
        # updating synaptic weights
        if block['lr_on'] and len(firedE)>0:
            See[:,firedE] += (lr['Altp']*np.exp(-lr['beta']*See[:,firedE]**2)*np.outer(xE*(xE>0), np.ones(len(firedE))) \
                + lr['Altd']*np.outer(xE*(xE<=0),np.ones(len(firedE))))*Pee[:,firedE]
            See[:,firedE] = np.maximum(See[:,firedE],0)
        # -----------------------------------------------------------------------------
        # record spiking data
        if len(fired) > 0: 
            V[fired] = net['Vr']
            [spkdata.append([t, idx]) for idx in fired]
        # -----------------------------------------------------------------------------
        # compute w0 and w1 
        if np.mod(t,10000) == 0:
            w0.append(np.mean(See[np.logical_and(sharedStimuli==0,Pee==1)]))
            w1.append(np.mean(See[np.logical_and(sharedStimuli>0,Pee==1)]))
        
    # save data
    spkdata = np.vstack(spkdata)
    w0 = np.vstack(w0)
    w1 = np.vstack(w1)
    savemat('spkdata/spkdata_%d.mat'%block_number, {'spkdata': spkdata, 'w0':w0, 'w1':w1})
    if block['savenet']:
        net['S'] = S
        net['P'] = P
        net['V'] = V 
        net['hsynE'] = hsynE 
        net['hsynI'] = hsynI 
        net['sE'] = sE
        net['vE'] = vE 
        net['uE'] = uE 
        savemat('block_%d.mat' % block_number, net)
    print('Block %d: ElapsedTime = %.1fs, w1=%.3f, w0=%.3f' % (block_number, time()-t_start, w1[-1], w0[-1]) )   
    block_number += 1