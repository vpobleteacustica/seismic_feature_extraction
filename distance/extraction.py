#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
from scipy import signal
from scipy import stats
import sklearn.metrics as mt
import re


def remove_peak(y_cepstrum, z_score = 0.20, max_lim = 3):
    count = 0 ## elementos que se quitarán
    w = 0 ## espera
    while count !=max_lim:
        z                 = np.abs(stats.zscore(np.abs(y_cepstrum)))
        masc              = z < z_score
        unique, counts    = np.unique(masc, return_counts= True)
        count = counts[0]
        
        if count < max_lim:
            z_score -= 1e-5  # 0.00001
            #w-=1
        elif count > max_lim:
            z_score += 1e-5  # 0.00001
            #w+=1
        #if w > 200:
        #    z_score += 1e-3
        #    w = 0
        #    
        #elif w < -200:
        #    z_score -= 1e-3
        #    w = 0
    #print('score encontrado')        

    abs_y_cepstrum    = np.abs(y_cepstrum)
    abs_y_cepstrum_np = abs_y_cepstrum[masc]
    masc_exit         = z > z_score
    mag_exit          = abs_y_cepstrum[masc_exit]
    #print('mascara aplicada')
    t_np              = np.linspace(0, len(abs_y_cepstrum_np), len(abs_y_cepstrum_np))
    auc = mt.auc(t_np, abs_y_cepstrum_np)
    #print('auc')
    return abs_y_cepstrum_np, mag_exit, np.asarray(auc)

def read_sac(tr, inv = None, remove_response = True):
    if remove_response:
        tr.remove_response(inventory = inv)  
    y    = tr.data
    sr   = int(tr.stats.sampling_rate)
    return y,sr

def sac_properties(tr):
    #Magnitude
    mag = tr.stats.sac.mag
    mag = float(str(round(mag, 1)))
    #Distance
    dist = tr.stats.sac.dist
    dist = float(str(round(dist,1)))
    return mag,dist
    
def inv_properties(inv):
    #Instrument sensitivity / Gain
    G = inv.networks[0][0][0].__dict__['response'].__dict__['instrument_sensitivity'].__dict__['value']
    return G

def temporal_energy(y,sr):
    E = sum(y**2)*(1/sr)
    return E

def fourier_transform(y,sr,nfft,frames=True):
    if frames:
        f,t,Y = signal.stft(y,fs=sr,nperseg=2*int(sr),nfft=nfft)
    else:
        Y = fft(y,nfft)/sr
    return Y
def mag_log_mag_fourier(Y):
    eps = 10**-3
    Y_mag = np.abs(Y) # |Y(w)|
    Y_hat = np.log(Y_mag + eps) # log(|Y(w)|)
    return Y_hat
def cepstrum(Y_hat,sr,npts,frames=True):
    if frames:
        y_hat = []
        for i in range(Y_hat.shape[1]):
            y_hat_i = ifft(Y_hat[:,i],npts)*sr
            y_hat.append(y_hat_i)
        y_hat = np.array(y_hat)
    else: 
        y_hat = ifft(Y_hat,npts)*sr
    return np.abs(y_hat)
def transform(y,sr,npts,frames=True):
    #print('true or false?sss',frames)
    Y = fourier_transform(y,sr,npts,frames)
    Y_hat = mag_log_mag_fourier(Y)
    y_hat = cepstrum(Y_hat,sr,npts,frames)
    shape_hat = y_hat.shape
    return y_hat

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def max_padding(y, sr, max_len):
    leny = len(y)/sr
    if leny < max_len:
        #y_new = np.zeros(max_len*sr)
        y_new = np.ones(max_len*sr)*min(np.abs(y))
        for i in range(len(y)):
            y_new[i] = y[i]
    else:
        y_new = y
    return y_new

def up_level_idx(y,n,p,l): #Subida
    nn=round((len(y)-n)/p)
    yy=np.zeros(nn)
    for i in range(nn):
        subarray=y[i*p:i*p+n]
        yy[i]=sum(abs(subarray))/n*p
    max_yy = max(yy)
    yy=yy/max_yy
    ix=np.zeros(l)
    vx=np.zeros(l)
    for i in range(l):
        i1=i+1
        ixi=np.argwhere(np.array(yy)>=i1/l)[0]
        ix[i]=ixi
        vx[i]=yy[ixi]
    return ix*p, max_yy*vx

def down_level_idx(y,n,p,l): #Bajada
    nn=round((len(y)-n)/p)
    yy=np.zeros(nn)
    for i in range(nn):
        subarray=y[i*p:i*p+n]
        yy[i]=sum(abs(subarray))/n*p
    max_yy = max(yy)
    yy = yy/max(yy)
    ix = np.zeros(l)
    vx = np.zeros(l)
    i1 = l
    for i in range(l):
        ixi=np.argwhere(np.array(yy)>=i1/l)[-1]
        ix[i]=ixi
        vx[i]=yy[ixi]
        i1  -= 1
    return ix*p, max_yy*vx