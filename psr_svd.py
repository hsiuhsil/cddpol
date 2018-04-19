import sys, os, argparse
import os.path

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
#from baseband import mark4
#from pulsar.predictor import Polyco
import astropy.units as u
import math
import random
from random import gauss
random.seed(3)
from scipy import fftpack, optimize, interpolate, linalg, integrate
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.optimize import curve_fit, minimize
from mpi4py import MPI

import copy
import glob
import paras

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

rawdata_folder = '/mnt/scratch-lustre/hhlin/Data/'
time_str_folder = '/mnt/raid-project/gmrt/hhlin/time_streams_1957/'
#time_str_patterns = time_str_folder + 'gp052*536s_h5'
time_str_patterns = time_str_folder + 'gp052*_ar_*536s_h5'
phase_amp_files = time_str_folder + 'gp052_fit_nodes_1_tint_8.0sec_npy/*nodes*npy'
TOAs_files = time_str_folder + 'gp052_TOA_nodes_1_tint_8.0sec_npy/*nodes*npy'
plots_path = '/mnt/raid-cita/hhlin/psr_1957/cddpol/profiles_fitting_plots/'
#TOAs_files = time_str_folder + 'gp052_TOA_nodes_1_tint_8.0sec_npy_zerocentre/*nodes*npy'

NHARMONIC = paras.NHARMONIC 
#NMODES = paras.NMODES

NPHASEBIN = paras.NPHASEBIN
NCENTRALBINS = paras.NCENTRALBINS
NCENTRALBINSMAIN = paras.NCENTRALBINSMAIN
chi2_samples_const = paras.chi2_samples_const

tint = paras.tint
#prof_stack = paras.prof_stack
#tint_stack = paras.tint*paras.prof_stack

SR = paras.SR
dt = paras.dt
N, DN = paras.N, paras.DN
block_length = paras.block_length
fedge = paras.fedge
fref = paras.fref
ngate = paras.ngate

with open("psrb1957+20.par") as fp:
    for i, line in enumerate(fp):
        if i == 6:
            F0 = float(str(line[15:35]))
        if i == 10:
            PEPOCH = int(str(line[15:20]))
P0 = 1/F0
print 'P0', P0
print 'PEPOCH', PEPOCH

#args = CL_Parser()
band = int(sys.argv[2])
print band

# list the epochs and sessions
dict = {}
epochs = ['a', 'b', 'c', 'd']
sessions = [3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22]
for i in range(len(epochs)*len(sessions)):
    dict[i] = epochs[i/len(sessions)]+str(sessions[i%len(sessions)])

start, end = 2, 3 # decide the sessions of starting and end(included)


def not_main():
    if False: #combine all folding data in one single array.

        single_fold_shape = (int(8*paras.T/paras.tint), NPHASEBIN*2)
        fold_data = np.zeros((len(glob.glob(time_str_patterns))*single_fold_shape[0], single_fold_shape[1]))

        count = 0
        for f in sorted(glob.glob(time_str_patterns)):
           ff = h5py.File(f, 'r')
           print('ff',ff)
           fold_data[count*single_fold_shape[0]:(count+1)*single_fold_shape[0], :] = ff['fold_data_int_'+str(paras.tint)+'_band_'+str(band)]            
           count += 1
        print 'finished combining folding data'

    if False: #reform the format of raw data
        '''raw_data is in the shape of (pulse rotation, phase, pol)'''
        raw_data = np.load('/scratch2/p/pen/hsiuhsil/psr_B1957+20/data_file/B1957pol3_512g_2014-06-15T07:06:23.00000+536s.npy')
        '''Separate raw__data into L, R pols, which in the shape of ((pulse rotation, phase, L/R pol))'''
        data = np.sum(raw_data.reshape(raw_data.shape[0], raw_data.shape[1], 3,2), axis=2)
        L_data = np.zeros((data[:,:,0].shape))
        R_data = np.zeros((data[:,:,1].shape))
        for ii in xrange(len(L_data)):
            L_data[ii] = data[ii,:,0] - np.mean(data[ii,:,0])
            R_data[ii] = data[ii,:,1] - np.mean(data[ii,:,1])
        B_data = np.concatenate((L_data, R_data), axis=1) # B means both of L and R
        np.save('B_data.npy', B_data)
        print 'save B_data'

#    B_data = np.load('/scratch2/p/pen/hsiuhsil/psr_B1957+20/data_file/B_data.npy')
    if False: # Stack the pulse profiles
        profile_stack = 50
        B_data_stack = stack(B_data, profile_stack)
        print 'B_data_stack.shape: ', B_data_stack.shape
        
    if False:
        rebin_pulse = 1
        filename = 'B_data_rebin_' + str(rebin_pulse)
        B_data_rebin = B_data_stack#rebin_spec(B_data, rebin_pulse, 1)
        np.save(filename + '.npy', B_data_rebin)
        print 'B_data_rebin.shape', B_data_rebin.shape
#        svd(B_data_rebin, rebin_pulse)
        plot_svd(B_data_rebin, rebin_pulse, filename)

    if False:
        t_end = 300 #sec
        print 't_end: ', t_end
        tint_stack = paras.tint*paras.prof_stack
        times = np.arange(0, t_end, tint_stack)
        phases_amps_1 = np.load('gp052a_ar_no0007fit_nodes_1_tint_30.0.npy')
        phases_amps_2 = np.load('gp052a_ar_no0007fit_nodes_2_tint_30.0.npy')

        plot_phase_lik(times, phases_amps_1[:len(times)], tint_stack, 'phase_lik_nodes_1_tint_'+str(tint_stack)+'sec.png')
        plot_phase_lik(times, phases_amps_2[:len(times)], tint_stack, 'phase_lik_nodes_2_tint_30.0sec.png')

#    process_profiles(profile, pattern, NMODES, tint_stack, V_recon=None, fit_profile=False, save_V=False)

def main():

    plot_spec_dedisperse()
    print 'done spec dedisperse'
#    plot_ut_bands_correlation()
#    print 'done cor plots'

    prof_stack = [1]#[512, 256, 128, 64, 32, 16, 8, 4, 2, 1]#[1, 2, 4, 8, 16, 32, 64]
    NMODES = [1,2]

#    rms_stack = np.zeros((len(prof_stack), len(NMODES), 3))

    rms_name = 'gp052_' + dict[start]+'_to_'+dict[end] + '_band_'+str(band)+'_rms'
    rms_file = rms_name+'.npy'

    if os.path.isfile(rms_file) is True:
        rms = np.load(rms_file).item()
    else:
        rms = {}
 
    for ii in xrange(len(prof_stack)):
        for jj in xrange(len(NMODES)):
#            pattern_refit = NMODES_stack(prof_stack[ii], NMODES[jj])[0]
            key = 'band_'+str(band)+'_modes_'+str(NMODES[jj])+'_tint_'+str(prof_stack[ii]*tint)
            rms[key] = NMODES_stack(prof_stack[ii], NMODES[jj])
        np.save(rms_file, rms)

    rms_stack = np.load(rms_file).item()        
    plot_rms_stack(band, prof_stack, NMODES, rms_stack, rms_name)        

def NMODES_stack(prof_stack, NMODES):

    '''Step 1: Get the raw / stacked profiles'''
    # list the epochs and sessions
#    dict = {}
#    epochs = ['a', 'b', 'c', 'd']
#    sessions = [3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22]
#    for i in range(len(epochs)*len(sessions)):
#        dict[i] = epochs[i/len(sessions)]+str(sessions[i%len(sessions)])

#    start, end = 18, 19 # decide the sessions of starting and end(included)

    pattern1 = 'gp052_' + dict[start] + '_to_' + dict[end]
    pattern2 = 'band_'+str(band)+'_modes_'+str(NMODES)+'_tint_'+str(tint)
    
    pattern = pattern1 + '_raw_' + pattern2
    pattern_move = pattern1 + '_move_' + pattern2

    t00 = []
    profile_raw=[]      
    for ii in xrange(start, end+1): #xrange(len(glob.glob(time_str_patterns))):
        hh = glob.glob(time_str_folder + str(sorted(glob.glob(time_str_patterns))[ii][47:63]) + '*h5')[0]
        this_file = h5py.File(hh, 'r')
        print this_file
        t00.append(this_file['t00'][0][0])
        stack_remainder = this_file['fold_data_int_0.125_band_'+str(band)].shape[0]%prof_stack
        if stack_remainder != 0:
            profile_raw.append(this_file['fold_data_int_0.125_band_'+str(band)][:-stack_remainder]) 
        else:
            profile_raw.append(this_file['fold_data_int_0.125_band_'+str(band)])

#    print 't00', t00
#    print 'len and type of profile_raw', profile_raw, len(profile_raw), type(profile_raw)
    profile_raw = np.concatenate((profile_raw))
    print 'rank, type(profile_raw), shape', rank, type(profile_raw), profile_raw.shape

    # check the existence of the new V modes, which were constructed by the shifted profiles with an integration time of 0.125 seconds.
    V_same = ('same_'+ pattern1 + '_move_' + 'band_'+ str(band)
             +'_modes_'+ str(1) + '_tint_' + str(tint) + '_norm_var_lik_V.npy')
    print 'V_same', V_same
    if os.path.isfile(V_same) is False:

        print 'the common V template does NOT exist, create the common V template'

        '''Step 2: Construct V modes, and use n-nodes to fit all raw / stacked profiles.
                   The result would be an array of [bin, amps, bin_err, amp_errs].
        '''
        #Initially check the parabola of profiles with an integration time of 8 seconds. 
        profile_raw_8sec = stack(profile_raw, 64)
        tint_8sec = tint * 64

        prof_measures = process_profiles(profile_raw_8sec, pattern, NMODES, tint_8sec, V_recon=None, fit_profile=True, save_V=False)
        print 'type and shape of prof_measures', type(prof_measures), prof_measures.shape
        phase_bins_8sec = prof_measures[:,0]
        phase_bins_8sec_errs = prof_measures[:, prof_measures.shape[1]/2]
                
        '''Step 3: Use raw_phase_bins to construct a parabola '''
        # subtract the parabola from the curve of TOAs
        # set the earlist TOA to be offset and combine all middle time of chunks.

        chunk_times_8sec = get_chunk_times(t00, tint_8sec)
        chunk_times_raw = get_chunk_times(t00, tint) 

        if False: # fit a n-order polynomial
            for n in xrange(9, 12):
                print 'poly order: ',n
                plot_raw = pattern+'n'+str(n)+'_poly.png'
                popt = plot_phase_poly(chunk_times_8sec, phase_bins_8sec, phase_bins_8sec_errs, band, n, plot_raw)
                move_phase_bins = np.asarray(poly_n(chunk_times_raw, *popt))
                profile_move = mpi_phase_move(profile_raw, -1*move_phase_bins)
                pattern_move_poly = pattern_move+'n'+str(n)+'_poly'
                prof_measures_move = process_profiles(profile_move, pattern_move_poly, NMODES, tint, V_recon=None, fit_profile=False, save_V=True)

        else: # fit a parabola and a sine wave.
            popt, pcov = curve_fit(parabola, chunk_times_8sec, phase_bins_8sec)
            print 'popt, pcov', popt, pcov
         
            # Plotting the parabola and the phase bins
            plot_raw = pattern+'phase_para.png'
            plot_phase_para(chunk_times_8sec, phase_bins_8sec, phase_bins_8sec_errs, band, plot_raw)
            move_phase_bins = np.asarray(poly_n(chunk_times_raw, *popt))       

            '''Step 4: Shift profiles that remove the parabola'''
            # move the raw profiles for removing the n-poly / parabola effect.
            profile_move = mpi_phase_move(profile_raw, -1*move_phase_bins)    

            '''Step 5: Use the shifted profiles to construct a common V modes'''
            prof_measures_move = process_profiles(profile_move, pattern_move, NMODES, tint, V_recon=None, fit_profile=False, save_V=True)
        print 'the construction of the common V mode finished.'

    else:
        print 'The common V mode exist.'
        V_same = np.load(V_same) #load the common V mode. 
        '''Step 6: Use the same/common V-modes to fit the profiles in step 1.
                   The common V-modes were construct by profile_move in step 4 without 
                   stacking profiles.
        '''
        if prof_stack != 1: # to stack profiles
            profile_raw = stack(profile_raw, prof_stack)
            tint_stack = tint * prof_stack # the length of each chunk (sec)
        else:
            tint_stack = paras.tint

        pattern3 = 'band_'+str(band)+'_modes_'+str(NMODES)+'_tint_'+str(tint_stack)
        pattern_refit = pattern1 + '_refit_' + pattern3        
        pattern_refit480 = pattern1 + '_refit480_' + pattern3

        print 'start refit 480'
        prof_measures_refit = process_profiles(profile_raw, pattern_refit, NMODES, tint_stack, V_recon=V_same, fit_profile=True, save_V=True)

        '''Step 7: get the refit TOAs, remove the parabola, and get the std.'''
        phase_bins_refit = prof_measures_refit[:,0]
        np.save('phase_bins_refit_test.npy', phase_bins_refit)
        phase_bins_refit_errs = prof_measures_refit[:, prof_measures_refit.shape[1]/2]
        np.save('phase_bins_refit_errs_test.npy', phase_bins_refit_errs)

        '''Step 8:  Plotting the parabola and the phase bins'''
        plot_refit = pattern_refit+'phase_para_refit.png'
        chunk_times_stack = get_chunk_times(t00, tint_stack, stack_remainder=stack_remainder)
        print 'len(chunk_times_stack), len(phase_bins_refit)', len(chunk_times_stack), len(phase_bins_refit)
        rms_refit, rms_refit_remove, rms_refit_remove2 = plot_phase_para(chunk_times_stack, phase_bins_refit, phase_bins_refit_errs, band, plot_refit)

        print 'finished steps 1-8'            
        return [rms_refit, rms_refit_remove, rms_refit_remove2]

def get_chunk_times(t00, tint_stack, stack_remainder=0):
    chunk_times = []
    for ii in xrange(len(t00)):
        times = np.arange(0 + tint_stack/2, 536-stack_remainder*tint, tint_stack) # all chunks of each data
        times_ii = times + 86400*(t00[ii]-t00[0])
        chunk_times.append(times_ii)
    chunk_times = np.concatenate((chunk_times))
    return chunk_times

def plot_n_res_var(n_values, res_values, var_values, plot_name):

    markersize, fontsize = 4.0, 12

    plt.close('all')
    x_axis = np.array(n_values)
    plt.plot(x_axis, np.array(res_values), 'ro', label='Residuals', markersize=markersize)
    plt.plot(x_axis, np.array(res_values), 'r')
    plt.plot(x_axis, np.array(var_values), 'bs', label='Variances', markersize=markersize)
    plt.plot(x_axis, np.array(var_values), 'b')
    plt.xlabel('order of the polynomial', fontsize=fontsize)
    plt.ylabel('Residuals', fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plot_name += 'n_res.png'
    plt.savefig(plot_name, bbox_inches='tight', dpi=300)


def fft_bandpass_filter(chunk_times, phase_bins_raw, phase_bins_raw_errs, tint_stack, plot_name):

    bin_size = P0 / ngate * 10**6 #microsecond
    # transfer the unit from phase bins to microseconds
    phase_bins_raw *= bin_size
    phase_bins_raw_errs *= bin_size

    # do the fft bandpass filter for even samples. If there are unevenly samples, split them into even samples for analysis and combine the results together.

    filter_phase_bins_raw = []

    split_number = int(len(chunk_times)/(536/tint_stack))
    split_chunk_times = np.split(chunk_times, split_number)
    split_phase_bins_raw = np.split(phase_bins_raw, split_number)
    split_phase_bins_raw_errs = np.split(phase_bins_raw_errs, split_number)

    fftfreq_threshold = 0.010 # Hz

    for i in range(split_number):
        W = fftfreq(int(536/tint_stack), d=tint_stack)
        print 'W',W
        fft_phase_bins_raw = rfft(split_phase_bins_raw[i])
        print 'fft_phase_bins_raw', fft_phase_bins_raw
        # If our original signal time was in seconds, this is now in Hz    
        cut_f_signal = fft_phase_bins_raw.copy()
        cut_f_signal[(np.abs(W)<fftfreq_threshold)] = 0
        cut_signal = irfft(cut_f_signal)
        filter_phase_bins_raw.append(cut_signal)

        # plot the signal and fftfreq before and after the bandpass
        markersize = 2.0
        fontsize = 12
        times = split_chunk_times[i]
        phase_bins = split_phase_bins_raw[i]
        phase_bin_errs = split_phase_bins_raw_errs[i]

        zeros_line = np.zeros(len(times))

        plt.close('all')
#        f, axarr = plt.subplots(2,2)
#        plt.subplot(221)
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(times, zeros_line, 'r--')
        ax1.plot(times, phase_bins, 'bo', markersize=markersize)
        ax1.errorbar(times, phase_bins, yerr= phase_bin_errs, fmt='none', ecolor='b')
        title1 = 'raw, std: '+str(np.round(np.std(phase_bins),3))+'('+r'$\mu$'+'s)'
        ax1.set_title(title1, fontsize=fontsize)
        ax1.set_xlim([min(times), max(times)])
        ax1.set_ylabel('Fitting TOAs '+'('+r'$\mu$'+'s)', fontsize=fontsize)
        ax1.set_xlabel('time (sec)', fontsize=fontsize)
        ax1.tick_params(axis='both', which='major', labelsize=fontsize)

#        plt.subplot(223)
        ax2.plot(times, zeros_line, 'r--')
        ax2.plot(times, cut_signal, 'bo', markersize=markersize)
        ax2.errorbar(times, cut_signal, yerr= phase_bin_errs, fmt='none', ecolor='b')
        title1 = 'fftfreq filter, std: '+str(np.round(np.std(cut_signal),3))+'('+r'$\mu$'+'s)'
        ax2.set_title(title1, fontsize=fontsize)
        ax2.set_xlim([min(times), max(times)])
        ax2.set_xlabel('time (sec)', fontsize=fontsize)
#        ax2.set_ylabel('Fitting TOAs '+'('+r'$\mu$'+'s)', fontsize=fontsize)
        ax2.tick_params(axis='both', which='major', labelsize=fontsize)

        ax3.plot(W,fft_phase_bins_raw, 'ks', markersize=markersize)
        ax3.set_xlabel('fft freq (Hz)', fontsize=fontsize)
        ax_xticks = [round(min(W),3), round(0.5*min(W),3), 0, round(0.5*max(W),3), round(max(W),3)]
        ax3.set_xticks(ax_xticks)
        ax3.set_xticklabels(map(str,ax_xticks))
        ax3.set_xlim([min(W), max(W)])
        ax3.tick_params(axis='both', which='major', labelsize=fontsize)

        ax4.plot(W,cut_f_signal, 'ks', markersize=markersize)
        ax4.set_xlabel('fft freq (Hz)', fontsize=fontsize)
        ax4.set_xticks(ax_xticks)
        ax4.set_xticklabels(map(str,ax_xticks))
        ax4.set_xlim([min(W), max(W)])
        ax4.tick_params(axis='both', which='major', labelsize=fontsize)
        plot_name_fft = plot_name + '_fftfreq_'+str(i)+'.png'
        plt.savefig(plot_name_fft, bbox_inches='tight', dpi=300)

    filter_phase_bins_raw = np.concatenate((filter_phase_bins_raw)) 
    return filter_phase_bins_raw

def average_bands():

    for f in sorted(glob.glob(time_str_patterns)):
        print f
        ff = h5py.File(f, 'a')
        first_data = ff['fold_data_int_0.125_band_0']
        dataset_name = 'fold_data_int_'+str(0.125)+'_band_'+str(3)
        ff.create_dataset(dataset_name, first_data.shape, maxshape = first_data.shape, dtype=first_data.dtype, chunks=True)
        ff['fold_data_int_0.125_band_3'][:] = (ff['fold_data_int_0.125_band_0'][:] + ff['fold_data_int_0.125_band_1'][:] + ff['fold_data_int_0.125_band_2'][:])/3
        print 'averaged data:', ff['fold_data_int_0.125_band_3'][:]

def find_outlier_index(phase_bins_raw, phase_bins_raw_errs):

    # remove outliers: 1) phase_bins_raw_errs > 2 phase bins 2) abs(phase_bins_raw - mean) > 5\sigma
    phase_bins_mean = np.mean(phase_bins_raw)
    phase_bins_std = np.std(phase_bins_raw)

    cond_1 = phase_bins_raw_errs > 2
#    print 'cond_1', cond_1
    cond_2 = np.abs(phase_bins_raw - phase_bins_mean) > 5*phase_bins_std
#    print 'cond_2', cond_2

    outliers_index = np.where(np.logical_or(cond_1, cond_2))
#    print 'outliers_index', outliers_index
    return outliers_index


def plot_phase_poly(times, phase_bins_raw, phase_bins_raw_errs, band, n, plot_name):

    if False:    # remove outliers:
        outliers_index = find_outlier_index(phase_bins_raw, phase_bins_raw_errs)
        outliers_count = len(outliers_index[0])
        print 'numbers and index of outlier:', outliers_count, outliers_index
        times = np.delete(times, outliers_index)
        phase_bins_raw = np.delete(phase_bins_raw, outliers_index)
        phase_bins_raw_errs = np.delete(phase_bins_raw_errs, outliers_index)

    # fit a parabola for the TOAs distribution
    popt, pcov = curve_fit(poly_n, times, phase_bins_raw, p0=[0]*(n+1))
    print 'popt, pcov', popt, pcov
#    fit_para = (["", "+"][popt[0] > 0]+'{:0.3e}'.format(popt[0])+r'$t^2$'
#              + ["", "+"][popt[1] > 0]+'{:0.3e}'.format(popt[1])+r'$t$'
#              + ["", "+"][popt[2] > 0]+'{:0.3e}'.format(popt[2]))
#    print 'fit_para',fit_para

    # move the raw profiles for removing the n-poly effect.
    poly_phase_bins = np.asarray(poly_n(times, *popt))
    phase_bins_remove = phase_bins_raw - poly_phase_bins

    # Plotting the parabola and the phase bins
    markersize = 2.0
    fontsize = 12
    zeros_line = np.zeros(len(times))
    xmin, xmax = np.amin(times), np.amax(times)

    bin_size = P0 / ngate * 10**6 #microsecond
    # transfer the unit from phase bins to microseconds
    phase_bins_raw *= bin_size
    phase_bins_raw_errs *= bin_size
    poly_phase_bins *= bin_size
    phase_bins_remove *= bin_size

    # calculate the std, which has a unit of microsecond
    std_raw = np.std(phase_bins_raw)
    std_remove = np.std(phase_bins_remove)

    plt.close('all')
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(times, zeros_line, 'r--')
    axarr[0].plot(times, phase_bins_raw, 'bo', markersize=markersize)
    axarr[0].errorbar(times, phase_bins_raw, yerr= phase_bins_raw_errs, fmt='none', ecolor='b')
    axarr[0].plot(times, poly_phase_bins, 'k')
    axarr[0].set_ylabel('Fitting TOAs '+'('+r'$\mu$'+'s)' , fontsize=fontsize)
    title_0 = 'Origin TOAs, std: '+str(round(std_raw, 4))+'('+r'$\mu$'+'s)'
    if outliers_count > 0:
        title_0 += ', outliers: '+str(outliers_count)
    axarr[0].set_title(title_0)
    axarr[0].tick_params(axis='both', which='major', labelsize=fontsize)

    axarr[1].plot(times, zeros_line, 'r--')
    axarr[1].plot(times, phase_bins_remove, 'bo', markersize=markersize)
    axarr[1].errorbar(times, phase_bins_remove, yerr= phase_bins_raw_errs, fmt='none', ecolor='b')
    axarr[1].set_xlabel('Times (sec)')
    axarr[1].set_ylabel('Fitting TOAs '+'('+r'$\mu$'+'s)' , fontsize=fontsize)
    title_1 = 'Subtract the poly (black) of order '+str(n)+', coef:'+str(np.round(popt,3))+', std:'+str(round(std_remove,4))+'('+r'$\mu$'+'s)'
    axarr[1].set_title(title_1, fontsize=fontsize-2)
    axarr[1].set_xlim([xmin, xmax])
    axarr[1].tick_params(axis='both', which='major', labelsize=fontsize)
#    plt.legend(loc='upper right', fontsize=fontsize-4)
    plt.savefig(plot_name, bbox_inches='tight', dpi=300)

    return popt



def plot_phase_para(times, phase_bins_raw, phase_bins_raw_errs, band, plot_name):

    # remove outliers:
    outliers_index = find_outlier_index(phase_bins_raw, phase_bins_raw_errs)
    outliers_count = len(outliers_index[0])
    print 'numbers and index of outlier:', outliers_count, outliers_index
    times = np.delete(times, outliers_index)
    phase_bins_raw = np.delete(phase_bins_raw, outliers_index)
    phase_bins_raw_errs = np.delete(phase_bins_raw_errs, outliers_index)

    # fit a parabola for the TOAs distribution
    popt_para, pcov_para = curve_fit(parabola, times, phase_bins_raw)
    print 'popt_para, pcov_para ', popt_para, pcov_para
    fit_para = (["", "+"][popt_para[0] > 0]+'{:0.3e}'.format(popt_para[0])+r'$t^2$' 
             + ["", "+"][popt_para[1] > 0]+'{:0.3e}'.format(popt_para[1])+r'$t$' 
             + ["", "+"][popt_para[2] > 0]+'{:0.3e}'.format(popt_para[2]))
    print 'fit_para',fit_para
    # move the raw profiles for removing the parabola effect.
    para_phase_bins = np.asarray(parabola(times, *popt_para))
    phase_bins_remove = phase_bins_raw - para_phase_bins

    # fit a parabola for the phase bins with removing parabola.
    if band == 0:
        p0 = [0.6, 0.02, 0.3,-0.1] # init parameters for band 0, a_06to07
    else:
        p0 = [0.6, 0.01, 0.3,-0.1] # init parameters
    popt_remove, pcov_remove = curve_fit(sine, times, phase_bins_remove, p0=p0)
    print 'popt_remove, pcov_remove', popt_remove, pcov_remove
    fit_sine = ('{:0.3e}'.format(popt_remove[0]) + '*sin('
             +'{:0.3e}'.format(popt_remove[1]) + r'$t$'
             + ["", "+"][popt_remove[2] > 0]+'{:0.3e}'.format(popt_remove[2]) + ')'
             + ["", "+"][popt_remove[3] > 0]+'{:0.3e}'.format(popt_remove[3]))

    para_phase_bins_remove = np.asarray(sine(times, *popt_remove))
    phase_bins_remove2 = phase_bins_remove - para_phase_bins_remove

    # Plotting the parabola and the phase bins
    markersize = 2.0
    fontsize = 12
    zeros_line = np.zeros(len(times))
    xmin, xmax = np.amin(times), np.amax(times)

    bin_size = P0 / ngate * 10**6 #microsecond
    # transfer the unit from phase bins to microseconds
    phase_bins_raw *= bin_size
    phase_bins_raw_errs *= bin_size
    para_phase_bins *= bin_size
    phase_bins_remove *= bin_size
    para_phase_bins_remove *= bin_size
    phase_bins_remove2 *= bin_size

    # calculate the std, which has a unit of microsecond 
    std_raw = np.std(phase_bins_raw)
    std_remove = np.std(phase_bins_remove)
    std_remove2 = np.std(phase_bins_remove2)

    plt.close('all')
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(times, zeros_line, 'r--')
    axarr[0].plot(times, phase_bins_raw, 'bo', markersize=markersize)
    axarr[0].errorbar(times, phase_bins_raw, yerr= phase_bins_raw_errs, fmt='none', ecolor='b')
    axarr[0].plot(times, para_phase_bins, 'k')
    axarr[0].set_ylabel('Fitting TOAs '+'('+r'$\mu$'+'s)' , fontsize=fontsize)
    title_0 = 'Origin TOAs, std: '+str(round(std_raw, 4))+'('+r'$\mu$'+'s)'
    if outliers_count > 0:
        title_0 += ', outliers: '+str(outliers_count)
    axarr[0].set_title(title_0)
    axarr[0].tick_params(axis='both', which='major', labelsize=fontsize)

    axarr[1].plot(times, zeros_line, 'r--')
    axarr[1].plot(times, phase_bins_remove, 'bo', markersize=markersize)
    axarr[1].errorbar(times, phase_bins_remove, yerr= phase_bins_raw_errs, fmt='none', ecolor='b')
    axarr[1].plot(times, np.asarray(sine(times, *p0)), 'g')
    axarr[1].plot(times, para_phase_bins_remove, 'k')
    axarr[1].set_ylabel('Fitting TOAs '+'('+r'$\mu$'+'s)' , fontsize=fontsize)
    title_1 = 'Subtract the parabola:'+fit_para+', std: '+str(round(std_remove,4))+'('+r'$\mu$'+'s)'
    axarr[1].set_title(title_1, fontsize=fontsize-2)
    axarr[1].tick_params(axis='both', which='major', labelsize=fontsize)


    axarr[2].plot(times, zeros_line, 'r--')
    axarr[2].plot(times, phase_bins_remove2, 'bo', markersize=markersize)
    axarr[2].errorbar(times, phase_bins_remove2, yerr= phase_bins_raw_errs, fmt='none', ecolor='b')
    axarr[2].set_xlabel('Times (sec)')
    axarr[2].set_ylabel('Fitting TOAs '+'('+r'$\mu$'+'s)' , fontsize=fontsize)
    title_2 = 'Subtract the sine (black):'+fit_sine+', std:'+str(round(std_remove2,4))+'('+r'$\mu$'+'s)'
    axarr[2].set_title(title_2, fontsize=fontsize-2)
    axarr[2].set_xlim([xmin, xmax])
    axarr[2].tick_params(axis='both', which='major', labelsize=fontsize)

#    plt.legend(loc='upper right', fontsize=fontsize-4)
    plt.savefig(plot_name, bbox_inches='tight', dpi=300)

    return std_raw, std_remove, std_remove2

def process_profiles(profile, pattern, NMODES, tint_stack, V_recon=None, fit_profile=False, save_V=False):
        # remove signal modes, and reconstruct noise profiles.
        U, s, V = svd(profile)
        plot_svd(profile, 'origin_')
        V0_raw = copy.deepcopy(V[0])
        V1_raw = copy.deepcopy(V[1])
        V[0] = 0
        V[1] = 0
        noise_profiles = reconstruct_profile(U,s,V)
        print 'noise_profiles.shape', noise_profiles.shape

        # get an estimate of the noise level, for each the R and L polarizations.
        noise_var_L = np.mean(np.var(noise_profiles[:, 0:noise_profiles.shape[1]/2],axis=1))
        noise_var_R = np.mean(np.var(noise_profiles[:, noise_profiles.shape[1]/2:noise_profiles.shape[1]],axis=1))

        # transform origin profiles with unit variance.
        profile_norm_var_L = np.zeros((profile.shape[0], profile.shape[1]/2))
        profile_norm_var_R = np.zeros((profile.shape[0], profile.shape[1]/2))

        for ii in xrange(profile.shape[0]):
            profile_norm_var_L[ii] = norm_variance_profile(profile[ii, 0:profile.shape[1]/2], noise_var_L)
            profile_norm_var_R[ii] = norm_variance_profile(profile[ii, profile.shape[1]/2:profile.shape[1]], noise_var_R)
        profile_norm_var = np.concatenate((profile_norm_var_L, profile_norm_var_R), axis=1)

        profile_norm_var = profile_norm_var[:, :]
        print 'line 614, profile_norm_var.shape', profile_norm_var.shape
#        plot_spec_dedisperse(profile_norm_var) # to check the spec
        np.save('profiles480_norm_var.npy', profile_norm_var)
        print 'line 617, finish profile_norm_var'
        # SVD on the normalized variance profile.
        U, s, V = svd(profile_norm_var)
        np.save('profiles480_U.npy', U)
        np.save('profiles480_s.npy', s)
        np.save('profiles480_V.npy', V)

        if save_V == True:
            save_V_name = 'same_'+ pattern + '_norm_var_lik_V.npy' 
            np.save(save_V_name, V)
            print 'finished saving same_V'
        plot_name_1 = pattern + '_norm_var_lik'
        plot_svd(profile_norm_var, plot_name_1)
        print 'line 522, finish SVD'
        check_noise(profile_norm_var)      
        print 'finish checking noise'

        if V_recon is not None:
            V_recon = V_recon.reshape(V_recon.shape[0], 2, V_recon.shape[1]/2)
            print 'load V_recon'
        else:
            V_recon = V.reshape(V.shape[0], 2, V.shape[1]/2)
            print 'done V_recon'

        if fit_profile == True:
            '''reshape profiles of L and R into periodic signals (pulse number, L/R, phases)'''
            profile_npy = np.zeros(profile.shape)
            profile_npy[:] = profile[:]
            profile_npy = profile_npy.reshape(profile_npy.shape[0], 2, profile_npy.shape[1]/2)   
            print 'profile_npy.shape', profile_npy.shape
            prof_measures = mpi_phase_fitting(profile_npy, V_recon, pattern, NMODES,tint_stack)
            return prof_measures

def fourier(x, tau, *a):
    ret = a[0] * np.cos(np.pi / tau * x)
    for deg in range(1, len(a)):
        ret += a[deg] * np.cos((deg+1) * np.pi / tau * x)
    return ret

def gaussian(t,a,t0,sigma):
    return a*np.exp(-(t-t0)**2/(2*sigma**2))

def sine(x, a, b, c, d):
    return a*np.sin(b*x+c) + d

def parabola(x, a, b, c):
    return a*x**2 + b*x + c      

def poly_n(x, *parameters):
    return sum([p*(x**i) for i, p in enumerate(parameters)])

def generate_tim_file():

    obs_code = 3 # AO

    TOA_err_threshold = 3 # in the unit of microsecond

#    ff = [np.load('gp052a_ar_no0003_TOAs_nodes_1_tint_8.0.npy')] 
    ff = [np.load(ii) for ii in sorted(glob.glob(TOAs_files))]
    tt = [ff[ii][jj] for ii in xrange(len(ff)) for jj in xrange(len(ff[0])) if ff[ii][jj][1] < TOA_err_threshold]
    errs = [tt[ii][1] for ii in xrange(len(tt))]
    if True:
        fontsize = 16
        plt.close('all')
        plt.figure()
        plt.hist(errs, 1000, normed=True )
        plt.xlim((min(errs), max(errs)))
#        plt.ylabel('V values', fontsize=fontsize)
        plt.gca().set_xscale("log")
        plt.gca().set_yscale("log")
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.savefig('TOAs_errs.png', bbox_inches='tight', dpi=300)       

    with open("1957_TOAs.tim", "a") as fp:    
        fp.write('MODE '+str(1)+"\n")
        fp.write('INFO '+str(0)+"\n")
        for ii in xrange(len(tt)):
            fp.write(str(obs_code)+str(' ')*14+("%.3f" % round(paras.fref.value,3))+str(' ')*2+("%.13f" % round(tt[ii][0],13))+str(' ')*1+("%.7f" % round(tt[ii][1],7))+"\n")
    print 'finished the tim file'

def TOAs_debug():
    '''the file is in the shape of(epochs, [TOAs, TOAs uncertainties in microseconds, TOAs phase predictor in phase bins, and fitting phase bins])'''
    ff = np.load('gp052a_ar_no0003_TOAs_nodes_1_tint_8.0.npy')

    times = ff[13:63,0]
    phase_centre = ff[13:63, 2]
    phase_fitting = ff[13:63, 3]
    color = ['bo','rs'] # 'g*'
    label = ['phase bin predictors', 'phase bin measurements']
    xmin = np.amin(times)
    xmax = np.amax(times)
    markersize = 4.0
    fontsize = 16

    plt.close('all')
#    for ii in xrange():#xrange((npy_lik_file.shape[1]-2)/2 -1):
    plt.plot(times, phase_centre, color[0], label=label[0], markersize=markersize)
    plt.plot(times, phase_fitting, color[1], label=label[1], markersize=markersize)
    plt.xlim([xmin, xmax])
    plt.xlabel('TOAs (MJD)', fontsize=fontsize)
    plt.ylabel('Phase bins', fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize-4)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.grid()
    plt.savefig('debug_TOAs.png', bbox_inches='tight', dpi=300)



def generate_TOAs():

    '''generate a file of TOAs for tempo to refine'''

    total_length = 8.*paras.T
    folding_intv_length = paras.tint*paras.prof_stack

    for ii in xrange(len(glob.glob(phase_amp_files))):
        if ii % size == rank:
            ff = np.load(sorted(glob.glob(phase_amp_files))[ii])
            pattern = str(sorted(glob.glob(phase_amp_files))[ii][81:97])
            print 'pattern',pattern
            hh = glob.glob(time_str_folder + pattern + '*h5')[0]
            ar_data = glob.glob(rawdata_folder + pattern)[0]
#        this_file = h5py.File(hh, 'r')
#        t00 = this_file['t00'][0]
#        # Define the t00_series is that the average of the time of beginning and ending of the folding (fold_t_steps)
#        fold_t_steps= np.linspace(0, total_length, total_length/folding_intv_length+1)
#        fold_t_series = np.array([np.average((fold_t_steps[ii], fold_t_steps[ii+1])) for ii in xrange(len(fold_t_steps)-1)])        
#        t00_series = t00 + fold_t_series/86400.

            if ff.shape[0] == paras.T: 
                # jj is the number of blocks
                # tt is the the average of the time of beginning and ending of the folding. For example, if the folding length is 8 sec, then tt is 4 sec.
                fold_t_steps = np.linspace(0, 8, 8./folding_intv_length+1)
                tt = np.array([np.average((fold_t_steps[ii], fold_t_steps[ii+1])) for ii in xrange(len(fold_t_steps)-1)])
                # create an array of TOAs and errs, which is in the shape of (TOAs, TOA_errs)
                TOAs = np.zeros((ff.shape[0], 4))

                print 'starting generating TOAs'
                for jj in xrange(ff.shape[0]): 
                    TOA, ph = TOA_predictor(ar_data, jj, tt, ff[jj,0])
                    TOAs[jj, 0] = TOA
                    TOAs[jj, 2] = ph
                    TOAs[jj, 3] = ff[jj,0]
                TOAs[:, 1] = ff[:,ff.shape[1]/2] * (P0/ngate) * 1e6 # in the unit of microseconds
            else: 
                print 'warning: check the phase amp file, which the shape is not consistent.'
                print sorted(glob.glob(phase_amp_files))[ii]   

            np.save(pattern + '_TOAs_nodes_'+str(NMODES)+'_tint_'+str(tint_stack)+'.npy', TOAs)


def TOA_predictor(ar_data, jj, tt, phase_correction):
    from astropy.time import Time
    # get the offset time
    fh = mark4.open(ar_data, 'rs', ntrack=64, decade=2010, sample_rate=SR)
    t_0 = fh.time0
    offset_time = Time(math.ceil(t_0.unix), format='unix', precision=9)
    offset = fh.seek(offset_time)

    # get the offset time in the block
    fh = mark4.open(ar_data, 'rs', ntrack=64, decade=2010, sample_rate=SR, thread_ids=[2*band, 2*band + 1])
    fh.seek(offset + jj*(N - DN))
    t0 = fh.tell(unit='time')
    t00 = t0.mjd

    # get the phase
    z_size = N-DN
    polyco = Polyco('/mnt/raid-cita/mahajan/Pulsars/B1957Timing/polycob1957+20_gpfit.dat')
    p = polyco.phasepol(t0, rphase='fraction', t0=t0, time_unit=u.second, convert=True)
    # note that p(times with unit sec after the offset time) will give the rotation phases at that times.
    ph = p(tt)
    ph = ((ph)*P0) / (P0/ngate) % ngate
    print 'ph',ph
    # combine the offset time, tt (time after the offset time), the predicting phase, and the correction of the predicting phase (produced by fitting) to get the TOA, which is in the unit of MJD.
    event_time = t00 + tt/86400.
    TOA = PEPOCH + ((event_time-PEPOCH)*86400//P0 + (ph + phase_correction)/ngate)*P0/86400.
#    TOA = PEPOCH + ((event_time-PEPOCH)*86400//P0 + ((ph + phase_correction) - (ngate/2))/ngate)*P0/86400.
#    TOA = PEPOCH + ((event_time-PEPOCH)*86400//P0 + ph/ngate)*P0/86400.

    return TOA, ph

def plot_spec_dedisperse():
#    this_file = h5py.File('/mnt/raid-project/gmrt/hhlin/time_streams_1957/gp052a_ar_no0007_512g_0b_56821.2537037+536s_h5','r')
#    spec = this_file['fold_data_int_0.125_band_0'][0:480]
    nt = 3200
    spec = np.load('profiles480_norm_var.npy')[0:nt]
    spec_L = spec[:, 0:spec.shape[1]/2]
    spec_R = spec[:, spec.shape[1]/2:]
    spec_mean = np.mean((spec_L,spec_R), axis=0)
    
    cmap = plt.cm.viridis
    vmin = np.mean(spec) + 1.5*np.std(spec)
    vmax = np.mean(spec) + -1*np.std(spec)
    extent = [0, paras.ngate, len(spec)*0.125, 0]

    plt.close('all')
    plt.imshow(spec_L, extent=extent, aspect='auto', cmap = cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel('Phase Bins')
    plt.ylabel('Time (Sec)')
    plt.savefig('spec_L.png', bbox_inches='tight', dpi=300)

    plt.close('all')
    plt.imshow(spec_R, extent=extent, aspect='auto', cmap = cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel('Phase Bins')
    plt.ylabel('Time (Sec)')
    plt.savefig('spec_R.png', bbox_inches='tight', dpi=300)

    plt.close('all')
    plt.imshow(spec_mean, extent=extent, aspect='auto', cmap = cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel('Phase Bins')
    plt.ylabel('Time (Sec)')
#    plt.ylim()
    plt.savefig('spec_mean.png', bbox_inches='tight', dpi=300)

    '''Plot the SVD decomposition for the 1st and 2nd mode in images. Raw data are: gp052_a6_to_a7_move_band_0_modes_1_tint_0.125, profiles[0:480]'''

    s = np.load('profiles480_s.npy')
    u = np.load('profiles480_U.npy')
    v =np.load('profiles480_V.npy')


    v_mean = np.zeros((v.shape[0], v.shape[1]/2))
    for i in xrange(len(v)):
        v_mean[i] = (v[i,0:v.shape[1]/2]+v[i, v.shape[1]/2:])/2

    imageall = -np.dot((u[0:nt,:]*s[:]).reshape(nt,1024), v_mean[:].reshape(1024,512))
    image1 = -np.dot((u[0:nt,0]*s[0]).reshape(nt,1), v_mean[0].reshape(1,512))
    image2 = -np.dot((u[0:nt,1]*s[1]).reshape(nt,1), v_mean[1].reshape(1,512))

    phase_range = np.arange(0, 512)
    time_range = np.arange(0,nt*0.125, 0.125)

    plt.close('all')
#    plt.plot(phase_range, v_mean[0]+0.5, 'g')
    plt.plot(phase_range, v_mean[0]+0.5, 'g')
    plt.plot(phase_range, v_mean[1], 'g')
#    plt.xlim()
#    plt.ylim([-0.4,0.9])
    plt.savefig('spec_v01.png', bbox_inches='tight', dpi=300)

    plt.close('all')
    plt.plot(time_range, -20*u[0:nt, 0]+1, 'b')
    plt.plot(time_range, -20*u[0:nt, 1]-1, 'b')
    plt.ylim([-2,2])
    plt.savefig('spec_u01.png', bbox_inches='tight', dpi=300)


    plt.close('all')
    plt.imshow(imageall, extent=extent, aspect='auto', cmap = cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel('Phase Bins')
    plt.ylabel('Time (Sec)')
    plt.savefig('spec_imageall.png', bbox_inches='tight', dpi=300)


    plt.close('all')
#    gs1 = gridspec.GridSpec(3, 3)
#    gs1.update(left=0.05, right=0.48, wspace=0.05)
#    ax1 = plt.subplot(gs1[0, :-1])
#    ax2 = plt.subplot(gs1[1:, :-1])
#    ax3 = plt.subplot(gs1[1:, -1])
    plt.imshow(image1, extent=extent, aspect='auto', cmap = cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel('Phase Bins')
    plt.ylabel('Time (Sec)')
    plt.savefig('spec_image1.png', bbox_inches='tight', dpi=300)

    plt.close('all')
    plt.imshow(image2, extent=extent, aspect='auto', cmap = cmap, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel('Phase Bins')
    plt.ylabel('Time (Sec)')
    plt.savefig('spec_image2.png', bbox_inches='tight', dpi=300)



def CL_Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--band", type=int, choices=[0, 1, 2], help="Select which frequency band to process. Integer from 0 to 2.")
    return parser.parse_args()

def MakeFileList(rank, size):
    import itertools
    epochs = ['d']
    nums = [3, 4, 6, 7]
#    nums = [7, 9, 10, 12, 13, 15, 16, 18, 19, 21]
#    nums = [3, 4, 6, 7, 9, 10]
#    epochs = ['a', 'b', 'c', 'd']
#    nums = [3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22]
    evn_gen = itertools.product(epochs, nums)
    evn = ['test_gp052{0}_ar_no00{1:02d}'.format(epoch, file_num) for epoch, file_num in evn_gen]
    print 'evn[rank::size]', evn[rank::size], rank, size
    return evn[rank::size]


def plot_fit_amps(times, phases_amps, plot_name):
    npy_lik_file = phases_amps
#    profile_numbers = np.arange(0, len(npy_lik_file)*tint*prof_stack, tint*prof_stack)
    xmin = np.amin(times)
    xmax = np.amax(times)
    zeros_line = np.zeros(len(npy_lik_file[:]))

    markersize = 4.0
    fontsize = 16

    plt.close('all')
    plt.plot(times, zeros_line, 'r--')
    color = ['bo','rs', 'g*']
    label = ['Amp1/Amp0', 'Amp2/Amp0', 'Amp3/Amp0']
    for ii in xrange(3):#xrange((npy_lik_file.shape[1]-2)/2 -1):
        plt.plot(times, (npy_lik_file[:,ii+2]/ npy_lik_file[:,1]), color[ii], label=label[ii], markersize=markersize)
    plt.xlim([xmin, xmax])
    plt.xlabel('time (MJD)', fontsize=fontsize)
    plt.ylabel('Fitting Amps ratio', fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize-4)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.savefig(plot_name, bbox_inches='tight', dpi=300)

    plt.ylim([-0.5,0.5])
    plt.savefig('zoom_'+plot_name, bbox_inches='tight', dpi=300)    

def remove_para(times, phase_bins):

    '''remove the parabola from the TOA distribution'''
    z_stacked = np.polyfit(times, phase_bins, 2)
    p_stacked = np.poly1d(z_stacked)
    parabola_stacked = p_stacked(times)
    phase_bins_para = phase_bins - parabola_stacked
    
    return phase_bins_para


def plot_phase_lik(times, phases_amps, tint_stack, plot_name):

    '''stacked profiles'''
    npy_lik_file = phases_amps
    xmin = np.amin(times)
    xmax = np.amax(times)
    zeros_line = np.zeros(len(npy_lik_file[:]))
    phase_bins = npy_lik_file[:,0]
    phase_bin_errs = npy_lik_file[:,npy_lik_file.shape[1]/2]

    '''remove the parabola from the TOA'''
    phase_bins_para = remove_para(times, phase_bins)


    '''rebinned the phase measurements'''
    times_100 = np.arange(0,300,tint_stack)
    if plot_name == 'phase_lik_nodes_1_tint_30.0sec.png':
        phase_bins_all = np.load('gp052a_ar_no0007fit_nodes_1_tint_0.125.npy')
    elif plot_name == 'phase_lik_nodes_2_tint_30.0sec.png':
        phase_bins_all = np.load('gp052a_ar_no0007fit_nodes_2_tint_0.125.npy')
    phase_bins_100 = np.mean(phase_bins_all[:2400,0].reshape(2400/paras.prof_stack, paras.prof_stack), axis=1)

    z_rebinned = np.polyfit(times_100, phase_bins_100, 2)
    p_rebinned = np.poly1d(z_rebinned)
    parabola_rebinned = p_stacked(times_100)
    phase_bins_100_para = phase_bins_100 - parabola_rebinned

#    plt.plot(times_100, phase_bins_100, 'ys', markersize=markersize+2)
    ut = np.load('UT_.npy')
    ut1 = ut[1]

    markersize = 2.0
    fontsize = 16

    # before subracting the fitting parabola
    plt.close('all')
    plt.plot(times, zeros_line, 'r--')
    plt.plot(times, phase_bins, 'bo', markersize=markersize)
    plt.errorbar(times, phase_bins, yerr= phase_bin_errs, fmt='none', ecolor='b')
    plt.plot(times, parabola_stacked, 'g', label='para_stacked')
    plt.plot(times_100, phase_bins_100, 'ys', markersize=markersize)
    plt.plot(times, parabola_rebinned, 'm', label='para_rebinned')
    title = 'RMS_stacked: '+str(np.round(np.std(phase_bins),3))+', RMS_rebinned: '+str(np.round(np.std(phase_bins_100),3))    
    plt.title(title, fontsize=fontsize)
    plt.xlim([xmin, xmax])
    plt.xlabel('time (sec)', fontsize=fontsize)
    plt.ylabel('Fitting phase bin errs', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize-4)
    plot_name = 'before_'+plot_name
    plt.savefig(plot_name, bbox_inches='tight', dpi=300)

    # after subracting the fitting parabola
    plt.close('all')
    plt.plot(times, zeros_line, 'r--')
    plt.plot(times, phase_bins_para, 'bo', markersize=markersize)
    plt.errorbar(times, phase_bins_para, yerr= phase_bin_errs, fmt='none', ecolor='b')
#    plt.plot(times, parabola_stacked, 'g', label='para_stacked')
    plt.plot(times_100, phase_bins_100_para, 'ys', markersize=markersize)
#    plt.plot(times, parabola_rebinned, 'm', label='para_rebinned')
    title = 'RMS_stacked: '+str(np.round(np.std(phase_bins_para),3))+', RMS_rebinned: '+str(np.round(np.std(phase_bins_100_para),3))
    plt.title(title, fontsize=fontsize)
    plt.xlim([xmin, xmax])
    plt.xlabel('time (sec)', fontsize=fontsize)
    plt.ylabel('Fitting phase bin errs', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize-4)
    plot_name = 'after_'+plot_name
    plt.savefig(plot_name, bbox_inches='tight', dpi=300)

def plot_ut_phase_time_streams():

    ut = np.load('UT_.npy')
    ut1 = ut[1]
    phase_file_1 = np.load('gp052a_ar_no0007fit_nodes_1_tint_0.125.npy')
    phase_file_2 = np.load('gp052a_ar_no0007fit_nodes_2_tint_0.125.npy')
    phase_bins_1 = phase_file_1[:2400, 0]
    phase_bins_2 = phase_file_2[:2400, 0]

    tot_chunk = 5
    for ii in xrange(tot_chunk):
        ut1_chunk = ut1[ii*(2400/tot_chunk):(ii+1)*(2400/tot_chunk)] * 25
        phase_bins_1_chunk = phase_bins_1[ii*(2400/tot_chunk):(ii+1)*(2400/tot_chunk)] -3
        phase_bins_2_chunk = phase_bins_2[ii*(2400/tot_chunk):(ii+1)*(2400/tot_chunk)] -6

        markersize = 4.0
        fontsize = 16

        plt.close('all')
        x_axis = np.arange(ii*(2400/tot_chunk)*0.125, (ii+1)*(2400/tot_chunk)*0.125, 0.125)
        plt.plot(x_axis, ut1_chunk, 'g', label='The U1 mode')
        plt.plot(x_axis, phase_bins_1_chunk, 'ro', markersize=markersize, label='1 mode')
        plt.plot(x_axis, phase_bins_1_chunk, 'r')
        plt.plot(x_axis, phase_bins_2_chunk, 'bs', markersize=markersize, label='2 modes')
        plt.plot(x_axis, phase_bins_2_chunk, 'b--')
        title = 'RMS of 1 mode: '+str(np.round(np.std(phase_bins_1_chunk),3))+', RMS of 2 modes: '+str(np.round(np.std(phase_bins_2_chunk),3))
        plt.title(title, fontsize=fontsize)
        plt.xlabel('Time (sec)', fontsize=fontsize)
        plt.ylabel('Time streams', fontsize=fontsize)
        plt.ylim([-10,4])
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.legend(loc='upper right', fontsize=fontsize-5)
        plot_name = 'u1_phase_stream_'+str(tot_chunk)+'_'+str(ii)+'.png'
        plt.savefig(plot_name, bbox_inches='tight', dpi=300)

def plot_ut_bands_correlation():

    # load the UT of each band:
    brange = 10
    b0 = np.load('gp052_b9_to_b10_move_band_0_modes_1_tint_0.125_norm_var_lik_UT.npy')[:brange]
    b1 = np.load('gp052_b9_to_b10_move_band_1_modes_1_tint_0.125_norm_var_lik_UT.npy')[:brange]
    b2 = np.load('gp052_b9_to_b10_move_band_2_modes_1_tint_0.125_norm_var_lik_UT.npy')[:brange]

    # correlation sets
    cor_sets = [[b0, b1], [b0, b2], [b1, b2]]

    for i in xrange(len(cor_sets)):
        #calculate the correlation matrix
        bcor = np.zeros((brange, brange))
        for ii in range(brange):
            for jj in range(brange):
#            for jj in range(brange-1, -1, -1):
                bcor[brange-ii-1, jj] = np.correlate(cor_sets[i][0][ii], cor_sets[i][1][jj])
        # plot the correlation
        cmap = cm.jet
        extent = [0, brange, 0, brange]
        plt.close('all')
        plt.imshow(bcor, extent=extent, aspect='auto', cmap = cmap)
        plt.colorbar()
        if i == 0:
            xlabel, ylabel = 'U modes of the 1st band', 'U modes of the 2nd band'
            plot_name = 'band_corr_'+'b0_b1.png'
        elif i == 1:
            xlabel, ylabel = 'U modes of the 1st band', 'U modes of the 3rd band'
            plot_name = 'band_corr_'+'b0_b2.png'
        elif i == 2:
            xlabel, ylabel = 'U modes of the 2nd band', 'U modes of the 3rd band'
            plot_name = 'band_corr_'+'b1_b2.png'

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(plot_name, bbox_inches='tight', dpi=300)

    # calculate and plot the correlation bandwidth.
    x_axis = np.arange(-40,41)*tint
    corbw01 = np.array([np.correlate(b0[1], np.roll(b1[1],i)) for i in range(-40,41)])
    corbw02 = np.array([np.correlate(b0[1], np.roll(b2[1],i)) for i in range(-40,41)])
    corbw12 = np.array([np.correlate(b1[1], np.roll(b2[1],i)) for i in range(-40,41)])

    popt01, pcov01 = curve_fit(gaussian, x_axis, np.array(corbw01[:,0]), p0=[1, 0, 1.5])
    popt02, pcov02 = curve_fit(gaussian, x_axis, np.array(corbw02[:,0]), p0=[1, 0, 1.5])
    popt12, pcov12 = curve_fit(gaussian, x_axis, np.array(corbw12[:,0]), p0=[1, 0, 1.5])

    bw01, bw02, bw12 = np.round(popt01[-1],4), np.round(popt02[-1],4), np.round(popt12[-1],4)
    print 'bw01, bw02, bw12', bw01, bw02, bw12

    label1 = 'decor. bandwith between the 1st and 2nd bands: '+str(bw01)+' sec.'
    label2 = 'decor. bandwith between the 1st and 3rd bands: '+str(bw02)+' sec.'
    label3 = 'decor. bandwith between the 2nd and 3rd bands: '+str(bw12)+' sec.'

    markersize = 4.0
    fontsize = 16

    plt.close('all')
    plt.plot(x_axis, corbw01, 'ro', markersize=markersize, label=label1)
    plt.plot(x_axis, corbw01, 'r')
    plt.plot(x_axis, corbw02, 'bs', markersize=markersize, label=label2)
    plt.plot(x_axis, corbw02, 'b--')
    plt.plot(x_axis, corbw12, 'g^', markersize=markersize, label=label3)
    plt.plot(x_axis, corbw12, 'g-.')
    plt.xlabel('Time (sec)', fontsize=fontsize)
    ylabel = 'Correlation of the 2nd U'
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlim([0, 5.0])
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize-4)
    plot_name = 'u_phase_bins_corbw.png'
    plt.savefig(plot_name, bbox_inches='tight', dpi=300)

def plot_ut_phase_correlation():
    
    ut = np.load('UT_.npy')
    phase_file_1 = np.load('gp052a_ar_no0007fit_nodes_1_tint_0.125.npy')
    phase_file_2 = np.load('gp052a_ar_no0007fit_nodes_2_tint_0.125.npy')
    phase_bins_1 = phase_file_1[:2400, 0]
    phase_bins_2 = phase_file_2[:2400, 0]    

    modes_U = 10
    cor_1 = np.zeros(modes_U) 
    cor_2 = np.zeros(modes_U)
    for ii in xrange(modes_U):
        cor_1[ii] = np.correlate(ut[ii], phase_bins_1)
        cor_2[ii] = np.correlate(ut[ii], phase_bins_2)

    markersize = 4.0
    fontsize = 16
    
    plt.close('all')
    x_axis = np.arange(modes_U)
    plt.plot(x_axis, cor_1, 'ro', markersize=markersize, label='1 mode')
    plt.plot(x_axis, cor_1, 'r')
    plt.plot(x_axis, cor_2, 'bs', markersize=markersize, label='2 modes')
    plt.plot(x_axis, cor_2, 'b--')
    plt.xlabel('U modes', fontsize=fontsize)
    plt.ylabel('Correlation', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize-4)
    plot_name = 'u_phase_bins_cor.png'
    plt.savefig(plot_name, bbox_inches='tight', dpi=300)

def plot_rms_stack(band, prof_stack, NMODES, rms_stack, pattern_refit):
    # note the rms_stack unit should be microsecond.
    # rms_stack is in the shape of (prof_stack, nmodes, (rms of refit, subtract para, subtract sine))

    x_axis = np.array(prof_stack)*tint
    tt = np.arange(0.125,65,1)

    plot_names = [pattern_refit + '_refit',
                  pattern_refit + '_remove_para',
                  pattern_refit + '_remove_sine']

    refit, para, sine = [], [], []
    refit_1, refit_2, para_1, para_2, sine_1, sine_2 = [], [], [], [], [], []
    for i in prof_stack:
        refit_1.append(rms_stack['band_'+str(band)+'_modes_'+str(1)+'_tint_'+str(i*tint)][0])
        refit_2.append(rms_stack['band_'+str(band)+'_modes_'+str(2)+'_tint_'+str(i*tint)][0])
        para_1.append(rms_stack['band_'+str(band)+'_modes_'+str(1)+'_tint_'+str(i*tint)][1])
        para_2.append(rms_stack['band_'+str(band)+'_modes_'+str(2)+'_tint_'+str(i*tint)][1])
        sine_1.append(rms_stack['band_'+str(band)+'_modes_'+str(1)+'_tint_'+str(i*tint)][2])
        sine_2.append(rms_stack['band_'+str(band)+'_modes_'+str(2)+'_tint_'+str(i*tint)][2])
    refit.append(refit_1)
    refit.append(refit_2)
    para.append(para_1)
    para.append(para_2)
    sine.append(sine_1)
    sine.append(sine_2)

    markersize = 4.0
    fontsize = 16

    for ii in xrange(len(plot_names)):
        if ii == 0:
            rms_points = refit
        elif ii == 1:
            rms_points = para
        elif ii == 2:
            rms_points = sine 
        max_value = max(max(rms_points))
        plt.close('all')
        tt_curve = 1/np.sqrt(tt)*(np.sqrt(0.125)*max_value)
        plt.loglog(tt, tt_curve, 'k', label='1 / sqrt(t)')
        plt.loglog(x_axis, rms_points[0], 'ro', markersize=markersize, label='1 mode')#, max: '+str(np.amax(np.round(phase_bins_1_rebin_rms,4)))+'('+r'$\mu$'+'s)')
        plt.loglog(x_axis, rms_points[0], 'r')
        plt.loglog(x_axis, rms_points[1], 'bs', markersize=markersize, label='2 modes')#, max: '+str(np.amax(np.round(phase_bins_2_rebin_rms, 4)))+'('+r'$\mu$'+'s)')
        plt.loglog(x_axis, rms_points[1], 'b--')
        plt.xlabel('Int. time for each stacking profile (s)', fontsize=fontsize)
        ylabel = 'RMS for stacking profiles '+'('+r'$\mu$'+'s)'
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.legend(loc='lower left', fontsize=fontsize-4)
        plt.xlim([0,70])
        plt.ylim([0.9*min(tt_curve), 1.1*max(tt_curve)])
        title = plot_names[ii]
        plt.title(title, fontsize=fontsize-4)
        plot_name = plot_names[ii]+'.png'
        plt.savefig(plot_name, bbox_inches='tight', dpi=300)


def plot_rms_binning():

    bin_size = P0 / ngate * 10**6 #microsecond
    phase_file_1 = np.load('gp052a_06to07_move_modes_1_tint_0.125.npy')
    phase_file_2 = np.load('gp052a_06to07_move_modes_2_tint_0.125.npy')

    print 'nonzero of phase_file_1', np.count_nonzero(phase_file_1)
    print 'nonzero of phase_file_2', np.count_nonzero(phase_file_2)

    phase_bins_1 = phase_file_1[:,0] * bin_size # change unit from phase bins to microseconds
    phase_bins_2 = phase_file_2[:,0] * bin_size

    rebin_factor = [1, 2, 4, 8, 16, 32, 64, 128]
    print 'rebin_factor', rebin_factor
    rebin_time = np.array(rebin_factor)*0.125

    tt = np.arange(0.125,60,1)

    phase_bins_1_rebin_rms = np.zeros(len(rebin_factor))
#    phase_bins_1_rebin_rms_para = np.zeros(len(rebin_factor))
    phase_bins_2_rebin_rms = np.zeros(len(rebin_factor))
#    phase_bins_2_rebin_rms_para = np.zeros(len(rebin_factor))

    for ii in xrange(len(rebin_factor)):
        # before subtracting parabola
        phase_bins_1_reshape = np.mean(phase_bins_1.reshape(len(phase_bins_1)/rebin_factor[ii], rebin_factor[ii]), axis=1)
        phase_bins_1_rebin_rms[ii] = np.std(phase_bins_1_reshape)
        phase_bins_2_reshape = np.mean(phase_bins_2.reshape(len(phase_bins_2)/rebin_factor[ii], rebin_factor[ii]), axis=1)
        phase_bins_2_rebin_rms[ii] = np.std(phase_bins_2_reshape)

#        # after subtracting parabola
#        times = np.arange(0, 300, rebin_time[ii])
#        print 'len(times)', len(times)
#        print 'len(phase_bins_1_reshape)', len(phase_bins_1_reshape)
#        z_1 = np.polyfit(times, phase_bins_1_reshape, 2)
#        print 'z_1',z_1
#        p_1 = np.poly1d(z_1)
#        para_1 = p_1(times)
#        phase_bins_1_reshape_para = phase_bins_1_reshape - para_1
#        print 'phase_bins_1_reshape_para', phase_bins_1_reshape_para
#        phase_bins_1_rebin_rms_para[ii] = np.std(phase_bins_1_reshape_para)

#        z_2 = np.polyfit(times, phase_bins_2_reshape, 2)
#        p_2 = np.poly1d(z_2)
#        para_2 = p_2(times)
#        phase_bins_2_reshape_para = phase_bins_2_reshape - para_2
#        phase_bins_2_rebin_rms_para[ii] = np.std(phase_bins_2_reshape_para)

#    print 'phase_bins_1_rebin_rms', phase_bins_1_rebin_rms
#    print 'phase_bins_1_rebin_rms_para', phase_bins_1_rebin_rms_para
#    print 'phase_bins_2_rebin_rms', phase_bins_2_rebin_rms
#    print 'phase_bins_2_rebin_rms_para', phase_bins_2_rebin_rms_para

    max_value = np.amax(np.concatenate((phase_bins_1_rebin_rms, 
                                        phase_bins_2_rebin_rms)))

    diff = np.amax(phase_bins_1_rebin_rms) - np.amax(phase_bins_2_rebin_rms)
#    diff_para = np.amax(phase_bins_1_rebin_rms_para) - np.amax(phase_bins_2_rebin_rms_para)

    markersize = 4.0
    fontsize = 16

    plt.close('all')
    x_axis = rebin_time
    plt.loglog(tt, 1/np.sqrt(tt)*(np.sqrt(0.125)*max_value), 'k', label='1 / sqrt(t), normalized to the 1 mode')
    plt.loglog(x_axis, phase_bins_1_rebin_rms, 'ro', markersize=markersize, label='1 mode, max: '+str(np.amax(np.round(phase_bins_1_rebin_rms,4)))+'('+r'$\mu$'+'s)')
    plt.loglog(x_axis, phase_bins_1_rebin_rms, 'r')
    plt.loglog(x_axis, phase_bins_2_rebin_rms, 'bs', markersize=markersize, label='2 modes, max: '+str(np.amax(np.round(phase_bins_2_rebin_rms, 4)))+'('+r'$\mu$'+'s)')
    plt.loglog(x_axis, phase_bins_2_rebin_rms, 'b--')
#    plt.loglog(x_axis, phase_bins_1_rebin_rms_para, 'g^', markersize=markersize, label='1 mode, after subtracting para, max: '+str(np.amax(np.round(phase_bins_1_rebin_rms_para, 4)))+'('+r'$\mu$'+'s)')
#    plt.loglog(x_axis, phase_bins_1_rebin_rms_para, 'g:')
#    plt.loglog(x_axis, phase_bins_2_rebin_rms_para, 'y*', markersize=markersize, label='2 modes, after subtracting para, max: '+str(np.amax(np.round(phase_bins_2_rebin_rms_para, 4)))+'('+r'$\mu$'+'s)')
#    plt.loglog(x_axis, phase_bins_2_rebin_rms_para, 'y-.')


    plt.xlabel('Int. time for each rebinning profile (s)', fontsize=fontsize)
    ylabel = 'RMS for rebinning phase measurements '+'('+r'$\mu$'+'s)'
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(loc='lower left', fontsize=fontsize-4)
    plt.xlim([0,20])
    title = 'diff: '+str(np.round(diff,3))+'('+r'$\mu$'+'s)'#, diff_para: +str(np.round(diff_para,3))+'('+r'$\mu$'+'s)'
    plt.title(title, fontsize=fontsize-4)
    plot_name = 'rebinning_rms_shift.png'
    plt.savefig(plot_name, bbox_inches='tight', dpi=300)



def reconstruct_V(V, V0_raw, V1_raw):
    '''reconstruct V modes by adding raw V0 and V1 to the V modes created without raw V0 and V1'''
    V[0] += V0_raw
    V[1] += V1_raw

    # check V_L and V_R
    V_L = V[:, 0: V.shape[1]/2]
    V_R = V[:, V.shape[1]/2:V.shape[1]]
    plot_V(V_L, 'recon_V_L.png')
    plot_V(V_R, 'recon_V_R.png')

    # reshape V into (mode numbers, L/R, phases)
    V_recon = V.reshape(V.shape[0], 2, V.shape[1]/2)

    return V_recon

def plot_V(V, plot_name_V): 
    '''plot V modes'''
    fontsize = 16
    plt.close('all')
    plt.figure()
    n_step = -0.3
    x_range = np.arange(0 , len(V[0]))
    color = ['r', 'g', 'b', 'y', 'c', '0.0', '0.2', '0.4', '0.6', '0.8']
    for ii in xrange(len(color)):
        plt.plot(x_range, np.roll(V[ii] + ii *n_step, 0), color[ii], linewidth=1.0)
    plt.xlim((0, len(V[0])))
    plt.ylabel('V values', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
#    plot_name_V = 'recon_V.png'
    plt.savefig(plot_name_V, bbox_inches='tight', dpi=300)

def reconstruct_profile(U,s,V):
    '''reconstruct a profile using U, s, and V.'''
    profile_recon = np.dot(U, np.dot(np.diag(s), V))
    return profile_recon 

def norm_variance_profile(profile, variance):
    '''transform an array into the array with zero mean and unit variance.'''
    profile_norm_var = (profile - np.mean(profile)) / np.sqrt(variance)
#    print 'mean ', np.mean(profile_norm_var)
#    print 'var ', np.var(profile_norm_var)
    return profile_norm_var

def check_noise(profile):

    # remove the signal modes, and check the noise modes only.
    U, s, V = svd(profile)
    V[0] = 0
    V[1] = 0
    noise_profiles = reconstruct_profile(U,s,V)

    #rebin
    profile_stack = 8
    print 'line 1192, profile.shape', profile.shape
#    profile = stack(profile, profile_stack)
#    print 'profile.shape', profile.shape

    var_L = np.zeros((profile.shape[0]))
    var_R = np.zeros((profile.shape[0]))
    for ii in xrange(profile.shape[0]):
        var_L[ii] = np.var(noise_profiles[ii, 0:profile.shape[1]/2])
        var_R[ii] = np.var(noise_profiles[ii, profile.shape[1]/2:profile.shape[1]])

    fontsize = 16

    plt.close('all')
    plt.figure()
    x_range = np.arange(0 , len(var_L))
    plt.plot(x_range, var_L, 'ro', linewidth=2.5, label='var_L')
    plt.plot(x_range, var_R, 'bs', linewidth=2.5, label='var_R')
    plt.xlim((0, len(var_L)))
    plt.xlabel('profile numbers', fontsize=fontsize)
    plt.ylabel('Variance', fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.savefig('variance_rl.png', bbox_inches='tight', dpi=300)

def mpi_random_noise(profiles):

    mpi_profiles = np.array_split(profiles, size)
    mpi_profiles = comm.scatter(mpi_profiles, root=0)
    profiles_random = random_noise(mpi_profiles)
    all_profiles_random = np.concatenate(comm.allgather(profiles_random))
    return all_profiles_random

def random_noise(profile_raw):

    noise_factor = 0.8
    profile_random = np.zeros(profile_raw.shape)
    for ii in xrange(profile_random.shape[0]):
        pro_raw_L = profile_raw[ii,0:profile_random.shape[1]/2]
        pro_L = pro_raw_L + np.array([random.gauss(0, noise_factor*np.std(pro_raw_L)) for i in range(profile_random.shape[1]/2)])
        pro_raw_R = profile_raw[ii,profile_random.shape[1]/2:]
        pro_R = pro_raw_R + np.array([random.gauss(0, noise_factor*np.std(pro_raw_R)) for i in range(profile_random.shape[1]/2)])
        profile_random[ii] = np.concatenate((pro_L, pro_R))

    return profile_random

def mpi_phase_move(profiles, para_phase_bins):

    '''Scattering profiles to each rank, moving phases of each profiles, and gathering the results to rank 0'''

    # reshpae profiles into (pulse numbers, L/R pol, phase bins)
    profiles = profiles.reshape(profiles.shape[0], 2, profiles.shape[1]/2)

    mpi_profiles = np.array_split(profiles, size)
    mpi_para_phase_bins = np.array_split(para_phase_bins, size)

    mpi_profiles = comm.scatter(mpi_profiles, root=0)
    mpi_para_phase_bins = comm.scatter(mpi_para_phase_bins, root=0)
    profiles_nopara = profiles_phase_move(mpi_profiles, mpi_para_phase_bins)
    all_profiles_nopara = np.concatenate(comm.allgather(profiles_nopara))

    all_profiles_nopara = all_profiles_nopara.reshape(len(profiles), profiles.shape[-1]*2)
    return all_profiles_nopara   

def profiles_phase_move(profiles, para_phase_bins):

    profiles_nopara = np.zeros(profiles.reshape(len(profiles), profiles.shape[-1]*2).shape)
    for ii, profile in list(enumerate(profiles)):
        print "Profile: ", ii
        profile_L = profile[0]
        profile_R = profile[1]

        profile_fft_L = fftpack.fft(profile_L)
        profile_fft_R = fftpack.fft(profile_R)

        profile_ifft_L_shift = fftpack.ifft(apply_phase_shift(profile_fft_L, para_phase_bins[ii]))
        profile_ifft_R_shift = fftpack.ifft(apply_phase_shift(profile_fft_R, para_phase_bins[ii]))
        profiles_nopara[ii] = np.concatenate((profile_ifft_L_shift, profile_ifft_R_shift))

    return profiles_nopara


def mpi_phase_fitting(profiles, V, patterns, NMODES, tint_stack):

    print 'Starting mpi_phase_fitting'
    print 'profiles.shape', profiles.shape


    '''Scattering profiles to each rank, fitting each profiles, and gathering the results to rank 0'''
#    patterns += 'modes_'+str(NMODES)+'_tint_'+str(tint_stack)
    remainder = len(profiles) % size
    if rank == 0:
        npy_lik_name = patterns+'.npy'
        mpi_profiles = np.array_split(profiles, size)
        print 'finished splitting'
    else:
        mpi_profiles = None

    mpi_profiles = comm.scatter(mpi_profiles, root=0)
    print 'finished scatter'
    phase_amp_bin_lik = phase_fitting(mpi_profiles, V, patterns, NMODES, tint_stack, remainder)
#    print 'rank, type and len of phase_amp_bin_lik', rank, type(phase_amp_bin_lik), len(phase_amp_bin_lik)
#   print 'rank, type and len of comm.allgather(phase_amp_bin_lik)', rank, type(comm.allgather(phase_amp_bin_lik)), len(comm.allgather(phase_amp_bin_lik))
    npy_lik_file = np.concatenate(comm.allgather(phase_amp_bin_lik))
    print 'rank, npy_lik_file.shape', rank, npy_lik_file.shape
  
    '''save the fitting amp and bin as [bin, amps, bin_err, amp_errs]''' 
    npy_lik_file = npy_lik_file.reshape(len(profiles), 2*(1+NMODES))
    return npy_lik_file
     
def phase_fitting(profiles, V, patterns, NMODES, tint_stack, remainder):
#    print 'NMODES', NMODES
    profile_numbers = []
    profile_numbers_lik = []
    phase_model = []
    phases = []
    phase_errors = []
    phases_lik = []
    phase_errors_lik = []

#    nprof = 10
    nprof = len(profiles)
    profiles = profiles[:nprof]
    npy_lik_file = np.zeros((len(profiles), (1+NMODES)*2))
 
    V_fft = fftpack.fft(V, axis=2)
    V_fft_L = fftpack.fft(V[:,0,:], axis=1)
    V_fft_R = fftpack.fft(V[:,1,:], axis=1)
 
    for ii, profile in list(enumerate(profiles)):
        ii_rank = ii + rank * len(profiles)
        if rank >= remainder:
            ii_rank += remainder
        print "Profile: ", ii_rank
        profile_numbers.append(ii)
        profile_L = profile[0]
        profile_R = profile[1]
 
        profile_fft = fftpack.fft(profile)
        profile_fft_L = fftpack.fft(profile_L)
        profile_fft_R = fftpack.fft(profile_R)

        phase_init = 0.
        phase_model.append(phase_init)
        amp_init = np.sqrt(np.sum(profile**2))
        pars_init = [phase_init, amp_init] + [0.] * (NMODES - 1)
#        pars_init = [phase_init, amp_init, amp_init/10]
        pars_fit, cov, infodict, mesg, ier = optimize.leastsq(
                residuals,
                pars_init,
                (NMODES, profile_fft, V_fft),
                full_output=1,
                )
        fit_res = residuals(pars_fit, NMODES, profile_fft, V_fft)

        chi2_fit = chi2(NMODES, pars_fit, profile_fft, V_fft)
        dof = len(fit_res) - len(pars_init)
        red_chi2 = chi2_fit / dof

#        print 'pars_fit, cov, infodict, mesg, ier',pars_fit, cov, infodict, mesg, ier
        print "chi1, dof, chi2/dof:", chi2_fit, dof, red_chi2
        print 'red_chi2', red_chi2
        cov_norm = cov * red_chi2

        errs = np.sqrt(cov_norm.flat[::len(pars_init) + 1])
        corr = cov_norm / errs[None,:] / errs[:,None]

        print "phase, amplidudes (errors):"
        print pars_fit
        print errs
#        print "correlations:"
#        print corr

        phases.append(pars_fit[0])
        phase_errors.append(errs[0])

        model_fft_L = model(NMODES, pars_fit, V_fft_L) 
        model_fft_R = model(NMODES, pars_fit, V_fft_R)
        print 'model_fft_L.shape', model_fft_L.shape
        print 'profile_fft_L.shape', profile_fft_L.shape
#        plot_phase_fft(pick_harmonics(profile_fft_L), model_fft_L, ii_rank, plots_path+patterns+'fit_L_')
#        plot_phase_fft(pick_harmonics(profile_fft_R), model_fft_R, ii_rank, plots_path+patterns+'fit_R_')
#        plot_phase_ifft(NMODES, pars_fit, profile_L, profile_R, V_fft_L, V_fft_R, ii_rank, plots_path+patterns+'fit_')

        if True:
            # Fix phase at set values, then fit for amplitudes. Then integrate
            # the likelihood over the parameters space to get mean and std of
            # phase.
            phase_diff_samples = np.arange(-20, 20, 0.15) * errs[0]
            chi2_samples = []
            for p in phase_diff_samples:
                this_phase = p + pars_init[0]
                if True:
                    # Linear fit.
#                    P = shift_trunc_modes(this_phase, V_fft)
#                    d = pick_harmonics(profile_fft)
                    P_L = shift_trunc_modes(NMODES, this_phase, V_fft_L)
                    P_R = shift_trunc_modes(NMODES, this_phase, V_fft_R)
                    P = np.concatenate((P_L, P_R), axis=1)

                    d_L = pick_harmonics(profile_fft_L)
                    d_R = pick_harmonics(profile_fft_R)
                    d = np.concatenate((d_L, d_R))

                    N = linalg.inv(np.dot(P, P.T))
                    this_pars_fit = np.dot(N, np.sum(P * d, 1))
                else:
                    # Nonlinear fit.
                    residuals_fix_phase = lambda pars: residuals(
                            [this_phase] + list(pars),
                            NMODES, 
                            profile_fft,
                            V_fft,
                            ) / red_chi2
                    #pars_sample = [p + pars_fit[0]] + list(pars_fit[1:])
                    this_pars_fit, cov, infodict, mesg, ier = optimize.leastsq(
                        residuals_fix_phase,
                        list(pars_init[1:]),
                        #(profile_fft, V_fft),
                        full_output=1,
                        )
                chi2_sample = chi2(NMODES, 
                        [this_phase] + list(this_pars_fit),
                        profile_fft,
                        V_fft,
                        1. / red_chi2,
                        )
                chi2_samples.append(chi2_sample)
            phase_diff_samples = np.array(phase_diff_samples)
            chi2_samples = np.array(chi2_samples, dtype=np.float64)
            chi2_samples -= chi2_samples_const # since chi2_samples are too large
#            print 'chi2_samples', chi2_samples
            while np.amax(chi2_samples) > 1400:
                chi2_samples -= 50
#            print 'np.amax(chi2_samples)', np.amax(chi2_samples)
#            print 'np.amin(chi2_samples)', np.amin(chi2_samples)

            if False:
                #Plot of chi-squared (ln likelihood) function.
                plt.figure()
                plt.plot(phase_diff_samples, chi2_samples - chi2_fit / red_chi2)
                plt.ylabel(r"$\Delta\chi^2$")
                plt.xlabel(r"$\Delta_{\rm phase}$")

            # Integrate the full liklihood, taking first and second moments to
            # get mean phase and variance.
            likelihood = np.exp(-chi2_samples / 2)
            norm = integrate.simps(likelihood) #* np.exp(-chi2_samples_const / 2)
            print 'norm', norm
            print 'Note: norm does not include exp('+str(-chi2_samples_const / 2)+')'
            mean = integrate.simps(phase_diff_samples * likelihood) / norm
            print 'mean', mean
            var = integrate.simps(phase_diff_samples**2 * likelihood) / norm - mean**2
            std = np.sqrt(var)
            print 'std', std
            print "Integrated Liklihood:", pars_init[0] + mean, std
            phases_lik.append(pars_init[0] + mean)
            phase_errors_lik.append(std)
            profile_numbers_lik.append(ii)
            # plot the chi2 distribution
#            plot_phase_diff_chi2(phase_diff_samples, likelihood, norm, ii_rank, plots_path+patterns)
            npy_lik_file[ii] = np.concatenate(([pars_init[0] + mean], pars_fit[1:], [std], errs[1:]))
        else:
            npy_lik_file[ii] = np.concatenate((pars_fit[:], errs[:]))

    return npy_lik_file
        

def stack(profiles, profile_stack):

    # for profile_stack > 64 (integration time > 8 seconds), we need to omit some profiles.
    len_1_prof = int(536/tint)
    n_sessions = int(len(profiles) / len_1_prof)
    stack_profiles = []
    for i in range(n_sessions):   
        profile = profiles[i*len_1_prof:(i+1)*len_1_prof]
        nprof = len(profile)
        nprof -= nprof % profile_stack
        profile = profile[:nprof].reshape(nprof // profile_stack, profile_stack, profile.shape[-1])
        profile = np.mean(profile, 1)
        print 'profile.shape', profile.shape
        stack_profiles.append(profile)
        print 'len(stack_profiles), len(stack_profiles[0])',len(stack_profiles), len(stack_profiles[0])
    stack_profiles = np.concatenate((stack_profiles))
    print 'stack_profiles.shape', stack_profiles.shape
    return stack_profiles


def residuals(parameters, NMODES, profile_fft, V_fft):
    res_L = pick_harmonics(profile_fft[0]) - model(NMODES, parameters, V_fft[:,0,:]) 
    res_R = pick_harmonics(profile_fft[1]) - model(NMODES, parameters, V_fft[:,1,:])
    res_all = np.concatenate((res_L, res_R))
    return res_all


def model(NMODES, parameters, V_fft):
    phase_bins = parameters[0]
    amplitudes = np.array(parameters[1:])
    shifted_modes = shift_trunc_modes(NMODES, phase_bins, V_fft)
    return np.sum(amplitudes[:,None] * shifted_modes, 0)


def shift_trunc_modes(NMODES, phase_bins, V_fft):
    V_fft_shift = apply_phase_shift(V_fft, phase_bins)
    V_harmonics = pick_harmonics(V_fft_shift)
    return V_harmonics[:NMODES]

def chi2(NMODES, parameters, profile_fft, V_fft, norm=1):
    return np.sum(residuals(parameters, NMODES, profile_fft, V_fft)**2) * norm


def pick_harmonics(profile_fft):
    harmonics = profile_fft[..., 1:NHARMONIC]
    harmonics = np.concatenate((harmonics.real, harmonics.imag), -1)
    return harmonics


def apply_phase_shift(profile_fft, phase_bins_shift):
    "Parameter *phase_shift* takes values [0 to 1)."

    phase_shift = phase_bins_shift / NPHASEBIN
    n = profile_fft.shape[-1]
    freq = fftpack.fftfreq(n, 1./n)
    phase = np.exp(-2j * np.pi * phase_shift * freq)
    return profile_fft * phase

def rebin_array(input_data, rebin_factor):
    xlen = input_data.shape[0] / rebin_factor
    output_data = np.zeros((xlen,))
    for ii in range(xlen):
        output_data[ii]=input_data[ii*rebin_factor:(ii+1)*rebin_factor].mean()
    return output_data
    
def rebin_spec(input_data, rebin_factor_0, rebin_factor_1):
    xlen = input_data.shape[0] / rebin_factor_0
    ylen = input_data.shape[1] / rebin_factor_1
    output_data = np.zeros((xlen, ylen))
    for ii in range(xlen):
        for jj in range(ylen):
            output_data[ii,jj]=input_data[ii*rebin_factor_0:(ii+1)*rebin_factor_0,jj*rebin_factor_1:(jj+1)*rebin_factor_1].mean()
    return output_data

def fft(file):
    profile_fft = np.fft.fft(file)
    profile_fft[0] = 0
    return profile_fft

def ifft(file):
    profile_ifft = np.fft.ifft(file)
    return profile_ifft

def fft_phase_curve_inverse(parameters, profile_fft):
    '''inverse phase for chaning 1.0j to -1 1.0j'''
    freq = np.fft.fftfreq(len(profile_fft))
    n= len(profile_fft)
    fft_model = parameters[1] * np.exp(-1.0j * 2 * np.pi * freq * ( n - parameters[0])) * profile_fft
    return fft_model

def svd(file):

    time_matrix = np.zeros(file.shape)
    for ii in xrange(len(time_matrix)):
        time_matrix[ii] = ifft(fft_phase_curve_inverse([0, 1], fft(file[ii]))).real

    U, s, V = np.linalg.svd(time_matrix, full_matrices=False)

    UT = U.T

    if np.abs(np.amax(V[0])) < np.abs(np.amin(V[0])):
        V[0] = -V[0]

    if True:
        np.save('UT_.npy', UT)
        np.save('s_.npy', s)
        np.save('V_.npy', V)

    return U, s, V

def plot_spec():

    time_length = B_data.shape[1]
    fontsize = 16

    plt.close('all')
    plt.figure()
    n_step = -5
    x_range = np.arange(0 , len(B_data[35197]))
    plt.plot(x_range, B_data[31597])
    plt.xlim((0, len(B_data[0])))
    plt.xlabel('Phase', fontsize=fontsize)
    plt.ylabel('B data values', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
#    plot_name_B_data = plot_name + '_V.png'
    plt.savefig('B_data_31597.png', bbox_inches='tight')


def plot_svd(file, plot_name):

    U, s, V= svd(file)

    UT = U.T

    V_name = plot_name + '_V.npy'
    UT_name = plot_name + '_UT.npy'
    np.save(V_name, V)
    np.save(UT_name, UT)
 
    print 'len(V[0])', len(V[0])
    print 'UT.shape', UT.shape
    print 's.shape', s.shape
    print 'V.shape', V.shape

    fontsize = 16

    plt.close('all')
    plt.figure()
    x_range = np.arange(0, len(s))
#    plt.semilogy(x_range, s, 'ro-')
    plt.plot(x_range, s, 'ro-')
#    plt.xlabel('Time')
    plt.ylabel('s values')
    plt.xlim((0, np.max(x_range)))
    plt.ylim((0, np.max(s)))
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plot_name_s = plot_name + '_s.png'
    plt.savefig(plot_name_s, bbox_inches=None, dpi=300)
#    plt.show()

#    print 'np.max(V[0])',np.max(V[0])
#    print 'np.max(V[1])',np.max(V[1])

    plt.close('all')
    plt.figure()
    n_step = -0.3
    x_range = np.arange(0 , len(V[0]))
    color = ['r', 'g', 'b', 'y', 'c', '0.0', '0.4', '0.8']
#    color = ['r', 'g', 'b']
    for ii in xrange(len(color)):
        plt.plot(x_range, np.roll(V[ii] + ii *n_step, 0), color[ii], linewidth=1.0)
    plt.xlim((0, len(V[0])))
#    plt.xlabel('Phase', fontsize=fontsize)
    plt.ylabel('V values', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plot_name_V = plot_name + '_V.png'
    plt.savefig(plot_name_V, bbox_inches='tight', dpi=300)

    plt.close('all')
    plt.figure()
    n_step = -0.3
    x_range = np.arange(0 , len(UT[0]))
    color = ['r', 'g', 'b', 'y', 'c', '0.0', '0.4', '0.8']
#    color = ['r', 'g', 'b']
    for ii in xrange(len(color)):
        plt.plot(x_range, np.roll(UT[ii] + ii *n_step, 0), color[ii], linewidth=1.0)
    plt.xlim((0, len(UT[0])))
#    plt.xlabel('Phase', fontsize=fontsize)
    plt.ylabel('U values', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plot_name_UT = plot_name + '_UT.png'
    plt.savefig(plot_name_UT, bbox_inches='tight', dpi=300)


def plot_rebin_ut(xmin, xmax):
    plot_name = 'B_rebin_ut'
    fontsize = 16

    rebin_u = np.load('/scratch2/p/pen/hsiuhsil/psr_B1957+20/data_file/B_rebin_U_t3334.npy')
    rebin_ut = rebin_u.T
    print 'rebin_ut.shape', rebin_ut.shape
#    np.save('rebin_ut.npy', rebin_ut)
    
    plt.close('all')
    plt.figure()
    n_step = -0.03
    x_range = np.arange(xmin, xmax)
    color = ['b']
#    color = ['r', 'g', 'b', 'y', 'c', '0.0', '0.2', '0.4', '0.6', '0.8']
#    color = ['r', 'g', 'b']
    for ii in xrange(len(color)):
        plt.plot(x_range, np.roll(rebin_ut[2,xmin:xmax] + ii *n_step, 0), color[ii], linewidth=1.0)
#    plt.xlim((0, len(rebin_ut[0])))
    plt.xlabel('Pulse numbers', fontsize=fontsize)
    plt.ylabel('U values', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plot_name_U = plot_name + '_' + str(xmin) + '_' + str(xmax) + '_rebin_ut.png'
    plt.savefig(plot_name_U, bbox_inches='tight', dpi=300)
    
def plot_phase_fft(data_fft, model_fft, ii, plot_name):

    freq = np.fft.fftfreq(len(data_fft))   
    '''Real part'''
    model_fft_real = np.concatenate((model_fft[:(len(model_fft)/2)], model_fft[:(len(model_fft)/2)][::-1]))
    data_fft_real = np.concatenate((data_fft[:(len(data_fft)/2)], data_fft[:(len(data_fft)/2)][::-1]))
    res_fft_real = data_fft_real - model_fft_real

    '''Imag part'''
    model_fft_imag = np.concatenate((model_fft[(len(model_fft)/2):], -model_fft[(len(model_fft)/2):][::-1]))
    data_fft_imag = np.concatenate((data_fft[(len(data_fft)/2):], -data_fft[(len(data_fft)/2):][::-1]))
    res_fft_imag = data_fft_imag - model_fft_imag

    freq_range = np.linspace(np.amin(np.fft.fftfreq(len(data_fft))), np.amax(np.fft.fftfreq(len(data_fft))), num=len(data_fft), endpoint=True)
    freq_min = np.amin(freq_range)
    freq_max = np.amax(freq_range)

#    plot_name = 'phase_fit_'
    plot_name += str(ii) + '_'
    fontsize = 16
    markersize = 4

    try:
        '''Plot for real and imag parts in the Fourier space.'''
        plt.close('all')
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(16,9))
        f.subplots_adjust(wspace=0.09, hspace=0.07)
        mode_range = np.linspace(-len(freq)/2, len(freq)/2, num=len(freq), endpoint=True)
        print 'len(freq)', len(freq)
        xmax = np.amax(mode_range)
        xmin = np.amin(mode_range)   
        ax1.plot(mode_range, np.roll(data_fft_real, -int(len(freq)/2)),'bo', markersize=markersize)
        ax1.plot(mode_range, np.roll(model_fft_real, -int(len(freq)/2)),'r-', markersize=markersize)
        ax1.set_title('Real', size=fontsize)
        ax1.set_xlim([xmin,xmax])
        ax1.tick_params(axis='both', which='major', labelsize=fontsize)

        ax2.plot(mode_range, np.roll(data_fft_imag, -int(len(freq)/2)),'bo', markersize=markersize)
        ax2.plot(mode_range, np.roll(model_fft_imag, -int(len(freq)/2)),'r-', markersize=markersize)
        ax2.set_title('Imag', size=fontsize)
        ax2.set_xlim([xmin,xmax])
        ax2.tick_params(axis='both', which='major', labelsize=fontsize)

        ax3.plot(mode_range, np.roll(res_fft_real, -int(len(freq)/2)),'gs', markersize=markersize)
        ax3.set_xlabel('Harmonic modes', fontsize=fontsize)
        ax3.set_ylabel('Residuals (T/Tsys)', fontsize=fontsize)
        ax3.set_xlim([xmin,xmax])
        ax3.tick_params(axis='both', which='major', labelsize=fontsize)

        ax4.plot(mode_range, np.roll(res_fft_imag, -int(len(freq)/2)),'gs', markersize=markersize)
        ax4.set_xlabel('Harmonic modes', fontsize=fontsize)
        ax4.set_xlim([xmin,xmax])
        ax4.tick_params(axis='both', which='major', labelsize=fontsize)     

        f.savefig(plot_name+'fft.png')
    except ValueError, IOError:
        pass

def plot_phase_ifft(NMODES, pars_fit, data_L, data_R, V_fft_L, V_fft_R, ii, plot_name):

    '''Plot for real part in real space'''
    fit_model_L = 0
    fit_model_R = 0
    V_fft_L_shift = apply_phase_shift(V_fft_L, pars_fit[0])
    V_fft_R_shift = apply_phase_shift(V_fft_R, pars_fit[0])
    for jj in range(NMODES):
        fit_model_L += pars_fit[jj+1] * fftpack.ifft(V_fft_L_shift[jj]).real
        fit_model_R += pars_fit[jj+1] * fftpack.ifft(V_fft_R_shift[jj]).real 

#    print 'fit_model_L.shape', fit_model_L.shape
#    print 'fit_model_L', fit_model_L

    fit_model = np.concatenate((fit_model_L, fit_model_R))
    data_L -= np.mean(data_L)
    data_R -= np.mean(data_R)
    data = np.concatenate((data_L, data_R))

    #rebin data
    rebin_factor = 4
    data_L_rebin = rebin_array(data_L, 4)
    data_R_rebin = rebin_array(data_R, 4)
    data_rebin = np.concatenate((data_L_rebin, data_R_rebin))

    model_ifft = fit_model
    data_ifft = data
    res_ifft = data_ifft - model_ifft

#    plot_name = 'phase_fit_'
    plot_name += str(ii) + '_'
    fontsize = 16
    markersize = 4

    plt.close('all')
    f, ((ax1, ax2)) = plt.subplots(2, 1, sharex='col', figsize=(8,9))
    f.subplots_adjust(hspace=0.07)
    phase_bins_range = np.linspace(0, len(data), num=len(data), endpoint=True)
    phase_bins_range_rebin = np.linspace(0, len(data), num=len(data)/rebin_factor, endpoint=True)
    xmax = np.amax(phase_bins_range)
    xmin = np.amin(phase_bins_range)
    ax1.plot(phase_bins_range, data_ifft,'bo', markersize=markersize/2)
    ax1.plot(phase_bins_range, model_ifft,'r-', markersize=markersize)
    ax1.plot(phase_bins_range_rebin, data_rebin,'ys', markersize=markersize)
    ax1.set_xlim([xmin,xmax])
    ax1.set_ylabel('T/Tsys', fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.plot(phase_bins_range, res_ifft,'gs', markersize=markersize)
    ax2.set_xlim([xmin,xmax])
    ax2.set_xlabel('Phase Bins', fontsize=fontsize)
    ax2.set_ylabel('Residuals (T/Tsys)', fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)

    try:
        plt.savefig(plot_name + 'ifft.png', bbox_inches='tight', dpi=300)    
    except ValueError, IOError:
        pass


def plot_phase_diff_chi2(phase_diff_samples, likelihood, norm, ii, patterns):
    plot_name = patterns + 'fit_'
    fontsize = 16
    plot_name += str(ii) + '_'
    plt.close('all')
    phase_diff_range = np.linspace(np.amin(phase_diff_samples), np.amax(phase_diff_samples), num=len(phase_diff_samples), endpoint=True)
    plt.semilogy(phase_diff_range, likelihood / norm / (0.02/NPHASEBIN))
    plt.xlabel('Phase Bins', fontsize=fontsize)
    plt.ylabel('log(Likelihood)', fontsize=fontsize)
    plt.xlim((phase_diff_range[np.where((likelihood / norm / (0.02/NPHASEBIN))>np.amax(likelihood / norm / (0.02/NPHASEBIN)) * 10**-4)[0][0]],phase_diff_range[np.where((likelihood / norm / (0.02/NPHASEBIN) )>np.amax(likelihood / norm / (0.02/NPHASEBIN)) * 10**-4)[0][-1]]))
    plt.ylim((np.amax(likelihood / norm / (0.02/NPHASEBIN)) * 10**-4, np.amax(likelihood / norm / (0.02/NPHASEBIN)) * 4.5))
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    try:
        plt.savefig(plot_name+'phase_chi2.png', bbox_inches='tight', dpi=300)
    except ValueError, IOError:
        pass


if __name__ == '__main__':
    main()


