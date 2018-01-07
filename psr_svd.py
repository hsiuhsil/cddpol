import sys, os, argparse
import os.path

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from baseband import mark4
from pulsar.predictor import Polyco
import astropy.units as u
import math
import random
from random import gauss
random.seed(3)
from scipy import fftpack, optimize, interpolate, linalg, integrate
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
time_str_patterns = time_str_folder + 'gp052a_ar_no0007*536s_h5'
phase_amp_files = time_str_folder + 'gp052_fit_nodes_1_tint_8.0sec_npy/*nodes*npy'
TOAs_files = time_str_folder + 'gp052_TOA_nodes_1_tint_8.0sec_npy/*nodes*npy'
#TOAs_files = time_str_folder + 'gp052_TOA_nodes_1_tint_8.0sec_npy_zerocentre/*nodes*npy'

NHARMONIC = paras.NHARMONIC 
NMODES = paras.NMODES

NPHASEBIN = paras.NPHASEBIN
NCENTRALBINS = paras.NCENTRALBINS
NCENTRALBINSMAIN = paras.NCENTRALBINSMAIN
chi2_samples_const = paras.chi2_samples_const

tint = paras.tint
prof_stack = paras.prof_stack
tint_stack = paras.tint*paras.prof_stack

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
        lists_amps = []
        lists_times = []
        for ii in xrange(len(glob.glob(phase_amp_files))):
            ff = np.load(sorted(glob.glob(phase_amp_files))[ii])
#            hh = sorted(glob.glob(time_str_patterns))[ii]
            hh = glob.glob(time_str_folder + str(sorted(glob.glob(phase_amp_files))[ii][81:97]) + '*h5')[0]
            this_file = h5py.File(hh, 'r')
            t00 = this_file['t00'][0]
            t00_series = t00 + np.arange(0, 8.*paras.T, paras.tint*paras.prof_stack)/86400.
            if ff.shape[0] == 67:
                lists_amps.append(ff)
                lists_times.append(t00_series)
        phases_amps = np.vstack(lists_amps)
        print 'phases_amps.shape', phases_amps.shape
        times = np.hstack(lists_times)
        print 'times.shape', times.shape
        print 'finished combining phases_amps and times data'

        plot_phase_lik(times, phases_amps, 'phase_lik.png')
        plot_fit_amps(times, phases_amps, 'amp_ratios.png')

    if False:
        t_end = 300 #sec
        print 't_end: ', t_end
        tint_stack = paras.tint*paras.prof_stack
        times = np.arange(0, t_end, tint_stack)
        phases_amps_1 = np.load('gp052a_ar_no0007fit_nodes_1_tint_30.0.npy')
        phases_amps_2 = np.load('gp052a_ar_no0007fit_nodes_2_tint_30.0.npy')

        plot_phase_lik(times, phases_amps_1[:len(times)], tint_stack, 'phase_lik_nodes_1_tint_'+str(tint_stack)+'sec.png')
        plot_phase_lik(times, phases_amps_2[:len(times)], tint_stack, 'phase_lik_nodes_2_tint_30.0sec.png')

def main():

#    plot_ut_phase_time_streams()
#    plot_rms_binning()
#    plot_ut_phase_correlation()
#    TOAs_debug()
#    generate_TOAs()
#    generate_tim_file()

    if True: 
        '''Reconstruct V modes'''

        this_file = h5py.File('/mnt/raid-project/gmrt/hhlin/time_streams_1957/gp052a_ar_no0007_512g_0b_56821.2537037+536s_h5','r')
        profile_raw = this_file['fold_data_int_0.125_band_0'][:2400,:] #5mins dataset

        # introduce Gaussian random noise. 
#        profile_random = np.zeros((profile_raw.shape))
#        for ii in xrange(profile_random.shape[0]):
#            pro_raw_L = profile_raw[ii,0:profile_random.shape[1]/2]
#            pro_L = pro_raw_L + np.array([random.gauss(np.mean(pro_raw_L), np.std(pro_raw_L)) for i in range(profile_random.shape[1]/2)])
#            pro_raw_R = profile_raw[ii,profile_random.shape[1]/2:]
#            pro_R = pro_raw_R + np.array([random.gauss(np.mean(pro_raw_R), np.std(pro_raw_R)) for i in range(profile_random.shape[1]/2)])
#            profile_random[ii] = np.concatenate((pro_L, pro_R))


        # introduce the noise from SVD analysis. 
        # remove signal modes, and reconstruct noise profiles.
        U, s, V = svd(profile_raw)
        V0_raw = copy.deepcopy(V[0])
        V1_raw = copy.deepcopy(V[1])
        V[0] = 0
        V[1] = 0
        noise_profiles_raw = reconstruct_profile(U,s,V)
        print 'noise_profiles_raw.shape', noise_profiles_raw.shape
        # transform noise_profiles_raw with a half period of each polarization to avoid the contribution of residuals    
        profile_svd_noise = np.zeros((profile_raw.shape))
        for ii in xrange(profile_svd_noise.shape[0]):
            pro_raw_L = profile_raw[ii,0:profile_svd_noise.shape[1]/2]
            pro_L = pro_raw_L + np.roll(noise_profiles_raw[ii,0:profile_svd_noise.shape[1]/2], ngate/2)
            pro_raw_R = profile_raw[ii,profile_svd_noise.shape[1]/2:]
            pro_R = pro_raw_R + np.roll(noise_profiles_raw[ii,profile_svd_noise.shape[1]/2:], ngate/2)
            profile_svd_noise[ii] = np.concatenate((pro_L, pro_R))

        # process profiles of raw, random noise, noise from svd
        print 'process profiles of raw, random noise, noise from svd'
        process_profiles(profile_raw, 'gp052a_07_raw_')
        process_profiles(profile_random, 'gp052a_07_random_')
        process_profiles(profile_svd_noise, 'gp052a_07_svdnoise_')


def process_profiles(profile, pattern):
        # remove signal modes, and reconstruct noise profiles.
        U, s, V = svd(profile)
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
        print 'profile_norm_var.shape', profile_norm_var.shape
        plot_spec_dedisperse(profile_norm_var)

        print 'finish profile_norm_var'
        # SVD on the normalized variance profile.
        U, s, V = svd(profile_norm_var)
        plot_name_1 = pattern + '_norm_var_lik'
        plot_svd(profile_norm_var, plot_name_1)
        print 'finish SVD'
        check_noise(profile_norm_var)      
        V_recon = V.reshape(V.shape[0], 2, V.shape[1]/2)
        print 'done V_recon'

        '''stack profiles'''
        if False:
            profile = stack(profile, paras.prof_stack)
            tint_stack = paras.tint*paras.prof_stack
        else:
            tint_stack = paras.tint

        '''reshape profiles of L and R into periodic signals (pulse number, L/R, phases)'''
        profile_npy = np.zeros(profile.shape)
        profile_npy[:] = profile[:]
        profile_npy = profile_npy.reshape(profile_npy.shape[0], 2, profile_npy.shape[1]/2)   
        print 'profile_npy.shape', profile_npy.shape
        mpi_phase_fitting(profile_npy, V_recon, pattern, tint_stack)
      

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

def plot_spec_dedisperse(spec):
#    this_file = h5py.File('/mnt/raid-project/gmrt/hhlin/time_streams_1957/gp052a_ar_no0007_512g_0b_56821.2537037+536s_h5','r')
#    spec = this_file['fold_data_int_0.125_band_0']
    spec_L = spec[:, 0:spec.shape[1]/2]
    spec_R = spec[:, spec.shape[1]/2:]
    spec_mean = np.mean((spec_L,spec_R), axis=0)
    
    cmap = cm.winter
    extent = [0, paras.ngate, len(spec)*0.125, 0]

    plt.close('all')
    plt.imshow(spec_L, extent=extent, aspect='auto', cmap = cmap)
    plt.colorbar()
    plt.xlabel('Phase Bins')
    plt.ylabel('Time (Sec)')
    plt.savefig('spec_L.png', bbox_inches='tight', dpi=300)

    plt.close('all')
    plt.imshow(spec_R, extent=extent, aspect='auto', cmap = cmap)
    plt.colorbar()
    plt.xlabel('Phase Bins')
    plt.ylabel('Time (Sec)')
    plt.savefig('spec_R.png', bbox_inches='tight', dpi=300)

    plt.close('all')
    plt.imshow(spec_mean, extent=extent, aspect='auto', cmap = cmap)
    plt.colorbar()
    plt.xlabel('Phase Bins')
    plt.ylabel('Time (Sec)')
    plt.savefig('spec_mean.png', bbox_inches='tight', dpi=300)




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


def plot_phase_lik(times, phases_amps, tint_stack, plot_name):

    '''stacked profiles'''
    npy_lik_file = phases_amps
    xmin = np.amin(times)
    xmax = np.amax(times)
    zeros_line = np.zeros(len(npy_lik_file[:]))
    phase_bins = npy_lik_file[:,0]
    phase_bin_errs = npy_lik_file[:,npy_lik_file.shape[1]/2]

    z_stacked = np.polyfit(times, phase_bins, 2)
    p_stacked = np.poly1d(z_stacked)
    parabola_stacked = p_stacked(times)
    phase_bins_para = phase_bins - parabola_stacked 


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

def plot_rms_binning():

    bin_size = P0 / ngate * 10**6 #microsecond
    phase_file_1 = np.load('gp052a_07_svdnoise_fit_nodes_1_tint_0.125.npy')
    phase_file_2 = np.load('gp052a_07_svdnoise_fit_nodes_2_tint_0.125.npy')
    phase_bins_1 = phase_file_1[:2400, 0] * bin_size # change unit from phase bins to microseconds
    phase_bins_2 = phase_file_2[:2400, 0] * bin_size

    rebin_factor = [1, 2, 4, 8, 16, 32, 60, 120, 240, 480]
    rebin_time = np.array(rebin_factor)*0.125

    tt = np.arange(0.125,60,1)

    phase_bins_1_rebin_rms = np.zeros(len(rebin_factor))
    phase_bins_1_rebin_rms_para = np.zeros(len(rebin_factor))
    phase_bins_2_rebin_rms = np.zeros(len(rebin_factor))
    phase_bins_2_rebin_rms_para = np.zeros(len(rebin_factor))

#    for ii in xrange(4, 8):
    for ii in xrange(len(rebin_factor)):
        # before subtracting parabola
        phase_bins_1_reshape = np.mean(phase_bins_1.reshape(len(phase_bins_1)/rebin_factor[ii], rebin_factor[ii]), axis=1)
        phase_bins_1_rebin_rms[ii] = np.std(phase_bins_1_reshape)
        phase_bins_2_reshape = np.mean(phase_bins_2.reshape(len(phase_bins_2)/rebin_factor[ii], rebin_factor[ii]), axis=1)
        phase_bins_2_rebin_rms[ii] = np.std(phase_bins_2_reshape)

        # after subtracting parabola
        times = np.arange(0, 300, rebin_time[ii])
#        print 'len(times)', len(times)
#        print 'len(phase_bins_1_reshape)', len(phase_bins_1_reshape)
        z_1 = np.polyfit(times, phase_bins_1_reshape, 2)
#        print 'z_1',z_1
        p_1 = np.poly1d(z_1)
        para_1 = p_1(times)
        phase_bins_1_reshape_para = phase_bins_1_reshape - para_1
#        print 'phase_bins_1_reshape_para', phase_bins_1_reshape_para
        phase_bins_1_rebin_rms_para[ii] = np.std(phase_bins_1_reshape_para)

        z_2 = np.polyfit(times, phase_bins_2_reshape, 2)
        p_2 = np.poly1d(z_2)
        para_2 = p_2(times)
        phase_bins_2_reshape_para = phase_bins_2_reshape - para_2
        phase_bins_2_rebin_rms_para[ii] = np.std(phase_bins_2_reshape_para)

    print 'phase_bins_1_rebin_rms', np.amax(phase_bins_1_rebin_rms)
    print 'phase_bins_1_rebin_rms_para', np.amax(phase_bins_1_rebin_rms_para)
    print 'phase_bins_2_rebin_rms', np.amax(phase_bins_2_rebin_rms)
    print 'phase_bins_2_rebin_rms_para', np.amax(phase_bins_2_rebin_rms_para)


    max_value = np.amax(np.concatenate((phase_bins_1_rebin_rms, phase_bins_1_rebin_rms_para, phase_bins_2_rebin_rms, phase_bins_2_rebin_rms_para)))

    markersize = 4.0
    fontsize = 16

    plt.close('all')
    x_axis = rebin_time
    plt.loglog(tt, 1/np.sqrt(tt)*(np.sqrt(0.125)*max_value), 'k', label='1 / sqrt(t), normalized')
    plt.loglog(x_axis, phase_bins_1_rebin_rms, 'ro', markersize=markersize, label='1 mode, before subtracting para')
    plt.loglog(x_axis, phase_bins_1_rebin_rms, 'r')
    plt.loglog(x_axis, phase_bins_2_rebin_rms, 'bs', markersize=markersize, label='2 modes, before subtracting para')
    plt.loglog(x_axis, phase_bins_2_rebin_rms, 'b--')
    plt.loglog(x_axis, phase_bins_1_rebin_rms_para, 'g^', markersize=markersize, label='1 mode, after subtracting para')
    plt.loglog(x_axis, phase_bins_1_rebin_rms_para, 'g:')
    plt.loglog(x_axis, phase_bins_2_rebin_rms_para, 'y*', markersize=markersize, label='2 modes, after subtracting para')
    plt.loglog(x_axis, phase_bins_2_rebin_rms_para, 'y-.')


    plt.xlabel('Int. time for each rebinning profile (sec)', fontsize=fontsize)
    plt.ylabel('RMS for rebinning phase measurements (micro-sec)', fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(loc='lower left', fontsize=fontsize-4)
    plt.xlim([0,65])
    title = 'Max RMS: '+str(np.round(max_value,3))+' (microsec)'
    plt.title(title, fontsize=fontsize)
    plot_name = 'rebinning_rms_svdnoise.png'
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
    profile_stack = 10
    profile = stack(profile, profile_stack)
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


def mpi_phase_fitting(profiles, V, patterns, tint_stack):

    '''save the fitting amp and bin as [bin, amps, bin_err, amp_errs]'''
    npy_lik_file = np.zeros((len(profiles), (1+NMODES)*2))
    npy_lik_name = patterns+'fit_nodes_'+str(NMODES)+'_tint_'+str(tint_stack)+'.npy'

    for ii, profile in list(enumerate(profiles)):
        if ii % size == rank:
            phase_amp_bin_lik = phase_fitting(profile, V, patterns, tint_stack, ii)
            npy_lik_file[ii] = phase_amp_bin_lik

    np.save(npy_lik_name, npy_lik_file)
     
def phase_fitting(profile, V, patterns, tint_stack, ii):
    profile_numbers = []
    profile_numbers_lik = []
    phase_model = []
    phases = []
    phase_errors = []
    phases_lik = []
    phase_errors_lik = []

#    nprof = 10
#    nprof = len(profiles)
#    profiles = profiles[:nprof]

    V_fft = fftpack.fft(V, axis=2)
    V_fft_L = fftpack.fft(V[:,0,:], axis=1)
    V_fft_R = fftpack.fft(V[:,1,:], axis=1)

    if True:
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
                (profile_fft, V_fft),
                full_output=1,
                )
        fit_res = residuals(pars_fit, profile_fft, V_fft)

        chi2_fit = chi2(pars_fit, profile_fft, V_fft)
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

        model_fft_L = model(pars_fit, V_fft_L) 
        model_fft_R = model(pars_fit, V_fft_R)
        print 'model_fft_L.shape', model_fft_L.shape
        print 'profile_fft_L.shape', profile_fft_L.shape
        plot_phase_fft(pick_harmonics(profile_fft_L), model_fft_L, ii, patterns+'fit_L_')
        plot_phase_ifft(pars_fit, profile_L, profile_R, V_fft_L, V_fft_R, ii, patterns+'fit_')
        plot_phase_fft(pick_harmonics(profile_fft_R), model_fft_R, ii, patterns+'fit_R_')
#        plot_phase_ifft(pars_fit, profile_R, V_fft_R, ii, 'phase_fit_R_')

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
                    P_L = shift_trunc_modes(this_phase, V_fft_L)
                    P_R = shift_trunc_modes(this_phase, V_fft_R)
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
                chi2_sample = chi2(
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
            plot_phase_diff_chi2(phase_diff_samples, likelihood, norm, ii, patterns)
            phase_amp_bin_lik = np.concatenate(([pars_init[0] + mean], pars_fit[1:], [std], errs[1:]))
        else:
            phase_amp_bin_lik = np.concatenate((pars_fit[:], errs[:]))

        return phase_amp_bin_lik
        

def stack(profile, profile_stack):
    nprof = len(profile)
    nprof -= nprof % profile_stack
    profile = profile[:nprof].reshape(nprof // profile_stack, profile_stack, profile.shape[-1])
    profile = np.mean(profile, 1)
    return profile


def residuals(parameters, profile_fft, V_fft):
    res_L = pick_harmonics(profile_fft[0]) - model(parameters, V_fft[:,0,:]) 
    res_R = pick_harmonics(profile_fft[1]) - model(parameters, V_fft[:,1,:])
    res_all = np.concatenate((res_L, res_R))
    return res_all


def model(parameters, V_fft):
    phase_bins = parameters[0]
    amplitudes = np.array(parameters[1:])
    shifted_modes = shift_trunc_modes(phase_bins, V_fft)
    return np.sum(amplitudes[:,None] * shifted_modes, 0)


def shift_trunc_modes(phase_bins, V_fft):
    V_fft_shift = apply_phase_shift(V_fft, phase_bins)
    V_harmonics = pick_harmonics(V_fft_shift)
    return V_harmonics[:NMODES]

def chi2(parameters, profile_fft, V_fft, norm=1):
    return np.sum(residuals(parameters, profile_fft, V_fft)**2) * norm


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
    np.save(V_name, V)
 
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
    color = ['r', 'g', 'b', 'y', 'c', '0.0', '0.2', '0.4', '0.6', '0.8']
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
    color = ['r', 'g', 'b', 'y', 'c', '0.0', '0.2', '0.4', '0.6', '0.8']
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

def plot_phase_ifft(pars_fit, data_L, data_R, V_fft_L, V_fft_R, ii, plot_name):

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


