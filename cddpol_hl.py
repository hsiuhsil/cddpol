from __future__ import print_function, division
import time, sys, os, argparse
import os.path
import numpy as np
import astropy.units as u
import math
from baseband import mark4
from pulsar.predictor import Polyco
from pyfftw.interfaces.numpy_fft import rfft, irfft
from mpi4py import MPI

import h5py
import matplotlib.pyplot as plt
import paras

_fftargs = {'threads': int(os.environ.get('OMP_NUM_THREADS', 4)),
            'planner_effort': 'FFTW_ESTIMATE',
            'overwrite_input': True}

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
in_folder = '/mnt/scratch-lustre/hhlin/Data'
#out_folder = '/mnt/raid-cita/hhlin/psr_1957/Ar_B1957_IndividualPulses'
out_folder = '/mnt/raid-project/gmrt/hhlin/time_streams_1957'
main_t0 = time.time()

def CL_Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--band", type=int, choices=[0, 1, 2], help="Select which frequency band to process. Integer from 0 to 2.")
    return parser.parse_args()

def TFmt(t):
    dt = time.time() - t
    return "{0:02d}:{1:02d}:{2:02d}".format(int(dt) // 3600, int(dt % 3600) // 60, int(dt % 60))

def MakeFileList(rank, size):
    import itertools
    epochs = ['d']
#    nums = [3, 4, 6, 7, 9, 10, 12, 13]
    nums = [15, 16, 18, 19, 21, 22]
#    nums = [7, 9, 10, 12, 13, 15, 16, 18, 19, 21]
#    nums = [3, 4, 6, 7, 9, 10]
#    epochs = ['a', 'b', 'c', 'd']
#    nums = [3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22]
    evn_gen = itertools.product(epochs, nums)
    evn = ['gp052{0}_ar_no00{1:02d}'.format(epoch, file_num) for epoch, file_num in evn_gen]
    return evn[rank::size]

def CoherentDD(fedge, fref, N, dt):
    try:
        dd_coh = np.load('CoherentDD/{0}+{1}_R{2}_N{3}.npy'.format(fedge.value, (1/(2*dt)).to(u.MHz).value, fref.value, N))
    except IOError:
        DM = 29.11680 * 1.00007 * u.pc / u.cm**3
        D = 4149. * u.s * u.MHz**2 * u.cm**3 / u.pc
        f = fedge + np.fft.rfftfreq(N, dt)
        dang = D * DM * u.cycle * f * (1./fref - 1./f)**2
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            dd_coh = np.exp(dang * 1j).conj().astype(np.complex64).value
    return dd_coh

def FindOffset(ar_data, SR):
    from astropy.time import Time
    fh = mark4.open(ar_data, 'rs', ntrack=64, decade=2010, sample_rate=SR)
    t0 = fh.time0
    offset_time = Time(math.ceil(t0.unix), format='unix', precision=9)
    offset = fh.seek(offset_time)
    t00 = np.float128(offset_time.mjd) # previous: isot
    return offset, t00

def FindPhase(t0, z_size, dt, ngate):
    polyco = Polyco('/mnt/raid-cita/mahajan/Pulsars/B1957Timing/polycob1957+20_gpfit.dat')
    p = polyco.phasepol(t0, rphase='fraction', t0=t0, time_unit=u.second, convert=True)
    ph = p(np.arange(z_size) * dt.to(u.s).value)
    ph -= np.floor(ph[0])
    print('ph1', ph)
    ncycle = int(np.floor(ph[-1])) + 1
    ph = np.remainder(ph*ngate, ngate*ncycle).astype(np.float64)
    print('ph2', ph)
    return ph, ncycle

def BasebandProcess(ar_data, band, SR, dt, N, DN, offset, i, dd_coh, ngate):
    fh = mark4.open(ar_data, 'rs', ntrack=64, decade=2010, sample_rate=SR, thread_ids=[2*band, 2*band + 1])
    fh.seek(offset + i*(N - DN))
    t0 = fh.tell(unit='time')
    ph, ncycle = FindPhase(t0, N - DN, dt, ngate)
    ph %= ngate
    print('ph3', ph)
    z = fh.read(N)
    z = rfft(z, axis=0, **_fftargs)
    z *= dd_coh[..., np.newaxis]
    z = irfft(z, axis=0, **_fftargs)[:-DN]
    z = z.astype(np.float32)
    z = z*z
    return ph, z

args = CL_Parser()
band = args.band
evn_files = MakeFileList(rank, size)
mpstr = '{0:02d}/{1:02d}'.format(rank+1, size)

ngate = paras.ngate
T = paras.T
fold_time_length = paras.tint #unit: sec

SR = 32. * u.MHz
dt = (1/SR).to(u.s)
N, DN = 2**28, 759*(2**14)
block_length = int((N - DN)/SR.decompose().value)
fedge = np.array([311.25, 327.25, 343.25, 359.25])*u.MHz
fref = 359.540 * u.MHz
print("[{0}: {1}] Using band {2}+{3} MHz (Ref. Freq. is {4})".format(mpstr, TFmt(main_t0), (fedge[band].value), (SR.value/2), (fref)))
dd_coh = CoherentDD(fedge[band], fref, N, dt)

for fi, evn_file in enumerate(evn_files):
    file_t0 = time.time()
    flstr = "{0:02d}/{1:02d}".format(fi+1, len(evn_files))
    ar_data = "{0}/{1}".format(in_folder, evn_file)
    offset, t00 = FindOffset(ar_data, SR)
    output_file = '{0}/{1}_{2}g_{3}b_{4}+{5}s_'.format(out_folder, evn_file, ngate, band, t00, T*block_length)
    print('output_file is', output_file)
#    zz = np.zeros(( int(T * 8 / fold_time_length), ngate * 2))
    zz = np.zeros(ngate * 2)

    '''create a dataset for each folding file'''
    files = {}
    keys = ['t00', 'fold_data_int_'+str(fold_time_length)+'_band_'+str(band)]
    if os.path.isfile(output_file) == True:
        files[output_file] = h5py.File(output_file + 'h5',"r+")
    else:
        this_file = h5py.File(output_file + 'h5',"w")
        for dataset_name in keys:
            if dataset_name == 't00':
                first_data = np.zeros(1, dtype=np.float64)
                this_file.create_dataset(dataset_name, (0,) + first_data.shape, maxshape = (None,) +first_data.shape, dtype=first_data.dtype, chunks=True)
#            elif dataset_name == 'ph':
#                first_data = np.zeros(int(8/dt.value))
#                this_file.create_dataset(dataset_name, (0,) + first_data.shape, maxshape = (None,) +first_data.shape, dtype=first_data.dtype, chunks=True)
            elif dataset_name == 'fold_data_int_'+str(fold_time_length)+'_band_'+str(band):
                first_data = zz
                this_file.create_dataset(dataset_name, (0,) + first_data.shape, maxshape = (None,) +first_data.shape, dtype=first_data.dtype, chunks=True)
            files[output_file] = this_file


    print("[{0}: {1}] ({2}: {3}) Begin processing, Output at {4}".format(mpstr, TFmt(main_t0), flstr, evn_file, output_file))

    for i in xrange(T):
        print('i: ',i)
        block_t0 = time.time()
        chnkstr = '{0:02d}/{1:02d}'.format(i+1, T)
        ph, z = BasebandProcess(ar_data, band, SR, dt, N, DN, offset, i, dd_coh, ngate)

        for dataset_name in keys[:-1]:
            current_len = files[output_file][dataset_name].shape[0]
            files[output_file][dataset_name].resize(current_len + 1, 0)
            if dataset_name == 't00':
                # the unit of the t00_offset is  MJD
                t00_offset = np.float64(t00 + i*8./86400)
                files[output_file]['t00'][current_len,...] = t00_offset
#            elif dataset_name == 'ph':
#                files[output_file]['ph'][current_len,...] = ph
            elif dataset_name == 'fold_data':
                    pass

        print('time to store t00 and ph', time.time() - block_t0)

        for jj in xrange(int(8./fold_time_length)):
            init = int(fold_time_length * len(z) / 8.) * jj
            final = int(fold_time_length * len(z) / 8.) * (jj+1)
            phase_bins = np.around(ph[init:final]).astype(np.int)
            phase_bins %= ngate
#            phase_bins = ph[init:final].astype(np.int)
            print('max of phase_bins', np.amax(phase_bins))
            print('min of phase_bins', np.amin(phase_bins))
            print('phase_bins', phase_bins)
            bin_counts = np.bincount(phase_bins)
            data_0 = z[init:final,0]
            data_1 = z[init:final,1]
            # Nikhil's folding:
            fold_data_0 = np.bincount(phase_bins, data_0)
            fold_data_1 = np.bincount(phase_bins, data_1)
            fold_data_0 /= bin_counts
            fold_data_1 /= bin_counts
            fold_data = np.concatenate((fold_data_0, fold_data_1), axis=0)           
            print('fold_data.shape', fold_data.shape)
            # store the fold_data into the h5py file.
            for dataset_name in keys:
                if dataset_name == 'fold_data_int_'+str(fold_time_length)+'_band_'+str(band):
                    current_len = files[output_file][dataset_name].shape[0]
#                    print('current_len', current_len)
                    files[output_file][dataset_name].resize(current_len + 1, 0)
                    files[output_file]['fold_data_int_'+str(fold_time_length)+'_band_'+str(band)][current_len,...] = fold_data
                else:
                    pass        
            print('time for a block', time.time()-block_t0) 

        print("[{0}: {1}] ({2}: {3}) Block {4} processed in {5}.".format(mpstr, TFmt(main_t0), flstr, evn_file, chnkstr, TFmt(block_t0)))
    print("[{0}: {1}] ({2}: {3}) All blocks processed in {4}.".format(mpstr, TFmt(main_t0), flstr, evn_file, TFmt(file_t0)))
print("[{0}: {1}] All done!".format(mpstr, TFmt(main_t0)))
