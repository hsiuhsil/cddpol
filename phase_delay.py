import numpy as np
from astropy.io import fits

def main():
    sn2 = fits.open('/mnt/raid-cita/hhlin/psr_1957/cddpol/slf_cal/A-slfCal.fits')['AIPS SN'].data
    ants = sn2['ANTENNA NO.']
#    ant_mask = ants == 3
#    r_real = (sn2['REAL1'])[ant_mask]
#    r_imag = (sn2['IMAG1'])[ant_mask]
#    r_phase = np.angle(r_imag)
#    l_real = (sn2['REAL2'])[ant_mask]
#    l_imag = (sn2['IMAG2'])[ant_mask]
#    l_phase = np.angle(l_imag)


    '''Step 1. Get the timestamps of telescops, which are the same as Arecibo's'''
    time_array = []
    time_ar = sn2['TIME'][sn2['ANTENNA NO.']==3]
    non_ar = [1,2,4,5,6,7,8,9,10,11,12,13] #indices of non-Ar telescopes
    for k in xrange(10):#(len(time_ar)):
        print 'k: ', k
        time_k = []
        for i in non_ar:
            for j in xrange(len(sn2['TIME'][sn2['ANTENNA NO.']==i])):
                if time_ar[k] == sn2['TIME'][sn2['ANTENNA NO.']==i][j]:
                    time_k.append([i,j])
        time_array.append(time_k)
#    np.save('time_array.npy', time_array)

    '''Step 2. Get the mean value of each set'''
    r_real = r_imag = l_real = l_imag = np.zeros((10,4))
    for k in xrange(10):
        r_real_tmp = r_imag_tmp = l_real_tmp = l_imag_tmp = []
        for i in xrange(len(time_array[k])):
            ant, ind = time_array[k][i][0], time_array[k][i][1]
            r_real_tmp.append(sn2['REAL1'][ants==ant][ind])
            r_imag_tmp.append(sn2['IMAG1'][ants==ant][ind])
            l_real_tmp.append(sn2['REAL2'][ants==ant][ind])
            l_imag_tmp.append(sn2['IMAG2'][ants==ant][ind])
        r_real[k] = np.nanmean(r_real_tmp, axis =0)
        r_imag[k] = np.nanmean(r_imag_tmp, axis =0)
        l_real[k] = np.nanmean(l_real_tmp, axis =0)
        l_imag[k] = np.nanmean(l_imag_tmp, axis =0)



if __name__ == '__main__':
    main()
