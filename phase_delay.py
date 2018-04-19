import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec


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
    num = 30 # number of data points

    '''Step 1. Get the timestamps of telescops, which are the same as Arecibo's'''
    time_array = []
    time_ar = sn2['TIME'][sn2['ANTENNA NO.']==3]
    non_ar = [1,2,4,5,6,7,8,9,10,11,12,13] #indices of non-Ar telescopes
    for k in xrange(num):#(len(time_ar)):
        print 'k: ', k
        time_k = []
        for i in non_ar:
            for j in xrange(len(sn2['TIME'][sn2['ANTENNA NO.']==i])):
                if time_ar[k] == sn2['TIME'][sn2['ANTENNA NO.']==i][j]:
                    time_k.append([i,j])
        time_array.append(time_k)
#    np.save('time_array.npy', time_array)

    '''Step 2. Get the mean value of each set'''
    r_real = r_imag = l_real = l_imag = np.zeros((num,4))
    for k in xrange(num):
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

    r_phase = np.angle(r_real + r_imag*1j) / (np.pi)
    l_phase = np.angle(l_real + l_imag*1j) / (np.pi)
#    print 'r_phase', r_phase

    markersize = 4.0
    fontsize = 16

    for ii in [0,1]:
        label = [['real, 1st band', 'imag, 1st band', 'norm. phase, 1st band'],
                 ['real, 2nd band', 'imag, 2nd band', 'norm. phase, 2nd band']]
        plt.close('all')
        x_axis = ((sn2['TIME'][ants==3] - sn2['TIME'][ants==3][0])*86400)[:num]
        plt.plot(x_axis, r_real[:,ii], 'ro', markersize=markersize, label=label[ii][0])
        plt.plot(x_axis, r_real[:,ii], 'r')
        plt.plot(x_axis, r_imag[:,ii], 'bs', markersize=markersize, label=label[ii][1])
        plt.plot(x_axis, r_imag[:,ii], 'b--')
        plt.plot(x_axis, r_phase[:,ii], 'g^', markersize=markersize, label=label[ii][2])
        plt.plot(x_axis, r_phase[:,ii], 'g--')
        title = 'R pol. Epoch A'
        plt.title(title, fontsize=fontsize)
        plt.xlabel('Time (sec)', fontsize=fontsize)
        plt.ylabel('Value', fontsize=fontsize)
        plt.ylim([0,3])
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.legend(loc='upper right', fontsize=fontsize-5)
        plot_name = 'r_pol_'+str(ii)+'.png'
        plt.savefig(plot_name, bbox_inches='tight', dpi=300)



if __name__ == '__main__':
    main()
