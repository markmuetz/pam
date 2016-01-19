from __future__ import division
import datetime as dt
import os.path
import math
import cPickle

import netCDF4 as nc
import pylab as plt
import numpy as np
from scipy.interpolate import interp1d

import tephi
from tephi._constants import CONST_K, CONST_KELVIN, CONST_L, CONST_MA, CONST_RV

# Appeared interesting at first pass, worth checking out further.
INTERESTING_INDICES = [17, 165, 169, 224, 238, 261, 263, 301, 309, 311, 325, 327, 330, 331, 333, 334, 338, 383, 389, 398, 507, 515, 526, 530, 608, 626, 632, 654, 682, 684, 685, 686, 693, 694, 698, 700, 702, 705, 706, 711, 741, 771, 933, 943, 945, 946, 947, 978, 979, 1002]

def google_maps_link(lat, lon, zoom='8'):
    return 'http://maps.google.com/maps?q={lat},{lon}&z={zoom}'.format(lat=lat,
                                                                       lon=lon,
                                                                       zoom=zoom)


def pressure_temp_to_mixing_ratio(pressure, T):
    es = 6.11 * np.exp((2.5e6 / 461) * (1 / 273.15 - 1 / (T + 273.15)))
    #rvs = 18.015 / 29 * es / p0
    rvs = 0.625 * es / pressure
    return rvs * 1000


def interp_p(T, p, full_p):
    full_T = np.zeros_like(full_p)
    full_T[0] = T[0]
    curr_sig_level = 1
    finished = False

    for i in range(1, len(full_p)):
        pp = full_p[i]
        while pp <= p[curr_sig_level]:
            curr_sig_level += 1
            if curr_sig_level == len(p):
                finished = True
                break
        if finished:
            break

        T0 = T[curr_sig_level - 1]
        T1 = T[curr_sig_level]
        if T1 == T0:
            full_T[i] = T0
        else:
            theta0 = T0 * (p[curr_sig_level - 1] / 1000)**-CONST_K
            theta1 = T1 * (p[curr_sig_level] / 1000)**-CONST_K
            a = (theta1 - theta0) / (T1 - T0)
            b = -a * T0 + theta0
            full_T[i] = b / ((pp/1000)**-CONST_K - a)

    return full_T


def calc_energies(full_p, full_T, full_dewT):
    p0 = full_p[0]
    Tparcel = full_T[0]
    theta = (Tparcel + 273.15) * (p0 / 1000)**-CONST_K
    ascentT = [Tparcel]

    dewT0 = full_dewT[0]
    rvs0 = pressure_temp_to_mixing_ratio(p0, dewT0)

    # print('p0={},rvs0={}'.format(p0, rvs0))
    LCL = None
    buoyancy = -1
    energies = []
    curr_energy = 0

    max_CIN = 0
    max_CAPE = 0
    for i in range(1, len(full_p) - 1):
        Tnext = (theta * (full_p[i] / 1000) ** CONST_K) - 273.15
        rvs = pressure_temp_to_mixing_ratio(full_p[i], Tnext)
        if rvs > rvs0:
            Tparcel = Tnext
        else:
            if not LCL:
                LCL = full_p[i]
                # print('p={},rvs={}'.format(full_p[i], rvs))
                #print('LCL={}'.format(LCL))
            dp = full_p[i] - full_p[i - 1]
            minT = -1000
            _, dT = tephi.isopleths._wet_adiabat_gradient(minT, 
                                                          full_p[i - 1], 
                                                          Tparcel, dp)
            Tparcel += dT

        Tdiff = Tparcel - full_T[i]
        delta_p = full_p[i - 1] - full_p[i]
        curr_energy += 287. * Tdiff * delta_p / ((full_p[i] + full_p[i - 1])/2)

        if full_T[i] > Tparcel:
            if buoyancy != -1:
                #print('New -ve level reached: p={}'.format(full_p[i]))
                #print('Tparcel={}'.format(Tparcel))
                #print('Tenv={}'.format(full_T[i]))
                buoyancy = -1
                energies.append((full_p[i], curr_energy, 'CAPE'))
                if curr_energy > max_CAPE:
                    max_CAPE = curr_energy
                curr_energy = 0
        else:
            if buoyancy != 1:
                #print('New +ve level reached: p={}'.format(full_p[i]))
                #print('Tparcel={}'.format(Tparcel))
                #print('Tenv={}'.format(full_T[i]))
                buoyancy = 1
                energies.append((full_p[i], curr_energy, 'CIN'))
                if abs(curr_energy) > abs(max_CIN):
                    max_CIN = curr_energy
                curr_energy = 0

        ascentT.append(Tparcel)
    return LCL, energies, max_CAPE, max_CIN, ascentT


def analyse_sounding(d, index):
    res = {}
    res['lat'] = d.variables['staLat'][index]
    res['lon'] = d.variables['staLon'][index]
    res['url'] = google_maps_link(res['lat'], res['lon'])
    try:
        res['time'] = dt.datetime.fromtimestamp(d.variables['synTime'][index])
    except ValueError:
        print('Time value wrong for {}'.format(index))
        return None

    numSigT = d.variables['numSigT'][index]
    if numSigT <= 2:
        print('Not enough significant levels')
        return None

    T = d.variables['tpSigT'][index, :numSigT]
    p = d.variables['prSigT'][index, :numSigT]
    dew_dep = d.variables['tdSigT'][index, :numSigT]
    dewT = T - dew_dep

    if p[0] < 200:
        print('Sounding starts too high up')
        return None
    full_p = np.linspace(p[0], 100, int(p[0]) - 100 + 1)

    if False:
        # Bad interpolation!
        # Can't interp straight onto T from p.
        interp_T = interp1d(p, T, kind='linear')
        interp_dewT = interp1d(p, dewT, kind='linear')
        full_T = interp_T(full_p) - 273.15
        full_dewT = interp_dewT(full_p) - 273.15
    else:
        full_T = interp_p(T, p, full_p) - 273.15
        full_dewT = interp_p(dewT, p, full_p) - 273.15

    LCL, energies, max_CAPE, max_CIN, ascentT = calc_energies(full_p, 
                                                              full_T, 
                                                              full_dewT)
    res['index'] = index
    res['p'] = p
    res['T'] = T
    res['dewT'] = dewT

    res['full_p'] = full_p
    res['full_T'] = full_T
    res['full_dewT'] = full_dewT
    res['ascentT'] = ascentT
    res['LCL'] = LCL
    res['energies'] = energies
    res['max_CAPE'] = max_CAPE
    res['max_CIN'] = max_CIN
    return res


def plot_results(results):
    for res in results:
        plot_tpg(res)
        print(res['max_CAPE'])
        print(res['energies'])
        print(res['url'])
        plt.pause(0.01)
        ri = raw_input()
        if ri == 'q':
            break

def plot_tpg(res):
    plt.clf()
    #tpg = tephi.Tephigram()
    tpg = tephi.Tephigram(anchor=[(1000, -20), (20, -20)])

    #tpg = tephi.Tephigram(anchor=[(1000, 0), (300, 0)])
    # plt.figure()

    #fig = plt.gcf()
    #fig.set_size_inches(7.5, 10.5)

    tpg.plot(zip(res['p'], res['T'] - 273.15), color='k', linestyle='-')
    tpg.plot(zip(res['p'], res['dewT'] - 273.15), color='k', linestyle='--')

    tpg.plot(zip(res['full_p'], res['full_T']), color='r', linestyle='-')
    tpg.plot(zip(res['full_p'], res['full_dewT']), color='r', linestyle='--')

    tpg.plot(zip(res['full_p'][1:], res['ascentT']), color='b')


def analyse_esrl_noaa_data(filename, indices=None, plot=False):
    d = nc.Dataset(filename)

    num_stations = d.variables['staLat'].shape[0]
    colours = ['r', 'b', 'g', 'k']
    if not indices:
        indices = range(num_stations)

    all_res = {}
    print(len(indices))
    for index in indices:
        print(index)
        res = analyse_sounding(d, index)
        if res:
            #print(res['energies'])
            all_res[index] = res
    return d, all_res


def save_results(results, filename):
    cPickle.dump(results, open(filename, 'w'))

def load_results(filename):
    return cPickle.load(open(filename, 'r'))

if __name__ == '__main__':
    plt.ion()
    #d, all_res = analyse_esrl_noaa_data('data/raob_soundings25458.cdf')
    #save_results(all_res, 'data/results/raob_soundings25458-results.pkl')
