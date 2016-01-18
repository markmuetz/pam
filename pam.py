from __future__ import division
import datetime as dt
import os.path

import netCDF4 as nc
import pylab as plt
import numpy as np
from scipy.interpolate import interp1d

import tephi
from tephi._constants import CONST_K, CONST_KELVIN, CONST_L, CONST_MA, CONST_RV

def google_maps_link(lat, lon, zoom='8'):
    return 'http://maps.google.com/maps?q={lat},{lon}&z={zoom}'.format(lat=lat,
                                                                       lon=lon,
                                                                       zoom=zoom)

def pressure_temp_to_mixing_ratio(pressure, T):
    es = 6.11 * np.exp((2.5e6 / 461) * (1 / 273.15 - 1 / (T + 273.15)))
    #rvs = 18.015 / 29 * es / p0
    rvs = 0.625 * es / pressure
    return rvs * 1000


def analyse_sounding(d, index):
    print('index: {}'.format(index))
    print('lat: {}'.format(d.variables['staLat'][index]))
    print('lon: {}'.format(d.variables['staLon'][index]))
    print('time: {}'.format(dt.datetime.fromtimestamp(d.variables['synTime'][index])))

    res = {}
    res['lat'] = d.variables['staLat'][index]
    res['lon'] = d.variables['staLon'][index]
    res['url'] = google_maps_link(res['lat'], res['lon'])
    res['time'] = dt.datetime.fromtimestamp(d.variables['synTime'][index])
    numSigT = d.variables['numSigT'][index]

    T = d.variables['tpSigT'][index, :numSigT]
    p = d.variables['prSigT'][index, :numSigT]
    dew_dep = d.variables['tdSigT'][index, :numSigT]
    dewT = T - dew_dep

    full_p = np.linspace(p[0], 100, int(p[0]) - 100 + 1)
    interp_T = interp1d(p, T)
    interp_dewT = interp1d(p, dewT)
    full_T = interp_T(full_p) - 273.15
    full_dewT = interp_dewT(full_p) - 273.15

    K = 287/1004
    p0 = full_p[0]
    Tnext = full_T[0]
    theta = (Tnext + 273.15) * (p0 / 1000)**-K
    ascentT = [Tnext]

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
        rvs = pressure_temp_to_mixing_ratio(full_p[i], Tnext)
        if rvs > rvs0:
            Tnext = (theta * (full_p[i] / 1000) ** K) - 273.15
        else:
            if not LCL:
                LCL = full_p[i]
                # print('p={},rvs={}'.format(full_p[i], rvs))
                print('LCL={}'.format(LCL))
            dp = full_p[i] - full_p[i - 1]
            minT = -1000
            _, dT = tephi.isopleths._wet_adiabat_gradient(minT, full_p[i - 1], Tnext, dp)
            Tnext += dT
            theta_parcel = (Tnext + 273.15) * (full_p[i] / 1000)**-K
            theta_env = (full_T[i] + 273.15) * (full_p[i] / 1000)**-K

            curr_energy += 287. * (full_T[i] - Tnext) * (full_p[i] - full_p[i - 1]) / full_p[i]
            # print(full_p[i], curr_energy)
            # raw_input()

            if theta_parcel < theta_env:
                if buoyancy != -1:
                    # print('New -ve level reached: p={}'.format(full_p[i]))
                    buoyancy = -1
                    energies.append((curr_energy, 'CAPE'))
                    if curr_energy > max_CAPE:
                        max_CAPE = curr_energy
                    curr_energy = 0
            elif theta_parcel > theta_env:
                if buoyancy != 1:
                    # print('New +ve level reached: p={}'.format(full_p[i]))
                    buoyancy = 1
                    energies.append((curr_energy, 'CIN'))
                    if abs(curr_energy) > abs(max_CIN):
                        max_CIN = curr_energy
                    curr_energy = 0

        ascentT.append(Tnext)

    res['index'] = index
    res['p'] = p
    res['T'] = T
    res['dewT'] = dewT

    res['full_p'] = full_p
    res['ascentT'] = ascentT
    res['LCL'] = LCL
    res['energies'] = energies
    res['max_CAPE'] = max_CAPE
    res['max_CIN'] = max_CIN
    return res


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
        try:
            res = analyse_sounding(d, index)
            print(res['energies'])
            if plot:
                plot_tpg(res)

                plt.pause(0.1)
                ri = raw_input()
                if ri == 'q':
                    break
            all_res[index] = res
        except:
            print('Problem with index {}'.format(index))
    return d, all_res


if __name__ == '__main__':
    plt.ion()
    #interesting_indices = [17, 165, 169, 224, 238, 261, 263, 301, 309, 311, 325, 327, 330, 331, 333, 334, 338, 383, 389, 398, 507, 515, 526, 530, 608, 626, 632, 654, 682, 684, 685, 686, 693, 694, 698, 700, 702, 705, 706, 711, 741, 771, 933, 943, 945, 946, 947, 978, 979, 1002]
    # analyse_esrl_noaa_data('data/raob_soundings27403.cdf', interesting_indices)
    #d, all_res = analyse_esrl_noaa_data('data/raob_soundings27403.cdf')
    d, all_res = analyse_esrl_noaa_data('data/raob_soundings25458.cdf')
