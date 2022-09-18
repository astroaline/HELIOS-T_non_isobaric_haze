from input import *
import struct
import numpy as np
import pdb



def load_sigma(molecule1, molecule2, x_full):
    file_name = molecule1+'-'+molecule2+'_2011.cia'

    with open(cia_path+file_name) as file:
        data = file.readlines()

    header = str(data[0])
    header_list = header.split(' ')
    header_list = [x for x in header_list if x]

    lines_per_temp = float(header_list[3])

    num_temps = len(temperature_array_cia)


    cia_data = []
    if molecule2 == 'H2':
        wavenumber_array = np.r_[x_full[0]:10001:res]
    else:
        wavenumber_array = np.r_[x_full[0]:17001:res]

    for i in range(int(num_temps)):
        j = int(i*(lines_per_temp+1))
        header = data[j]
        header_list = header.split(' ')
        header_list = [x for x in header_list if x]


        start_v = float(header_list[1])

        min_v = int(wavenumber_array[0] - start_v)
        max_v = int(wavenumber_array[-1] - start_v)

        cia_line = []
        cia_wavenumber = []

        for k in range(min_v, max_v+1):           # For CIA, temperature goes down rows, wavenumber goes across columns
            data_line = data[j+k].split(' ')
            data_line = [x for x in data_line if x]
            cia_line.append(float(data_line[1][:-1]))
            cia_wavenumber.append(float(data_line[0]))

        cia_line = np.array(cia_line)
        cia_wavenumber = np.array(cia_wavenumber)
        cia_line_hr = np.interp(wavenumber_array, cia_wavenumber, cia_line)
        # cia_line = cia_line[::res_cia]
        cia_data.append(cia_line_hr)

    cia_data = np.array(cia_data)

    wavenumber_array = np.array(wavenumber_array)


    # pad_start = int((x_full[0] - wavenumber_array[0]) / res)
    pad_end = int((x_full[-1] - wavenumber_array[-1]) / res)


    # if pad_start > 0:
    #     cia_data = cia_data[:, pad_start:]
    # else:
    #     cia_data = np.pad(cia_data, ((0, 0), (-pad_start, 0)), 'constant')

    if pad_end < 0:
        cia_data = cia_data[:, :pad_end]
    else:
        cia_data = np.pad(cia_data, ((0, 0), (0, pad_end)), 'constant')

    return cia_data
