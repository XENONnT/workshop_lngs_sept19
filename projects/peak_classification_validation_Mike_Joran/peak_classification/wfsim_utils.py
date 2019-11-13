import numpy as np
import os
import pandas as pd
import time
from threading import Thread

instruction_dtype = [('event_number', np.int), ('type', np.int), ('t', np.int),
                     ('x', np.float32), ('y', np.float32), ('z', np.float32),
                     ('amp', np.int), ('recoil', '<U2')]


def rand_instructions(input_inst):
    n = input_inst['nevents'] = input_inst['event_rate'] * input_inst[
        'chunk_size'] * input_inst['nchunk']
    input_inst['total_time'] = input_inst['chunk_size'] * input_inst['nchunk']

    inst = np.zeros(2 * n, dtype=instruction_dtype)
    uniform_times = input_inst['total_time'] * (np.arange(n) + 0.5) / n

    inst['t'] = np.repeat(uniform_times, 2) * int(1e9)
    inst['event_number'] = np.digitize(inst['t'],
                                       1e9 * np.arange(input_inst['nchunk']) *
                                       input_inst['chunk_size']) - 1
    inst['type'] = np.tile([1, 2], n)
    inst['recoil'] = ['er' for i in range(n * 2)]

    r = np.sqrt(np.random.uniform(0, 48 ** 2, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    inst['x'] = np.repeat(r * np.cos(t), 2)
    inst['y'] = np.repeat(r * np.sin(t), 2)
    inst['z'] = np.repeat(np.random.uniform(-100, 0, n), 2)

    nphotons = np.random.uniform(200, 2050, n)
    nelectrons = 10 ** (np.random.uniform(1, 4, n))
    inst['amp'] = np.vstack([nphotons, nelectrons]).T.flatten().astype(int)

    return inst


def inst_to_csv(instructions, csv_file):
    pd.DataFrame(rand_instructions(instructions)).to_csv(csv_file, index=False)


def get_timing_grid(input_inst):
    n = input_inst['nevents'] = input_inst['event_rate'] * input_inst[
        'chunk_size'] * input_inst['nchunk']
    input_inst['total_time'] = input_inst['chunk_size'] * input_inst['nchunk']
    timing_grid = np.linspace(0, input_inst['total_time'], n + 1) * 1e9
    return timing_grid


def check_for_strax_data():
    strax_folder = "strax_data"
    if os.path.exists(strax_folder):
        if input(f"Data found in '{strax_folder}', press [y] to remove and "
                 f"create new data\n").lower() == 'y':
            return True
        # return False
    return False


def timed_check_for_strax_data():
    time_out = 5 #s
    answer = False
    print(f"Please answer within {time_out} seconds.")
    def check():
        time.sleep(time_out)
        if answer is True:
            return
        else:
            print("Too Slow")
            return False

    Thread(target=check).start()
    answer = check_for_strax_data()
    return answer

