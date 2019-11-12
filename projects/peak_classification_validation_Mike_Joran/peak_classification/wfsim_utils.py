import numpy as np

instruction_dtype = [('event_number', np.int), ('type', np.int), ('t', np.int),
                     ('x', np.float32), ('y', np.float32), ('z', np.float32),
                     ('amp', np.int), ('recoil', '<U2')]


def rand_instructions(c):
    n = c['nevents'] = c['event_rate'] * c['chunk_size'] * c['nchunk']
    c['total_time'] = c['chunk_size'] * c['nchunk']

    instructions = np.zeros(2 * n, dtype=instruction_dtype)
    uniform_times = c['total_time'] * (np.arange(n) + 0.5) / n

    instructions['t'] = np.repeat(uniform_times, 2) * int(1e9)
    instructions['event_number'] = np.digitize(instructions['t'],
                                               1e9 * np.arange(c['nchunk']) *
                                               c['chunk_size']) - 1
    instructions['type'] = np.tile([1, 2], n)
    instructions['recoil'] = ['er' for i in range(n * 2)]

    r = np.sqrt(np.random.uniform(0, 2500, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    instructions['x'] = np.repeat(r * np.cos(t), 2)
    instructions['y'] = np.repeat(r * np.sin(t), 2)
    instructions['z'] = np.repeat(np.random.uniform(-100, 0, n), 2)

    nphotons = np.random.uniform(200, 2050, n)
    nelectrons = 10 ** (np.random.uniform(1, 4, n))
    instructions['amp'] = np.vstack([nphotons, nelectrons]).T.flatten().astype(
        int)

    return instructions
