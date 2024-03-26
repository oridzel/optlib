import multiprocessing
import optlib.seemc as mc
import matplotlib.pyplot as plt

num_cores = multiprocessing.cpu_count()
energies = [10, 20, 50, 100, 200, 500, 1000]

def run_seemc(i):
    energy = energies[i]
    print(energy)
    s = mc.SEEMC([energy], 'Au', 0, 10000, False, False)
    s.run_simulation(False)
    s.calculate_yield()
    return s.sey[0], s.bse[0], s.sey[0]+s.bse[0]


if __name__ == '__main__':

    with multiprocessing.Pool(num_cores) as pool:
        res = pool.map(run_seemc, range(len(energies)))
        sey, bsey, tey = zip(*res)

    plt.figure()
    plt.plot(energies, tey, label='TEY')
    plt.plot(energies, sey, label='SEY')
    plt.plot(energies, bsey, label='BSEY')
    plt.xlabel('Energy')
    plt.ylabel('Yield')
    plt.show()
