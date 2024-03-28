import optlib.seemc as mc

if __name__ == '__main__':
    energy = [10, 20, 50, 100, 500, 1000, 2000]
    s = mc.SEEMC(energy, 'Au', 0, 1000, False, False)
    s.run_parallel_simulation()
    # s.run_simulation(False)
    s.calculate_yield()
    # s.plot_yield()
    # s.plot_trajectories(0)