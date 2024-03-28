import optlib.seemc as mc

if __name__ == '__main__':
    energy = [1000]
    s = mc.SEEMC(energy, 'Au', 0, 500, False, False)
    # s.run_parallel_simulation()
    s.run_simulation(False)
    s.calculate_yield()
    # s.plot_yield()
    # s.plot_trajectories(0)
