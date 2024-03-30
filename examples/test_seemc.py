import optlib.seemc as mc

if __name__ == '__main__':
    energy = [100]
    s = mc.SEEMC(energy, 'Au', 0, 1000, False, False)
    s.run_parallel_simulation()
    # s.run_simulation(False)
    # s.calculate_yield()
    # s.plot_yield()

    # s.plot_trajectories(0)

    s.calculate_coincidence_histogram(False)
    s.plot_coincidence_histogram(100)
