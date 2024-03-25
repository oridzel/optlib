import optlib.seemc as mc

energy = [20, 50, 100, 200, 500, 100]
s = mc.SEEMC(energy, 'Au', 0, 10000, False, False)
s.run_simulation(True)
# s.plot_trajectories(0)
