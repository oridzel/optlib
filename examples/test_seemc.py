import optlib.seemc as mc

energy = [1000]
s = mc.SEEMC(energy, 'Au', 0, 10000, False, False)
s.run_simulation(True)
# s.plot_trajectories(0)
