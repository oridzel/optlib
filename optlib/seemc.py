import numpy as np
from optlib.constants import *
import pickle
from scipy import integrate
from scipy.interpolate import RectBivariateSpline
import random
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
from tqdm import tqdm
import multiprocessing
from types import SimpleNamespace
import gc


class Sample:
    material_data = {}
    T = 300  # K
    k_B = 8.617e-5  # eV/K

    def __init__(self, name):
        with open('MaterialDatabase.pkl', 'rb') as fp:
            data = pickle.load(fp)
        if name in [sub['name'] for sub in data]:
            self.name = name
            self.material_data = data[next((i for i, item in enumerate(data) if item["name"] == name), None)]
        else:
            raise ValueError('Allowed sample names are ' + str([sub['name'] for sub in data]))
        self.is_metal = self.material_data['is_metal']

    def get_imfp(self, energy):
        return np.interp(energy, self.material_data['energy'], self.material_data['imfp'])

    def get_emfp(self, energy):
        return np.interp(energy, self.material_data['energy'], self.material_data['emfp'])

    def get_phmfp(self, energy):
        if 'phonon' in self.material_data.keys():
            eloss = np.array(self.material_data['phonon']['eloss'])
            de_over_e = eloss / energy
            n_lo = 1.0 / (np.exp(eloss / (self.k_B * self.T / h2ev)) - 1.0)
            ln_plus = (1.0 + np.sqrt(np.abs(1.0 - de_over_e))) / (1.0 - np.sqrt(np.abs(1.0 - de_over_e)))
            term_plus = (n_lo + 1.0) * (
                    1.0 / self.material_data['phonon']['eps_inf'] - 1.0 / self.material_data['phonon'][
                'eps_zero']) * de_over_e / 2.0
            return term_plus * np.log(ln_plus)
        else:
            return 0

    def linterp(self, energy, y):
        ind = np.absolute(self.material_data['energy'] - energy).argmin()
        if self.material_data['energy'][ind + 1] > energy >= self.material_data['energy'][ind]:
            return y[:, ind] + (y[:, ind + 1] - y[:, ind])*(energy - self.material_data['energy'][ind]) / (
                self.material_data['energy'][ind + 1] - self.material_data['energy'][ind])
        else:
            return y[:, ind - 1] + (y[:, ind] - y[:, ind - 1]) * (energy - self.material_data['energy'][ind - 1]) / (
                self.material_data['energy'][ind] - self.material_data['energy'][ind - 1])

    def get_diimfp(self, energy):
        if self.is_metal:
            eloss = np.linspace(0, energy - self.material_data['e_fermi'],
                                self.material_data['diimfp'][:, 1, 1].shape[0])
        else:
            eloss = np.linspace(self.material_data['e_gap'],
                                energy - self.material_data['e_gap'] - self.material_data['e_vb'],
                                self.material_data['diimfp'][:, 1, 1].shape[0])
        diimfp = self.linterp(energy, self.material_data['diimfp'][:, 1, :])
        # f_rbs = RectBivariateSpline(eloss, self.material_data['energy'], self.material_data['diimfp'][:, 1, :])
        # x, y = np.meshgrid(eloss, energy, indexing="ij")
        # diimfp = np.squeeze(f_rbs(x, y, grid=False))
        return eloss, diimfp

    def get_angular_iimfp(self, energy, eloss):
        theta = np.linspace(0, math.pi / 2, 100)
        eloss /= h2ev
        if self.is_metal:
            energy /= h2ev
        else:
            energy = (energy - self.material_data['e_gap']) / h2ev
        q_squared = 4 * energy - 2 * eloss - 4 * np.sqrt(energy * (energy - eloss)) * np.cos(theta)
        f_rbs = RectBivariateSpline(self.material_data['omega'] / h2ev, self.material_data['q'] * a0,
                                    self.material_data['elf'])
        x, y = np.meshgrid(eloss, np.sqrt(q_squared), indexing="ij")
        return theta, np.squeeze(1 / (math.pi ** 2 * q_squared) * np.sqrt(1 - eloss / energy) * f_rbs(x, y, grid=False))

    def get_decs(self, energy):
        return self.linterp(energy, self.material_data['decs'])
        # f_rbs = RectBivariateSpline(self.material_data['decs_theta'], self.material_data['energy'],
        #                             self.material_data['decs'])
        # x, y = np.meshgrid(self.material_data['decs_theta'], energy, indexing="ij")
        # return np.squeeze(f_rbs(x, y, grid=False))


class Electron:

    def __init__(self, sample, energy, cb_ref, save_coord, xyz, uvw, gen, se, ind):
        self.sample = sample
        if self.sample.is_metal:
            self.inner_potential = self.sample.material_data['e_fermi'] + self.sample.material_data['work_function']
            self.energy_se = self.sample.material_data['e_fermi']
        else:
            self.inner_potential = self.sample.material_data['e_vb'] + self.sample.material_data['e_gap'] + \
                                   self.sample.material_data['affinity']
            self.energy_se = self.sample.material_data['e_vb']
        self.conduction_band_reference = cb_ref
        self.save_coordinates = save_coord
        self.xyz = xyz
        self.uvw = uvw
        self.generation = gen
        self.is_secondary = se
        if self.is_secondary:
            self.energy = energy
        else:
            self.energy = energy + self.inner_potential
        self.initial_energy = self.energy
        self.initial_depth = xyz[2]
        self.parent_index = ind
        self.coordinates = []
        if save_coord:
            coord_vector = [round(elem, 2) for elem in self.xyz + [self.energy]]
            self.coordinates.append(coord_vector)
        self.inside = True
        self.dead = False
        self.scattering_type = -1
        self.n_secondaries = 0
        self.energy_loss = 0
        self.path_length = 0
        self.deflection = [0, 0]
        self.sc_type_list = []
        # self.print_info()

    @property
    def iimfp(self):
        return 1 / self.sample.get_imfp(self.energy)

    @property
    def iemfp(self):
        if self.sample.is_metal:
            return 1 / self.sample.get_emfp(self.energy)
        else:
            return 1 / self.sample.get_emfp(
                self.energy - self.sample.material_data['e_gap'] - self.sample.material_data['e_vb'])

    @property
    def iphmfp(self):
        if self.sample.is_metal or 'phonon' not in self.sample.material_data.keys():
            return 0
        else:
            return 1 / self.sample.get_phmfp(
                self.energy - self.sample.material_data['e_gap'] - self.sample.material_data['e_vb'])

    @property
    def itmfp(self):
        return self.iimfp + self.iemfp + np.sum(self.iphmfp)

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self, value):
        if not isinstance(value, Sample):
            raise ValueError("The sample must be of type Sample")
        self._sample = value

    def print_info(self):
        print("Sample:", self.sample)
        print("Inner potential:", self.inner_potential)
        print("Energy reference:", self.conduction_band_reference)
        print("Track coordinates:", self.save_coordinates)
        print("xyz:", self.xyz)
        print("uvw:", self.uvw)
        print("Generation:", self.generation)
        print("Is secondary:", self.is_secondary)
        print("Energy:", self.energy)
        print("Initial energy:", self.initial_energy)
        print("Parent index:", self.parent_index)
        print("Coordinates:", self.coordinates)
        print("Inside:", self.inside)
        print("Is dead:", self.dead)
        print("Scat type:", self.scattering_type)
        print("Number se:", self.n_secondaries)
        print("Energy se:", self.energy_se)
        print("Energy loss:", self.energy_loss)
        print("Path length:", self.path_length)
        print("Deflection:", self.deflection)
        print("Scat type list:", self.sc_type_list)

    def travel(self):
        s = -(1 / self.itmfp) * np.log(random.random())
        if self.xyz[2] + self.uvw[2] * s < 0:
            s = np.abs(self.xyz[2] / self.uvw[2]) + 0.0001
        self.path_length += s
        self.xyz[0] = self.xyz[0] + self.uvw[0] * s
        self.xyz[1] = self.xyz[1] + self.uvw[1] * s
        self.xyz[2] = self.xyz[2] + self.uvw[2] * s
        if self.save_coordinates:
            coord_vector = [round(elem, 2) for elem in self.xyz + [self.energy]]
            self.coordinates.append(coord_vector)

    def get_scattering_type(self):
        rn = random.random()
        if rn < self.iemfp / self.itmfp:
            self.scattering_type = 0
        elif rn < (self.iemfp + self.iimfp) / self.itmfp:
            self.scattering_type = 1
        else:
            self.scattering_type = 2
        self.sc_type_list.append(self.scattering_type)

    def scatter(self):
        self.deflection[1] = random.random() * 2 * math.pi
        ind = np.absolute(self.sample.material_data['energy'] - self.energy).argmin()
        if self.scattering_type == 0:
            # decs = self.sample.get_decs(self.energy)
            decs = np.squeeze(self.sample.material_data['decs'][:, ind])
            cumsigma = integrate.cumtrapz(2 * math.pi * decs * np.sin(self.sample.material_data['decs_theta']),
                                          self.sample.material_data['decs_theta'], initial=0)
            self.deflection[0] = np.interp(random.random() * cumsigma[-1], cumsigma,
                                           self.sample.material_data['decs_theta'])
            # self.uvw = self.update_direction(self.uvw, self.deflection, 0)
            self.uvw = self.change_direction(self.uvw,self.deflection)
            return False
        elif self.scattering_type == 1:
            # [eloss, diimfp] = self.sample.get_diimfp(self.energy)
            eloss = np.squeeze(self.sample.material_data['diimfp'][:, 0, ind])
            diimfp = np.squeeze(self.sample.material_data['diimfp'][:, 1, ind])
            cumdiimfp = integrate.cumtrapz(diimfp, eloss, initial=0)
            while True:
                self.energy_loss = np.interp(random.random() * cumdiimfp[-1], cumdiimfp, eloss)
                if self.energy_loss < self.energy:
                    break
            self.energy -= self.energy_loss
            self.is_dead()
            if not self.dead:
                self.feg_dos()
                if self.sample.is_metal:
                    min_energy = 1
                else:
                    min_energy = self.sample.material_data['e_gap']
                if self.energy > min_energy:
                    theta, angdist = self.sample.get_angular_iimfp(self.energy + self.energy_loss, self.energy_loss)
                    cumdiimfp = integrate.cumtrapz(angdist * np.sin(theta), theta, initial=0)
                    self.deflection[0] = np.interp(random.random() * cumdiimfp[-1], cumdiimfp, theta)
                    if self.deflection[0] == np.nan:
                        print("nan")
                        self.deflection[0] = math.asin(math.sqrt(self.energy_loss / (self.energy + self.energy_loss)))
                    # self.uvw = self.update_direction(self.uvw, self.deflection, 0)
                    self.uvw = self.change_direction(self.uvw,self.deflection)
                else:
                    print("Should not have happened but...")
            return True
        else:
            rn = random.random()
            e = (self.energy - self.sample.material_data['e_gap'] - self.sample.material_data['e_vb']) / h2ev
            if random.random() < self.iphmfp[0] / np.sum(self.iphmfp):
                de = self.sample.material_data['phonon']['eloss'][0] / h2ev
            else:
                de = self.sample.material_data['phonon']['eloss'][1] / h2ev
            if e - de > 0:
                bph = (e + e - de + 2 * math.sqrt(e * (e - de))) / (e + e - de - 2 * math.sqrt(e * (e - de)))
                self.deflection[0] = math.acos(
                    (e + e - de) / (2 * math.sqrt(e * (e - de))) * (1 - bph ** rn) + bph ** rn)
                self.energy -= de * h2ev
                self.is_dead()
                if not self.dead:
                    self.uvw = self.update_direction(self.uvw, self.deflection, 0)
            else:
                self.dead = True
            return False

    def is_dead(self):
        if self.energy < self.inner_potential:
            self.dead = True

    def feg_dos(self):
        if self.sample.is_metal:
            e_ref = self.sample.material_data['e_fermi']
        else:
            e_ref = self.sample.material_data['e_vb']

        ener = np.linspace(0,e_ref,100)
        dist = np.sqrt(ener * (ener + self.energy_loss))
        cumdos = integrate.cumtrapz(dist, ener, initial=0)
        self.energy_se = np.interp(random.random() * cumdos[-1], cumdos, ener)

    def linterp(self, ind, y):
        if self.sample.material_data['energy'][ind + 1] > self.energy >= self.sample.material_data['energy'][ind]:
            return y[:, ind] + (y[:, ind + 1] - y[:, ind])*(self.energy - self.sample.material_data['energy'][ind]) / (
                self.sample.material_data['energy'][ind + 1] - self.sample.material_data['energy'][ind])
        else:
            return y[:, ind - 1] + (y[:, ind] - y[:, ind - 1]) * (self.energy - self.sample.material_data['energy'][ind - 1]) / (
                self.sample.material_data['energy'][ind] - self.sample.material_data['energy'][ind - 1])

    def update_direction(self, uvw, deflection, algorithm=0):
        base_uvw = [0.0, 0.0, 0.0]
        if algorithm == 0:
            cdt = math.cos(deflection[0])
            sdf = math.sin(deflection[1])
            cdf = math.cos(deflection[1])
            dxy = uvw[0] ** 2 + uvw[1] ** 2
            dxyz = dxy + uvw[2] ** 2
            if abs(dxyz - 1.0) > 1e-9:
                fnorm = 1.0 / math.sqrt(dxyz)
                uvw[0] = fnorm * uvw[0]
                uvw[1] = fnorm * uvw[1]
                uvw[2] = fnorm * uvw[2]
                dxy = uvw[0] ** 2 + uvw[1] ** 2

            if dxy > 1e-9:
                sdt = math.sqrt((1.0 - cdt ** 2) / dxy)
                up = uvw[0]
                base_uvw[0] = uvw[0] * cdt + sdt * (up * uvw[2] * cdf - uvw[1] * sdf)
                base_uvw[1] = uvw[1] * cdt + sdt * (uvw[1] * uvw[2] * cdf + up * sdf)
                base_uvw[2] = uvw[2] * cdt - dxy * sdt * cdf
            else:
                sdt = math.sqrt(1.0 - cdt ** 2)
                base_uvw[1] = sdt * sdf
                if uvw[2] > 0:
                    base_uvw[0] = sdt * cdf
                    base_uvw[2] = cdt
                else:
                    base_uvw[0] = -1.0 * sdt * cdf
                    base_uvw[2] = -1.0 * cdt
        elif algorithm == 1:
            if abs(uvw[2]) > 0.99:
                base_uvw[0] = math.sin(deflection[0]) * math.cos(deflection[1])
                base_uvw[1] = math.sin(deflection[0]) * math.sin(deflection[1])
                base_uvw[2] = math.cos(deflection[0])
            else:
                sq1 = math.sqrt(1 - math.cos(deflection[0]) ** 2)
                sqw = math.sqrt(1 - uvw[2] ** 2)
                base_uvw[0] = uvw[0] * math.cos(deflection[0]) + sq1 / sqw * (
                        uvw[0] * uvw[2] * math.cos(deflection[1]) - uvw[1] * math.sin(deflection[1]))
                base_uvw[1] = uvw[1] * math.cos(deflection[0]) + sq1 / sqw * (
                        uvw[1] * uvw[2] * math.cos(deflection[1]) + uvw[0] * math.sin(deflection[1]))
                base_uvw[2] = uvw[2] * math.cos(deflection[0]) - sq1 * sqw * math.cos(deflection[1])
        else:
            theta_old = math.acos(uvw[2])
            phi_old = math.atan2(uvw[1], uvw[0])
            theta = math.acos(math.cos(theta_old) * math.cos(deflection[0]) - math.sin(theta_old) * math.sin(
                deflection[0]) * math.cos(deflection[1]))
            phi = math.asin(math.sin(deflection[0]) * math.sin(deflection[1]) / math.sin(theta)) + phi_old

            base_uvw[0] = math.sin(theta) * math.cos(phi)
            base_uvw[1] = math.sin(theta) * math.sin(phi)
            base_uvw[2] = math.cos(theta)

        return base_uvw

    def change_direction(self, uvw, deflection):
        new_uvw = [0.0, 0.0, 0.0]
        sin_psi = math.sin(deflection[0])
        cos_psi = math.cos(deflection[0])
        sin_fi = math.sin(deflection[1])
        cos_fi = math.cos(deflection[1])
        cos_theta = uvw[2]
        sin_theta = math.sqrt(uvw[0] ** 2 + uvw[1] ** 2)
        if sin_theta > 1e-10:
            cos_phi = uvw[0] / sin_theta
            sin_phi = uvw[1] / sin_theta
        else:
            cos_phi = 1.0
            sin_phi = 0.0
        # calculate new direction
        h0 = sin_psi * cos_fi
        h1 = sin_theta * cos_psi + h0 * cos_theta
        h2 = sin_psi * sin_fi
        new_uvw[0] = h1 * cos_phi - h2 * sin_phi
        new_uvw[1] = h1 * sin_phi + h2 * cos_phi
        new_uvw[2] = cos_theta * cos_psi - h0 * sin_theta
        # to guarantee  dirx**2 + diry**2 + dirz**2 = 1
        delta = 1.5 - 0.5 * (new_uvw[0] ** 2 + new_uvw[1] ** 2 + new_uvw[2] ** 2)
        new_uvw[0] *= delta
        new_uvw[1] *= delta
        new_uvw[2] *= delta
        return new_uvw

    def calculate_direction(self, theta, phi):
        self.uvw[0] = math.sin(theta) * math.cos(phi)
        self.uvw[1] = math.sin(theta) * math.sin(phi)
        self.uvw[2] = -math.cos(theta)
        norm = math.sqrt(self.uvw[0] ** 2 + self.uvw[1] ** 2 + self.uvw[2] ** 2)
        self.uvw[0] /= norm
        self.uvw[1] /= norm
        self.uvw[2] /= norm

    def escape(self):
        theta = math.acos(self.uvw[2])
        phi = math.atan2(self.uvw[1], self.uvw[0])
        if self.xyz[2] < 0:
            if not self.sample.is_metal and self.conduction_band_reference:
                ecos = (self.energy - self.sample.material_data['e_gap'] - self.sample.material_data['e_vb']) * \
                       self.uvw[2] ** 2
                ui = self.sample.material_data['affinity']
            else:
                ecos = self.energy * self.uvw[2] ** 2
                ui = self.inner_potential
            if ecos > ui:
                t = 4 * math.sqrt(1 - ui / ecos) / ((1 + math.sqrt(1 - ui / ecos)) ** 2)
            else:
                t = 0
            if random.random() < t:
                self.inside = False
                theta_vacuum = math.asin(math.sin(theta) * math.sqrt(self.energy / (self.energy - ui)))
                self.energy -= self.inner_potential
                self.calculate_direction(theta_vacuum, phi)
                if self.save_coordinates:
                    self.xyz[0] += 100 * self.uvw[0]
                    self.xyz[1] += 100 * self.uvw[1]
                    self.xyz[2] += 100 * self.uvw[2]
                    coord_vector = [round(elem, 2) for elem in self.xyz + [self.energy]]
                    self.coordinates.append(coord_vector)
                return True
            else:
                self.uvw[2] *= -1
                self.xyz[2] = 1e-10
                if self.save_coordinates:
                    coord_vector = [round(elem, 2) for elem in self.xyz + [self.energy]]
                    self.coordinates.append(coord_vector)
                return False


class SEEMC:
    def __init__(self, energy_array, sample_name, angle, n_traj, cb_ref, track):
        self.coincidence_histogram = None
        self.bse = None
        self.sey = None
        self.tey = None
        self.energy_array = energy_array
        self.current_energy = None
        self.sample = Sample(sample_name)
        self.n_trajectories = n_traj
        self.cb_ref = cb_ref
        self.track_trajectories = track
        self.electron_list = []
        self.incident_angle = angle


    def run_trajectory(self, trajectory):
        electron_array = []
        electron_data = []
        i = 0
        electron_array.append(Electron(self.sample, self.current_energy, self.cb_ref, self.track_trajectories, [0, 0, 0],
                                       [math.sin(self.incident_angle), 0, math.cos(self.incident_angle)], 0,
                                       False, -1))
        while i < len(electron_array):
            while electron_array[i].inside and not electron_array[i].dead:
                se_xyz = [0, 0, 0]
                electron_array[i].travel()
                if not electron_array[i].escape():
                    electron_array[i].get_scattering_type()
                    if electron_array[i].scatter():
                        se_energy = electron_array[i].energy_loss + electron_array[i].energy_se
                        if se_energy > electron_array[i].inner_potential:
                            se_xyz[0] = electron_array[i].xyz[0]
                            se_xyz[1] = electron_array[i].xyz[1]
                            se_xyz[2] = electron_array[i].xyz[2]
                            se_uvw = electron_array[i].change_direction(electron_array[i].uvw, [
                                math.asin(math.cos(electron_array[i].deflection[0])),
                                electron_array[i].deflection[1] + math.pi])
                            electron_array[i].n_secondaries += 1
                            electron_array.append(
                                Electron(self.sample, se_energy, self.cb_ref, self.track_trajectories, se_xyz,
                                         se_uvw, electron_array[i].generation + 1, True, i))

            res = {'energy': electron_array[i].energy, 'inner_potential': electron_array[i].inner_potential,
                   'energy_loss': electron_array[i].energy_loss, 'energy_se': electron_array[i].energy_se,
                   'is_secondary': electron_array[i].is_secondary, 'generation': electron_array[i].generation,
                   'parent_index': electron_array[i].parent_index, 'dead': electron_array[i].dead,
                   'inside': electron_array[i].inside}
            if self.track_trajectories:
                res['coordinates'] = electron_array[i].coordinates
            electron_data.append(SimpleNamespace(**res))
            electron_array[i] = None
            i += 1
        gc.collect()
        return electron_data


    def run_parallel_simulation(self):
        start_time = time.time()

        for energy in self.energy_array:
            print(energy)
            self.current_energy = energy
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                res = pool.map(self.run_trajectory, range(self.n_trajectories))
            # self.electron_list.append([element for innerList in res for element in innerList])
            self.electron_list.append(res)

        print("--- %s seconds ---" % (time.time() - start_time))

    def run_simulation(self, plot_yields=False):
        start_time = time.time()

        for energy in self.energy_array:
            print(energy)
            self.current_energy = energy
            e_statistics = []
            for traj in tqdm(range(self.n_trajectories)):
                e_statistics.append(self.run_trajectory(traj))
            self.electron_list.append(e_statistics)
        print("--- %s seconds ---" % (time.time() - start_time))
        if plot_yields:
            self.calculate_yield()
            self.plot_yield()

    def calculate_yield(self):
        self.tey = np.zeros(len(self.energy_array))
        self.sey = np.zeros(len(self.energy_array))
        self.bse = np.zeros(len(self.energy_array))

        for i in range(len(self.energy_array)):
            for j in range(len(self.electron_list[i])):
                for e in self.electron_list[i][j]:
                    if not e.inside and not e.dead:
                        self.tey[i] += 1
                        if e.is_secondary:
                            self.sey[i] += 1
                        else:
                            self.bse[i] += 1

        self.tey /= self.n_trajectories
        self.sey /= self.n_trajectories
        self.bse /= self.n_trajectories

        print("Energy:", self.energy_array)
        print("TEY:", self.tey)
        print("SEY:", self.sey)
        print("BSE:", self.bse)

    def calculate_energy_histogram(self, energy_index=0):
        self.se_energy_histogram = []
        self.pe_energy_histogram = []
        for e in self.electron_list[energy_index]:
            if not e.inside:
                if e.is_secondary:
                    self.se_energy_histogram.append(e.energy)
                else:
                    self.pe_energy_histogram.append(e.energy)

    def calculate_depth_histogram(self, energy_index=0):
        self.se_depth_histogram = []
        self.pe_depth_histogram = []
        for e in self.electron_list[energy_index]:
            if not e.inside:
                if e.is_secondary:
                    self.se_depth_histogram.append(e.initial_depth)
                else:
                    self.pe_depth_histogram.append(e.initial_depth)


    def calculate_coincidence_histogram(self, true_pairs=True):
        self.coincidence_histogram = []
        for j in range(len(self.electron_list[0])):
            for e in self.electron_list[0][j]:
                if not e.dead and e.is_secondary:
                    if not self.electron_list[0][j][e.parent_index].dead:
                        if true_pairs:
                            if (e.energy + e.inner_potential == self.electron_list[0][j][e.parent_index].energy_loss +
                                    self.electron_list[0][j][e.parent_index].energy_se):
                                self.coincidence_histogram.append(
                                    [self.electron_list[0][j][e.parent_index].energy, e.energy])
                        else:
                            self.coincidence_histogram.append(
                                [self.electron_list[0][j][e.parent_index].energy, e.energy])


    def plot_coincidence_histogram(self, n_bins):
        coincidences = np.array(self.coincidence_histogram)
        plt.figure()
        plt.hist2d(coincidences[:, 0], coincidences[:, 1])
        plt.set_cmap('turbo')
        plt.xlabel('Energy of PE (eV)')
        plt.ylabel('Energy of SE (eV)')


    def plot_yield(self):
        plt.figure()
        plt.plot(self.energy_array, self.tey, label="TEY")
        plt.plot(self.energy_array, self.sey, label="SEY")
        plt.plot(self.energy_array, self.bse, label="BSE")
        plt.xlabel('Energy (eV)')
        plt.ylabel('Yield')
        plt.title(self.sample.name)
        plt.legend()
        plt.show()

    def plot_trajectories(self, energy_index=0):
        fig = go.Figure()
        for e in self.electron_list[energy_index]:
            x_c = [x[0] for x in e.coordinates]
            y_c = [x[1] for x in e.coordinates]
            z_c = [x[2] for x in e.coordinates]
            c = [x[3] for x in e.coordinates]

            fig.add_traces(
                go.Scatter(
                    x=x_c,
                    y=z_c,
                    mode='markers+lines',
                    showlegend=False,
                    marker=dict(color=c, coloraxis='coloraxis'),
                    line_color='black',
                    line={'width': 0.5}
                )
            )

        fig.update_layout(
            autosize=False,
            width=1100,
            height=850,
            showlegend=False,
            coloraxis=dict(colorscale='Turbo'),
            coloraxis_colorbar=dict(title='Energy (eV)'),
            title=self.sample.name,
            xaxis_title="X axis",
            yaxis_title="Z axis"
        )

        # for i, data in enumerate(fig['data']):
        #     fig.update_traces(marker_color=data['line']['color'],
        #                       selector=dict(name=data['name']))

        fig.update_yaxes(autorange="reversed")
        fig.update_coloraxes(colorscale='Turbo')
        fig.show()
