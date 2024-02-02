import numpy as np
import math
from optlib.constants import *
import pickle
from scipy import integrate
from scipy.interpolate import RectBivariateSpline
import random
import matplotlib.pyplot as plt
import time

class RealNumber:
    def __get__(self, obj, objtype = None):
        return obj._value

    def __set__(self, obj, value):
        if isinstance(value, complex):
            raise ValueError("The value must be real")
        obj._value = value

class Sample:
    material_data = {}
    kbt = 9.445e-4
    
    def __init__(self,name,is_metal = True):
        with open('MaterialDatabase.pkl','rb') as fp:
            data = pickle.load(fp)
        if name in [sub['name'] for sub in data]:
            self.name = name
            self.material_data = data[next((i for i, item in enumerate(data) if item["name"] == name), None)]
        else:
            raise ValueError('Allowed sample names are ' + str([sub['name'] for sub in data]))
        self.is_metal = is_metal

    def get_imfp(self,energy):
        return np.interp(energy,self.material_data['energy'],self.material_data['imfp'])

    def get_emfp(self,energy):
        return np.interp(energy,self.material_data['energy'],self.material_data['emfp'])

    def get_phmfp(self,energy):
        if 'phonon' in self.material_data.keys():
            de = self.material_data['phonon']['eloss']/energy
            sqe = np.sqrt(1 - de)
            return (self.material_data['phonon']['eps_zero'] - self.material_data['phonon']['eps_inf'])/ \
            (self.material_data['phonon']['eps_zero']*self.material_data['phonon']['eps_inf'])*de* \
            ( (1/(np.exp(self.material_data['phonon']['eloss']/self.kbt)-1))+1 )/2*np.log( (1.0+sqe)/(1.0-sqe) )
        else:
            return 0

    def get_diimfp(self,energy):
        eloss = np.linspace(0,energy - self.material_data['e_fermi'],self.material_data['diimfp'][:,1,1].shape[0])
        f_rbs = RectBivariateSpline(eloss,self.material_data['energy'], self.material_data['diimfp'][:,1,:])
        x,y = np.meshgrid(eloss,energy,indexing="ij")
        return eloss,np.squeeze(f_rbs(x,y, grid=False))

    def get_angular_iimfp(self,energy,eloss):
        theta = np.linspace(0,math.pi/2,100)
        q_squared = 4*energy/h2ev - 2*eloss/h2ev - 4*np.sqrt(energy/h2ev*(energy - eloss)/h2ev)*np.cos(theta)
        f_rbs = RectBivariateSpline(self.material_data['omega']/h2ev,self.material_data['q']*a0,self.material_data['elf'])
        x,y = np.meshgrid(eloss/h2ev,np.sqrt(q_squared),indexing="ij")
        return theta,np.squeeze(1/(math.pi**2*q_squared)*np.sqrt(1 - eloss/energy)*f_rbs(x,y, grid=False))

    def get_decs(self,energy):
        f_rbs = RectBivariateSpline(self.material_data['decs_theta'],self.material_data['energy'],self.material_data['decs'])
        x,y = np.meshgrid(self.material_data['decs_theta'],energy,indexing="ij")
        return np.squeeze(f_rbs(x,y, grid=False))
 

class Electron:
    
    def __init__(self,sample,energy,cb_ref,save_coord,xyz,uvw,gen,se,ind):
        self.sample = sample
        self.inner_potential = self.sample.material_data['e_fermi'] + self.sample.material_data['work_function']
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
        self.parent_index = ind
        self.coordinates = []
        if save_coord:
            coord_vector = [ round(elem, 2) for elem in self.xyz + [self.energy] ]
            self.coordinates.append(coord_vector)
        self.inside = True
        self.dead = False
        self.scattering_type = 0
        self.n_secondaries = 0
        self.energy_se = self.sample.material_data['e_fermi']
        self.energy_loss = 0
        self.path_length = 0
        self.deflection = [0,0]

    def __str__(self):
        return f"Electron in {self.matname} with energy {self.energy} eV."

    @property
    def iimfp(self):
        return 1/self.sample.get_imfp(self.energy)

    @property
    def iemfp(self):
        return 1/self.sample.get_emfp(self.energy)

    @property
    def iphmfp(self):
        if self.sample.is_metal:
            return 0
        else:
            return 1/self.sample.get_phmfp(self.energy)

    @property
    def itmfp(self):
        return self.iimfp + self.iemfp + self.iphmfp

    @property
    def sample(self):
        return self._sample

    @sample.setter
    def sample(self,value):
        if not isinstance(value, Sample):
            raise ValueError("The sample must be of type Sample")
        self._sample = value

    def travel(self):
        s = -(1/self.itmfp)*np.log(random.random())
        # if self.xyz[2] + self.uvw[2]*s < 0:
        #     s = np.abs(self.xyz[2]/self.uvw[2]) + 0.0001
        self.path_length += s
        self.xyz[0] = self.xyz[0] + self.uvw[0]*s
        self.xyz[1] = self.xyz[1] + self.uvw[1]*s
        self.xyz[2] = self.xyz[2] + self.uvw[2]*s
        if self.save_coordinates:
            coord_vector = [ round(elem, 2) for elem in self.xyz + [self.energy] ]
            self.coordinates.append(coord_vector)

    def get_scattering_type(self):
        rn = random.random()
        if rn < self.iemfp/self.itmfp:
            self.scattering_type = 0
        elif rn < (self.iemfp + self.iimfp)/self.itmfp:
            self.scattering_type = 1
        else:
            self.scattering_type = 2

    def scatter(self):
        self.deflection[1] = random.random()*2*math.pi
        if self.scattering_type == 0:
            loss = False
            # decs = self.sample.get_decs(self.energy)
            ind = np.absolute(self.sample.material_data['energy'] - self.energy).argmin()
            decs = np.squeeze(self.sample.material_data['decs'][:,ind])
            cumsigma = integrate.cumtrapz(2*math.pi*decs*np.sin(self.sample.material_data['decs_theta']),self.sample.material_data['decs_theta'],initial=0)
            self.deflection[0] = np.interp(random.random()*self.sample.material_data['sigma_el'][ind],cumsigma,self.sample.material_data['decs_theta'])
            self.uvw = self.update_direction(self.uvw,self.deflection)
        elif self.scattering_type == 1:
            loss = True
            # [eloss,diimfp] = self.sample.get_diimfp(self.energy)
            ind = np.absolute(self.sample.material_data['energy'] - self.energy).argmin()
            eloss = np.squeeze(self.sample.material_data['diimfp'][:,0,ind])
            diimfp = np.squeeze(self.sample.material_data['diimfp'][:,1,ind])
            cumdiimfp = integrate.cumtrapz(diimfp,eloss,initial=0)
            cumdiimfp = (cumdiimfp - cumdiimfp[0])/(cumdiimfp[-1] - cumdiimfp[0])
            self.energy_loss = np.interp(random.random(),cumdiimfp,eloss)
            self.energy -= self.energy_loss
            self.is_dead()
            if not self.dead:
                theta,angdist = self.sample.get_angular_iimfp(self.energy + self.energy_loss,self.energy_loss)
                cumdiimfp = integrate.cumtrapz(angdist,theta,initial=0)
                self.deflection[0] = np.interp(random.random()*cumdiimfp[-1],cumdiimfp,theta)
                if self.deflection[0] == np.nan:
                    self.deflection[0][0] = math.asin(math.sqrt(self.energy_loss/(self.energy + self.energy_loss)))
                self.uvw = self.update_direction(self.uvw,self.deflection)
        return loss

    def is_dead(self):
        if self.energy < self.inner_potential:
            self.dead = True

    def update_direction(self,uvw,deflection):
        base_uvw = [0.0, 0.0, 0.0]
        sdt = math.sin(deflection[0])
        cdt = math.cos(deflection[0])
        sdf = math.sin(deflection[1])
        cdf = math.cos(deflection[1])
        dxy = uvw[0]**2 + uvw[1]**2
        dxyz = dxy + uvw[2]**2
        if np.abs(dxyz - 1.0) > 1e-9:
            fnorm = 1.0/math.sqrt(dxyz)
            uvw[0] = fnorm*uvw[0]
            uvw[1] = fnorm*uvw[1]
            uvw[2] = fnorm*uvw[2]
            dxy = uvw[0]**2 + uvw[1]**2

        if dxy > 1e-9:
            sdt = math.sqrt((1.0 - cdt**2)/dxy)
            up = uvw[0]
            base_uvw[0] = uvw[0]*cdt + sdt*(up*uvw[2]*cdf - uvw[1]*sdf)
            base_uvw[1] = uvw[1]*cdt + sdt*(uvw[1]*uvw[2]*cdf + up*sdf)
            base_uvw[2] = uvw[2]*cdt - dxy*sdt*cdf
        else:
            sdt = math.sqrt(1.0-cdt**2)
            base_uvw[1] = sdt*sdf
            if uvw[2] > 0: 
                base_uvw[0] = sdt*cdf
                base_uvw[2] = cdt
            else:
                base_uvw[0] = -1.0*sdt*cdf
                base_uvw[2] = -1.0*cdt
        return base_uvw

    def escape(self):
        theta = math.acos(self.uvw[2])
        phi = math.atan2(self.uvw[1],self.uvw[0])
        if self.xyz[2] < 0:
            beta = math.pi - theta
            ecos = self.energy*beta**2
            if ecos > self.inner_potential:
                t = 4*math.sqrt(1 - self.inner_potential/ecos)/(1 + math.sqrt(1 - self.inner_potential/ecos))**2
            else:
                t = 0
            if random.random() < t:
                self.inside = False
                self.xyz[0] += math.sin(beta)*math.cos(phi)*self.xyz[2]/math.cos(beta)
                self.xyz[1] += math.sin(beta)*math.sin(phi)*self.xyz[2]/math.cos(beta)
                self.xyz[2] = 0.0
                if self.save_coordinates:
                    self.coordinates[-1] = [ round(elem, 2) for elem in self.xyz + [self.energy] ]
                theta = np.pi - np.arcsin(np.sin(beta)*np.sqrt( self.energy/(self.energy - self.inner_potential) ))                
                self.energy -= self.inner_potential
                self.uvw[0] = math.sin(theta)*math.cos(phi)
                self.uvw[1] = math.sin(theta)*math.sin(phi)
                self.uvw[2] = math.cos(theta)
                if self.save_coordinates:
                    self.xyz[0] += 100*self.uvw[0]
                    self.xyz[1] += 100*self.uvw[1]
                    self.xyz[2] += 100*self.uvw[2]
                    coord_vector = [ round(elem, 2) for elem in self.xyz + [self.energy] ]
                    self.coordinates.append(coord_vector)
            else:
                self.uvw[2] *= -1
                self.xyz[2] *= -1
                if self.save_coordinates:
                    coord_vector = [ round(elem, 2) for elem in self.xyz + [self.energy] ]
                    self.coordinates.append(coord_vector)

class SEEMC:
    def __init__(self,energy_array,sample_name,n_traj,cb_ref,track):
        self.energy_array = energy_array
        self.sample = Sample(sample_name)
        self.n_trajectories = n_traj
        self.cb_ref = cb_ref
        self.track_trajectories = track
        self.electron_list = []

    def run_simulation(self):
        start_time = time.time()
        
        for energy in self.energy_array:
            print(energy)
            e_statistics = []
            i = -1
            for traj in range(self.n_trajectories):
                e_statistics.append(Electron(self.sample,energy,self.cb_ref,self.track_trajectories,[0,0,0],[0,0,1],0,False,-1))
                while i < len(e_statistics)-1:
                    i += 1
                    e = e_statistics[i]
                    while e.inside and not e.dead:
                        e.travel()
                        e.escape()
                        if e.inside and not e.dead:
                            e.get_scattering_type()
                            if e.scatter():
                                se_energy = e.energy_loss + e.sample.material_data['e_fermi']
                                if se_energy > e.inner_potential:
                                    se_uvw = e.update_direction(e.uvw,[math.asin(math.cos(e.deflection[0])),e.deflection[1] + math.pi])
                                    e.n_secondaries += 1
                                    e_statistics.append(Electron(self.sample,se_energy,self.cb_ref,self.track_trajectories,e.xyz,se_uvw,e.generation + 1,True,i))
            self.electron_list.append(e_statistics)
        
        print("--- %s seconds ---" % (time.time() - start_time))

    def calculate_yield(self):
        self.tey = np.zeros(len(self.energy_array))
        self.sey = np.zeros(len(self.energy_arrayy))
        self.bse = np.zeros(len(self.energy_array))
        
        for i in range(len(self.energy_array)):
            for e in self.electron_list[i]:
                if not e.inside:
                    self.tey[i] += 1
                    if e.is_secondary and e.energy < 50:
                        self.sey[i] += 1
                    else:
                        self.bse[i] += 1

        self.tey /= self.n_trajectories
        self.sey /= self.n_trajectories
        self.bse /= self.n_trajectories
        print("TEY:",self.tey)
        print("SEY:",self.sey)
        print("BSE:",self.bse)

    def plot_trajectories(self,energy_index):
        plt.figure()
        energy = self.energy_array[energy_index]
        for e in self.electron_list[energy_index]:
            x_c = [x[0] for x in e.coordinates]
            y_c = [x[1] for x in e.coordinates]
            z_c = [x[2] for x in e.coordinates]
            c = [x[3] for x in e.coordinates]
            # plt.plot(x_c,z_c,linewidth=1)
            plt.scatter(y_c,z_c,s=5, c=c, cmap='jet')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.show()
        