import numpy as np
import math
from optlib.constants import *
import pickle
from scipy import integrate
from scipy.interpolate import RectBivariateSpline
import random

class PositiveNumber:
    def __get__(self, obj, objtype = None):
        return obj._value

    def __set__(self, obj, value):
        if not isinstance(value, int | float) or value <= 0:
            raise ValueError("The value must be positive")
        obj._value = value

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
    energy = PositiveNumber()
    energy_loss = 0
    inner_potential = 0
    path_length = 0
    deflection = [0,0]
    coordinates = []
    inside = True
    dead = False
    scattering_type = 0
    n_secondaries = 0
    energy_se = 0
    
    def __init__(self,sample,energy,cb_ref = False,save_coord = False,xyz = [0,0,0],uvw = [0,0,1],gen = 0,se = False,ind = -1):
        self.sample = sample
        self.energy = energy
        self.conduction_band_reference = cb_ref
        self.save_coordinates = save_coord
        self.xyz = xyz
        self.uvw = uvw
        self.generation = gen
        self.is_secondary = se
        self.parent_index = ind

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
        if self.xyz[2] + self.uvw[2]*s < 0:
            s = np.abs(self.xyz[2]/self.uvw[2]) + 0.0001
        self.path_length += s
        print(self.energy)
        self.xyz[0] = self.xyz[0] + self.uvw[0]*s
        self.xyz[1] = self.xyz[1] + self.uvw[1]*s
        self.xyz[2] = self.xyz[2] + self.uvw[2]*s

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
            decs = self.sample.get_decs(self.energy)
            cumsigma = integrate.cumtrapz(2*math.pi*decs*np.sin(self.sample.material_data['decs_theta']),self.sample.material_data['decs_theta'],initial=0)
            cumsigma = (cumsigma - cumsigma[0])/(cumsigma[-1] - cumsigma[0])
            self.deflection[0] = np.interp(random.random(),cumsigma,self.sample.material_data['decs_theta'])
        elif self.scattering_type == 1:
            loss = True
            [eloss,diimfp] = self.sample.get_diimfp(self.energy)
            cumdiimfp = integrate.cumtrapz(diimfp,eloss,initial=0)
            cumdiimfp = (cumdiimfp - cumdiimfp[0])/(cumdiimfp[-1] - cumdiimfp[0])
            self.energy_loss = np.interp(random.random(),cumdiimfp,eloss)
            self.energy -= self.energy_loss
        return loss