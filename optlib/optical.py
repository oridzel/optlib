from cmath import isinf
import subprocess
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import pandas as pd
import os
import time
from tqdm import tqdm
import re
from scipy import optimize, special, stats
from scipy.interpolate import RectBivariateSpline, interp1d
from optlib.constants import *

def wavelength2energy(wavelength):
	# wavelength in Angstroem
    return hc / wavelength  * 1e3

def energy2wavelength(energy):
	return hc / energy * 1e3


def tpp(e,e_p,e_gap,rho):
    # imfp in nm
    beta = -1.0 + 9.44/((e_p**2 + e_gap**2)**0.5) + 0.69*(rho**0.1)
    gamma = 0.191*rho**(-0.5)
    u = (e_p/28.816)**2
    c = 19.7 - 9.1*u
    d = 534 - 208*u
    alpha = (1+e/1021999.8)/(1+e/510998.9)**2
    return alpha*e/((e_p**2)*( beta*np.log(gamma*alpha*e) - (c/e) + (d/e**2) ))


def linspace(start, stop, step=1.):
	num = int((stop - start) / step + 1)
	return np.linspace(start, stop, num)


def check_list_type(lst, type_to_check):
	if lst and isinstance(lst, list):
		return all(isinstance(elem, type_to_check) for elem in lst)
	elif lst:
		return isinstance(lst, type_to_check)
	else:
		return True


def is_list_of_int_float(lst):
	if lst and isinstance(lst, list):
		return all(isinstance(elem, int) or isinstance(elem, float) for elem in lst)
	elif lst:
		return (isinstance(lst, int) or isinstance(lst, float))
	else:
		return True

def conv(x1, x2, de, mode='right'):
	n = x1.size
	a = np.convolve(x1, x2)
	if mode == 'right':
		return a[0:n] * de
	elif mode == 'left':
		return a[a.size-n:a.size] * de
	else:
		return a * de

def gauss(x, a1, b1, c1):
    return a1*np.exp(-((x-b1)/c1)**2)

class Error(Exception):
	"""Base class for exceptions in this module."""
	pass


class InputError(Error):
	"""Exception raised for errors in the input.

	Attributes:
		message -- explanation of the error
	"""

	def __init__(self, message):
		self.message = message


class Composition:
	def __init__(self, elements, indices, atomic_numbers):
		if not check_list_type(elements, str):
			raise InputError(
				"The array of elements passed must be of the type string!")
		if not is_list_of_int_float(indices):
			raise InputError(
				"The array of indices passed must be of the type int or float!")
		if isinstance(elements, list) and isinstance(indices, list) and len(elements) != len(indices):
			raise InputError(
				"The number of elements and indices must be the same!")
		self.elements = elements
		self.indices = indices
		self.atomic_numbers = atomic_numbers


class Oscillators:
	def __init__(self, model, A, gamma, omega, alpha = 1.0, eps_b = 1.0):
		if not is_list_of_int_float(omega):
			raise InputError("The array of omega passed must be of the type int or float!")
		if not is_list_of_int_float(gamma):
			raise InputError("The array of gamma passed must be of the type int or float!")
		if not is_list_of_int_float(A):
			raise InputError("The array of A passed must be of the type int or float!")
		if omega and A and gamma:
			if len(omega) != len(A) != len(gamma):
				raise InputError("The number of oscillator parameters must be the same!")
		self.model = model
		self.A = np.array(A)
		self.gamma = np.array(gamma)
		self.omega = np.array(omega)
		self.alpha = alpha
		self.eps_b = eps_b
	
	def __str__(self):
		return f'A = [{", ".join([str(i) for i in self.A.round(3).tolist()])}]\n' \
			+ f'gamma = [{", ".join([str(i) for i in self.gamma.round(3).tolist()])}]\n' \
			+ f'omega = [{", ".join([str(i) for i in self.omega.round(3).tolist()])}]\n' \
			+ f'alpha = {self.alpha}'

class Material:
	'''Here will be some description of the class Material'''

	def __init__(self, name, oscillators, composition, eloss, q, xraypath):
		if not isinstance(oscillators, Oscillators):
			raise InputError("The oscillators must be of the type Oscillators")
		if not isinstance(composition, Composition):
			raise InputError("The composition must be of the type Composition")
		self.name = name
		self.oscillators = oscillators
		self.composition = composition
		self.eloss = np.array(eloss)
		self.eloss[eloss == 0] = 1e-5
		self.xraypath = xraypath
		self.e_gap = 0
		self.e_fermi = 0
		self.u = 0
		self.width_of_the_valence_band = None
		self.atomic_density = None
		self.static_refractive_index = None
		self.electron_density = None
		self.Z = None
		self.elf = None
		self.elf_extended_to_henke = None
		self.elf_henke = None
		self.eloss_henke = None
		self.surfaceELF = None
		self.diimfp = None
		self.diimfp_e = None
		self.decs = None
		self.decs_mu = None
		self.e0 = None
		self.imfp = None
		self.imfp_e = None
		self.emfp = None
		self.sigma_el = None
		self.q_dependency = None
		self.q = q
		self.use_kk_constraint = False
		self.use_henke_for_ne = False
		self.electron_density_henke = 0
		self.use_kk_relation = False
		self.is_metal = True

	@property
	def q(self):
		if isinstance(self._q, np.ndarray):
			return self._q
		elif isinstance(self._q,list):
			return np.array(self._q)
		else:
			return self._q
	
	@q.setter
	def q(self, q):
		try: 
			self.size_q = q.shape[1]
		except:
			try:
				self.size_q = q.shape[0]
			except:
				self.size_q = 1
		finally:
			self._q = q

	@property
	def epsilon(self):
		self.calculate_dielectric_function()
		return self._epsilon

	def kramers_kronig(self, epsilon_imag):
		eps_real = np.zeros_like(self.eloss)
		for i in range(self.eloss.size):
			omega = self.eloss[i]
			ind = np.all([self.eloss != omega, self.eloss > self.e_gap], axis=0)
			if len(epsilon_imag.shape) > 1:
				kk_sum = np.trapz(self.eloss[ind] * epsilon_imag[ind,0] / (self.eloss[ind] ** 2 - omega ** 2), self.eloss[ind])
			else:
				kk_sum = np.trapz(self.eloss[ind] * epsilon_imag[ind] / (self.eloss[ind] ** 2 - omega ** 2), self.eloss[ind])
			eps_real[i] = 2 * kk_sum / math.pi + 1
		return eps_real

	def calculate_dielectric_function(self):
		if self.oscillators.model == 'Drude':
			self._epsilon = self.calculate_drude_dielectric_function()
		elif self.oscillators.model == 'DrudeLindhard':
			self._epsilon = self.calculate_dl_dielectric_function()
		elif self.oscillators.model == 'Mermin':
			self._epsilon = self.calculate_mermin_dielectric_function()
		elif self.oscillators.model == 'MLL':
			self._epsilon = self.calculate_mll_dielectric_function()
		else:
			raise InputError("Invalid model name. The valid model names are: Drude, DrudeLindhard, Mermin, and MLL")

	def calculate_drude_dielectric_function(self):
		self._convert2au()

		for i in range(len(self.oscillators.A)):
			eps_drude_real, eps_drude_imag = self._calculate_drude_oscillator(
				self.oscillators.omega[i], self.oscillators.gamma[i], self.oscillators.alpha)
			if i == 0:
				eps_real = self.oscillators.eps_b * np.ones_like(eps_drude_real)
				eps_imag = np.zeros_like(eps_drude_imag)
				epsilon = np.zeros_like(eps_real, dtype=complex)
			eps_real -= self.oscillators.A[i] * eps_drude_real
			eps_imag += self.oscillators.A[i] * eps_drude_imag

		if self.e_gap > 0:
			if len(eps_imag.shape) > 1:
				eps_imag[self.eloss <= self.e_gap,0] = 1e-5
			else:
				eps_imag[self.eloss <= self.e_gap] = 1e-5
		if self.use_kk_relation:
			if len(eps_real.shape) > 1:
				eps_real[:,0] = self.kramers_kronig(eps_imag)
			else:
				eps_real = self.kramers_kronig(eps_imag)

		epsilon.real = eps_real
		epsilon.imag = eps_imag

		self._convert2ru()
		return epsilon

	def _calculate_drude_oscillator(self, omega0, gamma, alpha):
		# if not self.q_dependency is None:
		# 	w_at_q = omega0 + 0.5 * alpha * self.q**0.5
			# w_at_q = omega0 + (self.q_dependency(self.q / a0)/h2ev - self.q_dependency(0)/h2ev)
		# else:
		w_at_q = omega0 + 0.5 * alpha * self.q**2
		if self.size_q == 1:
			omega = np.squeeze(np.array([self.eloss, ] * self.size_q).transpose())
			# if self.q <= np.sqrt(2*self.width_of_the_valence_band):
			# 	w_at_q = omega0 + alpha * 0.5 * self.q**2
			# else:
			# 	w_at_q = omega0 + alpha * self.e_gap + 0.5 * self.q**2
		else:
			omega = np.expand_dims(self.eloss, axis=tuple(range(1,self.q.ndim)))
			# ind_less = self.q <= np.sqrt(2*self.width_of_the_valence_band)
			# ind_more = self.q > np.sqrt(2*self.width_of_the_valence_band)
			# w_at_q = np.zeros_like(self.q)
			# w_at_q[ind_less] = omega0 + alpha * 0.5 * self.q[ind_less]**2
			# w_at_q[ind_more] = omega0 + alpha * self.e_gap + 0.5 * self.q[ind_more]**2

		mm = omega**2 - w_at_q**2
		divisor = mm**2 + omega**2 * gamma**2

		eps_real = mm / divisor
		eps_imag = omega*gamma / divisor

		return eps_real, eps_imag

	def calculate_dl_dielectric_function(self):
		self._convert2au()
		epsilon = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))
		sum_oneover_eps = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))
		oneover_eps = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))

		for i in range(len(self.oscillators.A)):
			oneover_eps = self._calculate_dl_oscillator(
				self.oscillators.omega[i], self.oscillators.gamma[i], self.oscillators.alpha)
			sum_oneover_eps += self.oscillators.A[i] * (oneover_eps - complex(1))

		sum_oneover_eps += complex(1)
		epsilon = complex(1) / sum_oneover_eps

		if self.use_kk_relation:
			eps_imag = epsilon.imag
			eps_real = self.kramers_kronig(eps_imag)
			epsilon.real = eps_real
			epsilon.imag = eps_imag

		self._convert2ru()
		return epsilon

	def _calculate_dl_oscillator(self, omega0, gamma, alpha):
		if not self.q_dependency is None:
			w_at_q = omega0 - self.q_dependency(0)/h2ev + self.q_dependency(self.q / a0)/h2ev
		else:
			w_at_q = omega0 + 0.5 * alpha * self.q**2
			# if self.q <= np.sqrt(2*self.width_of_the_valence_band):
			# 	w_at_q = omega0 + alpha * 0.5 * self.q**2
			# else:
			# 	w_at_q = omega0 + alpha * self.e_gap + 0.5 * self.q**2

		omega = np.squeeze(np.array([self.eloss, ] * self.size_q).transpose())

		mm = omega**2 - w_at_q**2
		divisor = mm**2 + omega**2 * gamma**2

		one_over_eps_imag = -omega0**2 * omega * gamma / divisor
		if self.e_gap > 0:
			one_over_eps_imag[self.eloss <= self.e_gap] = 1e-5
		if self.use_kk_relation:
			one_over_eps_real = self.kramers_kronig(one_over_eps_imag)
		else:
			one_over_eps_real = 1.0 + omega0**2 * mm / divisor

		one_over_eps = np.squeeze(np.apply_along_axis(lambda args: [complex(
			*args)], 0, np.array([one_over_eps_real, one_over_eps_imag])))

		return one_over_eps

	def calculate_mermin_dielectric_function(self):
		if self.size_q == 1 and self.q == 0:
			self.q = 0.01
		self._convert2au()
		epsilon = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))
		oneovereps = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))

		for i in range(len(self.oscillators.A)):
			if all(np.abs((self.eloss - self.oscillators.omega[i]) / self.oscillators.gamma[i]) < 100000):
				eps_mermin = self._calculate_mermin_oscillator(
				self.oscillators.omega[i], self.oscillators.gamma[i])
			else:
				eps_mermin = complex(1)
			oneovereps += self.oscillators.A[i] * (complex(1) / eps_mermin)
		oneovereps += complex(1) - complex(np.sum(self.oscillators.A))
		
		if self.e_gap > 0:
			oneovereps.imag[self.eloss <= self.e_gap] = 1e-5

		epsilon = complex(1) / oneovereps
		if self.use_kk_relation:
			eps_imag = epsilon.imag
			eps_real = self.kramers_kronig(eps_imag)
			epsilon.real = eps_real
			epsilon.imag = eps_imag

		self._convert2ru()
		return epsilon

	def _calculate_linhard_oscillator(self, omega, gamma, omega0):
		n_dens = omega0**2 / (4*math.pi)
		E_f = 0.5 * (3 * math.pi**2 * n_dens)**(2.0 / 3.0)
		v_f = (2*E_f)**0.5
		
		z = self.q / (2 * v_f);  
		chi = np.sqrt(1.0 / (math.pi * v_f))
		
		z1_1 = omega / (self.q * v_f)
		z1_1[np.isnan(z1_1)] = 1e-5
		
		gq = np.zeros_like(self.q)
		gq = gamma / (self.q * v_f)
		vos_g_array = np.vectorize(self._vos_g)
		reD1, imD1 = vos_g_array(z1_1 + z, gq)
		reD2, imD2 = vos_g_array(z1_1 - z, gq)
		
		red1_d2 = reD1 - reD2
		imd1_d2 = imD1 - imD2
		
		chizzz = chi**2 / (z**3 * 4)
		epsreal = 1 + red1_d2 * chizzz
		epsimag = imd1_d2 * chizzz
		complex_array = np.vectorize(complex)
		return complex_array(epsreal, epsimag)

	def _vos_g(self, z, img_z):
		zplus1 = z + 1
		zminus1 = z - 1
		
		if img_z != 0:
			imgZ2 = img_z**2
			dummy1 = math.log( np.sqrt((zplus1 * zplus1 + imgZ2) / (zminus1 * zminus1 + imgZ2)) )
			dummy2 = math.atan2(img_z, zplus1) - math.atan2(img_z, zminus1)

			reim1 = 1 - (z**2 - imgZ2)

			outreal_1 = z + 0.5 * reim1 * dummy1
			outreal = outreal_1 + z *img_z * dummy2

			outimag_1 = img_z + 0.5 * reim1 * dummy2
			outimag = outimag_1 - z * img_z * dummy1
		else:
			dummy1 = math.log( abs(zplus1) / abs(zminus1) )
			dummy2 = math.atan2(0, zplus1) - math.atan2(0, zminus1)

			reim1 = 1 - z**2

			outreal_1 = z + 0.5 * reim1 * dummy1
			outreal = outreal_1

			outimag_1 = 0.5 * reim1 * dummy2
			outimag = outimag_1
		return outreal, outimag

	def _calculate_mermin_oscillator(self, omega0, gamma):
		omega = np.squeeze(np.array([self.eloss, ] * self.size_q).transpose())
		gammma_over_omega = gamma / omega
		complex_array = np.vectorize(complex)
		z1 = complex_array(1, gammma_over_omega)
		z2 = self._calculate_linhard_oscillator(omega, gamma, omega0) - complex(1)
		z3 = self._calculate_linhard_oscillator(np.zeros_like(omega), 0, omega0) - complex(1)
		top = z1 * z2
		bottom = complex(1) + complex_array(0, gammma_over_omega) * z2 / z3
		return complex(1) + top / bottom

	def calculate_mll_dielectric_function(self):
		if self.u == 0:
			raise InputError("Please specify the value of U")
		if self.size_q == 1 and self.q == 0:
			self.q = 0.01
		self._convert2au()
		epsilon = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))
		oneovereps = np.squeeze(
			np.zeros((self.eloss.shape[0], self.size_q), dtype=complex))

		for i in range(len(self.oscillators.A)):
			eps_mll = self._calculate_mll_oscillator(
				self.oscillators.omega[i], self.oscillators.gamma[i])
			oneovereps += self.oscillators.A[i] * (complex(1) / eps_mll - complex(1))
		oneovereps += complex(1)
		epsilon = complex(1) / oneovereps
		self._convert2ru()
		return epsilon

	def convert_to_mll(self):
		current_u = 0.0
		new_sum_A = 0.0
		stopthis = 0
		u_step_size = 1.0
		counter = 0
		while (new_sum_A < 1) and (current_u < 1000):
			old_sum_A = new_sum_A
			new_sum_A = 0.0
			current_u = current_u + u_step_size
			print("current_u", current_u, "u_step_size", u_step_size)
			for i in range(len(self.oscillators.A)):
				old_A = self.oscillators.A[i]
				if old_A > 0.0:
					old_omega = self.oscillators.omega[i]
					if old_omega <= current_u:
						stopthis = 1
						break
					new_omega = math.sqrt(old_omega**2 - current_u**2)
					new_A = old_A * (old_omega / new_omega)**2
					new_sum_A = new_sum_A + new_A
					if new_sum_A > 1:
						stopthis = 1
						break

			if stopthis:
				current_u = current_u - u_step_size  # go back to previous U value
				new_sum_A = old_sum_A  # make step size 10 times smaller
				u_step_size = u_step_size / 10.0
				stopthis = 0
				print("new_sum_A", new_sum_A)
				counter = counter + 1
				if counter > 100:
					break
				if  new_sum_A > 0.99:
					break

		for i in range(len(self.oscillators.A)):  # now calculate new values for optimum U
			old_A = self.oscillators.A[i]
			if old_A > 0.0:
				old_omega = self.oscillators.omega[i]
				if old_omega < current_u:
					new_omega = 0.001
				else:
					new_omega = math.sqrt(old_omega**2 - current_u**2)
				new_A = old_A * (old_omega / new_omega)**2
				self.oscillators.A[i] = new_A
				self.oscillators.omega[i] = new_omega
		self.u = current_u

	def _calculate_mll_oscillator(self, omega0, gamma):
		omega = np.squeeze(np.array([self.eloss, ] * self.size_q).transpose())
		gammma_over_omega = gamma / omega
		complex_array = np.vectorize(complex)
		z1 = complex_array(1, gammma_over_omega)
		z2 = self._eps_llx(omega, gamma, omega0) - complex(1)
		z3 = self._eps_llx(np.zeros_like(omega), 0, omega0) - complex(1)
		top = z1 * z2
		bottom = complex(1) + complex_array(0, gammma_over_omega) * z2 / z3
		return complex(1) + top / bottom

	def _eps_llx(self, omega, gamma, omega0):
		complex_array = np.vectorize(complex)
		omega_minus_square = complex_array(omega**2 - self.u**2 - gamma**2, 2.0 * omega * gamma)
		r = np.abs(omega_minus_square)
		theta = np.arctan2(omega_minus_square.imag, omega_minus_square.real)
		omega_minus = complex_array(np.sqrt(r) * np.cos(theta / 2.0), np.sqrt(r) * np.sin(theta / 2.0))
		epsilon = np.zeros_like(omega_minus)
		ind_ge = omega_minus.real >= 0
		ind_lt = omega_minus.real < 0
		ge = any(ind_ge.flatten())
		lt = any(ind_lt.flatten())
		if ge:
			epsilon[ind_ge] = self._calculate_linhard_oscillator(omega_minus.real, omega_minus.imag, omega0)[ind_ge]      
		if lt:
			n_dens = omega0**2 / (4.0 * math.pi)
			E_f = 0.5 * (3.0 * math.pi**2 * n_dens)**(2.0 / 3.0)
			v_f = (2 * E_f)**0.5
			deltaSquare = -omega_minus_square / E_f**2
			r = abs(deltaSquare)
			# theta = atan2(deltaSquare.imag, deltaSquare.real)
			theta = np.arctan2(deltaSquare.imag, deltaSquare.real)
			delta = complex_array(np.sqrt(r) * np.cos(theta / 2.0), np.sqrt(r) * np.sin(theta / 2.0))
			QQ = self.q / v_f
			z1 = 2.0 * QQ + QQ**2
			res1 = z1 / delta
			res1 = self._c_arctan(res1)
			z2 = 2.0 * QQ + QQ**2
			res2 = z2 / delta
			res2 = self._c_arctan(res2)
			res1 = res1 + res2
			res2 = res1 * delta
			
			z1 = complex_array( deltaSquare.real + (2 * QQ + QQ**2)**2 , deltaSquare.imag)
			z2 = complex_array( deltaSquare.real + (2 * QQ - QQ**2)**2 , deltaSquare.imag)
			z1 = z1 / z2
			z1 = np.log(z1)
			z2 = deltaSquare * z1
			
			p1 = res2.imag / (2 * QQ**3)
			p2 = z2.imag / (8 * QQ**5)
			p3 = z1.imag / (2 * QQ**3)
			p4 = z1.imag / (8 * QQ)    
			eps_imag = 2 / (math.pi * v_f) * (-p1 + p2 + p3 - p4)
			
			t1 = res2.real / (2 * QQ**3)
			t2 = z2.real / (8 * QQ**5)
			t3 = z1.real / (2 * QQ**3)
			t4 = z1.real / (8 * QQ)
			t5 = 1 / QQ**2 - t1
			eps_real = 1 + 2 / (math.pi * v_f) * (t5 + t2 + t3 - t4)
			
			epsilon[ind_lt] = complex_array(eps_real, eps_imag)[ind_lt]
		return epsilon

	def _c_arctan(self, z):
		complex_array = np.vectorize(complex)
		reres = np.zeros_like(z)
		x = z.real
		y = z.imag
		imres = -1.0 / 4.0 * np.log( (1 - x**2 - y**2)**2 + 4 * x**2) \
				+ 1.0 / 2.0 * np.log( (1 + y)**2 + x**2 )
		reres[x != 0] = math.pi / 4.0 - 0.5 * np.arctan( (1 - x**2 - y**2) / (2.0 * x) )
		reres[np.logical_and(x > 0, x < 0)] = math.pi / 2.0
		return complex_array(reres.real, imres.imag)
	
	def g(self,x):
		return (1-x**2)*np.log(np.abs(1+x)/np.abs(1-x))
	
	def lindhard_epsilon(self,omega,q,omega_pl):
		k_f = np.sqrt( (3*math.pi/4)**(1/3) * omega_pl**(2/3) )
		u = omega[:,np.newaxis,np.newaxis]/(q[:,np.newaxis]*k_f)
		z = q[:,np.newaxis]/(2*k_f)
		chi = np.sqrt(1/(math.pi*k_f))

		f_1 = 1/2 + 1/(8*z)*(self.g(z-u) + self.g(z+u))
		f_2 = math.pi/2*u*np.heaviside(1-z-u,0.5) + math.pi/(8*z)*(1-(z-u)**2)*np.heaviside(1-np.abs(z-u),0.5)*np.heaviside(z+u-1,0.5)
		return 1 + chi**2/z**2*(f_1+f_2*1j)
	
	def calculate_fpa_elf(self, omega_pl_max = 5000):
		self._convert2au()
		elf_pl = np.squeeze(np.zeros((self.eloss.shape[0], self.size_q)))
		elf_se = np.squeeze(np.zeros((self.eloss.shape[0], self.size_q)))
		omega_0 = np.zeros_like(self.eloss)
		omega_pl = linspace(1e-5,omega_pl_max,0.01)/h2ev

		start_time = time.time()
		for k in tqdm(range(self.size_q)):
			epsilon = self.calculate_lindhard_dielectric_function(self.q[k],omega_pl)
			q_m = self._q_minus(omega_pl)
			q_p = self._q_plus(omega_pl)
			se = self._g(omega_pl,self.optical_omega,self.optical_elf) * (-1/epsilon).imag * np.heaviside(q_p - self.q[k],1) * np.heaviside(self.q[k] - q_m,1)
			se[np.isnan(se)] = 0

			for i in range(len(self.eloss)):
				try:
					if self.q[k] == 0:
						interval = [0, self.eloss[i] + 0.1]
					else:
						interval = [0, min(self.eloss[i],np.abs(self.q[k]/2 - self.eloss[i]/self.q[k]))]
					omega_0[i] = optimize.root_scalar(self._epsilon_real_lindhard,args=(self.q[k],self.eloss[i]),bracket=interval, method='brentq').root
				except:
					omega_0[i] = self._find_zero(omega_pl,epsilon.real[i,:],self.q[k],self.eloss[i])

			g_coef = self._g(omega_0,self.optical_omega,self.optical_elf)
			de_eps_real = np.abs(self._calculate_linhard_derivative(self.q[k],omega_0))
			elf_pl[:,k] = g_coef * math.pi/de_eps_real * np.heaviside(self._q_minus(omega_0) - self.q[k],1)
			
			elf_se[:,k] = np.trapz(se,omega_pl)

		elf_pl[np.isnan(elf_pl)] = 0
		print("--- %s seconds ---" % (time.time() - start_time))
		self._convert2ru()
		return elf_pl + elf_se
	
	def calculate_fpa_elf_for_diimfp(self, omega_pl_max = 2000):
		self._convert2au()
		elf_pl = np.squeeze(np.zeros((self.eloss.shape[0], self.size_q)))
		elf_se = np.squeeze(np.zeros((self.eloss.shape[0], self.size_q)))
		omega_0 = np.zeros_like(self.eloss)
		omega_pl = linspace(1e-5,omega_pl_max,0.01)/h2ev

		start_time = time.time()
		for k in range(self.size_q):
			epsilon = self._calculate_lindhard_epsilon(self.q[:,k],omega_pl)
			q_m = self._q_minus(omega_pl)
			q_p = self._q_plus(omega_pl)
			qq = self.q[:,k]
			se = self._g(omega_pl,self.optical_omega,self.optical_elf) * (-1/epsilon).imag * np.heaviside(q_p - qq[:,np.newaxis],1) * np.heaviside(qq[:,np.newaxis] - q_m,1)
			se[np.isnan(se)] = 0

			for i in range(len(self.eloss)):
				try:
					if self.q[i,k] == 0:
						interval = [0, self.eloss[i] + 0.1]
					else:
						interval = [0, min(self.eloss[i],np.abs(self.q[i,k]/2 - self.eloss[i]/self.q[i,k]))]
					omega_0[i] = optimize.root_scalar(self._epsilon_real_lindhard,args=(self.q[i,k],self.eloss[i]),bracket=interval, method='brentq').root
				except:
					omega_0[i] = self._find_zero(omega_pl,epsilon.real[i,:],self.q[i,k],self.eloss[i])

			g_coef = self._g(omega_0,self.optical_omega,self.optical_elf)
			de_eps_real = np.abs(self._calculate_linhard_derivative(self.q[:,k],omega_0))
			elf_pl[:,k] = g_coef * math.pi/de_eps_real * np.heaviside(self._q_minus(omega_0) - self.q[:,k],1)
			
			elf_se[:,k] = np.trapz(se,omega_pl)

		elf_pl[np.isnan(elf_pl)] = 0
		print("--- %s seconds ---" % (time.time() - start_time))
		self._convert2ru()
		return elf_pl + elf_se
	
	def _calculate_linhard_derivative(self,q,omega_pl):
		kf = self._k_f(omega_pl)
		x = 2*self.eloss / kf**2
		z = q / (2*kf)
		u = x / (4*z)
		y_plus = z + u
		y_minus = z - u

		ind = np.logical_not(x > 100*z,z > 100*x)
		de_eps_real = np.where(ind,(np.log(np.abs((y_minus+1)/(y_minus-1))) + np.log(np.abs((y_plus+1)/(y_plus-1))))/(4*kf**(5/2)*math.sqrt(3*math.pi)*z**3),0)
		a = z / x
		de_eps_real = np.where(x > 100*z,16/(kf**(5/2)*math.sqrt(3*math.pi)*x**2)*(-1 - 16*a**2 - 16*a**4*(16 + x**2) - 512/3*a**6*(24 + 5*x**2)),de_eps_real)
		b = x / (z*(z**1-1))
		de_eps_real = np.where(z > 100*x,(np.log(((z+1)/(z-1))**2) + 4*z*b**2*(1 + (1+z**2)*b**2 + 1/3*(3+z**2)*(1+3*z**2)*b**4))/(4*kf**(5/2)*math.sqrt(3*math.pi)*z**3),de_eps_real)
		return de_eps_real
	
	def _g(self,omega_pl,opt_omega,opt_elf):
		return 2/(math.pi*omega_pl)*np.interp(omega_pl*h2ev,opt_omega,opt_elf)
	
	def _q_plus(self,omega_pl):
		kf = self._k_f(omega_pl)
		return kf + np.sqrt(kf**2 + 2*self.eloss[:,np.newaxis])

	def _q_minus(self,omega_pl):
		kf = self._k_f(omega_pl)
		if kf.shape == self.eloss.shape:
			return -kf + np.sqrt(kf**2 + 2*self.eloss)
		else:
			return -kf + np.sqrt(kf**2 + 2*self.eloss[:,np.newaxis])

	def _k_f(self,omega_pl):
		return (3*math.pi/4)**(1/3)*omega_pl**(2/3)
	
	def calculate_lindhard_dielectric_function(self,q,omega_pl):
		kf = self._k_f(omega_pl)
		x = 2*self.eloss[:,np.newaxis] / kf**2
		x[np.isnan(x)] = 0
		z = q / (2*kf)
		z[np.isnan(z)] = 0
		
		if np.all(x == 0):
			eps_real = np.ones_like(x)
			eps_imag = np.zeros_like(x)
		elif all(z == 0):
			eps_real = 1 - 16/(3*kf*math.pi*x**2)
			eps_imag = np.zeros_like(x)
		else:
			u = x / (4*z)
			coef = 1/(8*kf*z**3)
		
			ind = np.logical_not(u < 0.01,u/(z+1) > 100)
			eps_real = np.where(ind,1 + 1/(math.pi*kf*z**2)*(1/2 + 1/(8*z)*(self._f(z - u) + self._f(z+u))),1)
			ind_1 = np.logical_and(x > 0,x < 4*z*(1-z))
			ind_2 = np.logical_and(x > np.abs(4*z*(1-z)),x < 4*z*(1+z))
			eps_imag = np.zeros_like(eps_real)     
			eps_imag = np.where(ind_1,coef*x,eps_imag)
			eps_imag = np.where(ind_2,coef*(1 - (z-u)**2),eps_imag)
			
			eps_real = np.where(u < 0.01,1 + 1/(math.pi*kf*z**2)*(1/2 + 1/(4*z)*((1-z**2-u**2)*np.log(np.abs((z+1)/(z-1))) + (z**2-u**2-1)*2*u**2*z/(z**2-1)**2)),eps_real)
			eps_imag = np.where(u < 0.01,u/(q*z),eps_imag)
		
			eps_real = np.where(u/(z+1) > 100,1 - 16/(3*kf*math.pi*x**2) - 256*z**2/(5*kf*math.pi*x**4) - 256*z**4/(3*kf*math.pi*x**4),eps_real)
			eps_imag = np.where(u/(z+1) > 100,0,eps_imag)
				
			eps_real = np.where(x == 0,1,eps_real)
			eps_imag = np.where(x == 0,0,eps_imag)
			
			ind = np.logical_and(z == 0,x != 0)
			eps_real = np.where(ind,1 - 16/(3*kf*math.pi*x**2),eps_real)
			eps_imag = np.where(ind,0,eps_imag)

		return eps_real + 1j * eps_imag
	
	def _calculate_lindhard_epsilon(self,q,omega_pl):
		kf = self._k_f(omega_pl)
		x = 2*self.eloss[:,np.newaxis] / kf**2
		x[np.isnan(x)] = 0
		z = q[:,np.newaxis] / (2*kf)
		z[np.isnan(z)] = 0
		
		if np.all(x == 0):
			eps_real = np.ones_like(x)
			eps_imag = np.zeros_like(x)
		elif np.all(z == 0):
			eps_real = 1 - 16/(3*kf*math.pi*x**2)
			eps_imag = np.zeros_like(x)
		else:
			u = x / (4*z)
			coef = 1/(8*kf*z**3)
		
			ind = np.logical_not(u < 0.01,u/(z+1) > 100)
			eps_real = np.where(ind,1 + 1/(math.pi*kf*z**2)*(1/2 + 1/(8*z)*(self._f(z - u) + self._f(z+u))),1)
			ind_1 = np.logical_and(x > 0,x < 4*z*(1-z))
			ind_2 = np.logical_and(x > np.abs(4*z*(1-z)),x < 4*z*(1+z))
			eps_imag = np.zeros_like(eps_real)     
			eps_imag = np.where(ind_1,coef*x,eps_imag)
			eps_imag = np.where(ind_2,coef*(1 - (z-u)**2),eps_imag)
			
			eps_real = np.where(u < 0.01,1 + 1/(math.pi*kf*z**2)*(1/2 + 1/(4*z)*((1-z**2-u**2)*np.log(np.abs((z+1)/(z-1))) + (z**2-u**2-1)*2*u**2*z/(z**2-1)**2)),eps_real)
			eps_imag = np.where(u < 0.01,u/(q[:,np.newaxis]*z),eps_imag)
		
			eps_real = np.where(u/(z+1) > 100,1 - 16/(3*kf*math.pi*x**2) - 256*z**2/(5*kf*math.pi*x**4) - 256*z**4/(3*kf*math.pi*x**4),eps_real)
			eps_imag = np.where(u/(z+1) > 100,0,eps_imag)
				
			eps_real = np.where(x == 0,1,eps_real)
			eps_imag = np.where(x == 0,0,eps_imag)
			
			ind = np.logical_and(z == 0,x != 0)
			eps_real = np.where(ind,1 - 16/(3*kf*math.pi*x**2),eps_real)
			eps_imag = np.where(ind,0,eps_imag)

		return eps_real + 1j * eps_imag

	def _f(self,t):
		return np.where(np.abs(t) != 1,(1 - t**2)*np.log(np.abs((t+1)/(t-1))),0)
	
	def _find_zero(self,x,y,q,omega):
		val = 0
		if any(y < 0):
			ind = np.where(y < 0)
			i = ind[0][0]
			# val = x[i-1] + (x[i] - x[i-1])/2
			val = optimize.root_scalar(self._epsilon_real_lindhard,args=(q,omega),bracket=[x[i-1], x[i]], method='brentq').root
		return val
	
	def _f_fzero(self,t):
		if np.abs(t) == 1:
			val = 0
		else:
			val = (1 - t**2)*np.log(np.abs((t+1)/(t-1)))
		return val
	
	def _epsilon_real_lindhard(self,omega_pl,q,omega):
		if omega_pl == 0:
			return 1
		else:
			kf = self._k_f(omega_pl)
			x = 2*omega / kf**2
			z = q / (2*kf)    
			if x == 0:
				eps_real = 1
			elif z == 0:
				eps_real = 1 - 16/(3*kf*math.pi*x**2)
			else:
				u = x / (4*z)
				if u < 0.01:
					eps_real = 1 + 1/(math.pi*kf*z**2)*(1/2 + 1/(4*z)*((1-z**2-u**2)*np.log(np.abs((z+1)/(z-1))) + (z**2-u**2-1)*2*u**2*z/(z**2-1)**2))
				elif u/(z+1) > 100:
					eps_real = 1 - 16/(3*kf*math.pi*x**2) - 256*z**2/(5*kf*math.pi*x**4) - 256*z**4/(3*kf*math.pi*x**4)
				else:
					eps_real = 1 + 1/(math.pi*kf*z**2)*(1/2 + 1/(8*z)*(self._f_fzero(z - u) + self._f_fzero(z + u)))
			
			return eps_real

	def _convert2au(self):
		if self.oscillators.model == 'Drude':
			self.oscillators.A = self.oscillators.A/h2ev/h2ev
		self.oscillators.gamma = self.oscillators.gamma/h2ev
		self.oscillators.omega = self.oscillators.omega/h2ev
		self.e_fermi = self.e_fermi/h2ev
		self.eloss = self.eloss/h2ev
		self.q = self.q*a0
		if (self.e_gap):
			self.e_gap = self.e_gap/h2ev
		if (self.u):
			self.u = self.u/h2ev
		if (self.width_of_the_valence_band):
			self.width_of_the_valence_band = self.width_of_the_valence_band/h2ev

	def _convert2ru(self):
		if self.oscillators.model == 'Drude':
			self.oscillators.A = self.oscillators.A*h2ev*h2ev
		self.oscillators.gamma = self.oscillators.gamma*h2ev
		self.oscillators.omega = self.oscillators.omega*h2ev
		self.e_fermi = self.e_fermi*h2ev
		self.eloss = self.eloss*h2ev
		self.q = self.q/a0
		if (self.e_gap):
			self.e_gap = self.e_gap*h2ev
		if (self.u):
			self.u = self.u*h2ev
		if (self.width_of_the_valence_band):
			self.width_of_the_valence_band = self.width_of_the_valence_band*h2ev

	def evaluate_f_sum(self):
		old_q = self.q
		self.q = 0
		self.extend_to_henke()
		ind = self.eloss_extended_to_henke >= self.e_gap
		fsum = 1 / (2 * math.pi**2 * (self.atomic_density * a0**3)) * np.trapz(self.eloss_extended_to_henke[ind]/h2ev * self.elf_extended_to_henke[ind], self.eloss_extended_to_henke[ind]/h2ev)
		self.q = old_q
		return fsum

	def evaluate_kk_sum(self):
		old_q = self.q
		self.eloss[self.eloss < 1e-5] = 1e-5
		self.q = 0
		if (self.oscillators.model == 'MLL'):
			self.q = 0.01
		self.extend_to_henke()
		div = self.elf_extended_to_henke / self.eloss_extended_to_henke
		div[((div < 0) | (np.isnan(div)))] = 1e-5
		kksum = 2 / math.pi * np.trapz(div, self.eloss_extended_to_henke)
		if self.e_gap != 0:
			if self.static_refractive_index == 0:
				kksum += 1 / self.epsilon.real[0]
			else:
				kksum += 1 / self.static_refractive_index**2
		self.q = old_q
		return kksum

	def extend_to_henke(self):
		self.calculate_elf()
		if self.eloss_henke is None and self.elf_henke is None:
			self.eloss_henke, self.elf_henke = self.mopt()
		ind = self.eloss < 100
		self.eloss_extended_to_henke = np.concatenate((self.eloss[ind], self.eloss_henke))
		self.elf_extended_to_henke = np.concatenate((self.elf[ind], self.elf_henke))

	def mopt(self):
		if self.atomic_density is None:
			raise InputError("Please specify the value of the atomic density atomic_density")
		numberOfElements = len(self.composition.elements)
		energy = linspace(100,30000)
		f1sum = np.zeros_like(energy)
		f2sum = np.zeros_like(energy)

		for i in range(numberOfElements):
			datahenke = self.readhenke(self.xraypath + self.composition.elements[i])
			f1 = np.interp(energy, datahenke[:, 0], datahenke[:, 1])
			f2 = np.interp(energy, datahenke[:, 0], datahenke[:, 2])
			f1sum += f1 * self.composition.indices[i]
			f2sum += f2 * self.composition.indices[i]

		lambda_ = hc/(energy/1000)
		if not self.is_metal:
			f1sum /= np.sum(self.composition.indices)
			f2sum /= np.sum(self.composition.indices)

		n = 1 - self.atomic_density * r0 * 1e10 * lambda_**2 * f1sum/2/math.pi
		k = -self.atomic_density * r0 * 1e10 * lambda_**2 * f2sum/2/math.pi

		eps1 = n**2 - k**2
		eps2 = 2*n*k

		return energy, -eps2/(eps1**2 + eps2**2)

	def readhenke(self, filename):
		henke = np.loadtxt(filename + '.nff', skiprows = 1)
		return henke

	def calculate_elf(self):
		if self.oscillators.model == 'FPA':
			if self.optical_omega is None and self.optical_elf is None:
				raise InputError("Provide optical data: self.optical_omega and self.optical_elf")
			else:
				if len(self.q.shape) > 1:
					elf = self.calculate_fpa_elf_for_diimfp()
				else:
					elf = self.calculate_fpa_elf()
		else:
			elf = (-1/self.epsilon).imag
			elf[np.isnan(elf)] = 1e-5
		self.elf = elf

	def calculate_surface_elf(self):
		if self.epsilon is None or self.epsilon.shape[0] != self.eloss.shape[0]:
			self.calculate_dielectric_function()
		eps_1 = self.epsilon.real
		eps_2 = self.epsilon.imag
		den = (eps_1**2 + eps_1 - eps_2**2)**2 + (2*eps_1*eps_2 + eps_2)**2
		enu = -eps_2*(2*eps_1 + 1)*((eps_1 - 1)**2 - eps_2**2)
		enu += 2*eps_2*(eps_1 - 1)*(eps_1*(eps_1 + 1) - eps_2**2)
		self.surfaceELF = enu/den

	def calculate_optical_constants(self):
		if self.epsilon is None or self.epsilon.shape[0] != self.eloss.shape[0]:
			self.calculate_dielectric_function()
		n_complex = np.sqrt(self.epsilon)
		self.refractive_index = n_complex.real
		self.extinction_coefficient = n_complex.imag

	def calculate_optical_constants_(self):
		if self.epsilon is None or self.epsilon.shape[0] != self.eloss.shape[0]:
			self.calculate_dielectric_function()
		first = np.sqrt(self.epsilon.real ** 2 + self.epsilon.imag ** 2) / 2
		second = self.epsilon.real / 2
		self.refractive_index = np.sqrt(first + second)
		self.extinction_coefficient = np.sqrt(first - second)
	
	def calculateLidiimfp_vs(self, e0, r, alpha, n_q=10, de=0.5):
		old_eloss = self.eloss
		old_q = self.q
		old_e0 = e0

		if (self.e_gap > 0):
			e0 = e0 - self.e_gap
			if old_e0 <= 100:
				eloss = linspace(self.e_gap, e0 - self.width_of_the_valence_band, de)
			elif old_e0 <= 1000:
				range_1 = linspace(self.e_gap, 100, de)
				range_2 = linspace(101, e0 - self.width_of_the_valence_band, 1)
				eloss = np.concatenate((range_1, range_2))
			else:
				range_1 = linspace(self.e_gap, 100, de)
				range_2 = linspace(110, 500, 10)
				range_3 = linspace(600, e0 - self.width_of_the_valence_band, 100)
				eloss = np.concatenate((range_1, range_2, range_3))
		else:
			if old_e0 <= 100:
				eloss = linspace(1e-5, e0 - self.e_fermi, de)
			elif old_e0 <= 1000:
				range_1 = linspace(1e-5, 100, de)
				range_2 = linspace(101, e0 - self.e_fermi, 1)
				eloss = np.concatenate((range_1, range_2))
			else:
				range_1 = linspace(1e-5, 100, de)
				range_2 = linspace(110, 500, 10)
				range_3 = linspace(600, e0 - self.e_fermi, 100)
				eloss = np.concatenate((range_1, range_2, range_3))

		self.eloss = eloss
		rel_coef = ((1 + (e0/h2ev)/(c**2))**2) / (1 + (e0/h2ev)/(2*c**2))

		theta = np.linspace(0, math.pi/2, 10)
		phi = np.linspace(0, 2*math.pi, 10)
		v = math.sqrt(2*e0/h2ev)
		r /= (a0 * np.cos(alpha))
		
		qm = np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) - np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
		qp = np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) + np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
		q = np.linspace(qm, qp, 2**(n_q - 1), axis = 1)
		if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
			q[q == 0] = 0.01

		q_ = np.expand_dims(q,axis=2) * np.sin(theta.reshape((1,1,-1)))
		qsintheta = np.expand_dims(q,axis=2) * np.sin(theta.reshape((1,1,-1)))**2
		omegawave = (self.eloss/h2ev).reshape((-1,1,1,1)) - np.expand_dims(np.expand_dims(q,axis=2) * v * np.sin(theta.reshape((1,1,-1))), axis=3) * np.cos(phi.reshape((1,1,1,-1))) * np.sin(alpha)
		qz = np.expand_dims(q,axis=2) * np.cos(theta.reshape((1,1,-1)))
		coe_fermi = ( np.expand_dims(qsintheta,axis=3) * np.cos(np.expand_dims(qz, axis=3)*r*np.cos(alpha)) * np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha)) ) / ( omegawave**2 + np.expand_dims(q_**2, axis=3) * (v*np.cos(alpha))**2 )

		if (r >= 0):					
			
			self.q = q / a0
			self.calculate_elf()
			integrand = self.elf / q
			integrand[q == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
				integrand[q == 0.01] = 0
			bulk = 1/(math.pi * (e0/h2ev)) * np.trapz( integrand, q, axis = 1 ) * (1/h2ev/a0) * np.heaviside(r, 0.5)

			self.q = q_ / a0
			self.calculate_dielectric_function()

			elf = (-1 / (self.epsilon + 1)).imag
			integrand = np.expand_dims(elf,axis=3)*coe_fermi
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
				integrand[q_ == 0.01] = 0
			surf_outside = 4*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(-r, 0.5)
			
			coe_fermi = ( np.expand_dims(qsintheta,axis=3) * np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha)) ) / ( omegawave**2 + np.expand_dims(q_**2, axis=3) * (v*np.cos(alpha))**2 )
			coe_fermi_ = 2 * np.cos(omegawave * r / v) - np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha))
			
			integrand = np.expand_dims(elf,axis=3)*coe_fermi*coe_fermi_
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
				integrand[q_ == 0.01] = 0
			surf_inside = 4*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(r, 0.5)
			
			elf = (-1 / self.epsilon).imag
			integrand = np.expand_dims(elf,axis=3)*coe_fermi*coe_fermi_
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
				integrand[q_ == 0.01] = 0
			bulk_reduced = 2*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(r, 0.5)

			dsep = rel_coef * ( surf_inside + surf_outside )
			diimfp = rel_coef * bulk
			total = rel_coef * ( bulk - bulk_reduced + surf_inside + surf_outside )
			self.bulk_reduced = bulk_reduced
		else:
			self.q = q_ / a0
			self.calculate_dielectric_function()
			elf = (-1 / (self.epsilon + 1)).imag

			integrand = np.expand_dims(elf,axis=3)*coe_fermi
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
				integrand[q_ == 0.01] = 0
			dsep = rel_coef * 4*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(-r, 0.5)
			diimfp = dsep
			total = dsep
			self.bulk_reduced = np.zeros_like(dsep)
		
		self.diimfp = diimfp
		self.totaldiimfp = total
		
		self.diimfp_e = eloss
		self.dsep = dsep
		self.e0 = old_e0
		self.eloss = old_eloss
		self.q = old_q
		self.sep = np.trapz(self.dsep, eloss, axis=0) 

	def calculateLidiimfp_sv(self, e0, r, alpha, n_q=10, de=0.5):
		old_eloss = self.eloss
		old_q = self.q
		old_e0 = e0

		if (self.e_gap > 0):
			e0 = e0 - self.e_gap
			if old_e0 <= 100:
				eloss = linspace(self.e_gap, e0 - self.width_of_the_valence_band, de)
			elif old_e0 <= 1000:
				range_1 = linspace(self.e_gap, 100, de)
				range_2 = linspace(101, e0 - self.width_of_the_valence_band, 1)
				eloss = np.concatenate((range_1, range_2))
			else:
				range_1 = linspace(self.e_gap, 100, de)
				range_2 = linspace(110, 500, 10)
				range_3 = linspace(600, e0 - self.width_of_the_valence_band, 100)
				eloss = np.concatenate((range_1, range_2, range_3))
		else:
			if old_e0 <= 100:
				eloss = linspace(1e-5, e0 - self.e_fermi, de)
			elif old_e0 <= 1000:
				range_1 = linspace(1e-5, 100, de)
				range_2 = linspace(101, e0 - self.e_fermi, 1)
				eloss = np.concatenate((range_1, range_2))
			else:
				range_1 = linspace(1e-5, 100, de)
				range_2 = linspace(110, 500, 10)
				range_3 = linspace(600, e0 - self.e_fermi, 100)
				eloss = np.concatenate((range_1, range_2, range_3))

		self.eloss = eloss
		rel_coef = ((1 + (e0/h2ev)/(c**2))**2) / (1 + (e0/h2ev)/(2*c**2))

		theta = np.linspace(0, math.pi/2, 10)
		phi = np.linspace(0, 2*math.pi, 10)
		v = math.sqrt(2*e0/h2ev)
		r /= (a0 * np.cos(alpha))
		
		qm = np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) - np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
		qp = np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) + np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
		q = np.linspace(qm, qp, 2**(n_q - 1), axis = 1)
		if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
			q[q == 0] = 0.01

		q_ = np.expand_dims(q,axis=2) * np.sin(theta.reshape((1,1,-1)))
		qsintheta = np.expand_dims(q,axis=2) * np.sin(theta.reshape((1,1,-1)))**2
		omegawave = (self.eloss/h2ev).reshape((-1,1,1,1)) - np.expand_dims(np.expand_dims(q,axis=2) * v * np.sin(theta.reshape((1,1,-1))), axis=3) * np.cos(phi.reshape((1,1,1,-1))) * np.sin(alpha)
	
		if (r >= 0):					
			qz = np.expand_dims(q,axis=2) * np.cos(theta.reshape((1,1,-1)))
			coe_fermi = ( np.expand_dims(qsintheta,axis=3) * np.cos(np.expand_dims(qz, axis=3)*r*np.cos(alpha)) * np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha)) ) / ( omegawave**2 + np.expand_dims(q_**2, axis=3) * (v*np.cos(alpha))**2 )

			self.q = q / a0
			self.calculate_elf()
			integrand = self.elf / q
			integrand[q == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
				integrand[q == 0.01] = 0
			bulk = 1/(math.pi * (e0/h2ev)) * np.trapz( integrand, q, axis = 1 ) * (1/h2ev/a0) * np.heaviside(r, 0.5)

			self.q = q_ / a0
			self.calculate_dielectric_function()

			elf = (-1 / self.epsilon).imag
			integrand = np.expand_dims(elf, axis=3) * coe_fermi
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
				integrand[q_ == 0.01] = 0
			bulk_reduced = 2*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(r, 0.5)

			elf = (-1 / (self.epsilon + 1)).imag
			integrand = np.expand_dims(elf,axis=3)*coe_fermi
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
				integrand[q_ == 0.01] = 0
			surf_inside = 4*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(r, 0.5)
			
			coe_fermi = ( np.expand_dims(qsintheta,axis=3) * np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha)) ) / ( omegawave**2 + np.expand_dims(q_**2, axis=3) * (v*np.cos(alpha))**2 )
			coe_fermi_ = 2 * np.cos(omegawave * r / v) - np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha))
			integrand = np.expand_dims(elf,axis=3)*coe_fermi*coe_fermi_
			integrand[q_ == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
				integrand[q_ == 0.01] = 0
			surf_outside = 4*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2), q) * (1/h2ev/a0) * np.heaviside(-r, 0.5)
			
			dsep = rel_coef * ( surf_inside + surf_outside )
			diimfp = rel_coef * bulk
			total = rel_coef * ( bulk - bulk_reduced + surf_inside + surf_outside )
			self.bulk_reduced = bulk_reduced
		else:
			self.q = q_ / a0
			self.calculate_dielectric_function()
			elf = (-1 / (self.epsilon + 1)).imag

			coe_fermi = ( np.expand_dims(qsintheta,axis=3) * np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha)) ) / ( omegawave**2 + np.expand_dims(q_**2, axis=3) * (v*np.cos(alpha))**2 )
			coe_fermi_ = 2 * np.cos(omegawave * r / v) - np.exp(-np.abs(r)*np.expand_dims(q_,axis=3)*np.cos(alpha))
			integrand = np.expand_dims(elf,axis=3)*coe_fermi*coe_fermi_
			dsep = rel_coef * 4*np.cos(alpha)/math.pi**3 * np.trapz(np.trapz(np.trapz(integrand, phi, axis=3), theta, axis=2),q) * (1/h2ev/a0) * np.heaviside(-r, 0.5)
			diimfp = np.zeros_like(dsep)
			total = dsep
			self.bulk_reduced = np.zeros_like(dsep)
		
		self.diimfp = diimfp
		self.totaldiimfp = total
		
		self.diimfp_e = eloss
		self.dsep = dsep
		self.e0 = old_e0
		self.eloss = old_eloss
		self.q = old_q
		self.sep = np.trapz(self.dsep, eloss, axis=0)


	def calculate_diimfp_ang(self, e0, de = 0.5, is_metal = True):
		if is_metal:
			self.eloss = linspace(1e-5, e0 - self.e_fermi, de)
		else:
			if e0 < 2*self.e_gap + self.width_of_the_valence_band:
				raise InputError("Please specify the value of energy greater than the 2*band gap + the width of the valence band")
			else:
				e0 -= self.e_gap
				self.eloss = linspace(self.e_gap, e0 - self.width_of_the_valence_band, de)

		theta = np.linspace(0,math.pi,100)
		q = 4*e0/h2ev - 2*np.expand_dims(self.eloss/h2ev,axis=1) - 4*np.sqrt(e0/h2ev*(e0/h2ev-np.expand_dims(self.eloss/h2ev,axis=1)))*np.cos(theta.reshape((1,-1)))
		self.q = q
		if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
			self.q[self.q == 0] = 0.01
		self.calculate_elf()
		integrand = self.elf * 1/q * np.sqrt(e0/h2ev*(e0/h2ev-np.expand_dims(self.eloss/h2ev,axis=1))) * np.sin(theta.reshape((1,-1)))
		res = 1/(math.pi * (e0/h2ev)) * 2*math.pi * np.trapz(integrand * np.sin(theta.reshape((1,-1))) * (1 - np.cos(theta.reshape((1,-1)))),theta,axis = 1)
		diimfp = res / (h2ev * a0)
		self.diimfp = diimfp
		self.diimfp_e = self.eloss
		self.diimfp_th = 1/(math.pi * (e0/h2ev)) * integrand
		self.theta = theta
		self.iimfp = res


	def calculate_trimfp(self, energy, de=0.5, is_metal = True):
		if is_metal and self.e_fermi == 0:
			raise InputError("Please specify the value of the Fermi energy e_fermi")
		elif not is_metal and self.e_gap == 0 and self.width_of_the_valence_band == 0:
			raise InputError("Please specify the values of the band gap e_gap and the width of the valence band width_of_the_valence_band")
		imfp = np.zeros_like(energy)
		for i in range(energy.shape[0]):
			print(energy[i])
			self.calculate_diimfp_ang(energy[i], de, is_metal = is_metal)
			imfp[i] = 1/np.trapz(self.iimfp, self.diimfp_e/h2ev)
		self.imfp = imfp*a0
		self.imfp_e = energy


	def calculate_diimfp(self, e0, de = 0.5, nq = 100, normalised = True):
		old_eloss = self.eloss
		old_q = self.q
		old_e0 = e0

		if self.is_metal:
			self.eloss = linspace(1e-5, e0 - self.e_fermi, de)
		else:
			if e0 < 2*self.e_gap + self.width_of_the_valence_band:
				raise InputError("Please specify the value of energy greater than the 2*band gap + the width of the valence band")
			else:
				e0 -= self.e_gap
				self.eloss = linspace(self.e_gap, e0 - self.width_of_the_valence_band, de)
		
		rel_coef = ((1 + (e0/h2ev)/(c**2))**2) / (1 + (e0/h2ev)/(2*c**2))

		if self.oscillators.alpha == 0 and self.oscillators.model != 'Mermin' and self.oscillators.model != 'MLL': # and self.q_dependency is None:
			qm = np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) - np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
			qp = np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) + np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
			self.extend_to_henke()
			int_limits = np.log(qp/qm)
			int_limits[np.isinf(int_limits)] = 1e-5
			interp_elf = np.interp(self.eloss, self.eloss_extended_to_henke, self.elf_extended_to_henke)
			interp_elf[np.isnan(interp_elf)] = 1e-5
			iimfp = rel_coef * 1/(math.pi*(e0/h2ev)) * interp_elf * int_limits
			diimfp = iimfp / (h2ev * a0)
		else:
			qm = np.log( np.sqrt( e0/h2ev * ( 2 + e0/h2ev/(c**2) ) ) - np.sqrt( ( e0/h2ev - self.eloss/h2ev ) * ( 2 + (e0/h2ev - self.eloss/h2ev)/(c**2) ) ) )
			qp = np.log( np.sqrt( e0/h2ev * ( 2 + e0/h2ev/(c**2) ) ) + np.sqrt( ( e0/h2ev - self.eloss/h2ev ) * ( 2 + (e0/h2ev - self.eloss/h2ev)/(c**2) ) ) )
			q = np.linspace(qm, qp, nq, axis = 1)
			self.q = np.exp(q)/a0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
				q[q == 0] = 0.01
			self.q = np.exp(q)/a0
			self.calculate_elf()
			integrand = self.elf
			integrand[self.q == 0] = 1e-5
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
				integrand[self.q == 0.01] = 1e-5
			iimfp = rel_coef * 1/(math.pi * (e0/h2ev)) * np.trapz( integrand, q, axis = 1 )
			diimfp = iimfp / (h2ev * a0)

		diimfp[np.isnan(diimfp)] = 1e-5
		iimfp[np.isnan(iimfp)] = 1e-5
		self.q = old_q

		if normalised:
			diimfp = diimfp / np.trapz(diimfp, self.eloss)
		
		self.diimfp = diimfp
		self.iimfp = iimfp
		self.diimfp_e = self.eloss
		self.e0 = old_e0
		self.eloss = old_eloss


	def diimfp_interp_fpa(self,e,rbs,nq = 100,de = 0.5):
		if self.is_metal:
			self.diimfp_e = linspace(1e-5, e - self.e_fermi, de)
		else:
			if e < 2*self.e_gap + self.width_of_the_valence_band:
				raise InputError("Please specify the value of energy greater than the 2*band gap + the width of the valence band")
			else:
				e0 = (e - self.e_gap)/h2ev
				self.diimfp_e = linspace(self.e_gap, e - self.e_gap - self.width_of_the_valence_band, de)
				
		omega = self.diimfp_e/h2ev
		c = 137.036
		qm = np.log( np.sqrt( e0 * ( 2 + e0/(c**2) ) ) - np.sqrt( ( e0 - omega ) * ( 2 + (e0 - omega)/(c**2) ) ) )
		qp = np.log( np.sqrt( e0 * ( 2 + e0/(c**2) ) ) + np.sqrt( ( e0 - omega ) * ( 2 + (e0 - omega)/(c**2) ) ) )
		q = np.linspace(qm, qp, nq, axis = 1)
		q_ru = np.exp(q)/a0
		q_ru[np.isnan(q_ru)] = 0
		
		eloss_ru = np.transpose(np.tile(self.diimfp_e,(nq,1)))
		elf_interp = rbs(eloss_ru,q_ru, grid=False)
		
		rel_coef = ((1 + e0/(c**2))**2) / (1 + e0/(2*c**2))
		iimfp = rel_coef * 1/(math.pi*e0) * np.trapz( elf_interp, q, axis = 1 )
		diimfp = iimfp / (h2ev * a0) # now in 1/(eV A)
		diimfp[np.isnan(diimfp)] = 1e-5
		iimfp[np.isnan(iimfp)] = 1e-5
		self.diimfp = diimfp
		self.iimfp = iimfp


	def diimfp_interp_fpa_mermin(self,fpa_eloss,e,rbs,nq = 100,de = 0.5):
		if self.is_metal:		
			e0 = e/h2ev
			self.diimfp_e = linspace(1e-5, e - self.e_fermi, de)
		else:
			if e < 2*self.e_gap + self.width_of_the_valence_band:
				raise InputError("Please specify the value of energy greater than the 2*band gap + the width of the valence band")
			else:
				e0 = (e - self.e_gap)/h2ev
				self.diimfp_e = linspace(self.e_gap, e - self.e_gap - self.width_of_the_valence_band, de)

		omega = self.diimfp_e/h2ev
		c = 137.036
		qm = np.log( np.sqrt( e0 * ( 2 + e0/(c**2) ) ) - np.sqrt( ( e0 - omega ) * ( 2 + (e0 - omega)/(c**2) ) ) )
		qp = np.log( np.sqrt( e0 * ( 2 + e0/(c**2) ) ) + np.sqrt( ( e0 - omega ) * ( 2 + (e0 - omega)/(c**2) ) ) )
		q = np.linspace(qm, qp, nq, axis = 1)
		q_ru = np.exp(q)/a0
		q_ru[np.isnan(q_ru)] = 0
		
		eloss_ru = np.transpose(np.tile(self.diimfp_e,(nq,1)))
		elf_interp = rbs(eloss_ru,q_ru, grid=False)

		ind = eloss_ru < fpa_eloss[-1]
		if np.any(ind):
			self.q = q_ru
			self.eloss = self.diimfp_e
			self.calculate_elf()
			elf_interp[ind] = self.elf[ind]
		
		rel_coef = ((1 + e0/(c**2))**2) / (1 + e0/(2*c**2))
		iimfp = rel_coef * 1/(math.pi*e0) * np.trapz( elf_interp, q, axis = 1 )
		diimfp = iimfp / (h2ev * a0) # now in 1/(eV A)
		diimfp[np.isnan(diimfp)] = 1e-5
		iimfp[np.isnan(iimfp)] = 1e-5
		self.diimfp = diimfp
		self.iimfp = iimfp


	def calculate_imfp(self, energy, de=0.5, nq=10):
		if self.is_metal and self.e_fermi == 0:
			raise InputError("Please specify the value of the Fermi energy e_fermi")
		elif not self.is_metal and self.e_gap == 0 and self.width_of_the_valence_band == 0:
			raise InputError("Please specify the values of the band gap e_gap and the width of the valence band width_of_the_valence_band")
		imfp = np.zeros_like(energy)
		for i in range(energy.shape[0]):
			self.calculate_diimfp(energy[i], de, nq, normalised = False)
			imfp[i] = 1/np.trapz(self.iimfp, self.diimfp_e/h2ev)
		self.imfp = imfp*a0
		self.imfp_e = energy


	def scat_rate_ph(self,E, loss, eps_zero, eps_inf):
		dephe = loss/E
		sq_e = np.sqrt(1-dephe)
		kbt = 9.445e-4 # room temp kbt in Ha
		n = 1.0/(np.exp((loss/h2ev)/kbt) - 1)
		return (eps_zero-eps_inf)/(eps_zero*eps_inf)*dephe*(n+1.0)/2*np.log((1.0+sq_e)/(1.0-sq_e))/a0


	def calculate_phonon_imfp(self, energy, eps_zero, e1, e2):
		T = 300.0 # K
		k_B = 8.617e-5 # eV/K
		eps_inf = self.static_refractive_index**2
		iimfp_phonon = []
		for loss in [e1, e2]:
			loss /= h2ev
			iimfp_plus = []
			iimfp_minus = []
			for e in energy:
				e /= h2ev
				N_lo = 1.0/(np.exp(loss/(k_B*T/h2ev))-1.0) # no need to convert to a.u.

				# //=========== phonon creation ===============
				ln_plus = (1.0 + np.sqrt(np.abs(1.0 - loss/e))) / (1.0 - np.sqrt(np.abs(1.0 - loss/e)))
				term_plus = (N_lo + 1.0)*(1.0/eps_inf - 1.0/eps_zero)*loss/(2.0*e)
				iimfp_plus.append(term_plus*np.log(ln_plus))

				# //=========== phonon annihilation ============
				ln_minus = (1.0 + np.sqrt(np.abs(1.0 + loss/e))) / (-1.0 + np.sqrt(np.abs(1.0 + loss/e)))
				term_minus = N_lo*(1.0/eps_inf - 1.0/eps_zero)*loss/(2.0*e)
				iimfp_minus.append(term_minus*np.log(ln_minus))

			iimfp_phonon.append(np.array(iimfp_minus) + np.array(iimfp_plus))

		self.imfp_phonon = 1/(np.array(iimfp_phonon[0]) + np.array(iimfp_phonon[1]))*a0
		self.imfp_phonon_plus = 1/np.array(iimfp_plus)*a0


	def _get_sigma(self, lines, line, pattern):
		start = lines[line].find(pattern) + len(pattern)
		end = lines[line].find(' cm**2')
		return float(lines[line][start:end])*1e16

	def calculate_elastic_properties(self,e0,mnucl,melec,mexch):
		self.e0 = e0
		sumweights = 0.0
		rmuf = 1

		for i in range(len(self.composition.elements)):
			sumweights += self.composition.indices[i]

			if self.composition.atomic_numbers[i] in gas_z:
				rmuf = 0

			fd = open('lub.in','w+')

			fd.write(f'IZ      {self.composition.atomic_numbers[i]}         atomic number                               [none]\n')
			fd.write(f'MNUCL   {mnucl}          rho_n (1=P, 2=U, 3=F, 4=Uu)                  [  3]\n')
			fd.write(f'NELEC   {self.composition.atomic_numbers[i]}         number of bound electrons                    [ IZ]\n')
			fd.write(f'MELEC   {melec}         rho_e (1=TFM, 2=TFD, 3=DHFS, 4=DF, 5=file)   [  4]\n')
			fd.write(f'MUFFIN  {rmuf}          0=free atom, 1=muffin-tin model              [  0]\n')
			# fd.write('RMUF    0          muffin-tin radius (cm)                  [measured]\n')
			fd.write('IELEC  -1          -1=electron, +1=positron                     [ -1]\n')
			fd.write(f'MEXCH   {mexch}          V_ex (0=none, 1=FM, 2=TF, 3=RT)              [  1]\n')
			fd.write(f'EV      {round(self.e0)}     kinetic energy (eV)                         [none]\n')

			fd.close()

			subprocess.run('/Users/olgaridzel/olgaridzel/dev/elsepa-2020/elsepa < lub.in',shell=True,capture_output=True)

			with open('dcs_' + '{:1.3e}'.format(round(self.e0)).replace('.','p').replace('+0','0') + '.dat','r') as fd:
				lines = fd.readlines()
				result = [ line for line in lines if "Total elastic cross section = " in line]
				self.sigma_el = float(re.findall(r"\d+\.\d+[E][+]\d+", result[0])[0])
				print("sigma_el = ",self.sigma_el)
				self.emfp = 1/(self.sigma_el*a0**2*self.atomic_density)
				# result = [ line for line in lines if "1st transport cross section = " in line]
				# self.sigma_tr = float(re.findall(r"\d+\.\d+[E][-]\d+", result[0])[0])
			
			data = np.loadtxt('dcs_' + '{:1.3e}'.format(round(self.e0)).replace('.','p').replace('+0','0') + '.dat', comments="#")
			if i == 0:
				self.decs = np.zeros_like(data[:,0])
				self.decs_a = np.zeros_like(data[:,0])
			self.decs_theta = np.deg2rad(data[:,0])
			self.decs_mu = data[:,1]
			self.decs_a += data[:,3]*self.composition.indices[i]
			self.decs += data[:,2]*self.composition.indices[i]
			
		self.decs_a /= sumweights
		self.decs /= sumweights
		self.norm_decs = self.decs / np.trapz(self.decs, self.decs_theta)

	def write_optical_data(self):
		self.calculate_elf()
		self.calculate_optical_constants()
		d = dict(E=np.round(self.eloss,2),n=np.round(self.re_fermiractive_index,2),k=np.round(self.extinction_coe_fermificient,2),eps1=np.round(self.epsilon.real,2), eps2=np.round(self.epsilon.imag,2), elf=np.round(self.elf,2))
		df = pd.DataFrame.from_dict(d, orient='index').transpose().fillna('')
		with open(f'{self.name}_{self.oscillators.model}_table_optical_data.csv', 'w') as tf:
			tf.write(df.to_csv(index=False))

class exp_data:
	def __init__(self):
		self.x_elf = []
		self.y_elf = []
		self.x_ndiimfp = []
		self.y_ndiimfp = []

class SAReflection:

	_theta_i = 0
	_theta_o = 0
	mu_i = np.cos(_theta_i)
	mu_o = np.cos(_theta_o)
	_n_in = 10
	_n_leg = 500
	_de = 0.5
	_n_q = 10
	mu_mesh = None
	energy_mesh = None
	x_l = None
	_tau = 0
	isinf = True
	_phi = 0
	n_spherical_harmonics = 1
	surface_done = False

	def __init__(self, material, e0):
		if not isinstance(material, Material):
			raise InputError("The material must be of the type Material")
		if e0 == 0:
			raise InputError("e0 must be non-zero")
		self.material = material
		self.e_primary = e0
		self._calculateAngularMesh()
		self._calculateLegendreCoefficients()

	@property
	def theta_i(self):
		return self._theta_i

	@theta_i.setter
	def theta_i(self, theta_i):
		self._theta_i = np.deg2rad(theta_i)
		self.mu_i = np.cos(self._theta_i)

	@property
	def theta_o(self):
		return self._theta_o

	@theta_o.setter
	def theta_o(self, theta_o):
		self._theta_o = np.deg2rad(theta_o)
		self.mu_o = np.cos(self._theta_o)

	@property
	def phi(self):
		return self._phi

	@phi.setter
	def phi(self, phi):
		self._phi = phi
		if phi > 0:
			self.psi = np.rad2deg(math.acos(self.mu_i * self.mu_o + math.sqrt(1 - self.mu_i**2) * math.sqrt(1 - self.mu_o**2) * math.cos(np.deg2rad(phi))))
			self.n_spherical_harmonics = math.floor(0.41 * self.psi - 6e-3 * self.psi**2 + 1.75e-10 * self.psi**6 + 0.8)
		else:
			self.n_spherical_harmonics = 1

	@property
	def tau(self):
		return self._tau

	@tau.setter
	def tau(self, tau):
		self._tau = tau
		if tau > 0:
			self.isinf = False
		else:
			self.isinf = True

	@property
	def e_primary(self):
		return self._e0

	@e_primary.setter
	def e_primary(self, e0):
		self._e0 = e0
		self.materialcalculate_elastic_properties(e0)
		self.material.calculate_imfp(np.array([e0]))

	@property
	def de(self):
		return self._de

	@de.setter
	def de(self, de):
		self._de = de

	@property
	def n_q(self):
		return self._n_q

	@n_q.setter
	def n_q(self, n_q):
		self._n_q = n_q

	@property
	def n_in(self):
		return self._n_in

	@n_in.setter
	def n_in(self, n_in):
		self._n_in = n_in

	@property
	def n_legendre(self):
		return self._n_leg

	@n_legendre.setter
	def n_legendre(self, n_leg):
		self._n_leg = n_leg
		self._calculateAngularMesh()
		self._calculateLegendreCoefficients()

	def _calculateAngularMesh(self):
		[x, s] = special.roots_legendre(self.n_legendre - 1)
		x = (np.concatenate((x, np.array([1]))) + 1) / 2
		self.mu_mesh = x[::-1]

	def _calculateLegendreCoefficients(self):
		N = max(2000, math.floor(self.n_legendre * 5))
		self.x_l = np.zeros(self.n_legendre + 1)
		mu = np.cos(np.deg2rad(self.material.decs))
		[x,s] = special.roots_legendre(N)
		decs = np.interp(x, mu[::-1], self.material.decs[::-1])
		decs /= np.trapz(decs, x)

		for l in range(self.n_legendre + 1):
			self.x_l[l] = np.trapz(decs * special.lpmv(0, l, x), x)

	def calculate(self):
		if self.isinf:
			self.material.TMFP = 1 / (1/self.materialimfp[0] + 1/self.material.emfp)
		else:
			imfp = np.mean((self.materialimfp_s_i, self.materialimfp_s_o))
			self.material.TMFP = 1 / (1/imfp + 1/self.material.emfp)
		self.material.albedo = self.material.TMFP / self.material.emfp
		norm = 1/2/math.pi

		mm = 1 / self.mu_mesh + 1 / self.mu_mesh.reshape((-1,1))
		B = norm / mm
		if not self.isinf: 
			e0 = special.exp1(self.tau * mm)
				
		norm_leg = (2 * linspace(0, self.n_legendre, 1) + 1) / 2
		self.R_l = np.zeros((self.mu_mesh.shape[0], self.mu_mesh.shape[0], self.n_in, self.n_legendre + 1))

		for l in range(self.n_legendre + 1):
			self.R_l[:,:,0,l] = -np.log(1 - self.material.albedo * self.x_l[l]) * B
			if not self.isinf:
				self.R_l[:,:,0,l] -= ( special.exp1(self.tau * (1 - self.material.albedo * self.x_l[l]) * mm) - e0 ) * B
			for k in range(1, self.n_in):
				if self.isinf:
					self.R_l[:,:,k,l] = (1 - self.material.albedo)**k / k * (1/(1 - self.material.albedo * self.x_l[l])**k - 1) * B
				else:
					self.R_l[:,:,k,l] = (1 - self.material.albedo)**k / k * ( special.gammainc(k, self.tau * (1 - self.material.albedo * self.x_l[l]) * mm) \
					/  (1 - self.material.albedo * self.x_l[l])**k - special.gammainc(k, self.tau * mm)) * B
				
		self.R_m = np.zeros((self.mu_mesh.shape[0], self.mu_mesh.shape[0], self.n_in, self.n_spherical_harmonics))

		for m in range(self.n_spherical_harmonics):
			for l in range(self.n_legendre + 1):
				P = special.lpmv(m, l, self.mu_mesh)
				PlP = P * norm_leg[l + m] * P.reshape((-1,1))
				for k in range(self.n_in):
					self.R_m[:,:,k,m] += (-1)**(l + m) * self.R_l[:,:,k,l] * PlP

		self.R_m[np.isnan(self.R_m)] = 0
		self.R_m[np.isinf(self.R_m)] = 0
		print('Yield calculated')

	def calculateAngularDistribution(self):
		self.angular_distribution = np.zeros((self.mu_mesh.shape[0], self.n_in))
		ind = self.mu_mesh == self.mu_i

		for k in range(self.n_in):
			if any(ind):
				self.angular_distribution[:, k] = self.R_m[ind,:,k,0] * 2 * math.pi
			else:
				f = interp1d(self.mu_mesh, self.R_m[:,:,k,0])
				self.angular_distribution[:, k] = f(self.mu_i) * 2 * math.pi
		
		print('Angular Distribution calculated')

	def calculate_partial_intensities(self):
		self.partial_intensities = np.zeros(self.n_in)
		ind = self.mu_mesh == self.mu_o

		for k in range(self.n_in):
			if any(ind):
				self.partial_intensities[k] = self.angular_distribution[ind,k]
			else:
				f = interp1d(self.mu_mesh, self.angular_distribution[:,k])
				self.partial_intensities[k] = f(self.mu_o)

		print('Partial Intensities calculated')

	def calculate_diimfp_convolutions(self):
		if self.material.diimfp is None:
			raise Error("The diimfp has not been calculated yet.")
		de = self.material.diimfp_e[2] - self.material.diimfp_e[1]
		convolutions = np.zeros((self.material.diimfp.size, self.n_in))
		convolutions[0, 0] = 1/de    
		
		for k in range(1, self.n_in):
			convolutions[:,k] = conv(convolutions[:,k-1], self.material.diimfp, de)
		
		return convolutions

	def calculateDsepConvolutions(self):
		if self.material.dsep is None:
			raise Error("The DSEP has not been calculated yet.")
		de = self.material.diimfp_e[2] - self.material.diimfp_e[1]
		convolutions = np.zeros((self.material.dsep.size, self.n_in))
		convolutions[0, 0] = 1/de    
		
		for k in range(1, self.n_in):
			convolutions[:,k] = conv(convolutions[:,k-1], self.material.dsep, de)
		
		return convolutions

	@staticmethod
	def fitElasticPeak(x_exp, y_exp):
		ind = np.logical_and(x_exp > x_exp[np.argmax(y_exp)] - 3, x_exp < x_exp[np.argmax(y_exp)] + 3)
		x = x_exp[ind]
		y = y_exp[ind]
		popt, pcov = optimize.curve_fit(gauss, x, y, [np.max(y), x[np.argmax(y)], 1])
		return popt

	def calculate_energy_distributionBulk(self, solid_angle=0):
		convs_b = self.calculate_diimfp_convolutions()
		if solid_angle > 0:
			if solid_angle == 90:
				self.energy_distribution_b = np.trapz(-2*math.pi * (np.mat(convs_b) * np.mat(self.angular_distribution.T)), self.mu_mesh, axis=1)
			else:
				solid_angle = np.deg2rad(solid_angle)
				mu_ = np.cos(np.linspace(np.arccos(self.mu_o), np.arccos(self.mu_o) + solid_angle, 100))
				f = interp1d(self.mu_mesh, self.angular_distribution, axis=0)
				self.energy_distribution_b = np.trapz(-2*math.pi * (np.mat(convs_b) * np.mat(f(mu_).T)), mu_, axis=1)
		else:
			# self.energy_distribution_b = np.sum(convs_b*np.squeeze(self.partial_intensities / self.partial_intensities[0]),axis=1)
			self.energy_distribution_b = np.sum(convs_b*np.squeeze(self.partial_intensities),axis=1)

	def prepareDSEP(self, depth):
		i = 0
		for r in depth:
			self.material.calculateLidiimfp_vs(self.e_primary, r, self.theta_i, self.n_q, self.de)

			if i == 0:
				dsep_i = np.zeros((len(depth),) + self.material.dsep.shape)
				dsep_o = np.zeros((len(depth),) + self.material.dsep.shape)
				total_i = np.zeros((len(depth),) + self.material.totaldiimfp.shape)
				total_o = np.zeros((len(depth),) + self.material.totaldiimfp.shape)

			dsep_i[i] = self.material.dsep - self.material.bulk_reduced
			total_i[i] = self.material.totaldiimfp

			self.material.calculateLidiimfp_sv(self.e_primary, r, self.theta_o, self.n_q, self.de)
			dsep_o[i] = self.material.dsep - self.material.bulk_reduced
			total_o[i] = self.material.totaldiimfp
			
			i += 1
		return dsep_i, dsep_o, total_i, total_o

	def calculate_energy_distributionSurface(self):
		depth = np.array([-6,-4,-2,0,2,4,6])
		dsep_i, dsep_o, total_i, total_o = self.prepareDSEP(depth)
		
		self.material.diimfp /= np.trapz(self.material.diimfp, self.material.diimfp_e)
		self.material.diimfp[np.isnan(self.material.diimfp)] = 1e-5
		
		self.material.SEP_i = np.trapz( np.trapz(total_i, self.material.diimfp_e,axis=1)[depth<=0], depth[depth<=0] / np.cos(self.theta_i))
		self.material.dsep_i = np.trapz(dsep_i, depth / np.cos(self.theta_i), axis=0)
		self.material.totaldiimfp_i = np.trapz(total_i, depth / np.cos(self.theta_i), axis=0)
		self.materialimfp_s_i = 1 / np.trapz(self.material.dsep_i, self.material.diimfp_e)
		self.materialimfp_t_i = 1 / np.trapz(self.material.totaldiimfp_i, self.material.diimfp_e)
		self.material.dsep = self.material.dsep_i / np.trapz(self.material.dsep_i, self.material.diimfp_e)
		self.material.dsep[np.isnan(self.material.dsep)] = 1e-5	
		# self.convs_s = self.calculateDsepConvolutions()
		self.material.totaldiimfp = self.material.totaldiimfp_i / np.trapz(self.material.totaldiimfp_i, self.material.diimfp_e)
		self.material.totaldiimfp[np.isnan(self.material.totaldiimfp)] = 1e-5	
		self.convs_s = self.calculateDsepConvolutions()
		self.partial_intensities_s_i = stats.poisson(self.material.SEP_i).pmf(range(self.n_in))
		# self.energy_distribution_s_i = np.sum(self.convs_s*np.squeeze(self.partial_intensities_s_i / self.partial_intensities_s_i[0]),axis=1)
		self.energy_distribution_s_i = np.sum(self.convs_s*np.squeeze(self.partial_intensities_s_i),axis=1)
		
		self.material.SEP_o = np.trapz( np.trapz(total_o, self.material.diimfp_e,axis=1)[depth<=0], depth[depth<=0] / np.cos(self.theta_o))	
		self.material.dsep_o = np.trapz(dsep_o, depth / np.cos(self.theta_o), axis=0)
		self.material.totaldiimfp_o = np.trapz(total_o, depth / np.cos(self.theta_o), axis=0)
		self.materialimfp_s_o = 1 / np.trapz(self.material.dsep_o, self.material.diimfp_e)
		self.materialimfp_t_o = 1 / np.trapz(self.material.totaldiimfp_o, self.material.diimfp_e)
		self.material.dsep = self.material.dsep_o / np.trapz(self.material.dsep_o, self.material.diimfp_e)
		self.material.dsep[np.isnan(self.material.dsep)] = 1e-5
		self.material.totaldiimfp = self.material.totaldiimfp_o / np.trapz(self.material.totaldiimfp_o, self.material.diimfp_e)
		self.material.totaldiimfp[np.isnan(self.material.totaldiimfp)] = 1e-5
		self.convs_s = self.calculateDsepConvolutions()
		self.partial_intensities_s_o = stats.poisson(self.material.SEP_o).pmf(range(self.n_in))
		# self.energy_distribution_s_o = np.sum(self.convs_s*np.squeeze(self.partial_intensities_s_o / self.partial_intensities_s_o[0]),axis=1)
		self.energy_distribution_s_o = np.sum(self.convs_s*np.squeeze(self.partial_intensities_s_o),axis=1)

		self.surface_done = True

	def calculate_energy_distribution(self, solid_angle=0):

		if not self.surface_done:
			self.calculate_energy_distributionSurface()

		self.calculate_energy_distributionBulk(solid_angle)

		self.tau = 6
		self.calculate()
		self.calculate_partial_intensities()
		# self.rs = np.sum(self.convs_s*np.squeeze(self.partial_intensities / self.partial_intensities[0]),axis=1)
		self.rs = np.sum(self.convs_s*np.squeeze(self.partial_intensities),axis=1)

		self.sb = conv(self.energy_distribution_s_i, self.energy_distribution_b, self.de)
		self.energy_distribution = self.rs + conv(self.sb, self.energy_distribution_s_o, self.de)
		
	def convolveGauss(self, x_exp, y_exp):
		extra = np.linspace(-10,-self.de, round(10/self.de))
		self.spectrum_e = np.concatenate((extra, self.material.diimfp_e))
		sbs = np.concatenate((np.zeros_like(extra), self.energy_distribution))
		coefs = self.fitElasticPeak(x_exp, y_exp)
		gaussian = gauss(linspace(-10,10,self.de), coefs[0], 0, coefs[2])
		self.spectrum = conv(sbs, gaussian, self.de)
