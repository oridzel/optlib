from cmath import isinf
import subprocess
import numpy as np
import math
import matplotlib.pyplot as plt
import nlopt
import copy
import pandas as pd
import os
from scipy import special, interpolate, sparse, stats, optimize
import time
from tqdm import tqdm

hc = 12.3981756608  # planck constant times velocity of light keV Angstr
r0 = 2.8179403227e-15
h2ev = 27.21184  # Hartree, converts Hartree to eV
a0 = 0.529177  # Bohr Radius in Angstroem
wpc = 4*math.pi * a0**3
avogadro = 6.02217e23
c = 137.036

def wavelength2energy(wavelength):
	# wavelength in Angstroem
    return hc / wavelength  * 1e3


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
	def __init__(self, elements, indices):
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
		self._q = q
		self.use_kk_constraint = False
		self.use_henke_for_ne = False
		self.electron_density_henke = 0
		self.use_kk_relation = False

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
				kk_sum = np.trapz(self.eloss[ind] * epsilon_imag[ind][0] / (self.eloss[ind] ** 2 - omega ** 2), self.eloss[ind])
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
			raise InputError("Invalid model name. The valid model names are: Drude, DrudeLindhard, Mermin and MLL")

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
			eps_imag[self.eloss <= self.e_gap] = 1e-5
		if self.use_kk_relation:
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
		else:
			omega = np.expand_dims(self.eloss, axis=tuple(range(1,self.q.ndim)))

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
		z1_1[np.isnan(z1_1)] = machine_eps
		
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
		fsum = 1 / (2 * math.pi**2 * (self.atomic_density * a0**3)) * np.trapz(self.eloss_extended_to_henke/h2ev * self.elf_extended_to_henke, self.eloss_extended_to_henke/h2ev)
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
		self.re_fermiractive_index = n_complex.real
		self.extinction_coe_fermificient = n_complex.imag

	def calculate_optical_constants_(self):
		if self.epsilon is None or self.epsilon.shape[0] != self.eloss.shape[0]:
			self.calculate_dielectric_function()
		first = np.sqrt(self.epsilon.real ** 2 + self.epsilon.imag ** 2) / 2
		second = self.epsilon.real / 2
		self.re_fermiractive_index = np.sqrt(first + second)
		self.extinction_coe_fermificient = np.sqrt(first - second)
	
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
		
		q_minus = np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) - np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
		q_plus = np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) + np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
		q = np.linspace(q_minus, q_plus, 2**(n_q - 1), axis = 1)
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
		
		q_minus = np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) - np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
		q_plus = np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) + np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
		q = np.linspace(q_minus, q_plus, 2**(n_q - 1), axis = 1)
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


	def calculate_diimfp(self, e0, de = 0.5, nq = 10, normalised = True):
		old_eloss = self.eloss
		old_q = self.q
		old_e0 = e0

		if (self.e_gap > 0):
			if e0 > self.e_gap + self.width_of_the_valence_band:
				e0 -= self.e_gap
			else:
				raise InputError("Please specify the value of energy greater than the band gap + the width of the valence band")
			if old_e0 <= 100 + self.width_of_the_valence_band:
				eloss = linspace(self.e_gap, e0 - self.width_of_the_valence_band, 0.1)
			elif old_e0 <= 1000 + self.width_of_the_valence_band:
				range_1 = linspace(self.e_gap, 100, de)
				range_2 = linspace(101, e0 - self.width_of_the_valence_band, 1)
				eloss = np.concatenate((range_1, range_2))
			else:
				range_1 = linspace(self.e_gap, 100, de)
				range_2 = linspace(110, 500, 10)
				range_3 = linspace(600, e0 - self.width_of_the_valence_band, 100)
				eloss = np.concatenate((range_1, range_2, range_3))
		else:
			if old_e0 <= 100 + self.e_fermi:
				eloss = linspace(1e-5, e0 - self.e_fermi, de)
			elif old_e0 <= 1000 + self.e_fermi:
				range_1 = linspace(1e-5, 100, de)
				range_2 = linspace(101, e0 - self.e_fermi, 1)
				eloss = np.concatenate((range_1, range_2))
			else:
				range_1 = linspace(1e-5, 100, de)
				range_2 = linspace(110, 500, 10)
				range_3 = linspace(600, e0 - self.e_fermi, 100)
				eloss = np.concatenate((range_1, range_2, range_3))
				# eloss = linspace(1e-5, e0 - self.e_fermi, de)

		self.eloss = eloss

		rel_coef = ((1 + (e0/h2ev)/(c**2))**2) / (1 + (e0/h2ev)/(2*c**2))

		if self.oscillators.alpha == 0 and self.oscillators.model != 'Mermin' and self.oscillators.model != 'MLL' and self.q_dependency is None:
			q_minus = np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) - np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
			q_plus =  np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) + np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
			self.extend_to_henke()
			int_limits = np.log(q_plus/q_minus)
			int_limits[np.isinf(int_limits)] = machine_eps
			interp_elf = np.interp(eloss, self.eloss_extended_to_henke, self.elf_extended_to_henke)
			interp_elf[np.isnan(interp_elf)] = machine_eps
			iimfp = rel_coef * 1/(math.pi*(e0/h2ev)) * interp_elf * int_limits
			diimfp = iimfp / (h2ev * a0)
		else:
			q_minus = np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) - np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
			q_plus =  np.sqrt(e0/h2ev * (2 + e0/h2ev/(c**2))) + np.sqrt((e0/h2ev - self.eloss/h2ev) * (2 + (e0/h2ev - self.eloss/h2ev)/(c**2)))
			q = np.linspace(q_minus, q_plus, 2**(nq - 1), axis = 1)
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
				q[q == 0] = 0.01
			self.q = q / a0
			self.calculate_elf()
			integrand = self.elf / q
			integrand[q == 0] = 0
			if (self.oscillators.model == 'Mermin' or self.oscillators.model == 'MLL'):
				integrand[q == 0.01] = 0
			iimfp = rel_coef * 1/(math.pi * (e0/h2ev)) * np.trapz( integrand, q, axis = 1 )
			diimfp = iimfp / (h2ev * a0)

		diimfp[np.isnan(diimfp)] = 1e-5
		iimfp[np.isnan(iimfp)] = 1e-5
		self.eloss = old_eloss
		self.q = old_q

		if normalised:
			diimfp = diimfp / np.trapz(diimfp, eloss)
		
		self.diimfp = diimfp
		self.iimfp = iimfp
		self.diimfp_e = eloss
		self.e0 = old_e0


	def calculate_imfp(self, energy, is_metal = True):
		if is_metal and self.e_fermi == 0:
			raise InputError("Please specify the value of the Fermi energy e_fermi")
		elif not is_metal and self.e_gap == 0 and self.width_of_the_valence_band == 0:
			raise InputError("Please specify the values of the band gap e_gap and the width of the valence band width_of_the_valence_band")
		imfp = np.zeros_like(energy)
		for i in range(energy.shape[0]):
			if not is_metal and energy[i] <= self.e_gap + self.width_of_the_valence_band:
				imfp[i] = np.Inf
			else:
				self.calculate_diimfp(energy[i], 9, normalised = False)
				# imfp[i] = 1/np.trapz(self.diimfp, self.diimfp_e)
				imfp[i] = 1/np.trapz(self.iimfp, self.diimfp_e/h2ev)
		self.imfp = imfp*a0
		self.imfp_e = energy


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


	def _get_sigma(self, lines, line, pattern):
		start = lines[line].find(pattern) + len(pattern)
		end = lines[line].find(' cm**2')
		return float(lines[line][start:end])*1e16

	def calculate_elastic_properties(self, e0):
		self.e0 = e0
		fd = open('lub.in','w+')

		fd.write(f'IZ      {self.Z}         atomic number                               [none]\n')
		fd.write('MNUCL   3          rho_n (1=P, 2=U, 3=F, 4=Uu)                  [  3]\n')
		fd.write(f'NELEC   {self.Z}         number of bound electrons                    [ IZ]\n')
		fd.write('MELEC   3          rho_e (1=TFM, 2=TFD, 3=DHFS, 4=DF, 5=file)   [  4]\n')
		fd.write('MUFFIN  0          0=free atom, 1=muffin-tin model              [  0]\n')
		fd.write('RMUF    0          muffin-tin radius (cm)                  [measured]\n')
		fd.write('IELEC  -1          -1=electron, +1=positron                     [ -1]\n')
		fd.write('MEXCH   1          V_ex (0=none, 1=FM, 2=TF, 3=RT)              [  1]\n')
		fd.write('MCPOL   2          V_cp (0=none, 1=B, 2=LDA)                    [  0]\n')
		fd.write('VPOLA  -1          atomic polarizability (cm^3)            [measured]\n')
		fd.write('VPOLB  -1          b_pol parameter                          [default]\n')
		fd.write('MABS    1          W_abs (0=none, 1=LDA-I, 2=LDA-II)            [  0]\n')
		fd.write('VABSA   2.0        absorption-potential strength, Aabs      [default]\n')
		fd.write('VABSD  -1.0        energy gap deLTA (eV)                    [default]\n')
		fd.write('IHe_fermi    2          high-E factorization (0=no, 1=yes, 2=Born)   [  1]\n')
		fd.write(f'EV      {round(self.e0)}     kinetic energy (eV)                         [none]\n')

		fd.close()

		# output = os.system('/Users/olgaridzel/Research/ESCal/src/MaterialDatabase/Data/Elsepa/elsepa-2020/elsepa-2020 < lub.in')
		x = subprocess.run('/Users/olgaridzel/Research/ESCal/src/MaterialDatabase/Data/Elsepa/elsepa-2020/elsepa-2020 < lub.in',shell=True,capture_output=True)

		with open('dcs_' + '{:1.3e}'.format(round(self.e0)).replace('.','p').replace('+0','0') + '.dat','r') as fd:
			if e0 < 100:
				self.sigma_el = self._get_sigma(fd.readlines(), 35, 'Total elastic cross section = ')
			else:
				self.sigma_el = self._get_sigma(fd.readlines(), 32, 'Total elastic cross section = ')
			self.emfp = 1/(self.sigma_el*self.atomic_density)
			# sigma_tr_1 = get_sigma(lines, 33, '1st transport cross section = ')
			# sigma_tr_2 = get_sigma(lines, 34, '2nd transport cross section = ')
		
		data = np.loadtxt('dcs_' + '{:1.3e}'.format(round(self.e0)).replace('.','p').replace('+0','0') + '.dat', skiprows=44)
		self.decs = data[:,0]
		self.decs_mu = data[:,1]
		self.decs = data[:,2]
		self.Ndecs = self.decs / np.trapz(self.decs, self.decs_mu)

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
				f = interpolate.interp1d(self.mu_mesh, self.R_m[:,:,k,0])
				self.angular_distribution[:, k] = f(self.mu_i) * 2 * math.pi
		
		print('Angular Distribution calculated')

	def calculate_partial_intensities(self):
		self.partial_intensities = np.zeros(self.n_in)
		ind = self.mu_mesh == self.mu_o

		for k in range(self.n_in):
			if any(ind):
				self.partial_intensities[k] = self.angular_distribution[ind,k]
			else:
				f = interpolate.interp1d(self.mu_mesh, self.angular_distribution[:,k])
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
				f = interpolate.interp1d(self.mu_mesh, self.angular_distribution, axis=0)
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

class OptFit:

	def __init__(self, material, exp_data, e0, de = 0.5, n_q = 10):
		if not isinstance(material, Material):
			raise InputError("The material must be of the type Material")
		if e0 == 0:
			raise InputError("e0 must be non-zero")
		self.material = material
		self.exp_data = exp_data
		self.e0 = e0
		self.de = de
		self.n_q = n_q
		self.count = 0
		
	def set_bounds(self):
		osc_min_A = np.ones_like(self.material.oscillators.A) * 1e-10
		osc_min_gamma = np.ones_like(self.material.oscillators.gamma) * 0.025
		osc_min_omega = np.ones_like(self.material.oscillators.omega) * self.material.e_gap
		
		if self.material.oscillators.model == 'Drude':
			osc_max_A = np.ones_like(self.material.oscillators.A) * 2e3
		else:
			osc_max_A = np.ones_like(self.material.oscillators.A)

		osc_max_gamma = np.ones_like(self.material.oscillators.gamma) * 100
		osc_max_omega = np.ones_like(self.material.oscillators.omega) * 500		

		if self.material.oscillators.model == 'MLL':
			osc_min_U = 0.0
			osc_max_U = 10.0
			self.lb = np.append( np.hstack((osc_min_A,osc_min_gamma,osc_min_omega)), osc_min_U )
			self.ub = np.append( np.hstack((osc_max_A,osc_max_gamma,osc_max_omega)), osc_max_U )
		elif self.material.oscillators.model == 'Mermin':
			self.lb = np.hstack((osc_min_A,osc_min_gamma,osc_min_omega))
			self.ub = np.hstack((osc_max_A,osc_max_gamma,osc_max_omega))
		else:
			osc_min_alpha = 0.0
			osc_max_alpha = 1.0
			self.lb = np.append( np.hstack((osc_min_A,osc_min_gamma,osc_min_omega)), osc_min_alpha )
			self.ub = np.append( np.hstack((osc_max_A,osc_max_gamma,osc_max_omega)), osc_max_alpha )

	def run_optimisation(self, diimfp_coef, elf_coef, maxeval = 1000, xtol_rel = 1e-6, is_global = False):
		print('Starting optimisation...')
		self.count = 0
		self.diimfp_coef = diimfp_coef
		self.elf_coef = elf_coef

		if is_global:
			opt_local = nlopt.opt(nlopt.LN_COBYLA, len(self.struct2vec(self.material)))
			opt_local.set_maxeval(maxeval)
			opt_local.set_xtol_rel(xtol_rel)
			opt_local.set_ftol_rel = 1e-20;

			opt = nlopt.opt(nlopt.AUGLAG, len(self.struct2vec(self.material)))
			opt.set_local_optimizer(opt_local)
			opt.set_min_objective(self.objective_function)
			self.set_bounds()
			opt.set_lower_bounds(self.lb)
			opt.set_upper_bounds(self.ub)

			# if self.material.use_henke_for_ne:
			# 	if self.material.eloss_henke is None and self.material.elf_henke is None:
			# 		self.material.eloss_henke, self.material.elf_henke = self.material.mopt()
			# 	self.material.electron_density_henke = self.material.atomic_density * self.material.Z * a0 ** 3 - \
			# 		1 / (2 * math.pi**2) * np.trapz(self.material.eloss_henke / h2ev * self.material.elf_henke, self.material.eloss_henke / h2ev)
			# 	print(f"Electron density = {self.material.electron_density_henke / a0 ** 3}")
			# 	opt.add_equality_constraint(self.constraint_function_henke)
			# 	if self.material.use_kk_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_re_fermiind_henke)
			# else:
			# 	opt.add_equality_constraint(self.constraint_function)
			# 	if self.material.use_kk_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_re_fermiind)

			opt.set_maxeval(maxeval)
			opt.set_xtol_rel(xtol_rel)

			x = opt.optimize(self.struct2vec(self.material))
			print(f"found minimum after {self.count} evaluations")
			print("minimum value = ", opt.last_optimum_value())
			print("result code = ", opt.last_optimize_result())

		else:
			opt = nlopt.opt(nlopt.LN_COBYLA, len(self.struct2vec(self.material)))
			opt.set_maxeval(maxeval)
			opt.set_xtol_rel(xtol_rel)
			opt.set_ftol_rel = 1e-20;
			if diimfp_coef == 0:
				opt.set_min_objective(self.objective_function_elf)
			elif elf_coef == 0:
				opt.set_min_objective(self.objective_function_ndiimfp)
			else:
				opt.set_min_objective(self.objective_function)
			self.set_bounds()
			opt.set_lower_bounds(self.lb)
			opt.set_upper_bounds(self.ub)

			# if self.material.use_henke_for_ne:
			# 	if self.material.eloss_henke is None and self.material.elf_henke is None:
			# 		self.material.eloss_henke, self.material.elf_henke = self.material.mopt()
			# 	self.material.electron_density_henke = self.material.atomic_density * self.material.Z * a0 ** 3 - \
			# 		1 / (2 * math.pi**2) * np.trapz(self.material.eloss_henke / h2ev * self.material.elf_henke, self.material.eloss_henke / h2ev)
			# 	print(f"Electron density = {self.material.electron_density_henke / a0 ** 3}")
			# 	opt.add_equality_constraint(self.constraint_function_henke)
			# 	if self.material.use_kk_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_re_fermiind_henke)
			# else:
			# 	opt.add_equality_constraint(self.constraint_function)
			# 	if self.material.use_kk_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_re_fermiind)

			x = opt.optimize(self.struct2vec(self.material))
			print(f"found minimum after {self.count} evaluations")
			print("minimum value = ", opt.last_optimum_value())
			print("result code = ", opt.last_optimize_result())

		return x

	def run_spectrum_optimisation(self, mu_i, mu_o, n_in, maxeval=1000, xtol_rel=1e-6, is_global=False):
		print('Starting spec optimisation...')
		self.bar = tqdm(total=maxeval)
		self.count = 0
		self.mu_i = mu_i
		self.mu_o = mu_o
		self.n_in = n_in

		if is_global:
			opt_local = nlopt.opt(nlopt.LN_COBYLA, len(self.struct2vec(self.material)))
			opt_local.set_maxeval(maxeval)
			opt_local.set_xtol_rel(xtol_rel)
			opt_local.set_ftol_rel = 1e-20;

			opt = nlopt.opt(nlopt.AUGLAG, len(self.struct2vec(self.material)))
			opt.set_local_optimizer(opt_local)
			opt.set_min_objective(self.objective_function_spec)
			self.set_bounds()
			opt.set_lower_bounds(self.lb)
			opt.set_upper_bounds(self.ub)

			# if self.material.use_henke_for_ne:
			# 	if self.material.eloss_henke is None and self.material.elf_henke is None:
			# 		self.material.eloss_henke, self.material.elf_henke = self.material.mopt()
			# 	self.material.electron_density_henke = self.material.atomic_density * self.material.Z * a0 ** 3 - \
			# 		1 / (2 * math.pi**2) * np.trapz(self.material.eloss_henke / h2ev * self.material.elf_henke, self.material.eloss_henke / h2ev)
			# 	print(f"Electron density = {self.material.electron_density_henke / a0 ** 3}")
			# 	opt.add_equality_constraint(self.constraint_function_henke)
			# 	if self.material.use_kk_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_re_fermiind_henke)
			# else:
			# 	opt.add_equality_constraint(self.constraint_function)
			# 	if self.material.use_kk_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_re_fermiind)

			opt.set_maxeval(maxeval)
			opt.set_xtol_rel(xtol_rel)

			self.material.calculate_elastic_properties(self.e0)
			self.material.calculateLegendreCoefficients(200)
			x = opt.optimize(self.struct2vec(self.material))
			print(f"found minimum after {self.count} evaluations")
			print("minimum value = ", opt.last_optimum_value())
			print("result code = ", opt.last_optimize_result())

		else:
			opt = nlopt.opt(nlopt.LN_COBYLA, len(self.struct2vec(self.material)))
			opt.set_maxeval(maxeval)
			opt.set_xtol_rel(xtol_rel)
			opt.set_ftol_rel = 1e-20
			opt.set_min_objective(self.objective_function_spec)
			self.set_bounds()
			opt.set_lower_bounds(self.lb)
			opt.set_upper_bounds(self.ub)

			# if self.material.use_henke_for_ne:
			# 	if self.material.eloss_henke is None and self.material.elf_henke is None:
			# 		self.material.eloss_henke, self.material.elf_henke = self.material.mopt()
			# 	self.material.electron_density_henke = self.material.atomic_density * self.material.Z * a0 ** 3 - \
			# 		1 / (2 * math.pi**2) * np.trapz(self.material.eloss_henke / h2ev * self.material.elf_henke, self.material.eloss_henke / h2ev)
			# 	print(f"Electron density = {self.material.electron_density_henke / a0 ** 3}")
			# 	opt.add_equality_constraint(self.constraint_function_henke)
			# 	if self.material.use_kk_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_re_fermiind_henke)
			# else:
			# 	opt.add_equality_constraint(self.constraint_function)
			# 	if self.material.use_kk_constraint and self.material.oscillators.model != 'Drude':
			# 		opt.add_equality_constraint(self.constraint_function_re_fermiind)

			self.material.calculate_elastic_properties(self.e0)
			self.material.calculateLegendreCoefficients(200)
			x = opt.optimize(self.struct2vec(self.material))
			self.bar.close()
			print(f"found minimum after {self.count} evaluations")
			print("minimum value = ", opt.last_optimum_value())
			print("result code = ", opt.last_optimize_result())

		return x

	def struct2vec(self, osc_struct):
		if osc_struct.oscillators.model == 'MLL':
			vec = np.append( np.hstack((osc_struct.oscillators.A,osc_struct.oscillators.gamma,osc_struct.oscillators.omega)), osc_struct.u )
		elif self.material.oscillators.model == 'Mermin':
			vec = np.hstack((osc_struct.oscillators.A,osc_struct.oscillators.gamma,osc_struct.oscillators.omega))
		else:
			vec = np.append( np.hstack((osc_struct.oscillators.A,osc_struct.oscillators.gamma,osc_struct.oscillators.omega)), osc_struct.oscillators.alpha )
		return vec

	def vec2struct(self, osc_vec):
		if self.material.oscillators.model == 'Mermin':
			oscillators = np.split(osc_vec[:],3)
		else:
			oscillators = np.split(osc_vec[0:-1],3)
		material = copy.deepcopy(self.material)
		material.oscillators.A = oscillators[0]
		material.oscillators.gamma = oscillators[1]
		material.oscillators.omega = oscillators[2]

		if self.material.oscillators.model == 'MLL':
			material.u = osc_vec[-1]
		elif self.material.oscillators.model != 'Mermin':
			material.oscillators.alpha = osc_vec[-1]
			
		return material

	def objective_function_spec(self, osc_vec, grad):
		self.count += 1
		alpha_i = np.rad2de_gap(np.arccos(self.mu_i))
		alpha_o = np.rad2de_gap(np.arccos(self.mu_o))
		ind = np.logical_and(self.exp_data.x_spec > self.exp_data.x_spec[np.argmax(self.exp_data.y_spec)] - 3, self.exp_data.x_spec < self.exp_data.x_spec[np.argmax(self.exp_data.y_spec)] + 3)
		x = self.exp_data.x_spec[ind]
		y = self.exp_data.y_spec[ind]
		exp_area = np.trapz(y, x)
		material = self.vec2struct(osc_vec)
		material.calculate(self.e0, self.n_in, 200, self.mu_i, self.mu_o)
		material.calculate_energy_distribution(self.e0, alpha_i, alpha_o, self.n_in, self.exp_data.x_spec, self.exp_data.y_spec, self.de, self.n_q)

		ind = self.exp_data.x_spec < self.exp_data.x_spec[np.argmax(self.exp_data.y_spec)] - 5
		spec_interp = np.interp(self.e0 - self.exp_data.x_spec[ind], material.spectrum_e - material.spectrum_e[np.argmax(material.spectrum)], material.spectrum)
		chi_squared = np.sum((self.exp_data.y_spec[ind] / exp_area - spec_interp / exp_area)**2 / self.exp_data.x_spec.size)
		# chi_squared = np.sum((self.exp_data.y_spec / exp_area - spec_interp / exp_area)**2)

		if grad.size > 0:
			grad = np.array([0, 0.5/chi_squared])

		self.bar.update(1)
		time.sleep(1)
		return chi_squared

	def objective_function_ndiimfp(self, osc_vec, grad):
		self.count += 1
		material = self.vec2struct(osc_vec)
		material.calculate_diimfp(self.e0, self.de, self.n_q)
		diimfp_interp = np.interp(self.exp_data.x_ndiimfp, material.diimfp_e, material.diimfp)
		chi_squared = np.sum((self.exp_data.y_ndiimfp - diimfp_interp)**2 / self.exp_data.x_ndiimfp.size)

		if grad.size > 0:
			grad = np.array([0, 0.5/chi_squared])
		return chi_squared

	def objective_function_elf(self, osc_vec, grad):
		self.count += 1
		material = self.vec2struct(osc_vec)
		material.calculate_elf()
		elf_interp = np.interp(self.exp_data.x_elf, material.eloss, material.elf)
		chi_squared = np.sum((self.exp_data.y_elf - elf_interp)**2 / self.exp_data.x_elf.size)

		if grad.size > 0:
			grad = np.array([0, 0.5/chi_squared])
		return chi_squared

	def objective_function(self, osc_vec, grad):
		self.count += 1
		material = self.vec2struct(osc_vec)
		material.calculate_diimfp(self.e0, self.de, self.n_q)
		diimfp_interp = np.interp(self.exp_data.x_ndiimfp, material.diimfp_e, material.diimfp)

		if material.oscillators.alpha == 0:
			elf_interp = np.interp(self.exp_data.x_elf, material.eloss_extended_to_henke, material.elf_extended_to_henke)
		else:
			elf_interp = np.interp(self.exp_data.x_elf, material.diimfp_e, material.elf[:,0])
		ind_ndiimfp = self.exp_data.y_ndiimfp >= 0
		ind_elf = self.exp_data.y_elf >= 0

		chi_squared = self.diimfp_coef*np.sqrt(np.sum((self.exp_data.y_ndiimfp[ind_ndiimfp] - diimfp_interp[ind_ndiimfp])**2 / len(self.exp_data.y_ndiimfp[ind_ndiimfp]))) + \
						self.elf_coef*np.sqrt(np.sum((self.exp_data.y_elf[ind_elf] - elf_interp[ind_elf])**2) / len(self.exp_data.y_elf[ind_elf]))

		if grad.size > 0:
			grad = np.array([0, 0.5/chi_squared])
		return chi_squared

	def constraint_function(self, osc_vec, grad):
		material = self.vec2struct(osc_vec)
		material._convert2au()
		if material.oscillators.model == 'Drude':
			cf = material.electron_density * wpc / np.sum(material.oscillators.A)
		else:
			cf = (1 - 1 / material.static_re_fermiractive_index ** 2) / np.sum(material.oscillators.A)
		val = np.fabs(cf - 1)

		if grad.size > 0:
			grad = np.array([0, 0.5 / val])

		return val

	def constraint_function_re_fermiind(self, osc_vec, grad):
		material = self.vec2struct(osc_vec)
		material._convert2au()
		if material.oscillators.model == 'Drude':
			cf = 1
		else:
			bethe_sum = np.sum((math.pi / 2) * material.oscillators.A * material.oscillators.omega ** 2)
			bethe_value = 2 * math.pi ** 2 * material.electron_density * a0 ** 3
			cf = np.sqrt(bethe_sum / bethe_value)

		val = np.fabs(cf - 1)

		if grad.size > 0:
			grad = np.array([0, 0.5 / val])

		return val

	def constraint_function_henke(self, osc_vec, grad):
		global iteration
		material = self.vec2struct(osc_vec)
		material._convert2au()

		if material.oscillators.model == 'Drude':
			cf = self.material.electron_density_henke * 4 * math.pi / np.sum(material.oscillators.A)
		else:
			cf = (1 - 1 / self.material.static_re_fermiractive_index ** 2) / np.sum(material.oscillators.A)

		val = np.fabs(cf - 1)
		if grad.size > 0:
			grad = np.array([0, 0.5 / val])

		return val

	def constraint_function_refind_henke(self, osc_vec, grad):
		global iteration
		material = self.vec2struct(osc_vec)
		material._convert2au()
		if material.oscillators.model == 'Drude':
			cf = 1
		else:
			bethe_sum = np.sum((math.pi / 2) * material.oscillators.A * material.oscillators.omega ** 2)
			bethe_value = 2 * math.pi ** 2 * self.material.electron_density_henke
			cf = np.sqrt(bethe_sum / bethe_value)
		val = np.fabs(cf - 1)
		if grad.size > 0:
			grad = np.array([0, 0.5 / val])

		return val