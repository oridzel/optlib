import subprocess
import numpy as np
import math
import copy
import os
import time
from tqdm import tqdm
import nlopt
from optlib.optical import Material
from optlib.constants import *

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
		osc_min_alpha = 0
		
		if self.material.oscillators.model == 'Drude':
			osc_max_A = np.ones_like(self.material.oscillators.A) * 2e3
		else:
			osc_max_A = np.ones_like(self.material.oscillators.A)

		osc_max_gamma = np.ones_like(self.material.oscillators.gamma) * 100
		osc_max_omega = np.ones_like(self.material.oscillators.omega) * 500
		osc_max_alpha = 1

		if self.material.oscillators.model == 'MLL':
			osc_min_U = 0.0
			osc_max_U = 10.0
			self.lb = np.append( np.hstack((osc_min_A,osc_min_gamma,osc_min_omega)), osc_min_U )
			self.ub = np.append( np.hstack((osc_max_A,osc_max_gamma,osc_max_omega)), osc_max_U )
		else:
			self.lb = np.append( np.hstack((osc_min_A,osc_min_gamma,osc_min_omega)), osc_min_alpha )
			self.ub = np.append( np.hstack((osc_max_A,osc_max_gamma,osc_max_omega)), osc_max_alpha )
			

	def run_optimisation(self, diimfp_coef, elf_coef, maxeval = 1000, xtol_rel = 1e-6, is_global = False):
		print('Starting optimisation...')
		self.bar = tqdm(total=maxeval)
		self.count = 0
		self.diimfp_coef = diimfp_coef
		self.elf_coef = elf_coef

		if is_global:
			opt_local = nlopt.opt(nlopt.LN_COBYLA, len(self.struct2vec(self.material)))
			opt_local.set_maxeval(maxeval)
			opt_local.set_xtol_rel(xtol_rel)
			opt_local.set_ftol_rel = 1e-15

			opt = nlopt.opt(nlopt.AUGLAG, len(self.struct2vec(self.material)))
			opt.set_local_optimizer(opt_local)
			if diimfp_coef == 0:
				opt.set_min_objective(self.objective_function_elf)
			elif elf_coef == 0:
				opt.set_min_objective(self.objective_function_ndiimfp)
			else:
				opt.set_min_objective(self.objective_function)
			self.set_bounds()
			opt.set_lower_bounds(self.lb)
			opt.set_upper_bounds(self.ub)

			if self.material.use_henke_for_ne:
				if self.material.eloss_henke is None and self.material.elf_henke is None:
					self.material.eloss_henke, self.material.elf_henke = self.material.mopt()
				self.material.electron_density_henke = self.material.atomic_density * self.material.Z * a0 ** 3 - \
					1 / (2 * math.pi**2) * np.trapz(self.material.eloss_henke / h2ev * self.material.elf_henke, self.material.eloss_henke / h2ev)
				opt.add_inequality_constraint(self.constraint_function_henke)
				if self.material.oscillators.model == 'Drude':
					opt.add_inequality_constraint(self.constraint_function_kk)
				else:
					opt.add_inequality_constraint(self.constraint_function_refind_henke)
			else:
				opt.add_inequality_constraint(self.constraint_function)
				if self.material.use_kk_constraint:
					if self.material.oscillators.model == 'Drude':
						opt.add_inequality_constraint(self.constraint_function_kk)
					else:
						opt.add_inequality_constraint(self.constraint_function_refind)

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
			opt.set_ftol_rel = 1e-15
			if diimfp_coef == 0:
				opt.set_min_objective(self.objective_function_elf)
			elif elf_coef == 0:
				opt.set_min_objective(self.objective_function_ndiimfp)
			else:
				opt.set_min_objective(self.objective_function)
			self.set_bounds()
			opt.set_lower_bounds(self.lb)
			opt.set_upper_bounds(self.ub)

			if self.material.use_henke_for_ne:
				if self.material.eloss_henke is None and self.material.elf_henke is None:
					self.material.eloss_henke, self.material.elf_henke = self.material.mopt()
				self.material.electron_density_henke = self.material.atomic_density * self.material.Z * a0 ** 3 - \
					1 / (2 * math.pi**2) * np.trapz(self.material.eloss_henke / h2ev * self.material.elf_henke, self.material.eloss_henke / h2ev)
				opt.add_inequality_constraint(self.constraint_function_henke)
				if self.material.oscillators.model == 'Drude':
					opt.add_inequality_constraint(self.constraint_function_kk)
				else:
					opt.add_inequality_constraint(self.constraint_function_refind_henke)
			else:
				opt.add_inequality_constraint(self.constraint_function)
				if self.material.use_kk_constraint:
					if self.material.oscillators.model == 'Drude':
						opt.add_inequality_constraint(self.constraint_function_kk)
					else:
						opt.add_inequality_constraint(self.constraint_function_refind)

			x = opt.optimize(self.struct2vec(self.material))
			self.bar.close()
			print(f"found minimum after {self.count} evaluations")
			print("minimum value = ", opt.last_optimum_value(), "%")
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

			if self.material.use_henke_for_ne:
				if self.material.eloss_henke is None and self.material.elf_henke is None:
					self.material.eloss_henke, self.material.elf_henke = self.material.mopt()
				self.material.electron_density_henke = self.material.atomic_density * self.material.Z * a0 ** 3 - \
					1 / (2 * math.pi**2) * np.trapz(self.material.eloss_henke / h2ev * self.material.elf_henke, self.material.eloss_henke / h2ev)
				opt.add_inequality_constraint(self.constraint_function_henke)
				if self.material.use_kk_constraint and self.material.oscillators.model != 'Drude':
					opt.add_inequality_constraint(self.constraint_function_refind_henke)
			else:
				opt.add_inequality_constraint(self.constraint_function)
				if self.material.use_kk_constraint and self.material.oscillators.model != 'Drude':
					opt.add_inequality_constraint(self.constraint_function_refind_henke)

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

			if self.material.use_henke_for_ne:
				if self.material.eloss_henke is None and self.material.elf_henke is None:
					self.material.eloss_henke, self.material.elf_henke = self.material.mopt()
				self.material.electron_density_henke = self.material.atomic_density * self.material.Z * a0 ** 3 - \
					1 / (2 * math.pi**2) * np.trapz(self.material.eloss_henke / h2ev * self.material.elf_henke, self.material.eloss_henke / h2ev)
				opt.add_inequality_constraint(self.constraint_function_henke)
				if self.material.use_kk_constraint and self.material.oscillators.model != 'Drude':
					opt.add_inequality_constraint(self.constraint_function_refind_henke)
			else:
				opt.add_inequality_constraint(self.constraint_function)
				if self.material.use_kk_constraint and self.material.oscillators.model != 'Drude':
					opt.add_inequality_constraint(self.constraint_function_refind_henke)

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
		elif self.material.oscillators.model == 'Mermin' or self.material.oscillators.model == 'Drude':
			vec = np.hstack((osc_struct.oscillators.A,osc_struct.oscillators.gamma,osc_struct.oscillators.omega,osc_struct.oscillators.alpha))
		# else:
		# 	vec = np.append( np.hstack((osc_struct.oscillators.A,osc_struct.oscillators.gamma,osc_struct.oscillators.omega)), osc_struct.oscillators.alpha )
		return vec

	def vec2struct(self, osc_vec):
		if self.material.oscillators.model == 'Mermin' or self.material.oscillators.model == 'Drude':
			oscillators = np.split(osc_vec[:],4)
		else:
			oscillators = np.split(osc_vec[0:-1],3)
		material = copy.deepcopy(self.material)
		material.oscillators.A = oscillators[0]
		material.oscillators.gamma = oscillators[1]
		material.oscillators.omega = oscillators[2]
		material.oscillators.alpha = oscillators[3]

		if self.material.oscillators.model == 'MLL':
			material.u = osc_vec[-1]
		# elif self.material.oscillators.model != 'Mermin' and self.material.oscillators.model != 'Drude':
		# 	material.oscillators.alpha = osc_vec[-1]
			
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
		return chi_squared

	def objective_function_ndiimfp(self, osc_vec, grad):
		self.count += 1
		material = self.vec2struct(osc_vec)
		material.calculate_diimfp(self.e0, self.de, self.n_q)
		diimfp_interp = np.interp(self.exp_data.x_ndiimfp, material.diimfp_e, material.diimfp)
		ind = self.exp_data.y_ndiimfp > 0
		rms = np.sum((self.exp_data.y_ndiimfp - diimfp_interp)**2 / self.exp_data.x_ndiimfp.size)
		# rms = 100*np.sqrt(np.sum(((diimfp_interp[ind]-self.exp_data.y_ndiimfp[ind])/self.exp_data.y_ndiimfp[ind])**2) / self.exp_data.x_ndiimfp.size)

		if grad.size > 0:
			grad = np.array([0, 0.5/rms])

		self.bar.update(1)
		return rms

	def objective_function_elf(self, osc_vec, grad):
		self.count += 1
		material = self.vec2struct(osc_vec)
		material.calculate_elf()
		elf_interp = np.interp(self.exp_data.x_elf, material.eloss, material.elf)
		ind = self.exp_data.y_elf > 0
		# chi_squared = np.sum((self.exp_data.y_elf - elf_interp)**2 / self.exp_data.x_elf.size)
		rms = 100*np.sqrt(np.sum(((elf_interp[ind]-self.exp_data.y_elf[ind])/self.exp_data.y_elf[ind])**2) / self.exp_data.x_elf.size)

		if grad.size > 0:
			grad = np.array([0, 0.5/rms])

		self.bar.update(1)
		return rms

	def objective_function(self, osc_vec, grad):
		self.count += 1
		material = self.vec2struct(osc_vec)
		material.calculate_diimfp(self.e0, self.de, self.n_q)
		diimfp_interp = np.interp(self.exp_data.x_ndiimfp, material.diimfp_e, material.diimfp)

		material.calculate_elf()
		elf_interp = np.interp(self.exp_data.x_elf, material.eloss, material.elf)
		ind_ndiimfp = self.exp_data.y_ndiimfp > 0
		ind_elf = self.exp_data.y_elf > 0

		rms = self.diimfp_coef*np.sqrt(np.sum(((self.exp_data.y_ndiimfp[ind_ndiimfp] - diimfp_interp[ind_ndiimfp])/self.exp_data.y_ndiimfp[ind_ndiimfp])**2 / len(self.exp_data.y_ndiimfp[ind_ndiimfp]))) + \
				self.elf_coef*np.sqrt(np.sum(((self.exp_data.y_elf[ind_elf] - elf_interp[ind_elf])/self.exp_data.y_elf[ind_elf])**2) / len(self.exp_data.y_elf[ind_elf]))
		
		if grad.size > 0:
			grad = np.array([0, 0.5/rms])

		self.bar.update(1)
		return rms

	def constraint_function(self, osc_vec, grad):
		material = self.vec2struct(osc_vec)
		material._convert2au()
		if material.oscillators.model == 'Drude':
			cf = material.electron_density * wpc / np.sum(material.oscillators.A)
		else:
			cf = (1 - 1 / material.static_refractive_index ** 2) / np.sum(material.oscillators.A)
		val = np.fabs(cf - 1)

		if grad.size > 0:
			grad = np.array([0, 0.5 / val])

		return val
	
	def constraint_function_kk(self, osc_vec, grad):
		material = self.vec2struct(osc_vec)
		material._convert2au()
		cf = ( material.static_refractive_index**2 - material.oscillators.eps_b ) / np.sum(material.oscillators.A/material.oscillators.omega ** 2)
		val = np.fabs(cf - 1)

		if grad.size > 0:
			grad = np.array([0, 0.5 / val])

		return val

	def constraint_function_refind(self, osc_vec, grad):
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
			cf = (1 - 1 / self.material.static_refractive_index ** 2) / np.sum(material.oscillators.A)

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
