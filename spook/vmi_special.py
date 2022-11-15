import numpy as np
from .lin_solve import SpookLinSolve
from .quad_program import SpookL1
import scipy.sparse as sps
from .utils import laplacian_square_S
from .contraction_utils import adaptive_contraction
import os
# from .utils import phspec_preproc

from pbasex import pbasex
from matplotlib import pyplot as plt

class PhotonFreqResVMI:
	"""
	This is a specially derived class for omega-resolved VMI
	Parameters
		vls_spec_dict:      Dictionary of preprocessed single-shot photon spectra
		processed_quad_dict:Dictionary of preprocessed single-shot image quadrants
		gData:				Pre-calculated G data by pBASEX
		precontractedData:	A dictionary or a npz file containing the following keys
								"A": 	VLS spectra
								"AtA":	pre-contracted A.T @ A
								"AtQuad": pre-contracted A.T @ B where B are the flattened images
								"vlsbounds" or "vlse_2bounds": boundary indices of the ROI along omega
		alpha_vmi:          Magnification factor of VMI
		pxWeights: 			Weights over pixels
	Keyword arguments
		spook_kwargs: kwargs that will be directly passed to SpookLinSolve
	"""
	def __init__(self, vls_spec_dict, processed_quad_dict, gData, 
		         precontractedData=None, alpha_vmi=1, pxWeights=None, 
		         sparsity="L2", **spook_kwargs):
		if precontractedData is not None:
			dat = precontractedData
			A = dat['A'] # This A is preferably in full range, but the cropped one is fine.
			Na_full = A.shape[1]
			keys = dat.files if isinstance(precontractedData, np.lib.npyio.NpzFile) else list(dat.keys())
			if "vlsbounds" in keys:
				bounds = dat['vlsbounds']
			elif "vlse_2bounds" in keys:
				bounds = dat['vlse_2bounds']
			else:
				bounds = (0,Na_full) # No cropping
# 			print(bounds)
			AtA = dat['AtA']
			AtQuad = dat['AtQuad']
			if AtA.shape[0] == Na_full:
				AtA = AtA[bounds[0]:bounds[1], bounds[0]:bounds[1]]
			if AtQuad.shape[0] == Na_full:
				AtQuad = AtQuad[bounds[0]:bounds[1],:]
			elif AtQuad.shape[0] != bounds[1]-bounds[0]:
				raise ValueError("Info loss in AtQuad: AtQuad.shape[0]=%d is neither ptp(bounds)=%d or A.shape[1]=%d."%(AtQuad.shape[0],bounds[1]-bounds[0],Na_full)
								+"Please verify precontractedData[vlsbounds] is properly set.")
			if AtA.shape[0] != AtQuad.shape[0]: # redo AtA contraction
				A1 = A[:,bounds[0]:bounds[1]]
				AtA = A1.T @ A1
		else:
			print("#Photon spectra:", len(vls_spec_dict.keys()), ". #VMI images", len(processed_quad_dict.keys()))
			BIDs = list(vls_spec_dict.keys())
			A = np.asarray([vls_spec_dict[b] for b in BIDs])
			Amean = A.mean(axis=0)
			bounds = np.argwhere(Amean > Amean.max() * np.exp(-2))
			bounds = (int(bounds.min()), int(bounds.max())+1)
			# cropped = True
			A1 = A[:,bounds[0]:bounds[1]]
			AtA =  A1.T @ A1
			AtQuad = adaptive_contraction(vls_spec_dict, processed_quad_dict, a_slice=bounds)
			npz_fname = "precontracted.npz"
			if os.path.exists(npz_fname):
				print("Overwriting", npz_fname)
			np.savez_compressed(npz_fname, A=A, BIDs=BIDs, AtA=AtA, AtQuad=AtQuad, vlse_2bounds=bounds)#, cropped=cropped)


		if pxWeights is None:
			GtG = gData['V'] @ np.diag(gData['S']**2) @ gData['V'].T
			AtBG = AtQuad @ (gData['V'] @ np.diag(gData['S']) @ gData['Up']).T
		else:
			w = pxWeights.ravel()
			tmp = gData['V'] @ np.diag(gData['S'])
			tmp2 = tmp @ (gData['Up'] * w)
			# calculating in this order is efficient when a lot of pixels are 0 weighted
			GtG = (tmp2 @ gData['Up'].T) @ tmp.T
			AtBG = AtQuad @ tmp2.T

		rsmoother = gData['frk'].T @ laplacian_square_S(gData['x'].size, True) @ gData['frk']
		rsmoother = sps.kron(sps.eye(gData['nl']), rsmoother)

		print(r"Tensor shapes: (A \otimes G).T@B, AtA, GtG, rsmoother")
		print(AtBG.shape, AtA.shape, GtG.shape, rsmoother.shape)
		if sparsity == 'L2':
			SpkCls = SpookLinSolve
		elif sparsity == 'L1':
			SpkCls = SpookL1
		else:
			raise ValueError("Unrecognized sparisty input: %s"%sparsity)
		self.__spook = SpkCls(AtBG, AtA, 'contracted', GtG, Bsmoother=rsmoother, **spook_kwargs)
		self.__gData = gData
		self._alpha = alpha_vmi 
		self.__vlsAxisInPX = np.arange(A.shape[1])[bounds[0]:bounds[1]]

	def getXopt_Ewl(self, l=None, **spook_kwargs):
		Xo = self.__spook.getXopt(**spook_kwargs)
		gData = self.__gData
		Xo_wlk = Xo.reshape(Xo.shape[0], gData['nl'], -1)
		ret = (Xo_wlk @ gData['frk'].T).transpose((2,0,1))
		self.Xo_Ewl = ((0.5/self._alpha) * gData['x'][:,None,None]) * ret
		if l is not None:
			return self.Xo_Ewl[:,l//2]
		return self.Xo_Ewl

	def show_res(self, l=0, ax=None):
		if ax is None:
			ax = plt.subplot(111)
		gData = self.__gData
		ax.pcolormesh(self._alpha*(gData['x']**2), self.__vlsAxisInPX,
			self.Xo_Ewl[:,:,l//2].T, shading='nearest')
		ax.set_ylabel("VLS pixel")
		ax.set_xlabel("E" + ("[px^2]" if self._alpha==1 else "[eV]"))
		return ax

	@property
	def keAxis(self):
		return self._alpha*(self.__gData['x']**2)

	@property
	def vlsAxis_px(self):
		return self.__vlsAxisInPX

	def getspook(self):
		return self.__spook
	