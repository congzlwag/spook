import numpy as np
from .lin_solve import SpookLinSolve
import scipy.sparse as sps
from .utils import laplacian_square_S
import os
# from .utils import phspec_preproc

from pbasex import loadG, pbasex
from matplotlib import pyplot as plt

class PhotonFreqResVMI:
	def __init__(self, vls_spec_dict, processed_quad_images_dict, gData, precontractedData=None, alpha_vmi=1, pxWeights=None, **spook_kwargs):
		if precontractedData is not None:
			dat = precontractedData
			bounds = dat['vlse_2bounds']
			AtA = dat['AtA']
			AtQuad = dat['AtQuad']
			A = dat['A']
		else:
			print("#Photon spectra:", len(vls_spec_dict.keys()), ". #VMI images", len(processed_quad_images_dict.keys()))
			BIDs = list(vls_spec_dict.keys())
			A = np.asarray([vls_spec_dict[b] for b in BIDs])
			Amean = A.mean(axis=0)
			bounds = np.argwhere(Amean > Amean.max() * np.exp(-2))
			bounds = (bounds.min(), bounds.max()+1)
			AtA =  A.T @ A
			AtQuad = dict_innerprod(vls_spec_dict, processed_quad_dict)
			npz_fname = "precontracted.npz"
			if os.path.exists(npz_fname):
				print("Overwriting", npz_fname)
			np.savez_compressed(npz_fname, A=A, BIDs=BIDs, AtA=AtA, AtQuad=AtQuad, vlse_2bounds=bounds)
		
		AtA = AtA[bounds[0]:bounds[1], bounds[0]:bounds[1]]
		AtQuad = AtQuad[bounds[0]:bounds[1]]

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

		print(r"Tensor shapes: (A \otimes G)t B, AtA, GtG, rsmoother")
		print(AtBG.shape, AtA.shape, GtG.shape, rsmoother.shape)
		self.__spook = SpookLinSolve(AtBG, AtA, 'contracted', GtG, Bsmoother=rsmoother, **spook_kwargs)
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

