# Spooktroscopy: Spectral-Domain Ghost Imaging

## Target Problem
Solve <img src="https://render.githubusercontent.com/render/math?math=(A \otimes G)X = B"> in the least-square way, under [regularizations](#Regularizations). A, G are matrices acting on the two dimensions of X, i.e. 
![(AG)X=B](https://latex.codecogs.com/svg.latex?\normalsize&space;\sum_{w,b}A_{iw}G_{bq}X_{wb}=B_{iq}) .
For spectral-domain ghost imaging, the dimension indexed by <img src="https://render.githubusercontent.com/render/math?math=w"> is photon energy, and the other dimension can be properties of the photoproduct, such as the electron kinetic energy.

`G` is the (optional) linear operator on the dimension indexed by <img src="https://render.githubusercontent.com/render/math?math=b">. By default (`G=None`), it is the identity, in which case this is the conventional Spooktroscopy, i.e. to solve <img src="https://render.githubusercontent.com/render/math?math=AX=B"> under regularizations. 
`G` can accommodate other linear operations on the $b$ dimension, to solve the two linear inversions in one step. 

With `mode='raw'`, pass in matrix $G_{bq}$ to G, and with `mode='contracted'`, pass in matrix $\sum_qG_{bq}G_{b'q}$. For example, for Abel transform in Velocity Map Imaging, [pBasex](https://github.com/e-champenois/CPBASEX) offers this G with `loadG`.

### Key Advantages
The key advantages of this package are
1. Efficient optimization: Contraction over shots is decoupled from optimization. It is **recommended** to input precontracted results when instantiating, or to save the caches using method `save_prectr` of a solver created with raw inputs. Once instantiated, the precontracted results are cached to be reused every time solving with a different hyperparameter. See `spook.contraction_utils.adaptive_contraction` .
2. Support dimension reduction on the dependent variable B, through basis functions in G. In that case, it is also recommended to contract (B,G) over the dependent variable space (index q) prior to instantiating a spook solver.
3. Support multiple combinations of regularizations. See [Solvers](#Solvers) .
4. Support time-dependent measurement (Beta): when each entry in the raw input A is a pair of (photon spectrum, delay bin), index w is the flattened axis of (<img src="https://render.githubusercontent.com/render/math?math=\omega">, <img src="https://render.githubusercontent.com/render/math?math=\tau">). In this case, the third smoothness hyperparameter is for the delay axis.

At the very bottom level, this package depends on either [OSQP](https://osqp.org) to solve a quadratic programming or LAPACK gesv through `numpy.linalg.solve` . 

## Installation
The stable version is on PyPI. Unfortunately in a different name.

    pip install FDGI

## Solvers

Different combinations of regularizations can lead to different forms of objective function. Solvers in package always formalize the specific problem into either a [Quadratic Programming](https://en.wikipedia.org/wiki/Quadratic_programming) or a linear equation. Examples can be found in [unit tests](#UnitTests) 

| Nonnegativity | Sparsity            | Smoothness | Solver              | Notes                                                        |
| ------------- | ------------------- | ---------- | ------------------- | ------------------------------------------------------------ |
| True          | L1 or False         | Quadratic  | `SpookPosL1`        | This solver can serve tasks like in [Li _et al_](https://iopscience.iop.org/article/10.1088/1361-6455/abcdf1) |
| True          | L2 squared          | Quadratic  | `SpookPosL2` |                                                              |
| False         | L2 squared or False | Quadratic  | `SpookLinSolve`     | This solver is so far the work-horse for SpookVMI            |
| False         | L1                  | Quadratic  | `SpookL1` |                                                              |


### Quadratic Programming

For cases where it can be formalized into a [Quadratic Programming](https://en.wikipedia.org/wiki/Quadratic_programming) , [OSQP](https://osqp.org) is the weight-lifter. The root numerical method is the alternating direction method of multipliers (ADMM). Looking into the [solver settings of OSQP](https://osqp.org/docs/interfaces/solver_settings.html) is always encouraged, but the default settings usually work fine for `spook` . If one needs to pass in settings, use `spk._prob.update_settings(**osqp_kwargs)` on your solver instance `spk`. 

### Linear Equation

A rare case that it can be formalized into a linear equation is the third line in the table above: no nonnegativity constraint, and the sparsity is L2 norm squared. This is implemented in `SpookLinSolve` , which calls `numpy.linalg.solve` or `scipy.sparse.linalg.spsolve` .

### Regularizations
Common regularizations are the following three types, all of which optional, depending on what _a prior_ knowledge one wants to enforce on the problem solving.

1. Nonnegativity: To constrain <img src="https://render.githubusercontent.com/render/math?math=X\succeq 0">
2. Sparsity: To penalize on <img src="https://render.githubusercontent.com/render/math?math=\|X\|_1"> or <img src="https://render.githubusercontent.com/render/math?math=\|X\|_2^2">
3. Smoothness: To penalize on roughness of X , along the two indices, independently. For <img src="https://render.githubusercontent.com/render/math?math=\omega">-axis, which is the photon energy axis, the form is fixed <img src="https://render.githubusercontent.com/render/math?math=\|(L_{N_w}\otimes I)X\|_2^2"> where <img src="https://render.githubusercontent.com/render/math?math=L_{N_w}"> is the [laplacian](https://en.wikipedia.org/wiki/Laplace_operator). Roughness along the second axis of X is customizable through parameter `Bsmoother`, which by default is laplacian squared too.

Sparsity and Smoothness are enforced through penalties in the total obejctive function, and the penalties are weighted by hyperparameters `lsparse` and `lsmooth`. `lsmooth` is a 2-tuple that weight roughness penalty along the two axes of X respectively. The hyperparameters can be passed in during instantiation and also updated afterwards. It is recommended to call method `getXopt` with the hyperparameter(s) to be updated, because it will update, solve, and return the optimal X in one step. Calling `solve` with  the hyperparameter(s) to be updated and then calling `getXopt()` without input is effectively the same, and the problem will be solved once as long as there is no update.

## Normalization Convention

The entries in <img src="https://render.githubusercontent.com/render/math?math=A^TA, G^TG"> are preferred to be on the order of unity, because regularization-related quadratic form matrices have their entries around unity. The scale factors are set as

$$s_a=\sqrt{\frac{1}{N_w}\mathrm{tr}(A^TA)},s_g=\sqrt{\frac{1}{N_q}\mathrm{tr}(G^TG)}$$

where <img src="https://render.githubusercontent.com/render/math?math=N_w, N_q"> are the dimensions along w-axis and q-axis, respectively. <img src="https://render.githubusercontent.com/render/math?math=s_as_g"> is an accessible property of the solver `AGscale`. To normalize or not is controlled by parameter `normalize` in creating a solver. 

**By default** `normalize=True`, i.e. `self._AtA` =<img src="https://render.githubusercontent.com/render/math?math=A^TA/s_a^2">, `self._GtG` =<img src="https://render.githubusercontent.com/render/math?math=G^TG/s_g^2">, and `self._Bcontracted` = <img src="https://render.githubusercontent.com/render/math?math=(A^T \otimes G^T)B/(s_as_g)">, and the normalization is not in-place starting from version 0.9.4 . In this case, the direct solution `self.res` is scaled as <img src="https://render.githubusercontent.com/render/math?math=s_as_g X_\mathrm{opt}"> , but the `getXopt` method returns the unscaled result of <img src="https://render.githubusercontent.com/render/math?math=X_\mathrm{opt}">. In v0.9.3 and before, the normalization is in-place, which means the first two arguments are modified in-place. In-place normalization saves memory but causes confusion when people reuses the precontracted results for other purposes after passing them to create a solver in the contracted mode. Therefore starting from v0.9.4, in-place normalization is only done when requested with `normalize='inplace'` in creating a solver.

## Citing
Please cite [Wang _et al_ 2023](https://iopscience.iop.org/article/10.1088/1367-2630/acc201) when using this package in your publishable work. If `SpookPosL1`, `SpookPosL2` or `SpookL1` is used, we also **strongly recommend** to cite the original OSQP paper as suggested [here](https://osqp.org/citing/).

## Unit Tests

`unittest/testPosL1.py` is a good example to play with `SpookPosL1`.
`unittest/testL1L2.ipynb` include good examples to play with `SpookPosL1`, `SpookPosL2`, and `SpookL1`.


## Dependencies

pip installation should manage the dependencies automatically. If not, check `requirements.txt` . 

## Acknowledgement
This work was supported by the U.S. Department of Energy (DOE), Office of Science, Office of Basic Energy Sciences (BES), Chemical Sciences, Geosciences, and Biosciences Division (CSGB).
