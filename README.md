# Spooktroscopy: Spectral-Domain Ghost Imaging

## Target Problem
Solve for $X$ from $(A \otimes G)X = B$ in the least-square manner, under [regularizations](#Regularizations). The full-index representation of the objective to minimize is
$\sum_{iq}\|\sum_{w,b}A_{iw}G_{bq}X_{wb} - B_{iq}\|^2$ .
|          Index     | $i$  |        $w$        |           $q$      |     $b$             |
|--------------------|------|-------------------|--------------------|---------------------|
|What does it number?| shot | photon energy bin | observed property  | interested property |
|Example 0           | shot | photon energy bin | kinetic energy     | kinetic energy      |
|Example 1           | shot | photon energy bin | projected momentum | coefficient on a basis function |

`G` is an optional linear operator from a function of the interested property (discretized by index $b$) to a function of the observed property (discretized by index $q$), e.g. from a KE spectrum to a KE spectrum.
By default `G=None`, $G=I$ is the identity matrix. This is the most common use case of spooktroscopy, i.e. to solve $AX=B$ without any extra linear mapping.
When `G` accommodates a linear operation that's not identity mapping, the two linear inversions are solved in a single step.

With `mode='raw'`, pass in matrix $G_{bq}$ to G, and with `mode='contracted'`, pass in matrix $\sum_qG_{bq}G_{b'q}$. For example, for Abel transform in Velocity Map Imaging, [pBasex](https://github.com/e-champenois/CPBASEX) offers this G with `loadG`.

### Key Advantages
The key advantages of this package are
1. Efficient optimization: Contraction over shots is decoupled from optimization. It is **recommended** to input precontracted results when instantiating, or to save the caches using method `save_prectr` of a solver created with raw inputs. Once instantiated, the precontracted results are cached to be reused every time solving with a different hyperparameter. See `spook.contraction_utils.adaptive_contraction` .
2. Support dimension reduction on the dependent variable B, through basis functions in G. In that case, it is also recommended to contract (B,G) over the dependent variable space (index q) prior to instantiating a spook solver.
3. Support multiple combinations of regularizations. See [Solvers](#Solvers) .
4. Support time-dependent measurement (Beta): when each entry in the raw input A is a pair of (photon spectrum, delay bin), index w is the flattened axis of ($\omega, \tau$). In this case, the third smoothness hyperparameter is for the delay axis.

At the very bottom level, this package depends on either [OSQP](https://osqp.org) to solve a quadratic programming or LAPACK gesv through `numpy.linalg.solve` .

## Installation
The stable version is on PyPI. Unfortunately in a different name.
```bash
pip install FDGI
```
## Solvers

Different combinations of regularizations can lead to different forms of objective function.
Solvers in package always formalize the specific problem into either a [Quadratic Programming](https://en.wikipedia.org/wiki/Quadratic_programming) or a linear equation.
Examples can be found in [unit tests](#UnitTests)

| Nonnegativity | Sparsity            | Smoothness | Solver              | Notes                                                        |
| ------------- | ------------------- | ---------- | ------------------- | ------------------------------------------------------------ |
| True          | L1                  | Quadratic  | `SpookPosL1`        | This solver can serve tasks like in [Li _et al_](https://iopscience.iop.org/article/10.1088/1361-6455/abcdf1) |
| True          | L2 squared          | Quadratic  | `SpookPosL2` |                                                              |
| False         | L2 squared          | Quadratic  | `SpookLinSolve`     | This solver is so far the work-horse for SpookVMI            |
| False         | L1                  | Quadratic  | `SpookL1` |                                                              |

A family tree of solvers is in `docs/figs/famtree.svg`.

### Quadratic Programming

For cases where it can be formalized into a [Quadratic Programming](https://en.wikipedia.org/wiki/Quadratic_programming) , [OSQP](https://osqp.org) is the weight-lifter. The root numerical method is the alternating direction method of multipliers (ADMM). Looking into the [solver settings of OSQP](https://osqp.org/docs/interfaces/solver_settings.html) is always encouraged, but the default settings usually work fine for `spook` . If one needs to pass in settings, use `spk._prob.update_settings(**osqp_kwargs)` on your solver instance `spk`.

### Linear Equation

A rare case that it can be formalized into a linear equation is the third line in the table above: no nonnegativity constraint, and the sparsity is L2 norm squared. This is implemented in `SpookLinSolve` , which calls `numpy.linalg.solve` or `scipy.sparse.linalg.spsolve` .

### Regularizations
Common regularizations are the following three types, all of which optional, depending on what _a prior_ knowledge one wants to enforce on the problem solving.

1. Nonnegativity: To constrain $X\geq 0$ everywhere
2. Sparsity: To penalize on $\|X\|_1$ or $\|X\|_2$
3. Smoothness: To penalize on roughness of X , along the two dimensions indexed by $w$ and $b$ independently. For the dimension indexed by $w$, the form is fixed to the laplacian-square $\|(L\otimes I) X\|^2_2$ where $L$ is the [laplacian](https://en.wikipedia.org/wiki/Laplace_operator). Roughness along the second dimension of X is customizable with input parameter `Bsmoother`, which by default is laplacian squared too.

Sparsity and Smoothness are enforced through penalties in the total obejctive function, and the penalties are weighted by hyperparameters `lsparse` and `lsmooth`. `lsmooth` is a 2-tuple that weight roughness penalty along the two axes of X respectively. The hyperparameters can be passed in during instantiation and also updated afterwards. It is recommended to call method `getXopt` with the hyperparameter(s) to be updated, because it will update, solve, and return the optimal X in one step. Calling `solve` with  the hyperparameter(s) to be updated and then calling `getXopt()` without input is effectively the same, and the problem will be solved once as long as there is no update.

### Evaluating Terms of Objective Function
| Term | Method | Notes |
|------|--------|-------|
| $\|(A \otimes G)X-B\|_2$ | `residueL2`| For full docstring, check `help(spk.residueL2)`, where `spk` is a solver instance. |
| $\|L X\|_2$ | `smoothness` | By default, $L$ is a 2nd-order finite difference operator along dimension specified by input argument `dim`. For full docstring, check `help(spk.smoothness)`.  |
| $\|X\|_1$ or $\|X\|_2$ | `sparsity` | Depends on solver class. |

* Method `residueL2` is based on function `utils.calcL2fromContracted`. This function evaluates the residual term from the precontracted results. Also see `XValidation.calc_residual` defined in `xval.py`.
* Function `utils.calcL2fromContracted` is also used in the cross-validation class `XValidation` to evaluate the residual on validation sets. See `xval.py` for more details.

## Normalization Convention

The entries in $A^TA, G^TG$ are preferred to be on the order of unity, because regularization-related quadratic form matrices have their entries around unity. The scale factors are set as

$$s_a=\sqrt{\frac{1}{N_w}\mathrm{tr}(A^TA)},s_g=\sqrt{\frac{1}{N_q}\mathrm{tr}(G^TG)}$$

where $N_w, N_q$ are the dimensions along the respective indices. $s_as_g$ is an accessible property of the solver `AGscale`. To normalize or not is controlled by parameter `normalize` in creating a solver.

**By default** `normalize=True`, i.e. `self._AtA` =$A^TA/s_a^2$, `self._GtG` =$G^TG/s_g^2$, and `self._Bcontracted` = $(A^T \otimes G^T)B/(s_as_g)$, and the normalization is not in-place starting from version 0.9.4 . In this case, the direct solution `self.res` is scaled as $s_as_g X_\mathrm{opt}$ , but the `getXopt` method returns the unscaled result of $X_\mathrm{opt}$. In v0.9.3 and before, the normalization is in-place, which means the first two arguments are modified in-place. In-place normalization saves memory but causes confusion when people reuses the precontracted results for other purposes after passing them to create a solver in the contracted mode. Therefore starting from v0.9.4, in-place normalization is only done when requested with `normalize='inplace'` in creating a solver.


With this normalization convention, the hyperparmeters whose associated regularization function is not quadratic in $X$ need scaling when changing the size of dataset.
If $h(s X) = s^p h(X)$, then the scaling power of $h(X)$ is $p$, e.g. $p=2$ for quadratic terms.
When duplicating the same dataset to $m$ times its original size, in order to keep the optimal point unchanged, we need to set $\lambda\mapsto m^{1-p/2} \lambda$. Therefore for $p=1$ regularization terms, the scaling is $\sqrt{m}$, and for $p=2$ terms they do not scale.

## Citing
Please cite [Wang _et al_ 2023](https://iopscience.iop.org/article/10.1088/1367-2630/acc201) when using this package in your publishable work. If `SpookPosL1`, `SpookPosL2` or `SpookL1` is used, we also **strongly recommend** to cite the original OSQP paper as suggested [here](https://osqp.org/citing/).

## Unit Tests
* `unittest/testPosL1.py` is a good example to play with `SpookPosL1`.
* `unittest/testL1L2.ipynb` includes good examples to play with `SpookPosL1`, `SpookPosL2`, and `SpookL1`.
* `unittest/test_resid_reg_eval.py` showcases how to evaluate the residual and regularization terms of the objective function.


## Dependencies

pip installation should manage the dependencies automatically. If not, check `requirements.txt` .

## Acknowledgement

This work was supported by the U.S. Department of Energy (DOE), Office of Science, Office of Basic Energy Sciences (BES), Chemical Sciences, Geosciences, and Biosciences Division (CSGB).

## Documentation

Full documentation is [here](https://congzlwag.github.io/spook/). Alternatively, clone this repo, switch to gh-pages, and open docs/index.html in a browser.
