# Spooktroscopy: Frequency-Domain Ghost Imaging

## Target Problem
Solve $(A \otimes G)X = B$​ under regularizations.
$A, G$​ are matrices acting on the two indices of $X$​, i.e. 
![(AG)X=B](https://latex.codecogs.com/svg.latex?\Large&space;\sum_{w,\beta}A_{iw}G_{jq}X_{wq}=B_{ij}) 

$G$ is optional, by default (`G=None`) it is identity, in which case this is more like the conventional Spooktroscopy, i.e. to solve $AX=B$ under regularizations. 

The reason for $G$ is to accommodate the combination with pBASEX, in which case this is solving the two linear inversions in one.

### Regularizations
Common regularizations are the following three types, all of which are optional. It depends on what _a prior_ knowledge one wants to enforce on the problem solving.

1. Nonnegativity: $X\succeq 0$​​​ 
2. Sparsity: To penalize on $\|X\|_1$​​​ or $\|X\|_2^2$​​​
3. Smoothness: To penalize on roughness of $X$ , along the two indices, independently. For $w$-axis, which is usually the photon energy axis, the form is fixed $\|(L_{N_w}\otimes I)X\|_2^2$ where $L_{N_w}$ is the laplacian. Roughness along the $q$​​​-axis is customizable through parameter `Bsmoother`, which by default is laplacian squared too.

## Solvers

Different combinations of regularizations can lead to different forms of objective function. Solvers in package always formalize the specific problem into either a [Quadratic Programming](https://en.wikipedia.org/wiki/Quadratic_programming) or a linear equation. 

| Nonnegativity | Sparsity            | Smoothness | Solver              | Notes                                                        |
| ------------- | ------------------- | ---------- | ------------------- | ------------------------------------------------------------ |
| True          | L1 or False         | Quadratic  | `SpookPosL1`        | This solver can serve tasks like in [Li _et al_](https://iopscience.iop.org/article/10.1088/1361-6455/abcdf1) |
| True          | L2 squared          | Quadratic  | Not implemented yet |                                                              |
| False         | L2 squared or False | Quadratic  | `SpookLinSolve`     | This solver is so far the work-horse for SpookVMI            |
| False         | L1                  | Quadratic  | Not implemented yet |                                                              |



### Quadratic Programming

For cases where it can be formalized into a [Quadratic Programming](https://en.wikipedia.org/wiki/Quadratic_programming) , [OSQP](https://osqp.org) does the job. Thus the root numerical method is alternating direction method of multipliers (ADMM). Looking into the [solver settings of OSQP](https://osqp.org/docs/interfaces/solver_settings.html) is always encouraged, but the default settings usually work fine for `spook` . If one needs to pass in settings, the OSQP solver is `SpookQPBase._prob` .

### Linear Equation

A rare case that it can be formalized into a linear equation is the third line in the table above: no nonnegativity constraint, and the sparsity is L2 norm squared. This is implemented in `SpookLinSolve` , which calls `numpy.linalg.solve` or `scipy.sparse.linalg.spsolve` .



## Unit Tests

`unittest/testPosL1.py` is a good example to play with `SpookPosL1`.



## Dependencies

numpy > 1.19

scipy > 1.7

osqp > 0.6.2