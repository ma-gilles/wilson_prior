''''
Functions to compute the hyperparameters of the prior, and use them in a Wiener filter
'''

import numpy as np
import scipy.ndimage
from fourier_transform_utils import get_inverse_fourier_transform, get_fourier_transform, get_1d_frequency_grid, compute_spherical_average, get_k_coordinate_of_each_pixel


## Helper functions for the Diagonal prior
def estimate_power_spectrum_function_from_image(image, voxel_size):
    image_shape = image.shape

    image_ft = get_fourier_transform(image, voxel_size)
    power_spectrum = np.real( compute_spherical_average(np.abs(image_ft)**2))

    # Put back on grid by interpolating
    freq =  get_1d_frequency_grid(4*image_shape[0], voxel_size = 1/4*voxel_size, scaled = True)
    freq = freq[freq >= 0 ]
    freq = freq[:power_spectrum.size]
    k_coords = get_k_coordinate_of_each_pixel(image_shape, voxel_size= voxel_size, scaled=True)
    r = np.linalg.norm(k_coords,axis = -1)
    power_spectrum_on_grid = np.interp( r, freq, power_spectrum )

    return power_spectrum_on_grid

def get_diagonal_prior_mean_covariance(clean_image, voxel_size):
    # Scheres, 2012 uses a factor T = 4 in the prior.
    T = 4
    power_spectrum = T * estimate_power_spectrum_function_from_image(clean_image, voxel_size).reshape(clean_image.shape)

    # Returns handle to diagonal matrix
    def relion_covariance_matvec_fn(x):
        return power_spectrum * x 

    relion_mean = np.zeros(clean_image.shape)
    return relion_mean, relion_covariance_matvec_fn


## Functions for the Wilson prior
def apply_wilson_covariance_matrix(voxel_size, N_atoms, g_real_on_grid, g_ft_on_grid, f_ft_on_grid, x_ft_vec ):
    '''
    Computes matrix-vector product Sigma_{wilson} x as described in Gilles-Singer, 2022
    '''
    # Computes is: y = D_f ( C - g g.H ) D_f.H * x
    image_shape = g_ft_on_grid.shape
    x_ft = x_ft_vec.reshape(image_shape)

    # Now precomputed and passed as an input to save 1 FFT.
    # g_ft_on_grid = get_fourier_transform(g_real_on_grid, voxel_size)

    # v = D_f.H @ x 
    Dx_ft = np.conj( f_ft_on_grid) * x_ft

    # p1 = g @ g.H @ v
    p1 = g_ft_on_grid * np.dot(np.conj(g_ft_on_grid).reshape(-1), Dx_ft.reshape(-1))

    # p2 = C @ v
    p2 = get_fourier_transform(g_real_on_grid * get_inverse_fourier_transform(Dx_ft, voxel_size) , voxel_size) * np.prod(image_shape)
    
    # y = N_atoms * D * ( p2 - p1)
    Cx =  N_atoms * f_ft_on_grid * ( p2 - p1) 

    return Cx.reshape(x_ft_vec.shape)

# Implementation is borrowed from https://gist.github.com/mblondel/6f3b7aaad90606b98f71
def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def estimate_g_from_image_est_and_f(N_atoms, f_fourier_on_grid, image, voxel_size, convolution_sigma):
    image_ft = get_fourier_transform(image, voxel_size)
    image_deconvolve = get_inverse_fourier_transform( image_ft / f_fourier_on_grid, voxel_size) / N_atoms
    
    convolved_image = scipy.ndimage.gaussian_filter(np.real(image_deconvolve),convolution_sigma/voxel_size ).reshape(-1)
    if np.abs(np.sum(convolved_image) - 1 ) > 0.1:
        print("convolved image sum not close to 1: sum", np.sum(convolved_image))

    estimated_g_function = projection_simplex_sort(convolved_image.reshape(-1) ,1).reshape(image.shape)
    return estimated_g_function

def get_wilson_prior_mean_covariance(N_atoms, atom_shape_fn, guess_image, voxel_size, convolution_sigma):

    image_shape  = guess_image.shape
    k_coords = get_k_coordinate_of_each_pixel(image_shape, voxel_size= voxel_size, scaled=True)
    f_fourier_on_grid = atom_shape_fn(k_coords).reshape(image_shape)
    g_real_on_grid = estimate_g_from_image_est_and_f(N_atoms, f_fourier_on_grid, guess_image, voxel_size, convolution_sigma)

    # This is precomputed to save 1FFT per CG iteration.
    g_fourier_on_grid = get_fourier_transform(g_real_on_grid, voxel_size)
    wilson_fourier_mean = N_atoms * f_fourier_on_grid * g_fourier_on_grid 
    
    def wilson_fourier_covariance_matvec_fn(x):
        return apply_wilson_covariance_matrix(voxel_size, N_atoms, g_real_on_grid, g_fourier_on_grid, f_fourier_on_grid, x )
    return wilson_fourier_mean, wilson_fourier_covariance_matvec_fn


## Wiener filter functions
def apply_weiner_matrix(x, population_cov_fn, noise_variance, filter_on_grid):
    x1 = noise_variance * x
    x2 = filter_on_grid*(population_cov_fn(np.conj(filter_on_grid) * x))
    return x1 + x2

def weiner_filter(observation, population_mean, population_cov_fn, noise_variance, filter_on_grid, diagonal_flag = False):
    image_shape  = observation.shape
    rhs = (observation - filter_on_grid * population_mean).reshape(-1)
        
    if diagonal_flag:
        bottom = filter_on_grid * population_cov_fn( np.conj(filter_on_grid) )  + noise_variance
        x_sol = rhs / bottom.reshape(-1)
    else:  # If not diagonal, solved with CG.
        def Afun(x):
            return apply_weiner_matrix(x.reshape(image_shape), population_cov_fn, noise_variance, filter_on_grid).reshape(-1)    
        dim = np.prod(filter_on_grid.shape)
        A = scipy.sparse.linalg.LinearOperator((dim,dim), matvec=Afun)
        x_sol , flag = scipy.sparse.linalg.cg(A, rhs, tol = 1e-8, maxiter = 100 , atol = "legacy")
        if flag != 0:
            residual = np.linalg.norm(Afun(x_sol) - rhs) / np.linalg.norm(rhs)
            print("cg did not converge. flag=", flag , ". Residual:", residual)

    CAx = population_cov_fn( np.conj(filter_on_grid) * x_sol.reshape(image_shape))
    return population_mean + CAx
