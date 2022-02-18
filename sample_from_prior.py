'''
Functions that sample from the priors described in the paper. See generate_sample_images.ipynb for usage.
'''

import numpy as np
from fourier_transform_utils import get_inverse_fourier_transform
import fourier_transform_utils as ftu
import generate_synthetic_data as gsd


def IFT_of_unit_ball(x):
    x_thresh = np.where(np.abs(x) > 1e-8, x, 1e-8)
    b1 = ( - 3* np.cos(2 * np.pi * x_thresh) + 3*np.sin ( 2* np.pi * x_thresh )  / (2 * np.pi * x_thresh ) ) / ( 4 * np.pi**2 * x_thresh**2)
    b1 = np.where(np.abs(x) > 1e-8, b1, 1)
    return b1 

def ball_g_hat(psi1, R):
    return IFT_of_unit_ball(np.linalg.norm(psi1, axis = -1)*R)


## Functions used for the Bag of atoms
def generate_spherical_bag_of_atom_projection(grid_size, radius, voxel_size, N_atoms, atom_shape_fn):
    ft_mol = generate_synthetic_spectrum_of_molecule(radius, grid_size, voxel_size, atom_shape_fn, N_atoms)
    image = np.real(get_inverse_fourier_transform(ft_mol[grid_size//2], voxel_size = voxel_size))
    return image


def get_random_points_in_unit_ball(N):
    p = np.random.normal(0,1,(N, 3))
    norms = np.linalg.norm(p, axis = 1)
    r = np.random.random(N)**(1./3)
    random_in_sphere = p * r[:,np.newaxis] / norms[:,np.newaxis]
    return random_in_sphere


def generate_synthetic_spectrum_of_molecule(radius, grid_size, voxel_size , atom_shape_fn , N_atoms ):

    assert( radius < (grid_size/ 2 * voxel_size ) )  

    atom_coords = get_random_points_in_unit_ball(N_atoms) * radius     
    return generate_spectrum_of_molecule_from_atom_coords(atom_coords, grid_size, voxel_size, atom_shape_fn)

def generate_spectrum_of_molecule_from_atom_coords(atom_coords, grid_size, voxel_size, atom_shape_fn ):
    weights = np.ones(atom_coords.shape[0], dtype = atom_coords.dtype ) + 1j*0 #atom_coords[:,0] * 0 + 1 + 1j * 0
    fourier_transform = gsd.get_fourier_transform_of_molecules_off_grid_on_k_grid(atom_coords, weights , grid_size, voxel_size )

    k_coords = ftu.get_k_coordinate_of_each_pixel(3*[grid_size], voxel_size= voxel_size, scaled=True)
    weight = atom_shape_fn(k_coords).reshape(fourier_transform.shape)        
    
    return fourier_transform * weight

## Functions to sample from the Wilson prior

def compute_wilson_covariance_matrix(image_shape, voxel_size, N_atoms, g_fn, f_fn ):
    
    coords = ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled = True)

    C1_x, C2_x = np.meshgrid(coords[:,0], coords[:,0], indexing = "ij")
    C1_y, C2_y = np.meshgrid(coords[:,1], coords[:,1], indexing = "ij")

    coords_2d = np.stack((C1_x - C2_x, C1_y - C2_y), axis = -1)
    C = N_atoms * np.conj(f_fn(coords)[np.newaxis]) * f_fn(coords)[:,np.newaxis] * \
        (g_fn(coords_2d) -  np.conj(g_fn(coords))[np.newaxis]* g_fn(coords)[:,np.newaxis]  )

    return C




def compute_wilson_relation_matrix(image_shape, voxel_size, N_atoms, g_fn, f_fn ):
    coords = ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled = True)
    C1_x, C2_x = np.meshgrid(coords[:,0], coords[:,0])
    C1_y, C2_y = np.meshgrid(coords[:,1], coords[:,1])
    coords_2d = np.stack((C1_x + C2_x, C1_y + C2_y), axis = -1)
    
    C = N_atoms * f_fn(coords)[np.newaxis] * f_fn(coords)[:,np.newaxis] * \
        (g_fn(coords_2d) -  g_fn(coords)[np.newaxis]* g_fn(coords)[:,np.newaxis]  )
    return C

def compute_wilson_mean_vector(image_shape, voxel_size, N_atoms, g_fn, f_fn ):
    coords = ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled = True)
    return N_atoms * f_fn(coords) * g_fn(coords )



def generate_GP_projection_from_covariance_and_mean_real(C_chol, mu, voxel_size, grid_size):
    image_shape = 2*[grid_size]
    Z = np.random.randn(*mu.shape) 
        
    X = mu + C_chol @ Z
    mol_ft = X.reshape(image_shape)
    
    return mol_ft

def get_wilson_mean_cov_chol(grid_size, voxel_size, N_atoms, g_fn , f_fn ):
    image_shape = 2*[grid_size]
    C = compute_wilson_covariance_matrix(image_shape, voxel_size, N_atoms, g_fn , f_fn)
    R = compute_wilson_relation_matrix(image_shape, voxel_size, N_atoms, g_fn , f_fn )
    mean = compute_wilson_mean_vector(image_shape, voxel_size, N_atoms, g_fn, f_fn )

    keps = 1e-9
    
    real_cov = 0.5 * np.real( C + R)
    imag_cov = 0.5 * np.real( C -  R)

    real_cov_chol = np.linalg.cholesky(real_cov + np.eye(mean.shape[0]) * keps ) 
    imag_cov_chol = np.linalg.cholesky(imag_cov + np.eye(mean.shape[0]) * keps ) 
    mean_real = np.real(mean)
    mean_imag = np.imag(mean)
    return mean_real, mean_imag, real_cov_chol, imag_cov_chol


def generate_wilson_projection(grid_size, voxel_size, N_atoms, real_cov_chol = None, imag_cov_chol = None, mean_real = None, mean_imag = None):

    mol_ft_real = generate_GP_projection_from_covariance_and_mean_real(real_cov_chol, mean_real, voxel_size, grid_size)    
    mol_ft_imag = generate_GP_projection_from_covariance_and_mean_real(imag_cov_chol, mean_imag, voxel_size, grid_size)    
    
    mol_ft = mol_ft_real + 1j* mol_ft_imag
    mol = np.real(get_inverse_fourier_transform(mol_ft, voxel_size = voxel_size))

    return mol

# Functions to sample from the diagonal prior

def compute_expected_power_spectrum(image_shape, voxel_size, N_atoms, g_fn, f_fn, from_unscaled_g = False):
    coords = ftu.get_k_coordinate_of_each_pixel(image_shape, voxel_size)    
    return np.abs(f_fn(coords))**2* ( N_atoms + N_atoms*(N_atoms -1) * np.abs(g_fn(coords))**2)

def symmetrize_ft(mol):
    return (mol + np.conj(np.flip(mol)))/2

def generate_GP_projection_from_covariance_and_mean_complex(C_chol, mu, voxel_size, grid_size):
    image_shape = 2*[grid_size]

    Z =np.random.randn(*mu.shape) + 1j * np.random.randn(*mu.shape)
    X = mu + C_chol @ Z
    mol_ft = X.reshape(image_shape)

    # We want to impose that the FT of of X is real. To account for this, need to scale |Z| = \sqrt{2}.
    sym_mol_ft = symmetrize_ft(mol_ft)
    
    mol = np.real(get_inverse_fourier_transform(sym_mol_ft, voxel_size = voxel_size))
    return mol

def compute_diagonal_prior_mean_cov_chol(grid_size, voxel_size, N_atoms, g_fn, f_fn):
    expected_power_spectrum = compute_expected_power_spectrum(2*[grid_size], voxel_size, N_atoms, g_fn , f_fn )
    cov_chol = np.diag(np.sqrt(expected_power_spectrum))
    mean = np.zeros(cov_chol.shape[0])       
    return mean, cov_chol

def generate_diagonal_projection(grid_size, voxel_size, N_atoms, cov_chol , mean):
    mol = generate_GP_projection_from_covariance_and_mean_complex(cov_chol, mean, voxel_size, grid_size)    
    return mol

# Functions for the spatially independent exponential prior

def get_sie_mean(grid_size, voxel_size, N_atoms, g_fn, f_fn):
    image_shape = 3*[grid_size]
    mean_ft = compute_wilson_mean_vector(image_shape, voxel_size, N_atoms, g_fn , f_fn )
    mean_ft = mean_ft.reshape(image_shape)
    mean = np.real(get_inverse_fourier_transform(mean_ft, voxel_size))

    # There might have spurious 0's due to spectrum truncation. Truncate them
    mean = np.where(mean > 0 , mean, 0 )
    return mean

def generate_sie_projection(grid_size, voxel_size, N_atoms, mean):
    mol = np.random.exponential(mean)
    mol_proj = np.sum(mol, axis = 0 )
    return mol_proj