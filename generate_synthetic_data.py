"""
Functions to generate the scattering potential of molecules using FINUFFT, and the CTFs used.
"""
import os
import json
from collections import defaultdict
import finufft
import numpy as np
from fourier_transform_utils import get_k_coordinate_of_each_pixel, get_grid_of_radial_distances


# These are the atomic scattering potential coefficients tabulated in:
# Peng, L-M., et al. "Robust parameterization of elastic and absorptive electron atomic scattering factors." 
# Acta Crystallographica Section A: Foundations of Crystallography 52.2 (1996): 257-276.
atom_coeff_path = 'atom_coeffs.json'
with open(os.path.join(os.path.dirname(__file__), atom_coeff_path), 'r') as f:
    atom_coeffs = json.load(f)


def get_exponent_and_constant_of_gaussian_FT(sigma, dim = 3):
    # real space exp
    a = 1/(2* sigma**2)
    tau = np.pi**2 / a

    cst = ((1/( sigma * np.sqrt(2*np.pi))) * np.sqrt(np.pi / a))**dim
    return tau, cst

def get_fourier_transform_of_molecules_off_grid_on_k_grid(atom_coords, weights, grid_size, voxel_size, eps = 1e-6 ):    
    normalized_atom_coords = (atom_coords)/grid_size * (2*np.pi) / voxel_size
    ft = finufft.nufft3d1(normalized_atom_coords[:,0], normalized_atom_coords[:,1], normalized_atom_coords[:,2], weights, (grid_size,grid_size,grid_size), isign=-1, eps = eps)
    return ft

def compute_gaussian_on_k_grid(sigma, grid_size, voxel_size):
    rs = get_grid_of_radial_distances(3*[grid_size], voxel_size = voxel_size, scaled = True)
    expo, cst = get_exponent_and_constant_of_gaussian_FT(sigma, dim = 3)
    return cst * np.exp(-rs**2 * expo)

def get_gaussian_fn_on_k(sigma):
    expo, cst = get_exponent_and_constant_of_gaussian_FT(sigma, dim = 3)
    def gaussian_fn(x):
        return cst * np.exp(-np.linalg.norm(x, axis = -1)**2 * expo)

    return gaussian_fn

def generate_spectrum_of_molecule_from_atom_coords(atom_coords, grid_size, voxel_size, atom_shape_fn ):
    weights = np.ones(atom_coords.shape[0], dtype = atom_coords.dtype ) 
    fourier_transform = get_fourier_transform_of_molecules_off_grid_on_k_grid(atom_coords, weights , grid_size, voxel_size )

    k_coords = get_k_coordinate_of_each_pixel(3*[grid_size], voxel_size= voxel_size, scaled=True)
    weight = atom_shape_fn(k_coords).reshape(fourier_transform.shape)        
    
    return fourier_transform * weight

def five_gaussian_atom_shape(psi, coeffs):
    a = coeffs[:5]
    b = coeffs[5:]
    if psi.ndim ==1:
        rs = psi
    else:
        rs = np.linalg.norm(psi, axis = -1)
        
    potential = np.zeros(psi.shape[0])
    for k in range(5):
        potential += a[k] * np.exp(- b[k] * rs**2)    
    return potential


def generate_volume_from_atom_positions_and_types(atom_coords, atom_types, grid_size, voxel_size ):
    
    atom_indices = defaultdict(list)
    [ atom_indices[atom_types[k]].append(k) for k in range(len(atom_types)) ]

    atoms_grouped_by_elements = {}
    for atom_name in atom_indices:
        atoms_grouped_by_elements[atom_name] = atom_coords[np.array(atom_indices[atom_name])]

    # Compute density for each kind of element
    density = np.zeros(3*[grid_size], dtype = complex)
    for atom_name in atoms_grouped_by_elements:
        atom_shape_fn = lambda x: five_gaussian_atom_shape(x, atom_coeffs[atom_name])
        density += generate_spectrum_of_molecule_from_atom_coords(atoms_grouped_by_elements[atom_name], grid_size, voxel_size, atom_shape_fn   )
    
    return density




def generate_volume_from_atoms(atoms, grid_size, voxel_size):

    atom_coords = atoms.getCoords()
    # Group atoms by elements
    atom_types = atoms.getData('element')
    return generate_volume_from_atom_positions_and_types(atom_coords, atom_types, grid_size, voxel_size)



def get_average_atom_shape_fn(atoms):

    atom_coords = atoms.getCoords()
    atom_coords = atom_coords - np.mean(atom_coords, axis = 0)

    # Group atoms by elements
    atom_names = atoms.getData('element')
    atom_indices = defaultdict(list)
    [ atom_indices[atom_names[k]].append(k) for k in range(len(atom_names)) ]


    atom_shape_fns = {}; atom_proportions = {}

    for atom_name in atom_indices:
        atom_shape_fns[atom_name] =  get_atom_shape_fn(atom_name) 
        atom_proportions[atom_name] = len(atom_indices[atom_name]) / atom_coords.shape[0]

    def average_atom_shape(psi):
        if psi.ndim ==1:
            rs = psi
        else:
            rs = np.linalg.norm(psi, axis = -1)

        density = np.zeros(rs.shape)
        for atom_name in atom_shape_fns:
            density += atom_proportions[atom_name] * atom_shape_fns[atom_name](rs)
        return density

    return average_atom_shape


def get_atom_shape_fn(atom_name):
    def shape_fn(x):
        return five_gaussian_atom_shape(x, atom_coeffs[atom_name])

    return shape_fn


def get_random_points_in_unit_ball(N):
    p = np.random.normal(0,1,(N, 3))
    norms = np.linalg.norm(p, axis = 1)
    r = np.random.random(N)**(1./3)
    random_in_sphere = p * r[:,np.newaxis] / norms[:,np.newaxis]
    return random_in_sphere



## This is a slighty changed implementation of the masking in EMDA here:
# https://emda.readthedocs.io/en/latest/rst/emda_methods.html?highlight=mask#emda_methods.mask_from_atomic_model
# It is only changed to take inputs in a more friendly format for this code.
def mask_from_coordinates(coords, grid_size, pixsize, atmrad=3, binary_mask=False):
    '''
    Generate a mask from ground truth coordinates
    '''
    import scipy.signal
    from emda.core import iotools, restools
    from scipy.ndimage.morphology import binary_dilation
    grid_3d = np.zeros(3*[grid_size], dtype='float')
    arr = grid_3d
    # Coords externally are centered at 0. 
    # This center them at the middle of the grid
    coords2= coords + grid_size / 2 * pixsize 
    x_np = coords2[:,0]
    y_np = coords2[:,1]
    z_np = coords2[:,2]
    uc = pixsize * grid_size
    # now map model coords into the 3d grid. (approximate grid positions)
    x = (x_np * arr.shape[0] / uc)
    y = (y_np * arr.shape[1] / uc)
    z = (z_np * arr.shape[2] / uc)
    for ix, iy, iz in zip(x, y, z):
        grid_3d[int(round(ix)), 
                int(round(iy)),
                int(round(iz))] = 1.0
    # now convolute with sphere
    kern_rad = int(atmrad / pixsize) + 1
    print("kernel radius: ", kern_rad)
    grid2 = scipy.signal.fftconvolve(
        grid_3d, restools.create_binary_kernel(kern_rad), "same")
    grid2_binary = grid2 > 1e-5
    # dilate
    dilate = binary_dilation(grid2_binary, iterations=1)
    # smoothening
    mask = scipy.signal.fftconvolve(
        dilate, restools.softedgekernel_5x5(), "same")
    mask = mask * (mask >= 1.e-5)
    mask = np.where(grid2_binary, 1.0, mask)
    return mask

### CTF functions 
def CTF_1D(k, defocus, wavelength, Cs, alpha, B):
    return np.sin(-np.pi*wavelength*defocus * k**2 + np.pi/2 * Cs * wavelength**3 * k **4  - alpha) * np.exp(- B * k**2 / 4)

def CTF(psi, defocus, wavelength, Cs, alpha, B):
    k = np.linalg.norm(psi, axis = -1)
    return CTF_1D(k, defocus, wavelength, Cs, alpha, B)

def get_CTF_on_grid(image_shape, voxel_size, defocus, wavelength, Cs, alpha, B):
    psi = get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled = True)
    return CTF(psi, defocus, wavelength, Cs, alpha, B)

def get_gaussian_on_grid(image_shape, voxel_size, width):
    psi = get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled = True)
    k = np.linalg.norm(psi, axis = -1)
    return np.exp(- width * k**2 )

def voltage_to_wavelength(voltage):
    # Borrowed from ASPIRE https://github.com/ComputationalCryoEM/ASPIRE-Python
    """
    Convert from electron voltage to wavelength.
    :param voltage: float, The electron voltage in kV.
    :return: float, The electron wavelength in nm.
    """
    return 12.2643247 / np.sqrt(voltage * 1e3 + 0.978466 * voltage ** 2)

def get_experiment_CTF(name, image_shape, voxel_size):
    if name == "denoising":
        return np.ones(image_shape)
    elif name == "deconvolution":
        # Specify the CTF parameters not used for this example
        # but necessary for initializing the simulation object
        voltage = 200  # Voltage (in KV)
        defocus = 1.5e4
        Cs = 2.0  # Spherical aberration
        alpha = 0.1  # Amplitude contrast
        wavelength = voltage_to_wavelength(voltage)
        Bfac = 80 
        CTF_on_grid = get_CTF_on_grid(image_shape, voxel_size, defocus, wavelength, Cs, alpha, B = Bfac ).reshape(image_shape)
        return CTF_on_grid
