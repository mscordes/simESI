"""
Transforms to ensure proper post-exchange geometries of product water and hydronium.
"""
import numpy as np
import math
import random
from coordinates import set_bond_length

def angle(v1, v2):
    """Calculates angle between two vectors.

    Args:
        v1 (np array): First vector.
        v2 (np array): Second vector.

    Outputs:
        Angle (in degrees) of angle between v1 and v2.
    """
    v1 = np.squeeze(np.asarray(v1))
    v2 = np.squeeze(np.asarray(v2))
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def plane_normal_vector(p1, p2, p3, p4):
    u = p1 - p2
    v = p3 - p4
    normal = np.cross(u, v)
    unit_normal = normal / math.sqrt((normal[0]**2) + (normal[1]**2) + (normal[2]**2))
    return unit_normal


def rotate_bond(p1, p2, p3, theta):
    """Defines plane of rotation between three points and rotates p3 to desired angle.

    Args:
        p1 (np array): Coord of point 1.
        p2 (np array): Coord of point 2.
        p3 (np array): Coord of point 3.
        theta (float): Angle to rotate

    Outputs:
        3D coords of new, rotated p3 loc. 
    """
    k = plane_normal_vector(p1, p2, p2, p3)
    u = p2 - p1
    unorm = math.sqrt(np.dot(u,u))
    v = p2 - p3
    vnorm = math.sqrt(np.dot(v,v))
    theta_old = math.acos(np.dot(u, v)/(unorm*vnorm)) 
    theta_new = math.radians(theta)
    theta = theta_old - theta_new
    #Rodriguez rotation formula
    vrot = v*math.cos(theta) + np.cross(k, v)*math.sin(theta) + k*np.dot(k, v)*(1-math.cos(theta))
    p3 = p2 - vrot
    return p3


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def hydronium_transform(hydronium, atoms):
    """Ensures proper geometry of newly formed hydronium ion product.

    Args:
        hydronium (Residue object): Hydronium resiude.
        atoms (list): Ordered list of Atom objects.

    Outputs:
        If atoms are passed, will output corrected atoms list, else, 
        returns only positionally corrected hydronium Resiude object.
    """
    points = np.array([atom.coord for atom in hydronium])

    #sets length of OH bonds to 0.09686 nm
    points[1] = set_bond_length(points[0], points[1], 0.09686)
    points[2] = set_bond_length(points[0], points[2], 0.09686)
    points[4] = set_bond_length(points[0], points[4], 0.09686)

    #defines plane of hydrogens and sets M to intersect plane with MO bond dist of 0.015nm
    k = plane_normal_vector(points[2], points[1], points[1], points[4])
    u = points[0] - points[2]
    unorm = np.dot(u, k) * k
    utan = u - unorm
    points[3] = utan + points[2] 
    rand = np.array([random.uniform(-0.001, 0.001), random.uniform(-0.001, 0.001), random.uniform(-0.001, 0.001)])
    points[3] = points[3] + rand
    points[3] = set_bond_length(points[0], points[3], 0.015)

    #defines MOH within plane and sets MOH bond to 74.4 degrees
    points[3] = rotate_bond(points[4], points[0], points[3], 74.4)
    points[3] = set_bond_length(points[0], points[3], 0.015)

    #delete two non-exchanged H's and replace by rotating exchanged H 120 degrees 2x
    OM_vector = points[0] - points[3]
    OH_vector = points[0] - points[4]
    points[1] = points[0] - np.dot(rotation_matrix(OM_vector, math.radians(120)), OH_vector)
    points[2] = points[0] - np.dot(rotation_matrix(OM_vector, math.radians(240)), OH_vector)
    points[1] = set_bond_length(points[0], points[1], 0.09686)
    points[2] = set_bond_length(points[0], points[2], 0.09686)

    #update positions
    if atoms is not None:
        for coord in range(5):
            atoms[hydronium[coord].atom_num].coord = points[coord]
        return atoms
    else:
        for coord in range(5):
           hydronium[coord].coord = points[coord]
        return hydronium


def water_transform(water, atoms):
    """Ensures proper geometry of newly formed water product.

    Args:
        water (Residue object): Hydronium resiude.
        atoms (list): Ordered list of Atom objects.

    Outputs:
        If atoms are passed, will output corrected atoms list, else, 
        returns only positionally corrected water Resiude object.
    """
    points = np.array([atom.coord for atom in water])

    #sets length of all OH bonds to 0.09686 nm
    points[1] = set_bond_length(points[0], points[1], 0.09686)
    points[2] = set_bond_length(points[0], points[2], 0.09686)

    #defines plane of HOH and sets virtual site bisecting two H's in plane at bond length of 0.015nm
    u = points[1] - points[0] 
    v = points[2] - points[0]
    un = u / np.linalg.norm(u)
    vn = v / np.linalg.norm(v)
    bisector = un + vn + points[0]
    points[3] = set_bond_length(points[0], bisector, 0.015)

    #sets water bond angle to 104.5, this may need to be tweaked later
    points[1] = rotate_bond(points[2], points[0], points[1], 104.5)
    points[1] = set_bond_length(points[0], points[1], 0.09686)

    #update positions
    if atoms is not None:
        for coord in range(4):
            atoms[water[coord].atom_num].coord = points[coord]
        return atoms
    else:
        for coord in range(4):
           water[coord].coord = points[coord]
        return water