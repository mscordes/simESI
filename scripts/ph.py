"""
Localized pH calculations.
"""
import math
import numpy as np
import hdbscan
from scipy.spatial.distance import cdist

def ph_calc(num_waters, num_hydronium, num_hydroxide):
    """pH calc considering number of waters and hydronium/hydroxide.

    Args:
       num_waters (int)
       num_hydronium (int)
       num_hydroxide (int)

    Outputs:
        Float of computed pH.
    """
    ions = num_hydronium - num_hydroxide
    #if ions negative, its a basic system
    if ions > 0:
        return (-1.0)*(math.log10(55.06794829*(ions/num_waters)))
    elif ions < 0:
        return 14.0+(math.log10(55.06794829*(abs(ions)/num_waters)))
    elif ions == 0:
        return 7.0


def find_clusterpH(waterO_coords, hydroniums, hydroxides):
    """Finds pH of each cluster in system.

    Args:
       waterO_coords (np array): Ordered array of all water oxygen coords.
       hydroniums (list): List of hydronium Resiude class objects.
       hydroxides (list): List of hydroxide Resiude class objects.

    Outputs:
        clusterer.labels_ (np array): Ordered list where each val is the corresponding cluster label of an individual
            water residue, in the same order as the waters in the coordinate file.
        cluster_ph (dict): pH of each cluster given cluster label as key, pH as val.
        cluster_waters (dict): Number of waters in each cluster given cluster label as key, num water as val.
    """
    if len(waterO_coords) < 2:
        clusters = [-1 for _ in waterO_coords]
        cluster_ph = {-1: 7.0, -2: 7.0}
        cluster_waters = {-1: 0, -2: 0}
        return clusters, cluster_ph, cluster_waters
    else:
        # Clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, cluster_selection_epsilon=0.5).fit(waterO_coords)

        # HDBSCAN has tendency to overassign outliers that in reality, should be part of a larger cluster so fix here
        outliers = np.where(clusterer.labels_ ==  -1)[0]
        outlier_coords = np.array([waterO_coords[outlier] for outlier in outliers])
        if len(waterO_coords) > 11 and len(outlier_coords) > 0:
            dist = cdist(outlier_coords, waterO_coords) 
            dist[dist == 0] = np.inf #void distance with self
            min_indices = np.argpartition(dist, 10, axis=1)[:,:10]
            min_dists = dist[np.arange(dist.shape[0])[:, None], min_indices]

            for index, outlier in enumerate(outliers):
                minimum = np.inf
                for nindex, min_dist in enumerate(min_dists[index]):
                    if min_dist < 0.50 and min_dist < minimum and clusterer.labels_[min_indices[index][nindex]] != -1:
                        clusterer.labels_[outlier] = clusterer.labels_[min_indices[index][nindex]]      
                        minimum = min_dist

        # Find number of waters in each cluster and initialize dict for later pH calc
        clusters, waters = np.unique(clusterer.labels_, return_counts=True)
        cluster_waters = dict(zip(clusters, waters))
        cluster_waters[-2] = 0 #empty for gas phase residues
        cluster_dict = {}
        for cluster in range(len(clusters)):
            cluster_name = clusters[cluster]
            cluster_data = {
                "water": waters[cluster],
                "hydronium": 0,
                "hydroxide": 0, 
            }
            cluster_dict[cluster_name] = cluster_data

        # Find number of H3O+ and OH- in each water cluster
        def update_ions(cluster_dict, waterO_coords, ions, name):
            if ions:
                ion_coords = np.array([ion.atoms[0].coord for ion in ions])
                dist = cdist(ion_coords, waterO_coords)
                min_indices = np.argmin(dist, axis=1)
                min_dists = dist[np.arange(len(ions)), min_indices]
                for ion, min_dist in enumerate(min_dists):
                    if min_dist < 0.50:
                        cluster_dict[clusterer.labels_[min_indices[ion]]][name] += 1
        update_ions(cluster_dict, waterO_coords, hydroniums, 'hydronium')
        update_ions(cluster_dict, waterO_coords, hydroxides, 'hydroxide')

        # Calcualte pH
        cluster_ph = {}
        for cluster in clusters:
            cluster_ph[cluster] = ph_calc(cluster_dict[cluster]['water'], cluster_dict[cluster]['hydronium'], cluster_dict[cluster]['hydroxide'])

        # Outliers as neutral pH
        cluster_ph[-1] = 7.0
        cluster_waters[-1] = 1
        return clusterer.labels_, cluster_ph, cluster_waters