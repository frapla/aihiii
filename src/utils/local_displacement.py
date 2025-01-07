import logging

import numpy as np

LOG: logging.Logger = logging.getLogger(__name__)


def get_displ_along_axis(
    nodes_coord: np.ndarray,
    root_coord: np.ndarray,
    direction_coord: np.ndarray,
    as_displacement: bool = True,
    log_lvl: int = 20,
    from_root: bool = False,
) -> np.ndarray:
    """Get local displacement of nodes along axis defined by root and direction point.

    Args:
        nodes_coord (np.ndarray): node coordinates of shape (m_nodes, n_timesteps, o_dimensions)
        root_coord (np.ndarray): node coordinates of root point of axis, shape (n_timesteps, o_dimensions)
        direction_coord (np.ndarray): node coordinates of direction point of axis, shape (n_timesteps, o_dimensions)
        as_displacement (bool, optional): make displacement from coordinate. Defaults to True.
        from_root (bool, optional): use root for distance else direction. Defaults to False.

    Returns:
        np.ndarray: node displacements along axis, shape (m_nodes, n_timesteps)
    """
    LOG.level = log_lvl
    LOG.debug("nodes_coord.shape: %s, expected (m_nodes, n_timesteps, o_dimensions)", nodes_coord.shape)
    LOG.debug("root_coord.shape: %s, expected (n_timesteps, o_dimensions)", root_coord.shape)
    LOG.debug("direction_coord.shape: %s, expected (n_timesteps, o_dimensions)", direction_coord.shape)

    # part vectors
    p1p2 = direction_coord - root_coord
    p1pn = nodes_coord - root_coord

    # project nodes on axis
    p1pn_dot_p1p2 = np.einsum("ijk,jk->ij", p1pn, p1p2)
    p1p2_dst = np.linalg.norm(p1p2, axis=1) ** 2
    divi = p1pn_dot_p1p2 / p1p2_dst
    divi_p1p2 = np.einsum("ij,jk->ijk", divi, p1p2)
    op_projected = root_coord + divi_p1p2

    # get distance
    p4p5 = (root_coord if from_root else direction_coord) - op_projected
    p4p5_dst = np.linalg.norm(p4p5, axis=2)

    # make displacement relative to t0
    if as_displacement:
        dst_t0 = p4p5_dst[:, 0]
        p4p5_dst = (p4p5_dst.T - dst_t0).T

    LOG.debug("distances.shape: %s, expected (m_nodes, n_timesteps)", p4p5_dst.shape)

    return p4p5_dst
