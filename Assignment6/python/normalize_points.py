import numpy as np

def normalize_points(pts):
    """ Computes a normalizing transformation of the points such that
    the points are centered at the origin and their mean distance from
    the origin is equal to sqrt(2).

    See HZ, Ch. 4.4.4: Normalizing transformations (p107).

    Args:
        pts:    Input 2D point array of shape n x 2

    Returns:
        pts_n:  Normalized 2D point array of shape n x 2
        T:      The normalizing transformation in 3x3 matrix form, such
                that for a point (x,y), the normalized point (x',y') is
                found by multiplying T with the point:

                    |x'|       |x|
                    |y'| = T * |y|
                    |1 |       |1|
    """
    u_dist = 0
    v_dist = 0
    tot_dist = 0
    cent_u = np.sum(pts[:, 0])/(len(pts))
    cent_v = np.sum(pts[:, 1])/(len(pts))
    for i in range(len(pts)):
        u_dist += abs(pts[i,0]-cent_u)
        v_dist += abs(pts[i,1]-cent_v)
        tot_dist += np.sqrt((pts[i,0]-cent_u) ** 2 + (pts[i,1]-cent_v) ** 2)
    print(u_dist)
    print(v_dist)
    u_dist = u_dist/len(pts)
    v_dist = v_dist/len(pts)
    tot_dist = tot_dist/len(pts)
    T = np.array([[np.sqrt(2)/tot_dist, 0, -cent_u*np.sqrt(2)/tot_dist],[0 , np.sqrt(2)/tot_dist, -cent_v*np.sqrt(2)/tot_dist],
    [0, 0, 1]])
    # todo: Compute pts_n and T
    pts_hom = np.ones((len(pts), 3))
    for i in range(len(pts)):
        pts_hom[i,0:2] = pts[i]
    pts_n = np.zeros_like(pts)
    for i in range(len(pts)):
        res = pts_hom[i,:]@np.transpose(T)
        pts_n[i,:] = res[0:2]
    print(pts_n)
    mean = np.mean(pts, axis=0)
    dist = np.mean(np.linalg.norm(pts - mean, axis=1))
    print(dist)
    return pts_n, T
