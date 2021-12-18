from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

from sklearn.manifold import TSNE
from sklearn import datasets
from matplotlib.tri import Triangulation



# Render the 2D scatterplot with crack/strain layers.
def draw_main(ax, vor, src, f_crack, f_strain, **kw):
    from matplotlib.collections import LineCollection
    
    # Frame viewport appropriately
    _adjust_bounds(ax, vor.points)
    
    #ax.tripcolor(triangularize(vor),np.clip(compute_vert_weights_adjusted(vor, src, f_strain),0,1),cmap='RdPu',shading='gouraud')
    #draw_strain_lines(ax, vor, compute_ridge_weights(vor, src, f_strain))
    
    ridge_weights = compute_ridge_weights(vor, src, f_crack)
    render_cracks(ax,vor,ridge_weights)
    
    strain_points = compute_strain_points(vor)
    strain_weights = compute_ridge_weights(vor, src, f_strain)
    
    strain_points = np.concatenate([strain_points,vor.points])
    
    strain_weights-= np.median(strain_weights)
    strain_weights=np.concatenate([strain_weights,np.zeros(len(vor.points))])
    
    render_strain_map(ax,kw.get('res',100),strain_points,strain_weights,kw.get('beta',1),kw.get('gamma',1))
    
    # Draw points
    point_size = kw.get('point_size', None)
    if 'values' in kw:
        point_values = kw.get('values', np.ones(vor.points.shape[0]))
        ax.scatter(vor.points[:,0], vor.points[:,1], s=point_size, c=point_values)
    else:
        point_color=kw.get('point_color','black')
        ax.scatter(vor.points[:,0], vor.points[:,1], s=point_size, color=point_color)
    
    return ax.figure

# Copied from source code of scipy.spatial.voronoi_plot_2d
def _adjust_bounds(ax, points):
    margin = 0.1 * points.ptp(axis=0)
    xy_min = points.min(axis=0) - margin
    xy_max = points.max(axis=0) + margin
    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])

# Compute and draw the strain image onto the specified axes.
def render_strain_map(ax,res,strain_points,strain_weights,beta,gamma):
    # Sample grid points within the plot
    xx = np.linspace(*ax.get_xlim(),res)
    yy = np.linspace(*ax.get_ylim(),res)
    
    zz = grid_map(xx, yy, lambda pt: compute_strain_field_value(pt,strain_points,strain_weights,beta,gamma))
    
    z_max = np.abs(np.max(zz))
    
    ax.imshow(zz,cmap='RdPu',vmin=0, vmax=z_max,interpolation='bicubic',alpha=0.5,extent=[*ax.get_xlim(),yy.max(),yy.min()])
    
def compute_ridge_weights(vor,src, f):
    weights = np.zeros(len(vor.ridge_points))
    for ridgeidx, pointidx in enumerate(vor.ridge_points):
        weights[ridgeidx] = f(vor, src, pointidx[0],pointidx[1])
    
    return weights

def compute_strain_points(vor):
    res = np.zeros((len(vor.ridge_points),2))
    
    for ridgeidx, pointidx in enumerate(vor.ridge_points):
        
        p0 = vor.points[pointidx[0]]
        p1 = vor.points[pointidx[1]]
        
        res[ridgeidx,:] = 0.5*(p0+p1)
    
    return res
    
# Render the crack layer.
# Based on source code of scipy.spatial.voronoi_plot_2d.
def render_cracks(ax, vor, ridge_weights):

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)
        
    normalizer = 1.0/np.mean(ridge_weights)
    
    for ridgeidx, (pointidx, simplex) in enumerate(zip(vor.ridge_points, vor.ridge_vertices)):
        simplex = np.asarray(simplex)
        
        # Opacity value of this line segment
        weight = np.clip(np.log(normalizer*ridge_weights[ridgeidx]),0.05,1)
        
        if np.all(simplex >= 0):
            # Ridge is line segment    
            vert = vor.vertices[simplex]
            l = mlines.Line2D(vert[:,0], vert[:,1], color=(0,0,0,weight))
            ax.add_line(l)
            
        else:
            # Ridge is an infinite ray
            
            i = simplex[simplex >= 0][0] # finite endpoint

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            
            # Compute virtual endpoint outside plot border
            far_point = vor.vertices[i] + direction * ptp_bound.max()
            
            vert = np.array([vor.vertices[i], far_point])
            
            l = mlines.Line2D(vert[:,0],vert[:,1], color=(0,0,0,weight))
            ax.add_line(l)

def f_standard_crack(vor, src, i0, i1):
    p0 = vor.points[i0]
    p1 = vor.points[i1]
    dist2d = np.linalg.norm(p1-p0)
    
    pp0 = src[i0,:]
    pp1 = src[i1,:]
    
    distNd = np.linalg.norm(pp1-pp0)
    
    return dist2d/distNd

def f_standard_strain(vor, src, i0, i1):
    p0 = vor.points[i0]
    p1 = vor.points[i1]
    dist2d = np.linalg.norm(p1-p0)
    
    pp0 = src[i0,:]
    pp1 = src[i1,:]
    
    distNd = np.linalg.norm(pp1-pp0)
    
    return np.log(distNd/dist2d)

def compute_strain_field_value(v, field_pt, field_w, beta, gamma):
    distances = np.linalg.norm(field_pt-v,axis=1)
    
    # Append synthetic zero-weighted source at distance gamma
    distances = np.append(distances,gamma)
    weights = np.append(field_w,0)
    
    # Exponential falloff
    scales = np.power(0.5,distances*beta)
    
    # Take weighted average
    return (np.sum(np.multiply(scales,weights)) / np.sum(scales))
    

def grid_map(xx,yy,f):
    res = np.zeros([len(yy),len(xx)])
    
    for idx in np.ndindex(res.shape):
        pt = np.array([xx[idx[1]],yy[idx[0]]])
        res[idx] = f(pt)
    
    return res

#########################

# [Unused]
def draw_strain_lines(ax,vor,ridge_weights):
    cue = interp1d([np.quantile(ridge_weights,0.4),np.quantile(ridge_weights,0.8)],[0,1],fill_value="extrapolate")
    
    for ridgeidx, pointidx in enumerate(vor.ridge_points):
        
        weight = np.clip(cue(ridge_weights[ridgeidx]),0,1)
        pt = vor.points[pointidx]
        l = mlines.Line2D(pt[:,0],pt[:,1], color=(1,0,0,weight), lw=2)
        ax.add_line(l)


# [Unused] Interpolate ridge weights to compute corner weights
def compute_vert_weights(vor,src,weight_fn):
    n_vert = len(vor.vertices)
    vert_weights = np.zeros(n_vert,dtype=float)
    vert_deposits = np.zeros(n_vert,dtype=float)
    
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        weight = weight_fn(vor,src,pointidx[0],pointidx[1])
        
        # normalize by edge length
        ridge_len = np.linalg.norm(vor.vertices[simplex[1]]-vor.vertices[simplex[0]])
        weight = weight / ridge_len;
        
        for i in [0,1]:
            if(simplex[i] >= 0):
                vert_weights[simplex[i]] += weight
                vert_deposits[simplex[i]] += 1
    
    return np.divide(vert_weights,vert_deposits,out=vert_weights,where=vert_deposits!=0)
    
# [Unused]
def compute_vert_weights_adjusted(vor,src,weight_fun):
    weights = compute_vert_weights(vor,src,weight_fun)
    weights = np.divide(weights,2*np.median(weights))
    return np.concatenate([weights,np.zeros(len(vor.points),dtype=float)])
    
# [Unused]
# Convert the Voronoi diagram to a triangular mesh.
def triangularize(vor):
    
    offset = len(vor.vertices)
    
    vert = np.concatenate([vor.vertices,vor.points])
    tri = []
    
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        
        if not np.all(simplex >= 0):
            continue
        
        tri.append([offset+pointidx[0],simplex[0],simplex[1]])
        tri.append([offset+pointidx[1],simplex[1],simplex[0]])
    
    return Triangulation(vert[:,0],vert[:,1],tri)
    
# [Unused]
# Compute the Voronoi diagram to a triangular mesh,
# with an additional vertex at the midpoint of each ridge.
def triangularize2(vor):
    
    offset1 = len(vor.vertices)
    offset2 =  offset1 + len(vor.points)
    
    mids = []
    tri = []
    
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        
        p0 = vor.points[pointidx[0]]
        p1 = vor.points[pointidx[1]]
        
        if np.all(simplex >= 0):
            midpoint =  0.5* (vor.vertices[simplex[0]]+vor.vertices[simplex[1]])
        else:
            midpoint = 0.5* (vor.points[pointidx[0]]+vor.points[pointidx[1]])
        
        if simplex[0] >= 0:
            tri.append([offset2+len(mids), offset1+pointidx[0], simplex[0]])
            tri.append([offset2+len(mids), offset1+pointidx[1], simplex[0]])
        if simplex[1] >= 0:
            tri.append([offset2+len(mids), offset1+pointidx[0], simplex[1]])
            tri.append([offset2+len(mids), offset1+pointidx[1], simplex[1]])
        
        mids.append(midpoint)
    
    vert = np.concatenate([vor.vertices,vor.points,np.array(mids)])
    
    return Triangulation(vert[:,0],vert[:,1],tri)

# [Unused]
def compute_vert_weights2(vor,src,weight_fn):
    offset1 = len(vor.vertices)
    offset2 =  offset1 + len(vor.points)
    
    n_vert = offset2 + len(vor.ridge_points)
    
    vert_weights = np.zeros(n_vert,dtype=float)
    vert_deposits = np.zeros(n_vert,dtype=float)
    
    t = offset2
    
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        if(simplex[0]>=0 and simplex[1] >= 0):
            weight = weight_fn(vor,src,pointidx[0],pointidx[1]) / np.linalg.norm(vor.vertices[simplex[1]]-vor.vertices[simplex[0]])
            
        else:
            weight = 0
        
        
        vert_weights[t] = weight
        vert_deposits[t] = 1
        t += 1
        
        for i in [0,1]:
            if(simplex[i] >= 0):
                vert_weights[simplex[i]] += weight
                vert_deposits[simplex[i]] += 1
        
        
    
    return np.divide(vert_weights,vert_deposits,out=vert_weights,where=vert_deposits!=0)
    
# [Unused]
def compute_vert_weights2_adjusted(vor,src, weight_fun):
    weights = compute_vert_weights2(vor,src,weight_fun)
    weights = np.divide(weights,2*np.mean(weights))
    return weights
        
        