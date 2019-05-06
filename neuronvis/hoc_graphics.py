from __future__ import print_function
from __future__ import absolute_import
import sys
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from . import mplcyl
from mayavi import mlab

Colors = { # colormap
    'b': np.array([0,0,255,255])/255.,
    'blue': np.array([0,0,255,255])/255.,
    'g': np.array([0,255,0,255])/255.,
    'green': np.array([0,255,0,255])/255.,
    'r': np.array([255,0,0,255])/255.,
    'red': np.array([255,0,0,255])/255.,
    'c': np.array([0,255,255,255])/255.,
    'cyan': np.array([0,255,255,255])/255.,
    'm': np.array([255,0,255,255])/255.,
    'magenta': np.array([255,0,255,255])/255.,
    'y': np.array([255,255,0,255])/255.,
    'yellow': np.array([255,255,0,255])/255.,
    'k': np.array([0,0,0,255])/255.,
    'black': np.array([0,0,0,255])/255.,
    'w': np.array([255,255,255,255])/255.,
    'white': np.array([255,255,255,255])/255.,
    'd': np.array([150,150,150,255])/255.,
    'dark': np.array([150,150,150,255])/255.,
    'l': np.array([200,200,200,255])/255.,
    'light': np.array([200,200,200,255])/255.,
    's': np.array([100,100,150,255])/255.,
    'powderblue': np.array([176,230,230,255])/255.,
    'brown': np.array([180,25,25,255])/255.,
    'orange': np.array([255,180,0,255])/255.,
    'pink': np.array([255,190,206,255])/255.,
}

colorMap = ['b', 'g', 'r', 'c', 'y', 'm', 'powderblue', 'brown', 'orange', 'pink']

def compute_cube(cube_definition):
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)
    edges = [
            [points[0], points[1]],
            [points[0], points[2]],
            [points[0], points[3]],
            [points[1], points[4]],
            [points[1], points[5]],
            [points[2], points[4]],
            [points[2], points[6]],
            [points[3], points[5]],
            [points[3], points[6]],
            [points[4], points[7]],
            [points[5], points[7]],
            [points[6], points[7]],
        ]
    return points, edges

# mayavi functions
def refaxes(scene=None, ext=[80, 75, 55]):      
    cube_definition = [
        (-ext[0],-ext[1],-ext[2]), (-ext[0],ext[1],-ext[2]), (ext[0],-ext[1],-ext[2]), (-ext[0],-ext[1],ext[2])
    ]
    points, edges = compute_cube(cube_definition)
    for j, e in enumerate(edges):
        xyz = []
        for k in range(3):
            xyz.append([e[0][k], e[1][k]])
        mlab.plot3d(xyz[0], xyz[1], xyz[2], color=(0.7, 0.7, 0.7), tube_radius=0.125)
        
    
def reflines(scene=None, ext=[80, 75, 55]):      
    x0 = [-ext[0], ext[0],     0.,    0.,    0.,   0.]
    y0 = [   0.,   0.,  -ext[1],  ext[1],    0.,   0.]
    z0 = [   0.,   0.,     0.,    0., -ext[2], ext[2]]
    colc = [(1,0,0), (0,1,0), (0,0,1)]
    # axisname = [f"x ({ext[0]:.0f})", 'y', 'z']
    axisl = ['x', 'y', 'z']
    for j, i in enumerate([0, 2, 4]):
        mlab.plot3d(x0[i:i+2], y0[i:i+2], z0[i:i+2], color=colc[j], tube_radius=0.1)
        axisname = f"{axisl[j]:s} ({ext[j]:+.0f})"
        mlab.text3d(x0[i+1], y0[i+1], z0[i+1], axisname, figure=scene, scale=3.0)
        


class HocGraphic(object):
    """
    Methods common to all Hoc graphical representation classes (HocVolume, 
    HocSurface, etc.)
    
    """
    def __init__(self, h, parentItem=None, **kwds):
        self.h = h

    def get_color_map(self, i):
        return colorMap[i]

    def set_section_colors(self, colors):
        """
        Recolor the graphic by section using the *colors* array. This method
        must be reimplemented by HocGraphic subclasses. The order of elements
        in the array must match the order of sections defined in 
        HocReader.sections.
        """
        raise NotImplementedError()
    
    def set_group_colors(self, colors, default_color=(0.5,0.5,0.5,0.5), alpha=None, mechanism=None, colormap=None):
        """
        Color the sections in the reconstruction according to their
        group name.
        Inputs: 
            colors: a dictionary of section group names and their associated colors
            default_color: color to use for any sections that are not included in the
                           groups listed in *colors*.
            alpha: If specified, this overrides the alpha value for all group colors.
        Side-effects: none.
        """
        sec_colors = np.zeros((len(self.h.sections), 4), dtype=float)
        sec_colors[:] = default_color
        mechmax = 0.
        for group_name, color in colors.items():
            try:
                sections = self.h.get_section_group(group_name)
            except KeyError:
                continue
            
            for sec_name in sections:
                if isinstance(color, str):
                    color = Colors[color]
                index = self.h.sec_index[sec_name]
                sec_colors[index] = color
                if mechanism is not None:
                    g = self.h.get_density(self.h.sections[sec_name], mechanism)
                    #print ('section: %s, gmax = %f' % (sec_name, g))
                    mechmax  = max(mechmax, g)
                    sec_colors[index,3] = g
                elif alpha is not None:
                    sec_colors[index, 3] = alpha
           # print (mechmax)
        if mechanism is not None and mechmax > 0:
            sec_colors[:,3] = 0.05 + 0.95*sec_colors[:,3]/mechmax # set alpha for all sections
        self.set_section_colors(sec_colors)

class HocGrid(HocGraphic, gl.GLGridItem):
    """
    subclass of GLGridItem to draw a grid on the field
    """
    def __init__(self, size=(40, 40, 40), spacing=(20, 20, 20)):
        super(HocGraphic, self).__init__()
        self.grid = gl.GLGridItem(color=pg.mkColor(128, 128, 128, 255))
        return
        self.grid.setSize(x=size[0], y=size[1], z=size[2])  # 100 um grid spacing
        self.grid.setSpacing(x=spacing[0], y=spacing[1], z=spacing[2])  # 10 um steps
        self.grid.scale(1,1,1)  # uniform scale
        self.grid.translate(100., 0., 0.)
        super(HocGrid, self).__init__(size, spacing, color=pg.mkColor(128, 128, 128, 255))


class HocGraph(HocGraphic, gl.GLLinePlotItem):
    """
    Subclass of GLLinePlotItem that draws a line representation of the geometry
    specified by a HocReader.
    
    Input:
        h: HocReader instance
    """
    def __init__(self, h, parentItem=None):
        super(HocGraphic, self).__init__()
        self.h = h
        # self.set_section_colors = self.set_section_colors
        verts, edges = h.get_geometry()
        # Prefer this method, but item does not support per-vertex width:
        #edges = edges.flatten()
        verts_indexed = verts[edges]
        self.vertex_sec_ids = verts_indexed['sec_index']
        
        # 
        self.lines = []
        for edge in edges:
            w = (verts['dia'][edge[0]] + verts['dia'][edge[1]]) * 0.5
            self.lines.append(gl.GLLinePlotItem(pos=verts['pos'][edge], width=w))
            self.lines[-1].setParentItem(self)
        super(HocGraph, self).__init__(h, pos=verts_indexed['pos'], mode='lines')

    def set_section_colors(self, sec_colors):
        colors = sec_colors[self.vertex_sec_ids]
        self.setData(color=colors)
        

class HocVolume(HocGraphic, gl.GLVolumeItem):
    """
    Subclass of GLVolumeItem that draws a volume representation of the geometry
    specified by a HocReader.
    
    Input:
        h: HocReader instance
    """
    def __init__(self, h):
        super(HocGraphic, self).__init__()
        self.h = h
        scfield, idfield, transform = self.h.make_volume_data()
        nfdata = np.empty(scfield.shape + (4,), dtype=np.ubyte)
        nfdata[...,0] = 255 #scfield*50
        nfdata[...,1] = 255# scfield*50
        nfdata[...,2] = 255# scfield*50
        nfdata[...,3] = np.clip(scfield*150, 0, 255)
        super(HocVolume, self).__init__(nfdata)
        self.setTransform(transform)


class mayavi_Volume(object):
    def __init__(self, h):
        self.h = h
        scfield, idfield, transform = self.h.make_volume_data()
        nfdata = np.empty(scfield.shape + (4,), dtype=np.ubyte)
        nfdata[...,0] = 255 #scfield*50
        nfdata[...,1] = 255# scfield*50
        nfdata[...,2] = 255# scfield*50
        nfdata[...,3] = np.clip(scfield*150, 0, 255)
        # f = mlab.figure() # Returns the current scene.
        # engine = mlab.get_engine() # Returns the running mayavi engine.
        # scene  = engine.new_scene()
        refaxes()
        mlab.pipeline.volume(mlab.pipeline.scalar_field(scfield), vmin=0.24, vmax=0.25)
        reflines()
    
class HocSurface(HocGraphic, gl.GLMeshItem):
    """
    Subclass of GLMeshItem that draws a surface representation of the geometry
    specified by a HocReader. The surface is generated as an isosurface of
    the scalar field generated by measuring the minimum distance from all
    cell membranes across the volume of the cell.
    
    Input:
        h: HocReader instance
    """
    def __init__(self, h):
        super(HocGraphic, self).__init__()
        super(HocSurface, self).__init__(h)
        self.h = h
        scfield, idfield, transform = self.h.make_volume_data()
        #scfield = scipy.ndimage.gaussian_filter(scfield, (0.5, 0.5, 0.5))
        #pg.image(scfield)
        verts, faces = pg.isosurface(scfield, level=0.0)
        self.verts = verts
        self.faces = faces
        vertexColors = np.empty((verts.shape[0], 4), dtype=float)
        md = gl.MeshData(vertexes=verts, faces=faces)

        # match vertexes to section IDs
        vox_locations = verts.astype(int)
        # get sction IDs for each vertex
        self.vertex_sec_ids = idfield[vox_locations[:,0], vox_locations[:,1], vox_locations[:,2]] 
        self.setMeshData(meshdata=md, smooth=True, shader='balloon')
        self.setTransform(transform)
        self.setGLOptions('additive')

    def show_section(self, sec_id, color=(1, 1, 1, 1), bg_color=(1, 1, 1, 0)):
        """
        Set the color of section named *sec_id* to *color*.
        All other sections are colored with *bg_color*.
        """
        colors = np.empty((len(self.vertex_sec_ids), 4))
        colors[:] = bg_color
        colors[sec_id] = color
        self.set_section_colors(colors)
    
    def set_section_colors(self, sec_colors):
        """
        Set the colors of multiple sections.
        
        Input:
            colors: (N,4) float array of (r,g,b,a) colors, in the order that
                    sections are defined by the HocReader.
        """
        colors = sec_colors[self.vertex_sec_ids]
        self.opts['meshdata'].setVertexColors(colors)

        self.meshDataChanged()


class mayavi_graph(object):
    def __init__(self, h, color=(0,0,1), label=None):
        self.h = h
        #plot_tc(p0=np.array([1, 3, 2]), p1=np.array([8, 5, 9]), R=[5.0, 2.0])
        verts, edges = h.get_geometry()
        sec_id = []
        XC = []
        YC = []
        ZC = []
        S = []
        connections = []
        index = 0
        for edge in edges:
            ends = verts['pos'][edge]  # xyz coordinate of one end [x,y,z]
            dia = verts['dia'][edge]  # diameter at that end
            sec_id = verts['sec_index'][edge[0]]  # save the section index
            
            dif = ends[1]-ends[0]  # distance between the ends
            length = (dif**2).sum() ** 0.5
            X = [ends[0][0], ends[1][0]]
            Y = [ends[0][1], ends[1][1]]
            Z = [ends[0][2], ends[1][2]]
            S.extend(dia)
            XC.extend(X)
            YC.extend(Y)
            ZC.extend(Z)
            N = len(X)
            connections.append(np.vstack(
                                   [np.arange(index,   index + N - 1.5),
                                    np.arange(index + 1, index + N - 0.5)]
                                        ).T)
            index += N

        XC = np.hstack(XC)
        YC = np.hstack(YC)
        ZC = np.hstack(ZC)
        S = np.hstack(S)
        # print(XC.shape, S.shape)
        connections = np.vstack(connections)
        # Create the points
        src = mlab.pipeline.scalar_scatter(XC, YC, ZC, S)
        # src.parent.parent.filter.vary_radius = 'vary_radius_by_scalar'

        # Connect them
        src.mlab_source.dataset.lines = connections
        src.update()

        # The stripper filter cleans up connected lines
        lines = mlab.pipeline.stripper(src)

        # Finally, display the set of lines
        t = mlab.pipeline.surface(lines, color=color, line_width=2.0, opacity=1.0)
        refaxes()
        reflines()
        if label is not None:
            mlab.text3d(XC[0], YC[0], ZC[0], f"{label:s}", figure=None, scale=1.5, color=color)



            
class mayavi_Cylinders(object):
    def __init__(self, h, color=(0,0,1), label=None, flags=None):
        self.h = h
        hcyl = mplcyl.TruncatedCone()
        #plot_tc(p0=np.array([1, 3, 2]), p1=np.array([8, 5, 9]), R=[5.0, 2.0])
        verts, edges = h.get_geometry()
        # print ('edges', edges)
        # print ('verts[pos]: ', verts['pos'])
        # print ('verts', verts)
        meshes = []
        sec_ids = []

        XC = []
        YC = []
        ZC = []
        S = []
        connections = []
        index = 0
        lastend = None
        ne = len(edges)
        # print('# edges: ', ne)
        ndone = 0
        for ind, edge in enumerate(edges): # # of edges corresponds to N-1 section indiators; and to # of cones
            ends = verts['pos'][edge]  # xyz coordinate of one end [x,y,z]
            dia = verts['dia'][edge]  # diameter at that end
            C, T, B = hcyl.make_truncated_cone(p0=ends[1], p1=ends[0], R=[dia[1]/2., dia[0]/2.])

            XC.extend(C[0])
            YC.extend(C[1])
            ZC.extend(C[2])
            XC.append(np.array([np.nan, np.nan]))  # disconnect adjacent cylinders to avoid "strings"
            YC.append(np.array([np.nan, np.nan]))
            ZC.append(np.array([np.nan, np.nan]))
            # mlab.mesh(XC, YC, ZC, color=color, line_width=0.0) # mesh so far
                                
            N = len(C[0])
            connections.append(np.vstack(
                                   [np.arange(index,   index + N - 1.5),
                                    np.arange(index + 1, index + N - 0.5)]
                                        ).T)            

            index += N
            # print('XC: ', len(XC), '\n', XC)
        # XC = np.hstack(XC)
        # YC = np.hstack(YC)
        # ZC = np.hstack(ZC)
        # S = np.hstack(S)

        # connections = np.vstack(connections)
        # # Create the points
        # src = mlab.pipeline.scalar_scatter(XC, YC, ZC)
        # src = mlab.pipeline.grid_source(XC, YC, ZC)
        # src.parent.parent.filter.vary_radius = 'vary_radius_by_scalar'

        # # Connect them
        # src.mlab_source.dataset.lines = connections
#         src.update()
#         # #
#         # # # The stripper filter cleans up connected lines
#         lines = mlab.pipeline.stripper(src)
## mesh = mlab.pipeline.delaunay3d(src)
        # Finally, display the set of lines
        # surface(lines... )
        # t = mlab.pipeline.surface(src, color=color, line_width=.0, opacity=1)
        mlab.mesh(XC, YC, ZC, color=color, line_width=0.0, opacity=1.0)
        # self.cursor3d = mlab.points3d(0., 0., 0., mode='2darrow',
        #                         color=(1, 1, 1), [-10, 10, -10, 10, -10, 10],
        #                         scale_factor=0.5)
        if flags is not None:
            if 'norefaxes' not in flags:
                refaxes()
            if 'norefline' not in flags:
                reflines()
        else:
            refaxes()
            reflines()
        
        if label is not None and 'text' in flags:
            # print('cylinder: ', label, XC[0], YC[0], ZC[0])
            mlab.text3d(XC[0][0]+5, YC[0][0], ZC[0][0], f"{label:s}", color=color, figure=None, scale=2.0)

        #mlab.plot3d(XC, YC, ZC, color=color, tube_radius=0.5)
       #  ext = 200.
       #  x0 = [-ext, ext,     0.,    0.,    0.,   0.]
       #  y0 = [   0.,   0.,  -ext,  ext,    0.,   0.]
       #  z0 = [   0.,   0.,     0.,    0., -ext, ext]
       #  colc = [(1,0,0), (0,1,0), (0,0,1)]
       #  axisname = [f"x ({ext:.0f})", 'y', 'z']
       #  for j, i in enumerate([0, 2, 4]):
       #      mlab.plot3d(x0[i:i+2], y0[i:i+2], z0[i:i+2], color=colc[j], tube_radius=1.0)
       #      mlab.text3d(x0[i], y0[i], z0[i], f"{axisname[j]:s}", scale=12.0)


        # mlab.mesh(X, Y, Z)


class mpl_Cylinders(object):
    """
    Input:
        h: HocReader instance
    """

    def __init__(self, h, mpl=True, fax=None):
        self.h = h
        hcyl = mplcyl.TruncatedCone()
        #plot_tc(p0=np.array([1, 3, 2]), p1=np.array([8, 5, 9]), R=[5.0, 2.0])
        verts, edges = h.get_geometry()
        # print 'verts', verts
        # print 'edges', edges
        # print verts['pos']
        # self.hg = HocGraphic(h)
        # self.hg.set_section_colors = self.set_section_colors  
        # super(HocCylinders, self).__init__()
        meshes = []
        sec_ids = []
        if mpl and fax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = fax[0]
            ax = fax[1]
        for edge in edges:
            ends = verts['pos'][edge]  # xyz coordinate of one end [x,y,z]
            dia = verts['dia'][edge]  # diameter at that end
            sec_id = verts['sec_index'][edge[0]]  # save the section index
            
            dif = ends[1]-ends[0]  # distance between the ends
            length = (dif**2).sum() ** 0.5
            C, T, B = hcyl.make_truncated_cone(p0=ends[0], p1=ends[1], R=[dia[0]/2., dia[1]/2.])
            mesh_verts = C
#            print mesh_verts
            
            # sec_id_array = np.empty(mesh_verts.shape[0]*3, dtype=int)
            # # sec_id_array[:] = sec_id
            # meshes.append(mesh_verts)
            # sec_ids.append(sec_id_array)
            meshes.append(mesh_verts)
            if mpl:
                ax.plot_surface(C[0], C[1], C[2], color='blue', linewidth=1, antialiased=False)

        if mpl:
            self.axisEqual3D(ax)
            if fax is None:
                plt.show()
        else:
            s = mlab.mesh(meshes[0], meshes[1], meshes[2])
            if fax is None:
                mlab.show()
        # exit(1)

    def axisEqual3D(self, ax):
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        ax.auto_scale_xyz(*np.column_stack((centers - r, centers + r)))

class mpl_Graph(object):
    """
    Input:
        h: HocReader instance
    """

    def __init__(self, h, mpl=True, fax=None, color='blue'):
        self.h = h
        # hcyl = mplcyl.TruncatedCone()

        #plot_tc(p0=np.array([1, 3, 2]), p1=np.array([8, 5, 9]), R=[5.0, 2.0])
        verts, edges = h.get_geometry()
        # print 'verts', verts
        # print ('edges', edges)
        # print (verts['pos'])
        # self.hg = HocGraphic(h)
        # self.hg.set_section_colors = self.set_section_colors  
        # super(HocCylinders, self).__init__()
        meshes = []
        sec_ids = []
        if mpl and fax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = fax[0]
            ax = fax[1]
        for edge in edges:
            ends = verts['pos'][edge]  # xyz coordinate of one end [x,y,z]
            dia = verts['dia'][edge]  # diameter at that end
            # print('ends: ', ends)
            # print(dia)
            sec_id = verts['sec_index'][edge[0]]  # save the section index
            X = [ends[0][0], ends[1][0]]
            Y = [ends[0][1], ends[1][1]]
            Z = [ends[0][2], ends[1][2]]
            if mpl:
                ax.plot(X, Y, Z, color=color, linewidth=0.5, antialiased=False)

        if mpl:
            self.axisEqual3D(ax)
            if fax is None:
                plt.show()
        # exit(1)

    def axisEqual3D(self, ax):
        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        ax.auto_scale_xyz(*np.column_stack((centers - r, centers + r)))


        
"""
        TRY VISPY
"""

# from vispy import app, gloo, visuals
# from vispy.geometry import create_sphere
# from vispy.geometry import create_cylinder
# from vispy.geometry import create_grid_mesh
#
# from vispy.visuals.transforms import (STTransform, MatrixTransform,
#                                       ChainTransform)
#
# class vispy_Cylinders(app.Canvas):
#     """
#     Input:
#         h: HocReader instance
#     """
#
#     def __init__(self, h):
#         self.h = h
#         app.Canvas.__init__(self, keys='interactive', size=(800, 550))
#
#         hcyl = mplcyl.TruncatedCone()
#         print('1')
#         #plot_tc(p0=np.array([1, 3, 2]), p1=np.array([8, 5, 9]), R=[5.0, 2.0])
#         verts, edges = h.get_geometry()
#
#         self.meshes = []
#         self.rotation = MatrixTransform()
#         sec_ids = []
#         s = 1.0
#         x, y = 0., 0.
#         for edge in edges:
#             ends = verts['pos'][edge]  # xyz coordinate of one end [x,y,z]
#             dia = verts['dia'][edge]  # diameter at that end
#             sec_id = verts['sec_index'][edge[0]]  # save the section index
#
#             dif = ends[1]-ends[0]  # distance between the ends
#             length = (dif**2).sum() ** 0.5
#             # print length
#             # print dia
#             #C, T, B = hcyl.make_truncated_cone(p0=ends[0], p1=ends[1], R=[dia[0]/2., dia[1]/2.])
#             mesh_verts =  create_cylinder(8, 8, radius=[dia[0]/2., dia[1]/2.], length=length, offset=False)
#             #mesh_verts = create_grid_mesh(C[0], C[1], C[2])
#
#
#             # sec_id_array = np.empty(mesh_verts.shape[0]*3, dtype=int)
#             # # sec_id_array[:] = sec_id
#             # meshes.append(mesh_verts)
#             # sec_ids.append(sec_id_array)
#             self.meshes.append(visuals.MeshVisual(meshdata=mesh_verts, color='r'))
#
# #             transform = ChainTransform([STTransform(translate=(x, y),
# #                                                     scale=(s, s, s)),
# #                                         self.rotation])
# #
# #         for i, mesh in enumerate(self.meshes):
# # #            x = 800. * (i % grid[0]) / grid[0] + 40
# #             mesh.transform = transform
# #             mesh.transforms.scene_transform = STTransform(scale=(1, 1, 0.01))
#
#         gloo.set_viewport(0, 0, *self.physical_size)
#         gloo.clear(color='white', depth=True)
#
#         for mesh in self.meshes:
#             mesh.draw()
#
#         print('running')
#         self.show()
#         if sys.flags.interactive != 1:
#             app.run()
#         #exit(1)
#
#     def rotate(self, event):
#         # rotate with an irrational amount over each axis so there is no
#         # periodicity
#         self.rotation.rotate(0.2 ** 0.5, (1, 0, 0))
#         self.rotation.rotate(0.3 ** 0.5, (0, 1, 0))
#         self.rotation.rotate(0.5 ** 0.5, (0, 0, 1))
#         self.update()
#
#     def on_resize(self, event):
#         # Set canvas viewport and reconfigure visual transforms to match.
#         vp = (0, 0, self.physical_size[0], self.physical_size[1])
#         self.context.set_viewport(*vp)
#
#         for mesh in self.meshes:
#             mesh.transforms.configure(canvas=self, viewport=vp)
#
#     def on_draw(self, ev):
#         gloo.set_viewport(0, 0, *self.physical_size)
#         gloo.clear(color='black', depth=True)
#
#         for mesh in self.meshes:
#             mesh.draw()
#
#


   
class HocCylinders(HocGraphic, gl.GLMeshItem):
    """
    Subclass of GLMeshItem that draws a cylinder representation of the geometry
    specified by a HocReader.
    
    Input:
        h: HocReader instance
    """
    def __init__(self, h):
        super(HocGraphic, self).__init__()
        self.h = h
        verts, edges = h.get_geometry()
        
        meshes = []
        sec_ids = []
        for edge in edges:
            ends = verts['pos'][edge]
            dia = verts['dia'][edge]
            sec_id = verts['sec_index'][edge[0]]
            
            dif = ends[1]-ends[0]
            length = (dif**2).sum() ** 0.5
            
            mesh = gl.MeshData.cylinder(rows=1, cols=8, radius=[dia[0]/2., dia[1]/2.], length=length)
            mesh_verts = mesh.vertexes(indexed='faces')
            
            # Rotate cylinder vertexes to match segment
            p1 = pg.Vector(*ends[0])
            p2 = pg.Vector(*ends[1])
            z = pg.Vector(0,0,1)
            axis = pg.QtGui.QVector3D.crossProduct(z, p2-p1)
            ang = z.angle(p2-p1)
            tr = pg.Transform3D()
            tr.translate(ends[0][0], ends[0][1], ends[0][2]) # move into position
            tr.rotate(ang, axis.x(), axis.y(), axis.z())
            mesh_verts = pg.transformCoordinates(tr, mesh_verts, transpose=True)
            sec_id_array = np.empty(mesh_verts.shape[0]*3, dtype=int)
            sec_id_array[:] = sec_id
            meshes.append(mesh_verts)
            sec_ids.append(sec_id_array)
        
        self.vertex_sec_ids = np.concatenate(sec_ids, axis=0)
        mesh_verts = np.concatenate(meshes, axis=0)

        md = gl.MeshData(vertexes=mesh_verts)
        gl.GLMeshItem.__init__(self, meshdata=md, shader='shaded')
            


#         verts, edges = h.get_geometry()
#         # self.hg = HocGraphic(h)
#         # self.hg.set_section_colors = self.set_section_colors
#         # super(HocCylinders, self).__init__()
#         meshes = []
#         sec_ids = []
#         for edge in edges:
#             ends = verts['pos'][edge]
#             dia = verts['dia'][edge]
#             sec_id = verts['sec_index'][edge[0]]
#
#             dif = ends[1]-ends[0]
#             length = (dif**2).sum() ** 0.5
#
#             mesh = gl.MeshData.cylinder(rows=1, cols=8, radius=[dia[0]/2., dia[1]/2.], length=length)
#             mesh_verts = mesh.vertexes(indexed='faces')
#
#             # Rotate cylinder vertexes to match segment
#             p1 = pg.Vector(*ends[0])
#             p2 = pg.Vector(*ends[1])
#             z = pg.Vector(0,0,1)
#             axis = pg.QtGui.QVector3D.crossProduct(z, p2-p1)
#             ang = z.angle(p2-p1)
#             tr = pg.Transform3D()
#             tr.translate(ends[0][0], ends[0][1], ends[0][2]) # move into position
#             tr.rotate(ang, axis.x(), axis.y(), axis.z())
#
#             mesh_verts = pg.transformCoordinates(tr, mesh_verts, transpose=True)
# #            print ('meshverts: ', mesh_verts.shape)
#             sec_id_array = np.empty(mesh_verts.shape[0]*3, dtype=int)
#             sec_id_array[:] = sec_id
#             meshes.append(mesh_verts)
#             sec_ids.append(sec_id_array)
#         #print mesh_verts
#         X = mesh_verts
#
#         print('edges done')
#         self.vertex_sec_ids = np.concatenate(sec_ids, axis=0)
#         mesh_verts = np.concatenate(meshes, axis=0)
#         print mesh_verts
#         md = gl.MeshData(vertexes=mesh_verts)
#         gl.GLMeshItem(meshdata=md, shader='shaded')
#
#         #super(HocCylinders, self).__init__()
#         self.setMeshData(meshdata=md, shader='shaded')
            
    def set_section_colors(self, sec_colors):
        colors = sec_colors[self.vertex_sec_ids]
        self.opts['meshdata'].setVertexColors(colors, indexed='faces')
        self.meshDataChanged()
        