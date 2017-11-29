import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui
import numpy as np
import scipy.ndimage
import mpl_colormaps as mpc

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


class HocGraphic(object):
    """
    Methods common to all Hoc graphical representation classes (HocVolume, 
    HocSurface, etc.)
    
    """
    def __init__(self, h, colormap='magma'):
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

    def set_group_colors(self, colors, default_color=(0.5,0.5,0.5,0.5), alpha=None, mechanism=None, colormap='magma'):
        """
        Color the sections in the reconstruction according to their
        group name.
        
        Parameters
        ----------
        colors: dict (no default)
            a dictionary of section group names and their associated colors
        
        default_color: tuple (default: (0.5, 0.5, 0.5, 0.5)
            color to use for any sections that are not included in the
            groups listed in *colors*.
        
        alpha: float (default: None)
            If specified, this overrides the alpha value for all group colors.
        
        mechanism: str (default: None)
            Name of the mechanism to use for coloring; scaled to maximum of mechanism
            conductance density in the hoc object.
            if None, we color by distance from soma
        
        Returns
        -------
        Nothing
        
        Side-effects: none.
        """
        self.h.h('access %s' % 'sections[0]') # reference point
        self.h.h.distance()
        self.distanceMap = {}
        for u in self.h.sections:
            self.h.h('access %s' % u)
            self.distanceMap[u] = self.h.h.distance(0.5) # should be distance from first point
        
        self.colormap = colormap
        sec_colors = np.zeros((len(self.h.sections), 4), dtype=float)
        sec_colors[:] = default_color
        mechmax = 0.
        for group_name, color in colors.items():  # identify max for scaling
            try:
                sections = self.h.get_section_group(group_name)
            except KeyError:
                continue            
            for sec_name in sections:
                if mechanism is not None:
                    g = self.h.get_density(self.h.sections[sec_name], mechanism)
                    mechmax  = max(mechmax, g)
                else:
                    g = self.distanceMap[sec_name]
                    mechmax = max(mechmax, g)
#        print colors.items()
        if mechanism is not None:
            for group_name, color in colors.items(): # now color the mechanisms... 
                try:
                    sections = self.h.get_section_group(group_name)
                except KeyError:
                    continue
                for i, sec_name in enumerate(sections):
                    if isinstance(color, basestring):
                        color = Colors[color]
                    index = self.h.sec_index[sec_name]
                    if mechanism[0] is not None:
                        g = self.h.get_density(self.h.sections[sec_name], mechanism)
                        mechmax  = max(mechmax, g)
                        scaled_g = (0.0 + (1.0*g/mechmax))
                        sec_colors[index, :] = [c/255. for c in mpc.mpl_cm[self.colormap].map(scaled_g)]
                    else:
                        sec_colors[index] = color
                        scaled_g = 0.01
        else:  # color by group type
            for group_name, color in colors.items():
                try:
                    sections = self.h.get_section_group(group_name)
                except KeyError:
                    continue
                for i, sec_name in enumerate(sections):
                    if isinstance(color, basestring):
                        color = Colors[color]
                    index = self.h.sec_index[sec_name]
                    sec_colors[index] = color

        self.set_section_colors(sec_colors)
        # make a new window with just the color scale on it in case we need it.
        # Assign color based on height
        # Make a 2D color bar using the same ColorMap
        colorBar = pg.GradientLegend(size=(50, 200), offset=(15, -25))
        colorBar.setGradient(mpc.mpl_cm[self.colormap].getGradient())
        if mechanism is not None:
            labels = dict([("%0.5f" % (v * scaled_g), v) for v in np.linspace(0, 1, 4)])
        else:
            labels = dict([("%s" % (gn), i) for i, gn in enumerate(colors)])
        colorBar.setLabels(labels)            
        w = QtGui.QWidget()
        layout = QtGui.QGridLayout()
        w.setLayout(layout)
        w.resize(200,200)
        w.show()
        view = pg.GraphicsView() 
        view.addItem(colorBar)
        layout.addWidget(view, 0, 0)
        print 'view show...'


class HocVolume(gl.GLVolumeItem, HocGraphic):
    """
    Subclass of GLVolumeItem that draws a volume representation of the geometry
    specified by a HocReader.
    
    Parameters
    ----------
        h: HocReader instance
    """
    
    def __init__(self, h):
        self.h = h
        scfield, idfield, transform = self.h.make_volume_data()
        nfdata = np.empty(scfield.shape + (4,), dtype=np.ubyte)
        nfdata[...,0] = 255
        nfdata[...,1] = 255
        nfdata[...,2] = 255
        nfdata[...,3] = np.clip(scfield*200, 0, 255)
        super(HocVolume, self).__init__(nfdata)
        self.setTransform(transform)

    def set_section_colors(self, sec_colors):
        """
        Set the colors of multiple sections.
        
        (inactive: there are no vertex section ids to work with...)
        Parameters
        ----------
        colors: tuple (N,4) (no default)
            float array of (r,g,b,a) colors, in the order that
            sections are defined by the HocReader.
        """
        return
        # colors = sec_colors[self.vertex_sec_ids]
        # self.opts['meshdata'].setVertexColors(colors)
        # self.meshDataChanged()

class HocSurface(gl.GLMeshItem, HocGraphic):
    """
    Subclass of GLMeshItem that draws a surface representation of the geometry
    specified by a HocReader. The surface is generated as an isosurface of
    the scalar field generated by measuring the minimum distance from all
    cell membranes across the volume of the cell.
    
    Parameters
    ----------
        h: HocReader instance
    
    """
    def __init__(self, h):
        self.h = h
        scfield, idfield, transform = self.h.make_volume_data()
        #scfield = scipy.ndimage.gaussian_filter(scfield, (0.5, 0.5, 0.5))
        #pg.image(scfield)
        verts, faces = pg.isosurface(scfield, level=-0.02)
        self.verts = verts
        self.faces = faces
        vertexColors = np.empty((verts.shape[0], 4), dtype=float)
        md = gl.MeshData(vertexes=verts, faces=faces)
        
        # match vertexes to section IDs
        vox_locations = verts.astype(int)
        # get sction IDs for each vertex
        self.vertex_sec_ids = idfield[vox_locations[:,0], vox_locations[:,1], vox_locations[:,2]] 
        
        super(HocSurface, self).__init__(meshdata=md, smooth=True, shader='shaded')  # 'normalColor', viewNormaCOlor, shaded, balloon
        self.setTransform(transform)
        self.setGLOptions('opaque')# ('additive') # ('opaque')

    def show_section(self, sec_id, color=(1, 1, 1, 1), bg_color=(1, 1, 1, 0)):
        """
        Set the color of section named *sec_id* to *color*.
        All other sections are colored with *bg_color*.
        
        Parameters
        ----------
        sec_id : int (no default)
            Section ID whose colors will be set
        
        color : tuple (default: (1,1,1,1))
            color in RGBA format (range [0-1]) to set the sections to
        
        bg_color : tuple (default: (1,1,1,0))
            background color for all sections that are not being colored
        
        """
        colors = np.empty((len(self.vertex_sec_ids), 4))
        colors[:] = bg_color
        colors[sec_id] = color
        self.set_section_colors(colors)
    
    def set_section_colors(self, sec_colors):
        """
        Set the colors of multiple sections.
        
        Parameters
        ----------
        colors: tuple (N,4) 
            float array of (r,g,b,a) colors, in the order that
                    sections are defined by the HocReader.
        """
        colors = sec_colors[self.vertex_sec_ids]
        self.opts['meshdata'].setVertexColors(colors)
        self.meshDataChanged()



class HocGraph(gl.GLLinePlotItem, HocGraphic):
    """
    Subclass of GLLinePlotItem that draws a line representation of the geometry
    specified by a HocReader.
    
    Parameters
    ----------
        h: HocReader instance
    """
    def __init__(self, h):
        gl.GLLinePlotItem.__init__(self)
        self.h = h
        verts, edges = h.get_geometry()
        
        # Prefer this method, but item does not support per-vertex width:

        verts_indexed = verts[edges.flatten()]
        self.vertex_sec_ids = verts_indexed['sec_index']
        super(HocGraph, self).__init__(pos=verts_indexed['pos'], mode='lines')
 
        self.lines = []
        for edge in edges:
            w = 2*(verts['dia'][edge[0]] + verts['dia'][edge[1]]) * 0.5
            self.lines.append(gl.GLLinePlotItem(pos=verts['pos'][edge], width=w, #color=(0,0,0,1),
                mode='line_strip', antialias=True))
            self.lines[-1].setParentItem(self)

    def set_section_colors(self, sec_colors):
        """
        Set the colors of multiple sections.
        
        Parameters
        ----------
        colors: tuple (N,4) 
            float array of (r,g,b,a) colors, in the order that
                    sections are defined by the HocReader.
        """
        colors = sec_colors[self.vertex_sec_ids]
        self.setData(color=colors)
        
    
class HocCylinders(gl.GLMeshItem, HocGraphic):
    """
    Subclass of GLMesgItem that draws a cylinder representation of the geometry
    specified by a HocReader.
    
    Parameters
    ----------
        h: HocReader instance
    """
    def __init__(self, h, facets=8):
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
            
            mesh = gl.MeshData.cylinder(rows=8, cols=facets, radius=[dia[0]/2., dia[1]/2.], length=length)
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
            
    def set_section_colors(self, sec_colors):
        """
        Set the colors of multiple sections.
        
        Parameters
        ----------
        colors: tuple (N,4) 
            float array of (r,g,b,a) colors, in the order that
                    sections are defined by the HocReader.
        """
        colors = sec_colors[self.vertex_sec_ids]
        self.opts['meshdata'].setVertexColors(colors, indexed='faces')
        self.meshDataChanged()
        
