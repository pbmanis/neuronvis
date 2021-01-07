import sys
import numpy as np
from typing import Union

import pyqtgraph as pg

import matplotlib.pyplot as mpl
import matplotlib.colors
import matplotlib.cm

from mpl_toolkits.mplot3d import axes3d
from . import mplcyl
from mayavi import mlab
from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

# This fix (patch)  for opengl on Mac OSX Big Sur seems to work as a patch. 
# could be put into pyqtgraph.opengl, or at top of pyqtgraph
# https://stackoverflow.com/questions/63475461/unable-to-import-opengl-gl-in-python-on-macos

try:
    import OpenGL
    try:
        import OpenGL.GL as OGL   # this fails in <=2020 versions of Python on OS X 11.x
    except ImportError:
        print('Drat, patching for Big Sur')
        from ctypes import util
        orig_util_find_library = util.find_library
        def new_util_find_library( name ):
            res = orig_util_find_library( name )
            if res: return res
            # return '/System/Library/Frameworks/'+name+'.framework/'+name
            return "/System/Library/Frameworks/{}.framework/{}".format(name,name)
        util.find_library = new_util_find_library
        import OpenGL.GL as OGL
except ImportError:
    print('Import of optngl Failed')
    pass

import pyqtgraph.opengl as gl

from . import xkcd_colors

Colors = xkcd_colors.get_colors()
colorMap = list(Colors.keys())


def compute_cube(cube_definition):
    cube_definition_array = [np.array(list(item)) for item in cube_definition]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0],
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


def setMapColors(colormapname: str, reverse: bool = False) -> object:
    """ matplotlib color schemes
    """
    cmnames = dir(matplotlib.cm)
    cmnames = [c for c in cmnames if not c.startswith("__")]
    if colormapname == "parula":
        colormapname = "snshelix"
    elif colormapname == "cubehelix":
        cm_sns = seaborn.cubehelix_palette(
            n_colors=6,
            start=0,
            rot=0.4,
            gamma=1.0,
            hue=0.8,
            light=0.85,
            dark=0.15,
            reverse=reverse,
            as_cmap=False,
        )
    elif colormapname == "snshelix":
        cm_sns = seaborn.cubehelix_palette(
            n_colors=64,
            start=3,
            rot=0.5,
            gamma=1.0,
            dark=0,
            light=1.0,
            reverse=reverse,
            as_cmap=True,
        )
    elif colormapname in cmnames:
        cm_sns = mpl.cm.get_cmap(colormapname)
    # elif colormapname == 'a':
    #     cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_a', colormaps.option_a.cm_data)
    # elif colormapname == 'b':
    #     cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_b', colormaps.option_b.cm_data)
    # elif colormapname == 'c':
    #     cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_c', colormaps.option_c.cm_data)
    # elif colormapname == 'd':
    #     cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('option_d', colormaps.option_d.cm_data)
    # elif colormapname == 'parula':
    #     cm_sns = matplotlib.colors.LinearSegmentedColormap.from_list('parula', colormaps.parula.cm_data)
    else:
        print(
            '(analyzemapdata) Unrecongnized color map {0:s}; setting to "snshelix"'.format(
                colormapname
            )
        )
        cm_sns = seaborn.cubehelix_palette(
            n_colors=64,
            start=3,
            rot=0.5,
            gamma=1.0,
            dark=0,
            light=1.0,
            reverse=reverse,
            as_cmap=True,
        )
    # elif colormapname == '
    return cm_sns


cm_sns = setMapColors("CMRmap")

# mayavi functions
def refaxes(scene=None, ext=[80, 75, 55], alpha=1):
    cube_definition = [
        (-ext[0], -ext[1], -ext[2]),
        (-ext[0], ext[1], -ext[2]),
        (ext[0], -ext[1], -ext[2]),
        (-ext[0], -ext[1], ext[2]),
    ]
    points, edges = compute_cube(cube_definition)
    for j, e in enumerate(edges):
        xyz = []
        for k in range(3):
            xyz.append([e[0][k], e[1][k]])
        mlab.plot3d(xyz[0], xyz[1], xyz[2], color=(0.7, 0.7, 0.7), opacity=0., tube_radius=0.125)


def reflines(scene=None, ext=[80, 75, 55]):
    x0 = [-ext[0], ext[0], 0.0, 0.0, 0.0, 0.0]
    y0 = [0.0, 0.0, -ext[1], ext[1], 0.0, 0.0]
    z0 = [0.0, 0.0, 0.0, 0.0, -ext[2], ext[2]]
    colc = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # axisname = [f"x ({ext[0]:.0f})", 'y', 'z']
    axisl = ["x", "y", "z"]
    for j, i in enumerate([0, 2, 4]):
        mlab.plot3d(
            x0[i : i + 2], y0[i : i + 2], z0[i : i + 2], color=colc[j], tube_radius=0.1
        )
        axisname = f"{axisl[j]:s} ({ext[j]:+.0f})"
        mlab.text3d(x0[i + 1], y0[i + 1], z0[i + 1], axisname, figure=scene, scale=3.0)

def scalebar(scene=None, length=20.):
     ext = [length, 0., 0.]
     x0 = [-ext[0], ext[0], 0.0, 0.0, 0.0, 0.0]
     y0 = [0.0, 0.0, -ext[1], ext[1], 0.0, 0.0]
     z0 = [0.0, 0.0, 0.0, 0.0, -ext[2], ext[2]]
     colc = [(1, 1, 1), (0, 1, 0), (0, 0, 1)]
     # axisname = [f"x ({ext[0]:.0f})", 'y', 'z']
     axisl = ["x", "y", "z"]
     opacity = 1
     for j, i in enumerate([0, 2, 4]):
         if j > 0:
             opacity = 0
         mlab.plot3d(
             x0[i : i + 2], y0[i : i + 2], z0[i : i + 2], color=colc[j], opacity=opacity, tube_radius=0.5
         )
         axisname = f"{axisl[j]:s} ({ext[j]:+.0f})"
         if j == 0:
             mlab.text3d(x0[i + 1], y0[i + 1], z0[i + 1], axisname, figure=scene, scale=3.0)

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

    def set_group_colors(
        self,
        colors: list,
        default_color: list = (0.5, 0.5, 0.5, 0.5),
        alpha: float = 1.0,
        mechanism: str = None,
        colormap: str = None,
    ):
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
        mechmax = 0.0
        cmap = pg.ColorMap(
            [0, 0.25, 0.6, 1.0], [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
        )
        # print('set group colors')
        dsecs = []
        for group_name, color in colors.items():
            sections = self.h.get_section_group(group_name)
            if sections is None:
                continue
            if isinstance(color, str):
                sec_color = Colors[color]
            else:
                sec_color = color
            for sec_name in sections:
                # print('setting color for Sec name: ', sec_name, 'group: ', group_name, 'to : ', sec_color)
                index = self.h.sec_index[sec_name]
                # print('mechanism: ', mechanism)
                if mechanism not in [None, "None"]:
                    gbar = self.h.get_density(self.h.sections[sec_name], mechanism)
                    mechmax = max(mechmax, gbar)
                    sec_colors[index, 3] = gbar  # use the alpha channel to set the color
                else:
                    sec_colors[index] = sec_color
                    # print('   seccolor: ', sec_color)
        mechmax = np.max(sec_colors[:, 3])
        mechmin = np.min(sec_colors[:, 3])
        # print('mechmax/min: ', mechmax, mechmin)
        #      print(sec_colors)
        if mechanism not in [None, "None"] and mechmax > 0.0:
            for i, c in enumerate(sec_colors):
                rgb = cmap.map(c[3] / mechmax, "float")
                c[:3] = rgb * 255.0  # set alpha for all sections
                c[3] = alpha
        self.set_section_colors(sec_colors)


class mplGraphic(object):
    """
    Methods common to all mpl graphical representation classes
    (mplCylinders, mplVolume, mplSurface, etc.)

    """

    def __init__(self, h, parentItem: Union[object, None] = None, **kwds):
        self.h = h
        self.cmx = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm_sns)
        print("init mplGraphic")

    def get_color_map(self, i):
        return colorMap[i]

    def set_section_colors(self, colors):
        """
        Recolor the graphic by section using the *colors* array. This method
        must be reimplemented by HocGraphic subclasses. The order of elements
        in the array must match the order of sections defined in
        HocReader.sections.
        """
        print("mplgraphic: set_section_colors")
        raise NotImplementedError()

    def set_group_colors(
        self,
        colors,
        default_color: list = (0.5, 0.5, 0.5, 1),
        alpha: float = 1,
        mechanism: Union[str, None] = None,
        colormap=None,
    ):
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
        mechmax = 0.0
        cmap = self.cmx
        # print('set group colors')
        dsecs = []
        for group_name, color in colors.items():
            sections = self.h.get_section_group(group_name)
            if sections is None:
                continue
            for sec_name in sections:
                print('Setting color for group: ', sec_name, 'to : ', color)
                if isinstance(color, str):
                    color = Colors[color]
                index = self.h.sec_index[sec_name]
                # print(index, color)
                if mechanism is None:
                    sec_colors[index] = color
                if mechanism not in [None, "None"]:
                    g = self.h.get_density(self.h.sections[sec_name], mechanism)
                    mechmax = max(mechmax, g)
                    sec_colors[index, 3] = g
                    # if group_name not in dsecs:
                    #      print ('section: %s, group: %s, mech: %s, gmax = %f' % (sec_name, group_name, mechanism, g))
                    #      print('group: ', group_name, sec_colors[index], g)
                    #      dsecs.append(group_name)
                    # if alpha is not None:
                    sec_colors[index, 3] = alpha
        # print (mechmax)
        # print('sec colors: ', sec_colors)
        mechmax = np.max(sec_colors[:, 3])
        mechmin = np.min(sec_colors[:, 3])
        if mechanism not in [None, "None"] and mechmax > 0.0:
            sec_colors = cmx.to_rgba(np.clip(sec_colors[3], 0.0, mechmax))

            # for i, c in enumerate(sec_colors):
            #     rgb = cmap.map(c[3]/mechmax, 'float')
            #     c[:3] = rgb*255. # set alpha for all sections
            #     c[3] = 1.0
        self.set_section_colors(sec_colors)


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
        print("HOC Cylinders")
        meshes = []
        sec_ids = []
        for edge in edges:
            ends = verts["pos"][edge]
            dia = verts["dia"][edge]
            sec_id = verts["sec_index"][edge[0]]

            dif = ends[1] - ends[0]
            length = (dif ** 2).sum() ** 0.5

            mesh = gl.MeshData.cylinder(
                rows=1, cols=8, radius=[dia[0] / 2.0, dia[1] / 2.0], length=length
            )
            mesh_verts = mesh.vertexes(indexed="faces")

            # Rotate cylinder vertexes to match segment
            p1 = pg.Vector(*ends[0])
            p2 = pg.Vector(*ends[1])
            z = pg.Vector(0, 0, 1)
            axis = pg.QtGui.QVector3D.crossProduct(z, p2 - p1)
            ang = z.angle(p2 - p1)
            tr = pg.Transform3D()
            tr.translate(ends[0][0], ends[0][1], ends[0][2])  # move into position
            tr.rotate(ang, axis.x(), axis.y(), axis.z())
            mesh_verts = pg.transformCoordinates(tr, mesh_verts, transpose=True)
            sec_id_array = np.empty(mesh_verts.shape[0] * 3, dtype=int)
            sec_id_array[:] = sec_id
            meshes.append(mesh_verts)
            sec_ids.append(sec_id_array)

        self.vertex_sec_ids = np.concatenate(sec_ids, axis=0)
        mesh_verts = np.concatenate(meshes, axis=0)

        md = gl.MeshData(vertexes=mesh_verts)
        """
        shaders:
        shaded : light (seems to come from inside? )
        normalColor: rainbow effect; varies with viewing angle
        balloon " hard color not "shaded" for dimension; a bit cartoonish
        edgeHilight like balloon, but highlight on edges
        heightColor  uggh
         """
        # gl.GLMeshItem.__init__(self, meshdata=md, shader='balloon', glOptions='additive') # 'balloon') # 'shaded')
        gl.GLMeshItem.__init__(
            self, meshdata=md, smooth=True, shader="balloon", glOptions="opaque"
        )  # 'balloon') # 'shaded')

    def set_section_colors(self, sec_colors):
        colors = sec_colors[self.vertex_sec_ids]
        self.opts["meshdata"].setVertexColors(colors, indexed="faces")
        self.opts["meshdata"].setFaceColors(colors, indexed="faces")
        self.meshDataChanged()


class mayavi_Cylinders(object):
    def __init__(
        self,
        h: object,
        color: Union[list, tuple] = (0, 0, 1),
        mechanism: Union[str, None] = None,
        camerapos: Union[list, tuple] = [200, 0.0, 0.0],
        label: Union[str, None] = None,
        flags=None,
    ):
        self.h = h

        hcyl = mplcyl.TruncatedCone(facets=24)
        verts, edges = h.get_geometry()
        meshes = []
        sec_ids = []
        # print(dir(verts))
        #         exit()
        connections = []
        index = 0
        lastend = None
        ne = len(edges)
        ndone = 0
        sectypes = []
        for ind, edge in enumerate(edges):
            # # of edges corresponds to N-1 section indiators; and to # of cones
            sectypes.append(verts["sec_type"][edge][0])

        secs = set(
            sectypes
        )  # find just the section types that are used in this representation
        nsecs = len(secs) + 1  # avoid extremes in color range

        # Assign a color to each section type
        # by giving it a floating value between 0 and 1
        # cs is a dictionary for each section type
        # to look up the named color and the scalar value that goes with that color
        # Colors can be mapped to another variable. Here we
        # support section type (anatomical type) and
        # channel density.

        # print('mechanism: ', mechanism)
        cs = {}
        scalars = {}
        if mechanism in [None, "None"]:  # coloring by section type
            for i, s in enumerate(secs):
                cs[s] = (color[s], float(i + 1) / (nsecs))

            for isec, s in enumerate(sectypes):
                c = color[s]
                scalars[s] = cs[s][1]  # assign a scalar to the section type
        else:  # coloring by a magnitude assigned to a section type
            mechmax = {}
            for group_name, colorn in color.items():
                # print('group name: ', group_name)
                mechgrp = 0.0
                sections = self.h.get_section_group(group_name)
                if sections is None:
                    continue
                for sec_name in sections:
                    g = self.h.get_density(self.h.sections[sec_name], mechanism)
                    # print("   g: ", g, mechanism)
                    mechgrp = max(mechgrp, g)
                mechmax[group_name] = mechgrp
            maxg = 0.05
            # print(mechmax)
            for g in list(mechmax.keys()):
                maxg = max(mechmax[g], maxg)
            if maxg == 0.0:
                maxg = 1.0
            for group_name in secs:
                scalars[group_name] = mechmax[group_name] / (1.2 * maxg)

        print(scalars)

        XC = []
        YC = []
        ZC = []
        S = []

        scalar = []

        # of edges corresponds to N-1 section indiators; and to # of cones
        for ind, edge in enumerate(edges):
            ends = verts["pos"][edge]  # xyz coordinate of one end [x,y,z]
            dia = verts["dia"][edge]  # diameter at that end
            sec_id = verts["sec_index"][edge]
            sec_type = verts["sec_type"][edge]

            C, T, B = hcyl.make_truncated_cone(
                p0=ends[1], p1=ends[0], R=[dia[1] / 2.0, dia[0] / 2.0]
            )
            XC.extend(C[0])
            YC.extend(C[1])
            ZC.extend(C[2])
            scalar.extend(
                np.broadcast_to(
                    [scalars[sec_type[0]], scalars[sec_type[0]]], (len(C[0]), 2)
                )
            )
            # disconnect adjacent cylinders to avoid "strings"

            XC.append(np.array([np.nan, np.nan]))
            YC.append(np.array([np.nan, np.nan]))
            ZC.append(np.array([np.nan, np.nan]))
            scalar.append(np.array([np.nan, np.nan]))
            N = len(C[0])
            connections.append(
                np.vstack(
                    [
                        np.arange(index, index + N - 1.5),
                        np.arange(index + 1, index + N - 0.5),
                    ]
                ).T
            )
            index += N
        # m = mlab.mesh(XC, YC, ZC, color=tuple(colors[240][:3]), line_width=0.0, opacity=1.0)
        self.g = mlab.mesh(
            XC,
            YC,
            ZC,
            scalars=scalar,
            colormap="gist_heat",
            representation="surface",
            line_width=0.1,
            opacity=1.0,
            vmin=0,
            vmax=1,
            resolution=24,
        )
        if mechanism not in [None, 'None']:
            mlab.title(mechanism, size=0.5)
        else:
            mlab.title('sec-type')
        fig = mlab.gcf()
        camera = fig.scene.camera
        # print(dir(camera))
        # print(camerapos)
        mlab.view(camerapos[1]+45, camerapos[2]+45, camerapos[0])
        # self.cursor3d = mlab.points3d(0., 0., 0., mode='2darrow',
        #                         color=(1, 1, 1), [-10, 10, -10, 10, -10, 10],
        #                         scale_factor=0.5)
        sb = mlab.scalarbar()
        # Scene.interactor.add_observer('KeyPressEvent', self.mayavi_report);
        
        if flags is not None:
            if "norefaxes" not in flags:
                refaxes()
            if "norefline" not in flags:
                reflines()
        else:
            refaxes()
            scalebar()
            # reflines()
            pass

        if label is not None and "text" in flags:
            # print('cylinder: ', label, XC[0], YC[0], ZC[0])
            mlab.text3d(
                XC[0][0] + 5,
                YC[0][0],
                ZC[0][0],
                f"{label:s}",
                color=color,
                figure=None,
                scale=2.0,
            )


        self.sec_ids = sec_ids

    def mayavi_report(self):
        print('view: ', self.g.view())
    
    def map_to_color(self, v: np.ndarray, v_min=None, v_max=None) -> np.ndarray:
        """
        Convert an array of Vm to array of representative colors
        """

        color = np.empty((v.shape[0], 4), dtype=float)
        if v_min is not None:
            v_min = -80  # mV
            v_range = 100.0  # mV range in scale
        else:
            v_min = np.min(v)
            v_max = np.max(v)
            v_range = v_max - v_min
        v = (v - v_min) / v_range
        color[:, 0] = v  # R
        color[:, 1] = 1.5 * abs(v - 0.5)  # G
        color[:, 2] = 1.0 - v  # B
        color[:, 3] = 0.1 + 0.8 * v  # alpha
        # for j in range(4):
        #            print(np.max(color[:,j]))
        print('VM: ', color)
        return color

    # def set_section_colors(self, sec_colors):
    #     print("SET SECTION COLORS")
    #
    #
    #     scs = []
    #     sc = self.sec_colors[self.sec_ids[0]]
    #     print(len(self.sec_ids))
    #     for i, secid in enumerate(self.sec_ids):
    #         self.sec_ids[i] = np.broadcast_to(self.sec_colors[secid[0]], (secid.shape[0], 4))
    #
    #     sc = np.zeros(2000)
    #     sc[0] = 1.0
    #     # print(sc)
    #     # print(self.g.mlab_source.dataset.cell_data)
    #     self.g.mlab_source.dataset.cell_data.scalars = sc
    #     self.g.mlab_source.dataset.cell_data.scalars.name = 'Section Type'
    #     self.g.mlab_source.update()
    #     # self.g.parent.update()
    #
    #     mesh2 = mlab.pipeline.set_active_attribute(self.g,
    #                     cell_scalars='Section Type')
    #     s2 = mlab.pipeline.surface(mesh2)
    #
    #     # for i, s in enumerate(self.g.surf):
    #     #     s.set_facecolors(colors[i])
    #
    # def set_group_colors(
    #     self,
    #     colors,
    #     default_color:list=(0.5, 0.5, 0.5, 1),
    #     alpha:float=1,
    #     mechanism:Union[str, None]=None,
    #     colormap=None,
    # ):
    #     """
    #     Color the sections in the reconstruction according to their
    #     group name.
    #     Inputs:
    #         colors: a dictionary of section group names and their associated colors
    #         default_color: color to use for any sections that are not included in the
    #                        groups listed in *colors*.
    #         alpha: If specified, this overrides the alpha value for all group colors.
    #     Side-effects: none.
    #     """
    #     sec_colors = np.zeros((len(self.h.sections), 4), dtype=float)
    #     sec_colors[:] = default_color
    #     mechmax = 0.0
    #     # cmap = self.cmx
    #     # print('set group colors')
    #     dsecs = []
    #     for group_name, color in colors.items():
    #         sections = self.h.get_section_group(group_name)
    #         if sections is None:
    #             continue
    #         for sec_name in sections:
    #             if isinstance(color, str):
    #                 color = Colors[color]
    #             index = self.h.sec_index[sec_name]
    #             # print(index, color)
    #             if mechanism is None:
    #                 sec_colors[index] = color
    #             if mechanism not in [None, "None"]:
    #                 g = self.h.get_density(self.h.sections[sec_name], mechanism)
    #                 mechmax = max(mechmax, g)
    #                 sec_colors[index, 3] = g
    #                 # if group_name not in dsecs:
    #                 #      print ('section: %s, group: %s, mech: %s, gmax = %f' % (sec_name, group_name, mechanism, g))
    #                 #      print('group: ', group_name, sec_colors[index], g)
    #                 #      dsecs.append(group_name)
    #                 # if alpha is not None:
    #                 sec_colors[index, 3] = alpha
    #                 print("  MAYAVI: set group colors alpha: ", alpha)
    #     # print (mechmax)
    #     # print('sec colors: ', sec_colors)
    #     mechmax = np.max(sec_colors[:, 3])
    #     mechmin = np.min(sec_colors[:, 3])
    #     if mechanism not in [None, "None"] and mechmax > 0.0:
    #         sec_colors = cmx.to_rgba(np.clip(sec_colors[3], 0.0, mechmax))
    #
    #         # for i, c in enumerate(sec_colors):
    #         #     rgb = cmap.map(c[3]/mechmax, 'float')
    #         #     c[:3] = rgb*255. # set alpha for all sections
    #         #     c[3] = 1.0
    #     self.sec_colors = sec_colors
    #     self.set_section_colors(sec_colors)


class mpl_Cylinders(mplGraphic):
    """
    Input:
        h: HocReader instance
    """

    def __init__(
        self,
        h,
        useMpl: bool = True,
        color: Union[str, None] = None,
        fax: Union[object, None] = None,
    ):
        super(mplGraphic, self).__init__()
        print("mpl cylinders")

        self.h = h
        hcyl = mplcyl.TruncatedCone()
        # plot_tc(p0=np.array([1, 3, 2]), p1=np.array([8, 5, 9]), R=[5.0, 2.0])
        verts, edges = h.get_geometry()
        # print 'verts', verts
        # print 'edges', edges
        # print verts['pos']
        # self.hg = HocGraphic(h)
        # self.hg.set_section_colors = self.set_section_colors
        # super(HocCylinders, self).__init__()
        if isinstance(color, str):
            if color in list(Colors.keys()):
                color = tuple(Colors[color])[0:3]
            else:
                raise ValueError("Color string %s not in Colors table" % color)
        meshes = []
        sec_ids = []
        self.surf = []
        if fax is None:
            fig = mpl.figure()
            ax = fig.gca(projection="3d")
        else:
            fig = fax[0]
            ax = fax[1]
        for edge in edges:
            ends = verts["pos"][edge]  # xyz coordinate of one end [x,y,z]
            dia = verts["dia"][edge]  # diameter at that end
            sec_id = verts["sec_index"][edge[0]]  # save the section index

            dif = ends[1] - ends[0]  # distance between the ends
            length = (dif ** 2).sum() ** 0.5
            C, T, B = hcyl.make_truncated_cone(
                p0=ends[0], p1=ends[1], R=[dia[0] / 2.0, dia[1] / 2.0]
            )
            mesh_verts = np.array(C)
            sec_id_array = np.empty(mesh_verts.shape[0] * 3, dtype=int)
            # # sec_id_array[:] = sec_id
            # meshes.append(mesh_verts)

            s = ax.plot_surface(
                C[0], C[1], C[2], color="blue", linewidth=1, antialiased=False
            )
            self.surf.append(s)

            sec_id_array[:] = sec_id
            meshes.append(mesh_verts)
            sec_ids.append(sec_id_array)

        self.vertex_sec_ids = np.concatenate(sec_ids, axis=0)
        mesh_verts = np.concatenate(meshes, axis=0)

        self.axisEqual3D(ax)
        self.set_group_colors()  # exit(1)
        if fax is None:
            mpl.show()

    def axisEqual3D(self, ax: object):
        extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
        sz = extents[:, 1] - extents[:, 0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize / 2
        ax.auto_scale_xyz(*np.column_stack((centers - r, centers + r)))

    def set_section_colors(self, sec_colors):
        print("SET SECTION COLORS")
        colors = sec_colors[self.vertex_sec_ids]
        for i, s in enumerate(self.surf):
            s.set_facecolors(colors[i])
        # (setVertexColors(colors, indexed='faces')
        # self.meshDataChanged()

    def set_group_colors(
        self,
        colors,
        default_color: list = (0.5, 0.5, 0.5, 1),
        alpha: float = 1,
        mechanism: Union[str, None] = None,
        colormap=None,
    ):
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
        mechmax = 0.0
        # cmap = self.cmx
        # print('set group colors')
        dsecs = []
        for group_name, color in colors.items():
            sections = self.h.get_section_group(group_name)
            if sections is None:
                continue
            for sec_name in sections:
                if isinstance(color, str):
                    color = Colors[color]
                index = self.h.sec_index[sec_name]
                # print(index, color)
                if mechanism is None:
                    sec_colors[index] = color
                if mechanism not in [None, "None"]:
                    g = self.h.get_density(self.h.sections[sec_name], mechanism)
                    mechmax = max(mechmax, g)
                    sec_colors[index, 3] = g
                    # if group_name not in dsecs:
                    #      print ('section: %s, group: %s, mech: %s, gmax = %f' % (sec_name, group_name, mechanism, g))
                    #      print('group: ', group_name, sec_colors[index], g)
                    #      dsecs.append(group_name)
                    # if alpha is not None:
                    sec_colors[index, 3] = alpha
                    print("  MPL: set group colors alpha: ", alpha)
        # print (mechmax)
        # print('sec colors: ', sec_colors)
        mechmax = np.max(sec_colors[:, 3])
        mechmin = np.min(sec_colors[:, 3])
        if mechanism not in [None, "None"] and mechmax > 0.0:
            sec_colors = cmx.to_rgba(np.clip(sec_colors[3], 0.0, mechmax))

            # for i, c in enumerate(sec_colors):
            #     rgb = cmap.map(c[3]/mechmax, 'float')
            #     c[:3] = rgb*255. # set alpha for all sections
            #     c[3] = 1.0
        self.sec_colors = sec_colors
        self.set_section_colors(sec_colors)


class HocGrid(HocGraphic, gl.GLGridItem):
    """
    subclass of GLGridItem to draw a grid on the field
    """

    def __init__(self, size: list = (250, 250, 250), spacing: list = (50, 50, 50)):
        super(HocGraphic, self).__init__()
        # grcolor = pg.mkColor(255, 255, 255, 255)
        grcolor = pg.mkColor("y")
        self.grid = gl.GLGridItem(color=grcolor)
        # return
        size = np.array(size)
        spacing = np.array(spacing)
        self.grid.setSize(x=size[0], y=size[1], z=size[2])  # 100 um grid spacing
        self.grid.setSpacing(x=spacing[0], y=spacing[1], z=spacing[2])  # 10 um steps
        self.grid.scale(1, 1, 1)  # uniform scale
        # self.grid.translate(100., 0., 0.)
        # super(HocGrid, self).__init__(size, spacing, color=grcolor)


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
        # edges = edges.flatten()
        verts_indexed = verts[edges]
        self.vertex_sec_ids = verts_indexed["sec_index"]

        #
        self.lines = []
        for edge in edges:
            w = (verts["dia"][edge[0]] + verts["dia"][edge[1]]) * 0.5
            self.lines.append(gl.GLLinePlotItem(pos=verts["pos"][edge], width=w))
            self.lines[-1].setParentItem(self)
        super(HocGraph, self).__init__(h, pos=verts_indexed["pos"], mode="lines")

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
        nfdata[..., 0] = 255  # scfield*50
        nfdata[..., 1] = 255  # scfield*50
        nfdata[..., 2] = 255  # scfield*50
        nfdata[..., 3] = np.clip(scfield * 150, 0, 255)
        super(HocVolume, self).__init__(nfdata)
        self.setTransform(transform)


class mayavi_Volume(object):
    def __init__(self, h):
        self.h = h
        scfield, idfield, transform = self.h.make_volume_data()
        nfdata = np.empty(scfield.shape + (4,), dtype=np.ubyte)
        nfdata[..., 0] = 255  # scfield*50
        nfdata[..., 1] = 255  # scfield*50
        nfdata[..., 2] = 255  # scfield*50
        nfdata[..., 3] = np.clip(scfield * 150, 0, 255)
        # f = mlab.figure() # Returns the current scene.
        # engine = mlab.get_engine() # Returns the running mayavi engine.
        # scene  = engine.new_scene()
        refaxes()
        g = mlab.pipeline.volume(
            mlab.pipeline.scalar_field(scfield), vmin=0.24, vmax=0.25
        )
        reflines()
        self.g = g


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
        # scfield = scipy.ndimage.gaussian_filter(scfield, (0.5, 0.5, 0.5))
        # pg.image(scfield)
        verts, faces = pg.isosurface(scfield, level=0.0)
        self.verts = verts
        self.faces = faces
        vertexColors = np.empty((verts.shape[0], 4), dtype=float)
        md = gl.MeshData(vertexes=verts, faces=faces)

        # match vertexes to section IDs
        vox_locations = verts.astype(int)
        # get sction IDs for each vertex
        self.vertex_sec_ids = idfield[
            vox_locations[:, 0], vox_locations[:, 1], vox_locations[:, 2]
        ]
        self.setMeshData(meshdata=md, smooth=True, shader="balloon")
        self.setTransform(transform)
        self.setGLOptions("additive")

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
        self.opts["meshdata"].setVertexColors(colors)

        self.meshDataChanged()


class mayavi_graph(object):
    def __init__(
        self,
        h: object,
        color: Union[list, tuple, str] = (0, 0, 1),
        label=None,
        flags=None,
    ) -> object:
        self.h = h
        # plot_tc(p0=np.array([1, 3, 2]), p1=np.array([8, 5, 9]), R=[5.0, 2.0])
        if isinstance(color, str):
            color = tuple(Colors[color][:3])
        verts, edges = h.get_geometry()
        sec_id = []
        XC = []
        YC = []
        ZC = []
        S = []
        connections = []
        index = 0
        for edge in edges:
            ends = verts["pos"][edge]  # xyz coordinate of one end [x,y,z]
            dia = verts["dia"][edge]  # diameter at that end
            sec_id = verts["sec_index"][edge[0]]  # save the section index

            dif = ends[1] - ends[0]  # distance between the ends
            length = (dif ** 2).sum() ** 0.5
            X = [ends[0][0], ends[1][0]]
            Y = [ends[0][1], ends[1][1]]
            Z = [ends[0][2], ends[1][2]]
            S.extend(dia)
            XC.extend(X)
            YC.extend(Y)
            ZC.extend(Z)
            N = len(X)
            connections.append(
                np.vstack(
                    [
                        np.arange(index, index + N - 1.5),
                        np.arange(index + 1, index + N - 0.5),
                    ]
                ).T
            )
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
            mlab.text3d(
                XC[0], YC[0], ZC[0], f"{label:s}", figure=None, scale=1.5, color=color
            )
        return t


class mpl_Graph(object):
    """
    Input:
        h: HocReader instance
    """

    def __init__(self, h, mpl=True, fax=None, color="blue"):
        self.h = h
        # hcyl = mplcyl.TruncatedCone()

        # plot_tc(p0=np.array([1, 3, 2]), p1=np.array([8, 5, 9]), R=[5.0, 2.0])
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
            fig = mpl.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = fax[0]
            ax = fax[1]
        for edge in edges:
            ends = verts["pos"][edge]  # xyz coordinate of one end [x,y,z]
            dia = verts["dia"][edge]  # diameter at that end
            # print('ends: ', ends)
            # print(dia)
            sec_id = verts["sec_index"][edge[0]]  # save the section index
            X = [ends[0][0], ends[1][0]]
            Y = [ends[0][1], ends[1][1]]
            Z = [ends[0][2], ends[1][2]]
            if mpl:
                ax.plot(X, Y, Z, color=color, linewidth=0.5, antialiased=False)

        if mpl:
            self.axisEqual3D(ax)
            if fax is None:
                mpl.show()
        # exit(1)

    def axisEqual3D(self, ax):
        extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
        sz = extents[:, 1] - extents[:, 0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize / 2
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
