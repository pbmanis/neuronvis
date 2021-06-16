from __future__ import print_function
from __future__ import absolute_import
import pyqtgraph as pg

import numpy as np
from .hoc_reader import HocReader
from .hoc_graphics import HocGrid, HocCylinders, HocSurface, HocGraph, HocVolume
from .hoc_graphics import mayavi_Cylinders
from .hoc_graphics import mpl_Cylinders
from pyqtgraph.Qt import QtGui
import matplotlib.pyplot as mpl
from mayavi import mlab

# This fix (patch)  for opengl on Mac OSX Big Sur seems to work as a patch. 
# could be put into pyqtgraph.opengl, I suspect. 
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


from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem


class GLAxisItem_r(GLGraphicsItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem>`
    
    Displays three lines indicating origin and orientation of local coordinate system. 
    
    """
    
    def __init__(self, size=None, antialias=True, glOptions='translucent'):
        GLGraphicsItem.__init__(self)
        if size is None:
            size = QtGui.QVector3D(1,1,1)
        self.antialias = antialias
        self.setSize(size=size)
        self.setGLOptions(glOptions)
    
    def setSize(self, x=None, y=None, z=None, size=None):
        """
        Set the size of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        if size is not None:
            x = size.x()
            y = size.y()
            z = size.z()
        self.__size = [x,y,z]
        self.update()
        
    def size(self):
        return self.__size[:]
    
    
    def paint(self):
            
        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glEnable( GL_BLEND )
        #glEnable( GL_ALPHA_TEST )
        self.setupGLState()

        if self.antialias:
            OGL.glEnable(OGL.GL_LINE_SMOOTH)
            OGL.glHint(OGL.GL_LINE_SMOOTH_HINT, OGL.GL_NICEST)

        OGL.glBegin( OGL.GL_LINES )

        x,y,z = self.size()
        OGL.glColor4f(0, 1, 0, .9)  # z is green
        OGL.glVertex3f(0, 0, 0)
        OGL.glVertex3f(0, 0, z)

        OGL.glColor4f(1, 1, 0, .9)  # y is yellow
        OGL.glVertex3f(0, 0, 0)
        OGL.glVertex3f(0, y, 0)

        OGL.glColor4f(0, 0, 1, .9)  # x is blue
        OGL.glVertex3f(0, 0, 0)
        OGL.glVertex3f(x, 0, 0)
        OGL.glEnd()
        
class HocViewer(gl.GLViewWidget):
    """
    Subclass of GLViewWidget that displays data from HocReader.
    This is a convenience class implementing boilerplate display code.

    Input:
        h: HocReader instance or "xxxx.hoc" file name
    """
    def __init__(self, hoc, camerapos=[200., 0., 0.], renderer='pyqtgraph', fighandle=None, 
                figsize=[720, 720], flags=None):
        if not isinstance(hoc, HocReader):
            hoc = HocReader(hoc)
        self.hr = hoc
        self.graphics = []
        self.flags = flags
        self.camerapos = camerapos
        self.video_file = None
        print('Renderer: ', renderer)
        if renderer == 'pyqtgraph' and fighandle == None:
            pg.mkQApp()  # make sure there is a QApplication before instantiating any QWidgets.
            super(HocViewer, self).__init__()
            self.resize(figsize[0], figsize[1])

            # self.setBackgroundColor(pg.glColor(pg.mkColor(255, 255, 255, 255)))
            self.setBackgroundColor(pg.glColor(pg.mkColor(0.1, 0.1, 0.1, 1)))
            color='w'
            # self.setBackgroundColor(0.2)
            self.show()
            self.setWindowTitle('hocViewer')
            self.setCameraPosition(distance=camerapos[0], elevation=camerapos[1], azimuth=camerapos[2])
        elif renderer == 'mayavi' and fighandle == None:
            fighandle = mlab.figure(figure=None, bgcolor=(0.6,0.6,0.6), fgcolor=None, size=(figsize[0], figsize[1]))
            super(HocViewer, self).__init__()

        elif renderer == 'mpl' and fighandle == None:
            super(HocViewer, self).__init__()
        ####
        # original grid code
        self.g = gl.GLGridItem()
        self.g.scale(10,10,10)
        self.g.color = [0, 0, 0, 1]
        # self.g.setColor(pg.mkColor('w'))
        self.addItem(self.g)
        self.ax = GLAxisItem_r()
        self.addItem(self.ax)
        self.ax.setSize(50, 50, 50)
        # print(dir(self))
        # self.mouseReleaseEvent(self.mouse_released2)
        # print(self.signalsBlocked())
        # print(dir(self.ax))
        # print(self.ax.childItems())

        # self.grid = self.draw_grid()
        # self.resetGrid()
        # self.grid = HocGrid()
        # self.graphics.append(self.grid)
        # gl.GLGridItem(color=pg.mkColor(128, 128, 128))

        # self.grid.setSize(x=40., y=40., z=40.)  # 100 um grid spacing
        # self.grid.setSpacing(x=20., y=20., z=20.)  # 10 um steps
        # self.grid.scale(1,1,1)  # uniform scale
        # self.grid.translate(0., 0., 0.)
        # self.addItem(self.grid)

    def mouse_released(self, event):
        print('released, event = ', event)

        
    def setBackcolor(self, color):
        self.setBackgroundColor(color)

    def setCamera(self, distance=200., elevation=45., azimuth=45.):
        self.camerapos = [distance, elevantion, azimuth]
        # for opengl:
        self.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)

    def draw_grid(self):
        g = HocGrid()
        self.graphics.append(g)
        self.addItem(g)
        return g

    def resetGrid(self, x=40., y=40., z=40.):
        self.grid.setSize(x=x, y=y, z=z)  # 100 um grid spacing

    def draw_volume(self):
        """
        Add a HocVolume graphic to this view.

        Returns
        -------
          HocVolume instance
        """
        raise Exception('Not implemented')
        g = HocVolume(self.hr)
        self.graphics.append(g)
        self.addItem(g)
        return g

    def draw_volume_mayavi(self):
        """
        Add a HocVolume graphic to this view.

        Returns
        -------
          HocVolume instance
        """
        MV = mayavi_Volume(self.hr)
        # g = HocVolume(self.hr)
        g = MV.g
        # self.graphics.append(g)
        # self.addItem(g)
        return g

    def draw_surface(self):
        """
        Add a HocSurface graphic to this view.

        Returns
        -------
        HocSurface instance
        """
        g = HocSurface(self.hr)
        self.graphics.append(g)
        self.addItem(g)
        return g

    def draw_graph(self):
        """
        Add a HocGraph graphic to this view.

        Returns
        -------
        HocGraph instance
        """
        g = HocGraph(self.hr)
        self.graphics.append(g)
        self.addItem(g)
        return g

    def draw_cylinders(self):
        """
        Add a HocCylinders graphic to this view.

        Returns
        -------
        HocCylinders instance
        """

        g = HocCylinders(self.hr) #, facets=24) ? no facets in call?
        self.graphics.append(g)
        self.addItem(g)
        return g

    def draw_mayavi_cylinders(self, color=(0,0,1), label=None, flags=None, mechanism=None):
        MC = mayavi_Cylinders(self.hr, color=color, label=label, flags=flags, mechanism=mechanism, 
        camerapos=self.camerapos)
        return(MC)

    def draw_mayavi_graph(self, color=None, label=None, flags=None) -> object:
        MG = mayavi_graph(self.hr, color=color, label=label, flags=flags)
        return(MG)

    def draw_mpl_cylinders(self, fax=None):
        """
        fax is figure and axes: [fig, ax] from 3d plot definition
        """
        mpl_Cylinders(self.hr, fax=fax)

    def draw_mpl_graph(self, fax=None, color='blue'):
        """
        fax is figure and axes: [fig, ax] from 3d plot definition
        """
        mpl_Graph(self.hr, fax=fax, color=color)

    def draw_vispy(self):
        vispy_Cylinders(self.hr)


    def save_frame(self, file_name=None):
        """
        Save the currently visible frame to a file.
        If no file name is given, then the frame is added on to the currently-
        accumulating video stack.

        Parameters
        ----------
        filename: str (default: None)
            filename to save frame to

        Returns
        -------
        Nothing
        """
        if file_name is None:
            if self.video_file is None:
                raise Exception("No file name specified and no video storage in progress.")
            img = pg.imageToArray(self.readQImage())
            self.video_file.write(img)
        else:
            self.readQImage().save(file_name)
        print('Saved frame to file: ', file_name)


    def begin_video(self, file_name, fps=25):
        """
        Begin storing a new video to *file_name*.
        New frames are added to the file when save_frame() is called.

        Parameters
        ----------
        filename: str (default None)
            filename to save video to

        fps: int (default 25)
            frames per second for storage

        Returns
        -------
        Nothing

        """
        import cv
        import cv2
        winsize = self.width(), self.height()
        self.video_file = cv2.VideoWriter()
        self.video_file.open(filename=file_name,
                             fourcc=cv.CV_FOURCC('M', 'P', '4', 'V'),
                             fps=fps,
                             frameSize=winsize,
                             isColor=False)
        print('opened video file: ', file_name)

    def save_video(self):
        """
        Finish storing the video created since the last call to begin_video()
        """
        print('Finished storing video file')
        self.video_file.release()
        self.video_file = None
