import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from hoc_reader import HocReader
from hoc_graphics import *

class HocViewer(gl.GLViewWidget):
    """
    Subclass of GLViewWidget that displays data from HocReader.
    This is a convenience class implementing boilerplate display code.
    
    Input:
        h: HocReader instance or "xxxx.hoc" file name
    """
    def __init__(self, hoc):
        if not isinstance(hoc, HocReader):
            hoc = HocReader(hoc)
        self.hr = hoc
        pg.mkQApp()  # make sure there is a QApplication before instantiating any QWidgets.
        super(HocViewer, self).__init__()
        self.resize(720,720)
        self.show()
        self.setWindowTitle('hocViewer')
        self.setCameraPosition(distance=200., elevation=45., azimuth=45.)
        self.setBackgroundColor((0.5, 0.5, 0.5, 0.25))

        self.g = gl.GLGridItem(color=pg.mkPen(0.5))
        self.g.setSize(x=100., y=100., z=100.)  # 100 um grid spacing
        self.g.setSpacing(x=10., y=10., z=10.)  # 10 um steps
        self.g.scale(1,1,1)  # uniform scale
        self.addItem(self.g)
        
        self.graphics = []
        self.video_file = None


    def draw_volume(self):
        """
        Add a HocVolume graphic to this view.
        
        Returns
        -------
          HocVolume instance
        """
        g = HocVolume(self.hr)
        self.graphics.append(g)
        self.addItem(g)
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
        g = HocCylinders(self.hr, facets=24)
        self.graphics.append(g)
        self.addItem(g)
        return g

    def save_frame(self, filename=None):
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
            self.readQImage().save(filename)
        print 'Saved frame to file: ', file_name

    
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
        print 'opened video file: ', file_name
        
    def save_video(self):
        """
        Finish storing the video created since the last call to begin_video()
        """
        print 'Finished storing video file'
        self.video_file.release()
        self.video_file = None
