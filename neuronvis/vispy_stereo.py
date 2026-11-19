"""
vispy_stereo : side-by-side stereo and lighting for vispy scenes.

StereoCanvas   one SceneCanvas holding two ViewBoxes, left eye | right eye.
               vispy 0.15 visuals have a single parent, so every visual is
               made twice (add_mesh does that; the data is shared on the CPU).
EyeCamera      an ArcballCamera turned by half the eye separation about the
               screen's vertical axis, so both eyes converge on the centre of
               rotation (that depth is at the screen; nearer parts come out).
               The two cameras are linked: dragging in either half turns both.
Lighting       one ShadingFilter setting for every mesh (both eyes, or a
               single view), with a headlight that follows the camera. Both
               eyes get the same light direction, or they are shaded differently.
merge_meshes() the tubes and spheres of a morphology as one MeshData (one draw
               call per eye instead of one per section).

For display glasses with a "3D" side-by-side mode (XREAL, Viture, ...): put
the canvas full screen on the glasses' display (3840 x 1080, each eye
1920 x 1080). Swap eyes to free-view (cross-eyed) on an ordinary screen.

Note on vispy's lighting (checked against vispy 0.15.2):
  - ShadingFilter.light_dir is the direction the light travels: the shader
    lights a face by dot(-light_dir, normal), in the ViewBox's data coordinates.
  - camera.transform maps the camera frame to the scene; in that frame x is
    right, y is up and z points towards the viewer.
"""

from typing import Union

import numpy as np
import vispy.scene
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from vispy.color import Color
from vispy.geometry import MeshData
from vispy.scene.cameras import ArcballCamera
from vispy.util import transforms

stereo_defaults = {
    "background": (0.0, 0.0, 0.0, 1.0),  # black: see-through in birdbath glasses, best contrast
    "separation": 3.0,  # degrees between the eyes' view directions
    "fov": 30.0,  # perspective is needed for depth; narrower than vispy's 45
    "brightness": 0.9,  # minimum HSV value of the colours on a black background
}

light_defaults = {
    "headlight": True,
    "up": 20.0,  # degrees above the line of sight ("over the shoulder")
    "left": 20.0,  # degrees to the left of it
    "ambient": 0.30,
    "diffuse": 0.80,
    "specular": 0.25,
    "shininess": 60.0,
}

fixed_light = (10.0, 5.0, -5.0)  # vispy's default light_dir, used when the headlight is off


class EyeCamera(ArcballCamera):
    """ArcballCamera seen from one eye: turned by *eye_angle* degrees about the screen's vertical"""

    def __init__(self, eye_angle: float = 0.0, **kwds):
        self.eye_angle = eye_angle
        super().__init__(**kwds)

    def _get_rotation_tr(self):
        # applied before the arcball rotation, in the camera's frame (z is up there)
        return transforms.rotate(self.eye_angle, (0, 0, 1)) @ super()._get_rotation_tr()


class Lighting:
    """ambient/diffuse/specular and a headlight, applied to every registered mesh"""

    def __init__(self, camera: object, **kwds):
        self.camera = camera  # the light follows this camera
        self.settings = dict(light_defaults, **kwds)
        self.meshes = []
        # whenever the camera moves (canvas.render(), used for Save image, emits no draw event)
        camera.transform.changed.connect(self.update_direction)

    def add(self, visual: object) -> None:
        """a MeshVisual, or a visual holding one (Sphere keeps it in _mesh)"""
        mesh_visual = getattr(visual, "_mesh", visual)
        if getattr(mesh_visual, "shading_filter", None) is not None:
            self.meshes.append(mesh_visual)
            self.apply_settings(mesh_visual)
            mesh_visual.shading_filter.light_dir = self.direction()

    def set(self, **kwds) -> None:
        self.settings.update(kwds)
        for m in self.meshes:
            self.apply_settings(m)
        self.update_direction()

    def apply_settings(self, mesh_visual: object) -> None:
        s = self.settings
        sf = mesh_visual.shading_filter
        sf.ambient_light = (1, 1, 1, s["ambient"])
        sf.diffuse_light = (1, 1, 1, s["diffuse"])
        sf.specular_light = (1, 1, 1, s["specular"])
        sf.shininess = s["shininess"]

    def direction(self) -> tuple:
        """
        the direction the light travels, in scene coordinates: from a lamp
        behind the viewer, raised and to the left; or vispy's fixed light
        when the headlight is off
        """
        if not self.settings["headlight"]:
            return fixed_light
        up, left = np.radians(self.settings["up"]), np.radians(self.settings["left"])
        lamp = np.array([-np.sin(left) * np.cos(up), np.sin(up), np.cos(left) * np.cos(up), 0.0], dtype=np.float32)
        d = -self.camera.transform.map(lamp)[:3]
        return tuple(d / max(np.linalg.norm(d), 1e-10))

    def update_direction(self, event: object = None) -> None:
        d = self.direction()
        for m in self.meshes:
            m.shading_filter.light_dir = d


class StereoCanvas:
    """two linked views of the same scene, side by side"""

    def __init__(
        self,
        separation: Union[float, None] = None,
        swap: bool = False,
        bgcolor: Union[tuple, None] = None,
        **canvas_kwds,
    ):
        separation = stereo_defaults["separation"] if separation is None else separation
        bgcolor = stereo_defaults["background"] if bgcolor is None else bgcolor
        canvas_kwds.setdefault("keys", "interactive")  # None: no vispy keys (its Esc closes the canvas)
        self.canvas = vispy.scene.SceneCanvas(bgcolor=bgcolor, **canvas_kwds)
        grid = self.canvas.central_widget.add_grid(spacing=0)
        self.views = [grid.add_view(row=0, col=0), grid.add_view(row=0, col=1)]
        self.cameras = [EyeCamera(fov=stereo_defaults["fov"]), EyeCamera(fov=stereo_defaults["fov"])]
        for view, camera in zip(self.views, self.cameras):
            view.camera = camera
        self.cameras[0].link(self.cameras[1])
        self.set_separation(separation, swap)
        self.lighting = Lighting(self.cameras[0])

    def set_separation(self, separation: float, swap: bool = False) -> None:
        """left eye on the left half (parallel viewing, glasses), or swapped for cross-eyed viewing"""
        sign = -1.0 if swap else 1.0
        self.cameras[0].eye_angle = -sign * separation / 2.0
        self.cameras[1].eye_angle = sign * separation / 2.0
        for camera in self.cameras:
            camera.view_changed()

    def add_mesh(
        self,
        vertices: Union[np.ndarray, None] = None,
        faces: Union[np.ndarray, None] = None,
        meshdata: Union[MeshData, None] = None,
        translucent: bool = False,
        **kwds,
    ) -> list:
        """the same mesh in both views, lit by self.lighting; returns the two visuals"""
        visuals = []
        for view in self.views:
            m = vispy.scene.visuals.Mesh(vertices=vertices, faces=faces, meshdata=meshdata, shading="smooth", **kwds)
            if translucent:
                m.set_gl_state("translucent", depth_test=True, cull_face=False)
            view.add(m)
            self.lighting.add(m)
            visuals.append(m)
        return visuals

    def add_visual(self, make: callable) -> list:
        """make(parent) creates one visual (e.g. markers, axes); called once for each eye"""
        return [make(view.scene) for view in self.views]

    def set_range(self, lo: np.ndarray, hi: np.ndarray) -> None:
        self.cameras[0].set_range(x=(lo[0], hi[0]), y=(lo[1], hi[1]), z=(lo[2], hi[2]))


def merge_meshes(visuals: list) -> MeshData:
    """
    one MeshData from several MeshVisuals (or visuals holding one in _mesh,
    like Sphere), keeping each one's vertex colours (or its single colour)
    """
    vertices, faces, colors, n = [], [], [], 0
    for visual in visuals:
        md = getattr(visual, "_mesh", visual).mesh_data
        v, f = md.get_vertices(), md.get_faces()
        if md.has_vertex_color():
            c = md.get_vertex_colors()
        elif md.has_face_color():
            c = np.tile(md.get_face_colors()[:1], (len(v), 1))
        else:
            c = np.tile(Color(getattr(visual, "color", (0.6, 0.6, 0.6, 1.0))).rgba, (len(v), 1))
        vertices.append(v)
        faces.append(f + n)
        colors.append(c)
        n += len(v)
    return MeshData(
        vertices=np.concatenate(vertices), faces=np.concatenate(faces), vertex_colors=np.concatenate(colors)
    )


def brighten(rgba: np.ndarray, value: float = stereo_defaults["brightness"]) -> np.ndarray:
    """raise the HSV value of each colour to at least *value* (for a black background); alpha is kept"""
    out = np.array(rgba, dtype=np.float32)
    hsv = rgb_to_hsv(np.clip(out[..., :3], 0.0, 1.0))
    hsv[..., 2] = np.maximum(hsv[..., 2], value)
    out[..., :3] = hsv_to_rgb(hsv)
    return out
