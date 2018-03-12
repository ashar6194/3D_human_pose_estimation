from vtk_visualizer import *
# import vtk_visualizer
import numpy as np

__colors = {
    'r': (1, 0, 0),
    'g': (0, 0.8, 0),
    'b': (0, 0, 1),
    'c': (0, 1, 1),
    'm': (1, 0, 1),
    'y': (1, 1, 0),
    'w': (1, 1, 1),
    #'k': (0, 0, 0),  # do we need black?
}


def _char2color(c):
    "Convert one-letter code to an RGB color"
    global __colors
    return __colors[c]


def _next_color(color):
    c = list(__colors.keys())
    i = c.index(color)
    i += 1
    if i >= len(c):
        i = 0
    return c[i]


def plotxyz(pts_l, color='g', hold=False, block=False):
    if not isinstance(pts_l, list):
        pts_l = [(pts_l)]
    vtkControl = get_vtk_control(block)
    if not (hold or is_hold_enabled()):
        vtkControl.RemoveAllActors()

    for pts in pts_l:
        vtkControl.AddPointCloudActor(pts)
        if pts.shape[1] <= 3:
            nID = vtkControl.GetLastActorID()
            vtkControl.SetWindowBackground(1, 1, 1)
            vtkControl.AddAxesActor(1)
            vtkControl.SetActorColor(nID, _char2color(color))
            vtkControl.Render()
            color = _next_color(color)

    if block:
        vtkControl.exec_()


if __name__ == '__main__':
    for i in range(1000):
        pointCloud = np.random.rand(100000, 3)*10
        # view_point_cloud(pointCloud)

        # vtkControl = VTKVisualizerControl()
        plotxyz(pointCloud, color='r', hold=False)


