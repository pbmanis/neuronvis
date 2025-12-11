import pandas as pd
import os
import swc_to_hoc
from pathlib import Path
import swc_sectypes
import numpy as np
import neuron
from neuron import h

def swc_match():

    df_counting = pd.read_csv('new_counting_points.csv', sep=',', header=1, comment='#', 
                              names = ['color', 'x', 'y', 'z'])
    df_counting = df_counting[df_counting['color'] == "Orange"]  # limit to one presynaptic point set
    # get the swc file
    swc_obj = swc_to_hoc.SWC(filename=Path("Granule_cell_1_um_v200000.swc"), section_map = "sbem3")

    # convert to dataframe
    df_swc = pd.DataFrame(swc_obj.data, columns=["id", "type", "x", "y", "z", "radius", "parent"])
    df_claw = df_swc[df_swc['type'] == 31]  # only look at claw nodes
    # make np arrays for computation
    nodes = np.array([df_claw['x'].values, df_claw['y'].values, df_claw['z'].values]).reshape(3, -1).T
    points = np.array([df_counting['x'].values, df_counting['y'].values, df_counting['z'].values]).reshape(3, -1).T
    for p in points:
        distances = np.linalg.norm(nodes - p, axis=1)
        closest_node_index = np.argmin(distances)
        closest_node = nodes[closest_node_index]
        p_str = ','.join([f"{coord:6.1f}" for coord in p])
        closest_node_str = ','.join([f"{coord:6.1f}" for coord in closest_node])
        print(f"Point [{p_str:s}] is closest to node [{closest_node_str:s}] at index {closest_node_index:>4d} with distance {distances[closest_node_index]:>6.2f} um")   

def hocx_match():

    df_counting = pd.read_csv('new_counting_points.csv', sep=',', header=1, comment='#', 
                              names = ['color', 'x', 'y', 'z'])
    df_counting = df_counting[df_counting['color'] == "Orange"]  # limit to one presynaptic point set
    hoc = "Granule_Cell_1_um_v200000.hocx"
    fullfile = Path(os.getcwd(), hoc)
    if not fullfile.exists():
        raise Exception("File not found: %s" % (str(fullfile)))
    if fullfile.suffix in [".hoc", ".hocx"]:
        neuron.h.hoc_stdout(
            "/dev/null"
        )  # prevent junk from printing while reading the file
        success = neuron.h.load_file(str(fullfile))
        neuron.h.hoc_stdout()
        if not success:
            raise Exception("Error reading file: %s" % (str(fullfile)))
    xyz = []
    sections = []
    for j, sec in enumerate(list(h.Dendrite_claw)):
        for i in np.arange(sec.n3d()):
            xyz.append([sec.x3d(i), sec.y3d(i), sec.z3d(i)])
            sections.append(sec.name())
            print(sec.nseg)
    # make np arrays for computation
    nodes = np.array(xyz)

    points = np.array([df_counting['x'].values, df_counting['y'].values, df_counting['z'].values]).reshape(3, -1).T
    for p in points:
        distances = np.linalg.norm(nodes - p, axis=1)
        closest_node_index = np.argmin(distances)
        closest_node = nodes[closest_node_index]
        p_str = ','.join([f"{coord:6.1f}" for coord in p])
        closest_node_str = ','.join([f"{coord:6.1f}" for coord in closest_node])
        closest_node_section = sections[closest_node_index]
        print(f"hoc Point [{p_str:s}] ({closest_node_section!s}) is closest to node [{closest_node_str:s}] at index {closest_node_index:>4d} with distance {distances[closest_node_index]:>6.2f} um")   


if __name__ == "__main__":
    hocx_match()