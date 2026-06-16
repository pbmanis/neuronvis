##########################################################
# Color maps for various SWC names. 
# # this is an extensible list...
# 
# colors are from XKCD color list. Sorry folks.
#



section_colors = {
    "Undefined": "black",

    "axon": "green",  # in this dict, we handle multiple labels for the same structure.

    "Axon_Initial_Segment": "cyan",
    "axon_initial_segment": "cyan",
    "Axon_initial_segment": "cyan",
    "initialsegment": "cyan",
    "initseg": "cyan",
    "ais": "cyan",

    "hillock": "red",
    "Axon_Hillock": "red",
    "Axon_hillock": "red",
    
    "myelinatedaxon": "dark red",
    "Myelinated_Axon": "dark red",
    "Axon_myelinated_segment": "dark red",
    "Axon_node": "light blue",
    "Axon_heminode": "light blue",
    "unmyelinatedaxon": "light cyan",
    "Unmyelinated_Axon": "light cyan",
    "Axon_unmyelinated": "light cyan",
    "Axon_spine": "dark blue",
    "Axon_en_passant_synapse": "blue green",
    "Terminal_swelling": "magenta",

    "soma": "dark grey",
    "Soma": "dark grey",
    "somatic": "dark grey",
    "Soma_spine": "baby blue",
    "Cilium": "white",
    "Soma_undefined_1": "dark grey",
    "Soma_undefined_2": "dark grey",
    "Soma_undefined_3": "dark grey",


# Dendrites
    "apic": "beige",
    "apical": "beige",
    "Dendrite_apical": "beige",
    "Distal_Dendrite": "beige",
    "Dendrite_distal": "beige",
    "dend": "grey",
    "dendrite": "grey",
    "Proximal_Dendrite": "teal",
    "Dendrite_proximal": "teal",
    "basal": "magenta",
    "basal_dendrite": "magenta",
    "Dendrite_basal": "magenta",
    "Dendrite_spine": "red",
# parts of dendrites
    "Dendrite_swelling": "yellow",
    "Dendritic_Swelling": "yellow",
    "Dendritic_Hub": "neon red",
    "Dendrite_hub": "neon red",  # "wintergreen",
    # "Dendrite_undefined_1": "darkgrey",  # Claude fixed 2026-06-16: "darkgrey" not in xkcd; xkcd uses "dark grey"
    # "Dendrite_undefined_2": "darkgrey",  # Claude fixed 2026-06-16: same
    # "Dendrite_undefined_3": "darkgrey",  # Claude fixed 2026-06-16: same
    "Dendrite_undefined_1": "dark grey",
    "Dendrite_undefined_2": "dark grey",
    "Dendrite_undefined_3": "dark grey",
    
    # granule cell
    "primarydendrite": "teal",
    "Dendrite_preclaw": "powder blue",
    "Dendrite_claw": "light blue",
    "Dendritic_Claw": "light blue",
    "dendriticclaw": "light blue",

    # calyx specific
    "heminode": "green",
    "stalk": "yellow",
    "branch": "blue",
    "neck": "brown",
    "swelling": "magenta",
    "tip": "powder blue",
    "parentaxon": "orange",
    "synapse": "white",
    "Terminal": "green",
    "Terminal_stalk": "yellow",
    "Terminal_neck": "brown",
    "Terminal_branch": "blue",
    "Terminal_undefined_1": "blue",
    "Terminal_undefined_2": "blue",
    "Terminal_undefined_3": "blue",
    
    # astrocytes
    "Astrocyte_soma": "orange",
    "Astrocyte_primary_branch": "pale orange",
    "Astrocyte_distal_process": "mustard",
    "Astrocyte_vellus_process": "goldenrod",
    "Astrocyte_blood_brain_barrier": "puke green",
    "Astrocyte_undefined": "white",
    
    # other (cortex dendrites)
    "dend1_*": "magenta",
    "dend2_*": "yellow",
    "dend3_*": "dandelion",
    "dend4_*": "beige",
    "apic": "dandelion",
    "dend_1*": "red",
    "dend_2*": "orange",
    "dend_3*": "yellow",
    "dend_4*": "green",
    "dend_5*": "blue",
    "dend_6*": "indigo",
    "dend_7*": "violet",
    "dendritic_0": "red",
    "dendritic_5": "yellow",
    "dendritic_7": "orange",
    "custom": "white",
    "unspecified neurites": "gold",
    "Neurite": "gold",
    "type_8": "red",
    "type_13": "orange",
    "type_14": "yellow",
    "type_15": "green",

}
