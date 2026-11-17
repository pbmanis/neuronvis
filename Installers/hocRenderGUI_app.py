"""
PyInstaller entry point for the hocRender GUI.

PyInstaller needs a script rather than a module with relative imports, so this
file only imports and runs neuronvis.hocRender_gui.main().
"""

from neuronvis.hocRender_gui import main

if __name__ == "__main__":
    main()
