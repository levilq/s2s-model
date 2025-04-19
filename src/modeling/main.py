import sys
import os
path_to_package = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','..'))
sys.path.append(path_to_package)

from matplotlib import pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

from lem import SourceToSinkSimulator
plt.switch_backend('TkAgg')

def main():
    """Main function to run the Source-to-Sink model."""

    path_to_topo = os.path.join(DATA_DIR, 'sarez500m_utm.tif')
    path_to_precip = os.path.join(DATA_DIR, 'precip500m_utm.tif')

    mg = SourceToSinkSimulator(path_to_topography=path_to_topo, path_to_precipitation=path_to_precip)
    mg.setNoData(nodata_value=0)
    mg.createRasterModelGrid()
    mg.setWatershedBoundaryConditions()
    mg.setUpErosionDepositionModel()
    mg.runSimulation(50)

if __name__ == "__main__":
    main()
