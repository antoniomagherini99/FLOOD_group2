The datasets used in this repository are based upon a research paper by Bentivoglio et al (2023). For training and validation purposes a dataset of 80 different generated digital elevation models are used to reflect different plausible topographies. The velocity in both x and y direction (VX and VY) as well as the water depth (WD) are known, based on numerical computations utilizing Delft3D.
For testing purposes three different datasets are used: 
	Dataset 1: 20 DEMs over a squared domain of 64x64 grids of length 100 m and a simulation time of 48 h. A fixed breach location is used.
	Dataset 2: 20 DEMs with the same domain as dataset 1, but now with a randomly varying breach location.
	Dataset 3: 10 DEMs now over a squared domain of 128x128 grids of length 100 m and a simulation time of 120 h. The breach locations also vary randomly across the domain.
