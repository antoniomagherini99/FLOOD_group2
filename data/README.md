The datasets used in this repository are based upon a research paper by Bentivoglio et al (2023). All datasets include DEMs, velcoity in x and y
and water depths at different time steps. Each folder has csvs which are created by the encode_to_csv function in the pre_processing folder.
These are sometimes split as the upload to github is limited to 100 mb in our case.

# dataset_train_val
80 DEMs over a squared domain of 64x64 grids of length 100m with a simulation time of 48h. Breach is always located in the bottom left corner.

# Dataset 1
20 DEMs over a squared domain of 64x64 grids of length 100 m and a simulation time of 48 h. A fixed breach location is used (bottom left corner)
	
# Dataset 2
20 DEMs with the same domain as dataset 1, but now with a randomly varying breach location.

# Dataset 3
10 DEMs now over a squared domain of 128x128 grids of length 100 m and a simulation time of 120 h. The breach locations also vary randomly across the domain.
