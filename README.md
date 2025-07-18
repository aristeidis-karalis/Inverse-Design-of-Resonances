This repository contains the codes for the results in the article:
* Mo Chen, Steven G. Johnson, Aristeidis Karalis, "Inverse design of multiresonance filters via quasi-normal mode theory", arxiv:2504.10219, April 2025

Before running any code, you need to change some files in the julia package "LeastSquaresOptim". In the julia package environment, do "dev LeastSquaresOptim". Then replace the files that you can locate in the "LeastSquaresOptim" branch of this repository.

To get the results in Figure 3, run the file "TopOptRes.jl" in a parallelized system with 24 nodes. The plots can be produced in the Jupyter notebook "TopOptRes_plot.ipynb".

To get the results in Figure 4, in the file "TopOptRes.jl", change the parameters to N = 4, σ1phase = 1im, αC = 0.022, and run again with 24 nodes.

To get the results in Figure 5, run the file "AlternateStackOpt.jl".
