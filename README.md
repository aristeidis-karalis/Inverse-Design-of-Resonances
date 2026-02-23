This repository contains the codes for the results in the article:
* Mo Chen, Steven G. Johnson, Aristeidis Karalis, "Inverse design of multiresonance filters via quasi-normal mode theory", Optics Express, Vol. 34, Issue 4, pp. 5729-5752 (2026) (doi: 10.1364/OE.579219)

Before running any code, you need to change some files in the julia package "LeastSquaresOptim" to enable a more efficient version of the Levenberg-Marquardt algorithm (see article Appendix B). In the julia package environment, do "dev LeastSquaresOptim". Within "~/.julia/dev/LeastSquaresOptim/", replace the appropriate files with those in the "LeastSquaresOptim/" folder of this repository.

To get the results in Figure 3, change to the "2dTopologyOpt/" directory and run the file "TopOptRes.jl" in a parallelized system with 24 nodes. The results will be saved to ".jld" files and the plots can then be produced in the Jupyter notebook "TopOptRes_plot.ipynb".

To get the results in Figure 4, in the file "TopOptRes.jl", change the parameters to N = 4, σ1phase = 1im, αC = 0.022, and run again with 24 nodes.

To get the results in Figure 5, change to the "1dAlternateStackOpt/" directory and run the file "AlternateStackOpt.jl".

To get the results in Figure 6, change to the "LCladderOpt/" directory and run the file "LCladderOpt.jl". For the standard 4th-order solution, change to σ1phase = -π/2.
