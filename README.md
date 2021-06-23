# AB_2021_MCMC

Stan code for multilevel constitutive data fitting of collagen scaffolds.

Publication: 

**Mechanobiological Wound Model for Improved Design and Evaluation of Collagen Dermal Replacement Scaffolds 
David Sohutskay, Sherry Voytik-Harbin, Adrian Buganza Tepole**

The code has been designed to run on Purdue Research Computing Clusters. It may not work as-is in your system. Here are a few tips to troubleshoot compiling: 

* The code requires Stan library. It currently runs witn BOOST 1.75 and EIGEN 3.2.10
* The code has a parallel version with OPEN MP. The parallel parts of the code are primarily in solver.cpp and can be easily commented out if desired 

Please get in touch in you have additional questions about using this code:
abuganza@purdue.edu
Adrian Buganza Tepole
Associate Professor, Purdue University 


