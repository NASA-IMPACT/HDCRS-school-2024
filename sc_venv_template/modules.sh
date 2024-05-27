module purge
module load Stages/2023
module load GCC OpenMPI
# Some base modules commonly used in AI
module load mpi4py numba tqdm matplotlib IPython SciPy-Stack bokeh git
module load Flask Seaborn OpenCV

# ML Frameworks
module load  scikit-learn
module load tensorboard
