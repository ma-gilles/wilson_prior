# Wilson prior

This is the companion code to the paper [A Molecular Prior Distribution for Bayesian Inference Based on Wilson Statistics](https://arxiv.org/abs/2202.09388) by Marc Aurèle Gilles and Amit Singer.

To run the examples, you can create a virtual environment as follows:

```
conda create -n wilson python=3.6 
conda activate wilson
pip install -r <THIS_DIRECTORY>/requirements.txt
python -m ipykernel install --user --name=wilson
```

and then use the kernel to run the notebooks:
- [generate_sample_images.ipynb](https://github.com/ma-gilles/wilsontest/blob/main/generate_sample_images.ipynb) for figure 1.
- [solve_linear_inverse_problem.ipynb](https://github.com/ma-gilles/wilsontest/blob/main/solve_linear_inverse_problem.ipynb) for figures 2 and 3.

