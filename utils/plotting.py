import os
import matplotlib.pyplot as plt

def use_earthy_style():
    here = os.path.dirname(__file__)
    style_path = os.path.join(here, "earthy.mplstyle")
    plt.style.use(style_path)
