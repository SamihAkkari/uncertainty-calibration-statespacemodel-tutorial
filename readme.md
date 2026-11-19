# Tutorial: Uncertainty Propagation, Calibration, and State-Space Modelling

## 1. Introduction and Purpose

This repository provides a hands-on tutorial on uncertainty propagation, calibration, and state-space modelling using the Kalman filter.

The objective is to introduce the underlying concepts and demonstrate their practical implementation in Python through a series of guided exercises. The repository includes datasets, Python scripts, and Jupyter notebooks designed to support a progressive learning experience.

The repository is organised around two complementary notebooks:

- **Exercise notebook (`GAPS*.ipynb`):** Contains the tutorial exercises, with selected sections of code intentionally left incomplete. Learners are expected to complete these sections, apply the concepts introduced, and explore the associated datasets.
- **Solution notebook :** Provides the completed version of the exercises, including the missing code and corresponding solutions.

The tutorial covers three main topics:

1. **Uncertainty propagation:** Understanding how uncertainties in input quantities propagate through mathematical models and affect the resulting estimates.
2. **Calibration:** Applying calibration methods to estimate model parameters and assess their associated uncertainties.
3. **State-space modelling and Kalman filtering:** Introducing state-space representations and using the Kalman filter to estimate the evolution of a system from noisy observations.

## 2. Environment Preparation

Before starting the exercises, you need to install Python and the required scientific computing libraries.

### 2.1. Required Software and Packages

The tutorial is developed in Python and uses the following packages:

- `numpy`: Numerical computations and array operations.
- `pandas`: Data manipulation and analysis.
- `matplotlib`: Data visualisation and plotting.
- `scipy`: Scientific computing and numerical methods.

We recommend using the **Anaconda Distribution**, which provides Python, Jupyter Notebook, and a convenient environment for managing scientific computing packages.

### 2.2. Installing Anaconda

#### Step 1 — Download Anaconda

1. Open your web browser.
2. Visit the official Anaconda download page: https://www.anaconda.com/download
3. Select your operating system (Windows, macOS, or Linux).
4. Download the Anaconda Distribution installer.

#### Step 2 — Install Anaconda

**Windows:**

1. Double-click the downloaded `.exe` installer.
2. Click Next and accept the licence agreement.
3. Select "Just Me" (recommended).
4. Keep the default installation location.
5. Leave "Add Anaconda to my PATH environment variable" unchecked.
6. Select "Register Anaconda as my default Python".
7. Click Install and wait for the installation to complete.
8. Click Finish.

**macOS:**

1. Open the downloaded `.pkg` installer.
2. Follow the installation instructions.
3. Click Continue until the installation is complete.

### 2.3. Opening the Anaconda Prompt or Terminal

Once Anaconda is installed, open the appropriate command-line interface:

- **Windows:** Open the Start Menu, search for `Anaconda Prompt`, and launch it.
- **macOS/Linux:** Open the Terminal application.

All subsequent commands should be entered in this window.

### 2.4. Activating the Python Environment

Anaconda automatically creates a default environment called `base`.

Activate it by running:

```bash
conda activate base
```

If the environment is already active, you should see `(base)` at the beginning of the command prompt.

### 2.5. Checking the Required Packages

Check whether the required packages are already installed by executing the following commands individually:

```bash
python -c "import numpy"
python -c "import pandas"
python -c "import matplotlib"
python -c "import scipy"
```

If a command executes without displaying an error, the corresponding package is installed.

If you encounter an error such as:

```text
ModuleNotFoundError: No module named 'numpy'
```

the corresponding package is missing and must be installed.

### 2.6. Installing Missing Packages

To install all the required packages, execute:

```bash
conda install numpy pandas matplotlib scipy
```

If some packages are already installed, Conda will simply check their dependencies and install or update what is necessary.

When prompted, type `y` to confirm the installation.

### 2.7. Verifying the Installation

To ensure that Python and all required libraries are working correctly, launch the Python interpreter:

```bash
python
```

You should see the Python version information followed by the interactive prompt:

```text
Python 3.x.x | Anaconda, Inc.
>>>
```

Next, execute:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

print("All packages imported successfully!")
```

If the success message appears without any errors, your environment is ready.

Exit the Python interpreter using:

```python
exit()
```

## 3. Getting Started with the Tutorial

Once the environment is prepared:

1. Download or clone this repository to your computer.
2. Navigate to the repository directory.
3. Launch Jupyter Notebook by running:

```bash
jupyter notebook
```

4. Open the exercise notebook (`TD.ipynb`).
5. Follow the instructions and complete the missing sections of code.
6. Use the solution notebook (`TD_Solutions.ipynb`) to check your work once you have attempted the exercises.

**Important:** To get the most out of this tutorial, try to complete the exercises independently before referring to the solutions. The objective is not only to obtain the correct numerical results but also to understand the methods and the reasoning behind their implementation.
