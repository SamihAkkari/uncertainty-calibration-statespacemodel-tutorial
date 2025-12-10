# Environment prepation 

## 1. What You Will Install

For the TD, you need Python and the following packages:

* numpy
* pandas
* matplotlib
* scipy

Anaconda normally includes all these packages in its base environment, but we will still check to be sure.

## 2. Step-by-Step Installation of Anaconda

### Step 1 — Download Anaconda

1. Open your web browser (Chrome, Firefox, etc.).
2. Go to: https://www.anaconda.com/download
3. Choose your operating system: Windows, Mac, or Linux.
4. Download the Anaconda Distribution (Python 3.x) installer.

### Step 2 — Install Anaconda

#### Windows:

1. Double-click the downloaded `.exe` file.
2. Click Next, accept the license agreement, then Next again.
3. Choose "Just Me" (recommended).
4. Leave the installation path as default.
5. Important: keep Add Anaconda to my PATH unchecked.
6. Make sure Register Anaconda as my default Python is checked.
7. Click Install and wait for the installation to finish.
8. Click Finish.

#### Mac:

1. Open the downloaded `.pkg` file.
2. Follow the instructions and click Continue until installation completes.

## 3. Opening the Anaconda Prompt

Once Anaconda is installed:

**Windows:**
* Open the Start Menu, type Anaconda Prompt, and click it.

**Mac/Linux:**
* Open Terminal (Anaconda automatically configures it).

This is where you will type commands.

## 4. Activate the Base Environment

Anaconda automatically creates a default environment called base.

In the prompt/terminal, type:
```bash
conda activate base
```

(If it is already active, you will see `(base)` at the beginning of the line.)

## 5. Check If the Required Packages Are Installed

Type each of the following commands inside Anaconda Prompt/Terminal:
```bash
python -c "import numpy"
python -c "import pandas"
python -c "import matplotlib"
python -c "import scipy"
```

If nothing happens and no error message appears → the package is installed.

If you get an error like:
```
ModuleNotFoundError: No module named 'xxx'
```

then the package is missing. Install it as described in the next section.

## 6. Installing Missing Packages

If a package is missing, install it using:
```bash
conda install numpy pandas matplotlib scipy
```

* You can install them all at once (recommended).
* Press y when asked to proceed.

If conda gives an environment-related error, try:
```bash
conda update --all
```

then repeat the installation command.

## 7. Test the Installation

Open Python inside Anaconda Prompt by typing:
```bash
python
```

You should now see something like:
```
Python 3.x.x | Anaconda, Inc.
>>>
```

Then type the following inside Python:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
print("All packages imported successfully!")
```

If you see the printed message and no error, everything is working.

Exit Python by typing:
```python
exit()
```