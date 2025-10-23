## Setup Instructions

To run the code and make changes, the following steps must be followed:

### 1. Clone this repository
```bash
git clone https://github.com/GortFan/VDH-Lab.git
```
This can be done in any location on your computer's disk, using the VSCode terminal, Windows command prompt, or Anaconda command prompt.

### 2. Create a virtual environment
```bash
conda create -n vdh-lab python=3.8
```
Python 3.8 is used for compatibility between OpenPNM v3.0.0 and its dependencies.

### 3. Activate the environment
```bash
conda activate vdh-lab
```

### 4. Open VSCode
```bash
code .
```
Then navigate to the cloned repository 

### 5. Clone OpenPNM
In the VSCode terminal, run:
```bash
git clone https://github.com/PMEAL/OpenPNM
cd OpenPNM
```

### 6. Checkout OpenPNM v3.0.0
```bash
git checkout v3.0.0
```
This changes the version of the package to the 3.0.0 release.

### 7. Install OpenPNM dependencies
```bash
conda install --file requirements/conda.txt -c conda-forge
```

### 8. Install OpenPNM in editable mode
```bash
pip install -e .
``` 