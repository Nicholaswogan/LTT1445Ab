```sh
setopt INTERACTIVE_COMMENTS

mamba create -n emission -c conda-forge -c bokeh -y photochem=0.6.7 numpy=1.24 mpi4py dill tqdm astropy=6.0 matplotlib pandas pip xarray pathos bokeh=2.4.3 wget unzip tar pymultinest=2.12 scipy=1.11
mamba activate emission

# Prereqs for Photochem build
mamba install -y scikit-build cmake=3 ninja cython fypp pip c-compiler cxx-compiler fortran-compiler git
# build Photochem
wget https://github.com/nicholaswogan/photochem/archive/eb8fd42adba64ebf68eb15fc22e81e50104abb6e.zip
unzip eb8fd42adba64ebf68eb15fc22e81e50104abb6e.zip
cd photochem-eb8fd42adba64ebf68eb15fc22e81e50104abb6e
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX"
python -m pip install --no-deps --no-build-isolation . -v
cd ..
rm -rf photochem-eb8fd42adba64ebf68eb15fc22e81e50104abb6e
rm eb8fd42adba64ebf68eb15fc22e81e50104abb6e.zip

# Update photochem_clima_data
wget https://github.com/nicholaswogan/photochem_clima_data/archive/cba05ffaae3ab7ef11e27bb40570807014d718c0.zip
unzip cba05ffaae3ab7ef11e27bb40570807014d718c0.zip
cd photochem_clima_data-cba05ffaae3ab7ef11e27bb40570807014d718c0
python -m pip install --no-deps --no-build-isolation . -v
cd ..
rm -rf photochem_clima_data-cba05ffaae3ab7ef11e27bb40570807014d718c0
rm cba05ffaae3ab7ef11e27bb40570807014d718c0.zip

# Install PICASO
wget https://github.com/natashabatalha/picaso/archive/4d5eded20c38d5e0189d49f643518a7b336a5768.zip
unzip 4d5eded20c38d5e0189d49f643518a7b336a5768.zip
cd picaso-4d5eded20c38d5e0189d49f643518a7b336a5768
python -m pip install . -v
# Get reference
cd ../
cp -r picaso-4d5eded20c38d5e0189d49f643518a7b336a5768/reference picasofiles/
rm -rf picaso-4d5eded20c38d5e0189d49f643518a7b336a5768
rm 4d5eded20c38d5e0189d49f643518a7b336a5768.zip

python input_files.py

# Exports
export picaso_refdata=$(pwd)"/picasofiles/reference/"
export PYSYN_CDBS="NOT_A_PATH_1234567890"
```