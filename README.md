```sh
setopt INTERACTIVE_COMMENTS

mamba create -n emission -c conda-forge -c bokeh photochem=0.6.7 numpy=1.24 mpi4py dill tqdm astropy=6.1 matplotlib pandas pip xarray pathos bokeh=2.4.3 wget unzip tar pymultinest=2.12 scipy=1.11
mamba activate emission

# Prereqs for Photochem build
mamba install scikit-build cmake=3 ninja cython fypp pip c-compiler cxx-compiler fortran-compiler git
# build Photochem
wget https://github.com/nicholaswogan/photochem/archive/3fe116524a8abd24d7f5afbdc39e44e32ef99dd2.zip
unzip 3fe116524a8abd24d7f5afbdc39e44e32ef99dd2.zip
cd photochem-3fe116524a8abd24d7f5afbdc39e44e32ef99dd2
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX"
python -m pip install --no-deps --no-build-isolation . -v
cd ..
rm -rf photochem-3fe116524a8abd24d7f5afbdc39e44e32ef99dd2
rm 3fe116524a8abd24d7f5afbdc39e44e32ef99dd2.zip

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

# Get the star stuff
wget http://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz
tar -xvzf synphot3.tar.gz
mv grp picasofiles/
rm synphot3.tar.gz

# Get more star stuff
wget https://archive.stsci.edu/hlsps/reference-atlases/hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar
tar -xvf hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar
mv grp/redcat/trds/grid/phoenix picasofiles/grp/redcat/trds/grid/
rm hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar

# Exports
export PYSYN_CDBS=$(pwd)"picasofiles/grp/redcat/trds"
export picaso_refdata=$(pwd)"/picasofiles/reference/"
```