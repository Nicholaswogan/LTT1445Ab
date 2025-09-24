from photochem.extensions import hotrocks

def main():
    hotrocks.download_sphinx_spectra(filename='data/sphinx.h5')

if __name__ == '__main__':
    main()