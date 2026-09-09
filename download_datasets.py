import os
import tarfile
import time
import requests
from pathlib import Path

cdir = os.path.dirname(__file__)

# flags for which datasets to pull
flag_mpimnist = False
flag_openmpi = False
flag_anisotropy = True

url_mpimnist = "https://zenodo.org/records/12799417"
url_openmpi = "https://media.tuhh.de/ibi/openMPIData/data/"
url_anisotropy = "https://zenodo.org/records/10646064/files/MDFStore.tar.gz?download=1"


def download_and_extract(url, dest_folder, max_retries=5, chunk_size=1024 * 1024):
    filename = url.split("/")[-1].split("?")[0]
    dest_path = Path(dest_folder) / filename

    # resume support: if a partial file exists, continue from where it left off
    resume_byte_pos = dest_path.stat().st_size if dest_path.exists() else 0

    for attempt in range(1, max_retries + 1):
        try:
            headers = {"Range": f"bytes={resume_byte_pos}-"} if resume_byte_pos else {}
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:
                    # already fully downloaded
                    break
                r.raise_for_status()

                mode = "ab" if resume_byte_pos else "wb"
                with open(dest_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            print(f"downloaded to {dest_path}")
            break
        except (requests.exceptions.RequestException,) as e:
            print(f"attempt {attempt}/{max_retries} failed: {e}")
            resume_byte_pos = dest_path.stat().st_size if dest_path.exists() else 0
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)

    if dest_path.suffixes[-2:] != ['.tar', '.gz']:
        print("not a tar.gz, skipping extraction")
        return

    print(f"extracting {filename}...")
    with tarfile.open(dest_path, "r:gz") as tar:
        tar.extractall(path=dest_folder)
    dest_path.unlink()
    print(f"extracted and removed archive: {dest_path}")

if __name__ == "__main__":


    data_root = os.path.join(cdir, "datasets")
    os.makedirs(data_root, exist_ok=True)

    if flag_mpimnist:
        '''
        This will create:

        datasets/
        └── mpimnist/
            ├── SM/
            |   ├── SM_fluid_equilibrium_coarse.mdf
            |   └── ...
            ├── test_gt/
            |   └── ...
            ├── test_obsnoisy/
            |   └── ...
            └── test_noise/
                └── ...
        '''
        mpimnist_root = os.path.join(data_root, 'mpimnist')
        os.makedirs(mpimnist_root, exist_ok=True)

        url_dataset = lambda name: f"https://zenodo.org/records/12799417/files/{name}.tar.gz?download=1"

        # modify this list to access other files (e.g. training sets) from the MPI-MNIST dataset
        file_names = ['SM', 'test_gt', 'test_obsnoisy', 'test_noise']  
        for name in file_names:
            download_and_extract(url_dataset(name), mpimnist_root)

    if flag_openmpi:
        '''
        This will create:

        datasets/
        ├── mpimnist/
        |   ├── SM/
        |   ├── test_gt/
        |   ├── test_obsnoisy/
        |   └── test_noise/
        └──  openmpi/
            ├── calibrations/
            |   └── 2.mdf
            └── measurements/
                ├── shapePhantom/
                |   └── 2.mdf
                ├── resolutionPhantom/
                |   └── 2.mdf
                └── concentrationPhantom/
                    └── 2.mdf
        '''
        openmpi_root = os.path.join(data_root, 'openmpi')
        os.makedirs(openmpi_root, exist_ok=True)

        path2calibration = os.path.join(openmpi_root, 'calibration')
        os.makedirs(path2calibration, exist_ok=True)
        download_and_extract(url_openmpi + 'calibrations/2.mdf', path2calibration)

        for name in ['shape', 'resolution', 'concentration']:
            path2measurement = os.path.join(openmpi_root, 'measurement', name + 'Phantom')
            os.makedirs(path2measurement, exist_ok=True)
            download_and_extract(url_openmpi + f'measurements/{name}Phantom/2.mdf', path2measurement)

    if flag_anisotropy:
        '''
        This will create:

        datasets/
        ├── mpimnist/
        |   ├── SM/
        |   ├── test_gt/
        |   ├── test_obsnoisy/
        |   └── test_noise/
        ├── openmpi/
        |   ├── calibrations/
        |   └── measurements/
        |       ├── shapePhantom/
        |       ├── resolutionPhantom/
        |       └── concentrationPhantom/
        └── EMWAdata/

        '''
        anisotropy_root = os.path.join(data_root, 'EMWAdata')
        os.makedirs(anisotropy_root, exist_ok=True)
        download_and_extract(url_anisotropy, anisotropy_root)

    print("done.")