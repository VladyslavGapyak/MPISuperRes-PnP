# MPISuperRes-PnP

This repository contains the code for the MPISuperRes-PnP algorithm, a super-resolution zero-shot plug-and-play algorithm for Magnetic Particle Imaging (MPI).

The methods corresponding to this repository are described in the associated publication:




# Installation

Follow the steps below to set up the environment, download the datasets, and download the denoiser model. These scripts were tested using Python 3.10.0.

## 1. Set up the environment

This code has been tested on Python 3.10. The requirements can be found in **`requirements.txt`:**
```
numpy==1.26.4
scipy==1.12.0
matplotlib==3.8.3
scikit-learn==1.4.1.post1
h5py==3.10.0
scikit-image==0.22.0
torch>=2.2.1
requests==2.34.2
```
torch: no strict version required, both CPU and CUDA builds work. Tested with CPU version `2.14.0+cpu` and CUDA version `2.1.2+cu118`.


Create and activate a virtual environment, then install the requirements:

```bash
python -m venv mpi_venv
source mpi_venv/bin/activate      # on Windows: mpi_venv\Scripts\Activate

pip install -r requirements.txt
```

## 2. Download the Deep Denoiser Prior

After intalling your environment you can download the deep denoiser prior by running `download_denoiser.py` in the root folder.

## 3. Download a dataset

### Please make sure to have enough storage space on your machine, the datasets are in the order of magnitute of Gigabytes

Download the dataset you want by running `download_datasets.py`. Select which dataset to download by setting the corresponding flag to `True` (and the others to `False`) at the top of the script:

```python
flag_mpimnist = True
flag_openmpi = False
flag_anisotropy = False
```

Available datasets:
- `mpimnist` — Iske, M., Albers, H., Kluth, T., & Knopp, T. (2025). *MPI-MNIST Dataset* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.12799417. License: CC BY 4.0
- `openmpi` — Knopp, T., Szwargulski, P., Griese, F., Gräser, M. (2020). *OpenMPIData: An initiative for freely accessible magnetic particle imaging data*, [Data in Brief], Volume 28, February 2020, 104971. https://doi.org/10.1016/j.dib.2019.104971. License: CC BY 4.0
- `anisotropy` — Knopp, T.,  Scheffler, K. (2024). *MPIData: EquilibriumModelWithAnisotropy* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.10646064. License: CC BY 4.0

Then run:

```bash
python download_datasets.py
```

This creates a `datasets/` folder with the following structure (only the folders for the flags you enabled will actually be downloaded):

```
datasets/
├── mpimnist/
│   ├── SM/
│   │   ├── SM_fluid_equilibrium_coarse.mdf
│   │   └── ...
│   ├── test_gt/
│   │   └── ...
│   ├── test_obsnoisy/
│   │   └── ...
│   └── test_noise/
│       └── ...
├── openmpi/
│   ├── calibrations/
│   │   └── 2.mdf
│   └── measurements/
│       ├── shapePhantom/
│       │   └── 2.mdf
│       ├── resolutionPhantom/
│       │   └── 2.mdf
│       └── concentrationPhantom/
│           └── 2.mdf
└── EMWAdata/
```

## 4. Run an example reconstruction

This repository contains examples of reconstructions for each of the downloadable datasets. 

### Example on the 'Equilibrium data with Anisotropy' Dataset

To run a reconstruction on the MPI-MNIST dataset, go to `[./reconstructions/data_EMWA/reconstruction.py]`. You can choose 

```bash
phantom_name = 'spiral'  
upscale_const = 5        
mode = 'bilinear'
```

where the variable `mode` corresponds to the interpolation scheme (`bilinear` and `repeat` are available), `upscale_const` is the upscaling factore for the super-resolution (e.g., 2,3,4,...). You can choose with phantom to reconstruct by setting `phantom_name` to be one of the following: `spiral`, `resolution1`, `resolution2`, `resolution3`, `icecream`, `dot`.

The reconstruction is stored in `[./reconstructions/data_EMWA/reconstruction_results]`.

### Example on the 'MPI-MNIST' Dataset

To run a reconstruction on the MPI-MNIST dataset, go to `[./reconstructions/data_mpimnist/reconstruction.py]`. You can choose 

```bash
sm_type = 'coarse'
mode = 'bilinear'
multi = 2
```

where the variable `mode` corresponds to the interpolation scheme (`bilinear` and `repeat` are available), `multi` is the upscaling factore for the super-resolution (e.g., 1,2,3,4,...) and `sm_type` corresponds to the choice of system matrices on the following grid sizes

```bash
{   'coarse' : "15 x 17",
    'int'    : "45 x 51",
    'fine'   : "75 x 85",
    }
```
The variable `exp_num = 0` can be changed to select which phantom in the in the dataset will be reconstructed. The reconstruction is stored in `[./reconstructions/data_mpimnist/reconstruction_results]`.

### Example on the '2D OpenMPI' Dataset

To run a reconstruction on the 2d OpenMPI dataset, go to `[./reconstructions/data_openmpi/reconstruction.py]`. You can choose 

```bash
phantom_name = 'shape'
upscale_const = 3
mode = 'bilinear'
```

where the variable `mode` corresponds to the interpolation scheme (`bilinear` and `repeat` are available), `upscale_const` is the upscaling factore for the super-resolution (e.g., 2,3,4,...). You can choose with phantom to reconstruct by setting `phantom_name` to be one of the following: `shape`, `resolution`, `concentration`.

The reconstruction is stored in `[./reconstructions/data_openmpi/reconstruction_results]`.


## 5. Reference to the denoiser model

In this code we use the *deep denoiser prior* by Zhang et al., (2021). The weights of the denoiser and reused code is stored in:

```
src/cldef_sm/models
```

We appropriately provide the reference to the *deep denoiser prior* by Zhang et al.:
```
@article{zhang2021plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={10},
  pages={6360-6376},
  year={2021}
}
 @inproceedings{zhang2017learning,
   title={Learning Deep CNN Denoiser Prior for Image Restoration},
   author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
   pages={3929--3938},
   year={2017},
 }
 ```

This part of the code is licensed under MIT (Copyright (c) 2020 Kai Zhang). The original license is included at `.\utils\clfunc_MPI2Dreco_sm\models\LICENSE.txt`.

## Citation
If you use this software in your research, please cite:

> Gapyak et al. (2026). MPISuperRes-PnP: A Super-Resolution Zero-Shot Plug-and-Play Reconstruction Algorithm for Magnetic Particle Imaging. *Physics in Medicine & Biology*. 

If citing the software itself, please use:

> VladyslavGapyak (2026) “VladyslavGapyak/MPISuperRes-PnP: MPISuperRes-PnP”. Zenodo. Available at: https://doi.org/10.5281/zenodo.22670938.


## License
This project is licensed under the [GNU GPL v3](LICENSE) — see the `LICENSE` file for details. 

Note: the `[.\utils\clfunc_MPI2Dreco_sm\models\]` subdirectory contains third-party code licensed separately under MIT — see the notes above and its own `LICENSE` file.

## References

If you use the datasets downloaded by this repository, please also cite the corresponding original sources:

- **MPI-MNIST**: Iske, M., Albers, H., Kluth, T., & Knopp, T. (2025). *MPI-MNIST Dataset* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.12799417
- **OpenMPI**: Knopp, T., Szwargulski, P., Griese, F., Gräser, M. (2020). *OpenMPIData: An initiative for freely accessible magnetic particle imaging data*, [Data in Brief], Volume 28, February 2020, 104971. https://doi.org/10.1016/j.dib.2019.104971.
- **Anisotropy**: Knopp, T.,  Scheffler, K. (2024). *MPIData: EquilibriumModelWithAnisotropy* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.10646064
Data: EquilibriumModelWithAnisotropy* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.10646064
