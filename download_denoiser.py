import os
import shutil
import subprocess
import sys

# set current folder
cdir = os.path.dirname(os.path.abspath(__file__))

# set path to the licenced script by Kai Zhang et al.
models_folder = os.path.join(cdir,"utils", "clfunc_MPI2Dreco_sm","models")
os.makedirs(models_folder, exist_ok=True)

download_script = os.path.join(models_folder ,"main_download_pretrained_models.py")
args = ["--models", "DPIR IRCNN", "--model_dir", "model_zoo"]

# start the download
cmd = [sys.executable, download_script] + args
print("running:", " ".join(cmd))
result = subprocess.run(cmd)

if result.returncode != 0:
    print("download script failed, stopping here")
    sys.exit(1)

# this is where main_download_pretrained_models.py dumps everything
folder_dump = os.path.join(cdir, "model_zoo")

# this is where we move the model were we need it
print('Moving the denoiser to the destination folder...')
model = os.path.join(folder_dump,'drunet_gray.pth')
shutil.move(model, models_folder)
shutil.rmtree(folder_dump)
print('Done!')
