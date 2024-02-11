FROM doduo1.umcn.nl/uokbaseimage/diag:tf2.8-pt1.10-v1

USER root
RUN pip install numpy
RUN pip install tables
RUN pip install h5py
RUN pip install opencv-python
RUN pip install scikit-image
RUN pip install pandas
RUN pip install sklearn
RUN pip install gorilla
RUN pip install xlsxwriter
RUN pip install openpyxl
RUN pip install nibabel
RUN pip install torchsummary
RUN pip install sewar
RUN pip install matplotlib
RUN pip install Pillow
RUN pip install scipy
RUN pip install tqdm
RUN pip install seaborn
RUN pip install moviepy
RUN pip install git+https://github.com/wbaek/theconf@de32022f8c0651a043dc812d17194cdfd62066e8
RUN pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git@08f7d5e
RUN pip install git+https://github.com/ildoonet/pystopwatch2.git
RUN pip install git+https://github.com/hyperopt/hyperopt.git
RUN pip install tensorboardx
RUN pip install ray
RUN pip install psutil
RUN pip install requests
RUN pip install --upgrade protobuf
RUN pip install -U hyperopt
RUN pip install --upgrade pip
RUN pip install wandb
RUN pip install tqdm
RUN pip install theconf
RUN pip install setproctitle
RUN pip install cython
RUN pip install -U albumentations


USER user
WORKDIR /data/pathology/

ENTRYPOINT /bin/bash projects/autoaugmentation/from_chansey_review/invariant/dockers/execute.sh $0 $@