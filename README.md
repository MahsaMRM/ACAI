# ACAI
a repository for adding the symbols and fonts work with acai source code

<h4>1) set up the virtualenv:</h4>

<p>Running on GPU:

sudo apt install virtualenv
cd <path_to_code>
virtualenv --system-site-packages env2
. env2/bin/activate
pip install -r requirements.txt

For running on CPU on local machine and testing:

Change tensorflow-gpu==1.8.0 in requirements.txt to tensorflow</p>

<h4>2) make your data sets using prepare_data</h4>

<h4>3) Set your data directory in lib/data.py:</h4>

<p>DATA_DIR ='./data' (line 36)</p>

<h4>4) In lib/data.py, add your images classes and datasets names to _NCLASS and _DATASETS dictionaries (line 232 and 242)</h4>

<h4>5) In runs/acai.sh you can add your bash run command:</h4>

<p>acai.py --dataset=fire_leaf --latent_width=4 --depth=4 --latent=2 --train_dir=TRAIN</p>


<h4>6) Training:</h4>
<p>
  On CPU:
  !python acai.py \
   --train_dir=TEMP \
   --latent=16 --latent_width=2 --depth=16 --dataset=fire_leaf
   
   on GPU: 
   CUDA_VISIBLE_DEVICES=0 python acai.py \</p>
