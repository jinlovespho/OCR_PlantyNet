

# OCR_PLANTYNET


### Code preparation

```shell
# 1. clone repo
git clone https://github.com/jinlovespho/OCR_PlantyNet.git
cd OCR_PlantyNet

# 2. create environment
conda create -n pho_ocr_plantynet python=3.10 -y
conda activate pho_ocr_plantynet

# 3-1. install torch first
pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# 3-2. then install other libraries
pip3 install -r requirements.txt

# 3-3. install detectron2 
cd detectron2 
pip3 install -e .

# 3-4. install testr
cd testr 
pip3 install -e .
```


### Run Validation script 
```shell
cd OCR_PlantyNet
bash run_script/val_script/run_val_diffbir_sam_try1.sh
```
