conda create -n weakness python=3.10 && conda activate weakness

# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt

git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .
cd ..

