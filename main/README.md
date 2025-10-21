Cross-Platform Real-Time On-Device Diarization (Voice-to-Text)

# Mac Setup Instructions
## Install pyenv and pyenv-virtualenv
```
brew install pyenv virtualenv pyenv-virtualenv portaudio
```

## Setup pyenv
```
pyenv install 3.13.9

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init - zsh)"' >> ~/.zshrc
```

## Init pyenv-virtualenv
```
eval "$(pyenv virtualenv-init -)"
```

## Create and activate venv
```
pyenv virtualenv 3.13.9 parakeet_env
pyenv activate parakeet_env
```

## Install Parekeet + dependencies
Make sure you are in the root directory of the git project
```
pip install -r requirements.txt
```

## Manually install model
Download models from [huggingface](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/tree/main) and place them into the "models" folder in the root directory
Required files:
- config.json
- decoder_joint-model.onnx
- encoder-model.onnx
- encoder-model.onnx.data
- vocab.txt

Optional INT8 support (required for CoreML, also provides up to 3x faster performance on CPU):
Create an 'INT8' sub directory inside the model folder, and place + rename the following files inside:
- config.json
- decoder_joint-model.int8.onnx (rename to decoder_joint-model.onnx)
- encoder-model.int8.onnx (rename to encoder-model.onnx)
- vocab.txt

If you want the experimental CoreML INT8 support, go to backend/Providers.py and change `allow_coreml=False` to `allow_coreml=True`


## Note
Anytime you want to work on this project, activate the venv with
```
pyenv activate parakeet_env
```
