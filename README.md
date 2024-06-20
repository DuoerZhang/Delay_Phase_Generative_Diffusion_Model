# Delay_Phase_Generative_Diffusion_Model
the Delay Phase Generative Diffusion Model (DPGDM) was produced to synthesize delayed phase images from the arterial phase and portal-venous phase images of contrast-enhanced multiphasic liver CT. 

## Usage
The following command will train the network using the parameters specified in the file:
```
python Main.py
```
You can change the mode by setting `state` to `Train` or `eval`

Training and generated outputs will be located in the model directory as specified in the param file.

## Acknowledgement
The codes use [guided-diffusion]([https://github.com/openai/guided-diffusion]) as code base.

