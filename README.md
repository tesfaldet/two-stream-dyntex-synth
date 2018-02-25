# Two-Stream Convolutional Networks for Dynamic Texture Synthesis

This readme is a rough draft. Example usage will be given soon. For now please read `synthesize.py` to understand basic usage.

Required:
- Tensorflow 1.3 (or latest, although not tested)
- Preferably a Titan X for synthesizing 12 frames
- Appearance-stream [tfmodel](https://drive.google.com/open?id=19KkFi92oWLzuOWnGo6Zsqe-2CCXFAoXZ)
- Dynamics-stream [tfmodel](https://drive.google.com/open?id=1DHnzoNO-iTgMUTbUOLrigEmpPHmn_mT1)
- [Dynamic textures](https://drive.google.com/open?id=0B5T9jWfa9iDySWJHZnpNZ2dHWUk)
- [Static textures](https://drive.google.com/open?id=11yMiPXiuYvLCyoLfQf_dEG6kuav8h6_3) (for dynamics style transfer)

Store the appearance-stream tfmodel in `/models`.

Store the dynamics-stream tfmodel in `/models`. The filepath to this model is your `--dynamics_model` path.

Store your chosen dynamic texture image sequence in a folder in `/data/dynamic_textures`. This folder is your `--dynamics_target` path.

Store your chosen static texture in `/data/textures`. The filepath to this texture is your `--appearance_target` path. This is only used for dynamics style transfer, static texture synthesis, and incremental dynamic texture synthesis.

The network's output will be saved at `data/out/<your_runid>`.

Use `/useful_scripts/makegif.sh` to create a gif from a folder of images, e.g., `./useful_scripts/makegif.sh "data/out/calm_water/iter_6000*" calm_water.gif` will create the gif `calm_water.gif` from the images `iter_6000*`.

Logs and snapshots are created and stored in `/logs` and `/snapshots`, respectively. You can view the loss progress in Tensorboard.
