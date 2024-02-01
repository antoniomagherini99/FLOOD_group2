This folder contains the convlstm py file obtained from github.
THis specific py file was modified (Antonio and Lucas) to produce predictions after each layer.
A 1x1 kernel is used to reduce the hidden_dim to output_dim.

The multistep_convlstm (Lucas) py file is a copy of convlstm, but uses layers for all time steps.
Residual connections were added.
A 1x1 kernel is added for the final point wise sum.
Different initialization were tested for the 1x1 kernel and the convolution layer.
However, no significant improvements were noticed, but these were kept as it helped train the current best model.

A relu can be placed on the output of both models, but it was noticed that it sometimes made it
harder for the model to train as it could output zeros for all timesteps. 