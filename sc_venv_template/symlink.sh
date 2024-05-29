#!/bin/bash

export PROJECT=/p/project/training2411

# setup symlink for courses
ln -s $PROJECT ~/course

# create cache directory
mkdir ~/course/$USER/.cache

# symlink for cache
ln -s ~/course/$USER/.cache $HOME/

# create config directory
mkdir ~/course/$USER/.config

# symlink for config directory
ln -s ~/course/$USER/.config $HOME/
