#!/bin/bash

jupyter nbextension enable vim_binding/vim_binding

jupyter lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password='' --notebook-dir='/home/nbooks'