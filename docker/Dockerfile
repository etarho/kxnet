FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

LABEL maintainer="etarho <etarho.py@gmail.com>"

USER root
ENV HOME /home
ENV WORKSPACE /home/workspace
ENV PATH /opt/conda/bin:$PATH
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN mkdir -p $WORKSPACE

RUN echo "Now building..."

RUN apt-get update \
 && apt-get install -y git build-essential vim

######################
# Package installation (conda)

RUN conda config --add channels pytorch \
 && conda update --all -y -q \
 && conda install -y -q \
    cython \
    ipywidgets \
    jupyterlab \
    matplotlib \
    nodejs==10.13.0 \
    opencv \
    pandas \
    pyyaml \
    scikit-image \
    scikit-learn \
    seaborn \
    scipy \
    tensorboard \
    yaml \
 && conda install -y -q xeus-python -c conda-forge \
 && conda install -y -q pycocotools -c conda-forge \
 && conda clean --all -f -y

# Package installation (pip)
RUN pip install -U pip \
 && pip install \
    bokeh \
    line-profiler \
    mlflow \
    optuna \
    pytorch_memlab \
    torch-summary
# RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
# RUN pip install 'git+https://github.com/facebookresearch/fvcore' \
# RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

######################
# Preferences (Comment the lines out, if not needed)

RUN pip install -q \
    black \
    jupyterlab_code_formatter \
    jupyterlab-git \
    jupyter-lsp \
    jupyter-tensorboard \
    jupytext \
    nbresuse \
    'python-language-server[all]' \
    yapf \
    ipympl


######################
# Jupyter setup

RUN jupyter notebook --generate-config \
 && echo "c.NotebookApp.ip = '*'" >> $HOME/.jupyter/jupyter_notebook_config.py \
 && echo "c.NotebookApp.port = 8888" >> $HOME/.jupyter/jupyter_notebook_config.py \
 && echo "c.NotebookApp.open_browser = False" >> $HOME/.jupyter/jupyter_notebook_config.py \
 && echo "c.NotebookApp.notebook_dir = '$WORKSPACE'" >> $HOME/.jupyter/jupyter_notebook_config.py \
 && echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"' >> $HOME/.jupyter/jupyter_notebook_config.py \
 && echo 'c.ContentsManager.default_jupytext_formats = "ipynb,py"' >> $HOME/.jupyter/jupyter_notebook_config.py


######################
# JupyterLab Extensions

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager \
 && jupyter labextension install @lckr/jupyterlab_variableinspector \
 && jupyter labextension install @krassowski/jupyterlab-lsp \
 && jupyter labextension install @jupyterlab/debugger \
 && jupyter labextension install @telamonian/theme-darcula \
 && jupyter labextension install @jupyterlab/toc \
 && jupyter labextension install jupyterlab_tensorboard \
 && jupyter labextension install jupyterlab-topbar-extension jupyterlab-system-monitor \
 && jupyter labextension install @ryantam626/jupyterlab_code_formatter \
 && jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib@0.7.4 \
 && jupyter labextension install @bokeh/jupyter_bokeh

RUN jupyter serverextension enable --py jupyterlab_code_formatter

RUN jupyter lab build
