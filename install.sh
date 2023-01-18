pip install -r requirements.txt

# ln -s "$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib/libnvrtc-builtins.so" "$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib/libnvrtc-builtins.so.11.1"

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'unset LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

echo "Install finish please deactivate and reactivate your conda environment to enable the env variables."