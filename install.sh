pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html --use-deprecated=legacy-resolver
pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html --use-deprecated=legacy-resolver
pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html --use-deprecated=legacy-resolver
pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html --use-deprecated=legacy-resolver

pip install torch-geometric==1.6.3
pip install -e .
