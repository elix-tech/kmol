

# Get the user id and group, and use it so that created files are not owned by root
HOST_UID = $(shell id -u)
HOST_GID = $(shell id -g)
HOST_USER = $(USER)
export HOST_UID
export HOST_GID
export HOST_USER

bash-alphafold:
	docker run --gpus all \
	-v /nasa/datasets/kyodai_alphafold/2022_10:/dataset \
	--user $(HOST_UID):$(HOST_GID) \
	--rm -it alphafold bash

bash-openfold:
	docker run \
	--gpus all \
	-v $(shell pwd)/:/data \
	-v /nasa/datasets/kyodai_federated/proj_202208_202210/activity/fasta_ready:/inputs \
	-v /nasa/datasets/kyodai_federated/proj_202208_202210/activity/msa_ready:/precomputed_alignments \
	-v /home/vincent/kmol-internal/data/debug:/outputs \
	-v /nasa/datasets/kyodai_alphafold:/database \
	--user $(HOST_UID):$(HOST_GID) \
	--rm -it \
	-ti openfold:latest \
	bash

wheel:
	python setup.py bdist_wheel
# python3 /opt/openfold/run_pretrained_openfold.py \
# /data/fasta_dir \
# /database/pdb_mmcif/mmcif_files/ \
# --uniref90_database_path /database/uniref90/uniref90.fasta \
# --mgnify_database_path /database/mgnify/mgy_clusters_2018_12.fa \
# --pdb70_database_path /database/pdb70/pdb70 \
# --uniclust30_database_path /database/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
# --output_dir /data \
# --bfd_database_path /database/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
# --model_device cuda:0 \
# --jackhmmer_binary_path /opt/conda/bin/jackhmmer \
# --hhblits_binary_path /opt/conda/bin/hhblits \
# --hhsearch_binary_path /opt/conda/bin/hhsearch \
# --kalign_binary_path /opt/conda/bin/kalign \
# --openfold_checkpoint_path /database/openfold_params/finetuning_ptm_2.pt