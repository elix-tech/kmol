import os
import time
import subprocess
from pathlib import Path
from mila.factories import AbstractScript


class GenerateMsaScript(AbstractScript):
    def __init__(self, path_protein_folder, output_dir_path, data_path, cpus, nb_thread):
        self.path_protein_folder = self.preprocess_path(path_protein_folder)
        self.output_dir_path = self.preprocess_path(output_dir_path)
        self.data_path = self.preprocess_path(data_path)
        self.cpus = cpus
        self.nb_thread = nb_thread

    def preprocess_path(self, path):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        return str(Path(path).absolute())

    def run(self):
        # compute_aligments()
        process = subprocess.Popen(
            [
                "docker",
                "run",
                "--gpus",
                "all",
                "-v",
                f"{self.path_protein_folder}:/inputs",
                "-v",
                f"{self.output_dir_path}:/outputs",
                "-v",
                f"{self.data_path}/:/database",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--rm",
                "-it",
                "-ti",
                "openfold:latest",
                "python3",
                "/opt/openfold/scripts/precompute_alignments.py",
                "/inputs",
                "/outputs",
                "--no_tasks",
                f"{self.nb_thread}",
                "--uniref90_database_path",
                "/database/uniref90/uniref90.fasta",
                "--mgnify_database_path",
                "/database/mgnify/mgy_clusters_2018_12.fa",
                "--pdb70_database_path",
                "/database/pdb70/pdb70",
                "--uniclust30_database_path",
                "/database/uniclust30/uniclust30_2018_08/uniclust30_2018_08",
                "--bfd_database_path",
                "/database/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt",
                "--cpus",
                f"{self.cpus}",
                "--jackhmmer_binary_path",
                "/opt/conda/bin/jackhmmer",
                "--hhblits_binary_path",
                "/opt/conda/bin/hhblits",
                "--hhsearch_binary_path",
                "/opt/conda/bin/hhsearch",
                "--kalign_binary_path",
                "/opt/conda/bin/kalign",
            ]
        )
        while process.poll() is None:
            time.sleep(10)
