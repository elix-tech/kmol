from typing import List


class AutoIncrement:

    def __init__(self, padding: int = 4):
        self._id = 0
        self._padding = padding

    def __call__(self) -> str:
        self._id += 1
        return str(self._id).zfill(self._padding)

    def __str__(self) -> str:
        return self()


REFERENCE = "data/configs/reference.json"
OUTPUT_PATH = "data/configs/gcn/"
REPLACEMENTS = {
    "{{{layer_type}}}": [
        "ARMAConv", "ChebConv", "ClusterGCNConv", "FeaStConv", "GATConv", "GENConv",
        "GraphConv", "HypergraphConv", "LEConv", "SAGEConv", "SGConv", "TAGConv"
    ],
    "{{{layers_count}}}": [2, 3, 5],
    "{{{output_path}}}": [AutoIncrement()]
}


def compile_(sample: str, targets: List[str], auto_increment: AutoIncrement) -> None:
    if targets:
        target = targets[0]
        targets = targets[1:]

        for option in REPLACEMENTS[target]:
            compile_(sample.replace(target, str(option)), targets, auto_increment)
    else:
        with open("{}/{}.json".format(OUTPUT_PATH, auto_increment()), "w") as write_buffer:
            write_buffer.write(sample)


with open(REFERENCE) as read_handle:
    template = read_handle.read()
    compile_(sample=template, targets=list(REPLACEMENTS.keys()), auto_increment=AutoIncrement())
