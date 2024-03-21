from typing import List

import hoomd
import numpy as np
from PIL import Image
from polychrom_hoomd.utils import get_chrom_bounds
from polykit.analysis.polymer_analyses import Rg2

from . import render

try:
    import wandb
except ImportError:
    pass


class RGWriter(hoomd.custom.Action):
    def __init__(
        self,
        project_name: str = 'hoomd',
        tags: List[str] | None = None,
        *,
        use_wandb: bool = True,
        new_run: bool = True,
    ):
        self.use_wandb = use_wandb
        if use_wandb:
            if wandb.run is None and new_run:
                self.run = wandb.init(project=project_name, tags=tags)
        self.scores = {}

    def act(self, timestep):
        """Write out a new frame to the trajectory."""
        snapshot = self._state.get_snapshot()

        positions = snapshot.particles.position.copy()
        box = np.asarray(snapshot.configuration.box)[None, :3]

        chrom_bounds = get_chrom_bounds(snapshot)
        rg = np.zeros(len(chrom_bounds))

        for i, bounds in enumerate(chrom_bounds):
            chrom_positions = positions[bounds[0]:bounds[1]+1]

            bond_vectors = chrom_positions[1:] - chrom_positions[:-1]
            PBC_shifts = np.round(bond_vectors / box)

            chrom_positions[1:] -= np.cumsum(PBC_shifts, axis=0) * box
            rg[i] = Rg2(chrom_positions)

        rg_mean = np.mean(rg)
        self.scores[timestep] = rg_mean
        if self.use_wandb:
            wandb.log({"timestep": timestep, "Rg": rg_mean})


class SnapshotWriter(hoomd.custom.Action):
    def __init__(
        self,
        project_name: str = 'hoomd',
        tags: List[str] | None = None,
        *,
        use_wandb: bool = True,
        new_run: bool = True,
    ):
        self.use_wandb = use_wandb
        if use_wandb and new_run:
            if wandb.run is None:
                self.run = wandb.init(project=project_name, tags=tags)

    def act(self, timestep):
        """Write out a new frame to the trajectory."""
        if self.use_wandb:
            snapshot = self._state.get_snapshot()
            image = render.fresnel(snapshot, cmap='jet').static(
                pathtrace=False, png_output_file='tmp.png',
            )
            image = Image.open('tmp.png')
            wandb.log({f"snapshot": wandb.Image(image)})
