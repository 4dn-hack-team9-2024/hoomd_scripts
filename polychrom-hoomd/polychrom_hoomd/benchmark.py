import json
import logging
from time import time
from typing import Any, Dict, List, Literal

import hoomd
import numpy as np
import polykit
import polykit.generators.initial_conformations as generator

import polychrom_hoomd.build as build
import polychrom_hoomd.forces as forces
import polychrom_hoomd.log as log
import polychrom_hoomd.render as render

from . import callbacks as CB

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

T_CONV = (1.67377 * 10**(-27) / (1.380649 * 10**(-23) * 300))**0.5


class Simulation:
    """General benchmarking class"""

    def __init__(
        self,
        chromosome_sizes: int | List[int],
        force_dict_path: str,
        *,
        use_wandb: bool = True,
        monomer_ids: Literal["tile"] | np.ndarray | None = None,
        monomer_kwargs: Dict[str, Any] = {},
        dT: float = 70 * T_CONV,
        add_dpd_forces: bool = False,
        add_attraction_forces: bool = False,
        density: float = 0.2,
        thermostat_type: str = 'Langevin',
        thermostat_kwargs: Dict[str, Any] = {},
        integrator_kwargs: Dict[str, Any] = {},
        device_kwargs: Dict[str, Any] = {},
        init: Literal['cubic', 'spiral', 'random_walk'] = 'cubic',
        init_kwargs: Dict[str, Any] = {},
        random_state: int | None = None,
        callbacks: List[str] | None = ['TableFormatter', 'RGWriter'],
        trigger: int = 10000,
    ):
        self.dT = dT
        self.chromosome_sizes = chromosome_sizes
        if isinstance(chromosome_sizes, int):
            self.chromosome_sizes = [chromosome_sizes]
        self.n_monomers = sum(self.chromosome_sizes)
        self.L = self.get_L(self.n_monomers, density)
        self.device = self.get_device(**device_kwargs)
        self.init_coords = self.get_init_coords(init, **init_kwargs)

        with open(force_dict_path, 'r') as f:
            self.force_dict = json.load(f)
        self.force_field = self.get_force_field(
            add_dpd_forces=add_dpd_forces,
            add_attraction_forces=add_attraction_forces,
        )

        self.system = self.create_simulation(
            monomer_ids, random_state, monomer_kwargs)
        self.callbacks = callbacks
        self.trigger = trigger

        self.system.operations.integrator = self.get_integrator(
            thermostat_type,
            thermostat_kwargs,
            **integrator_kwargs,
        )

        self.initial_snapshot = self.system.state.get_snapshot()
        self.add_callbacks(
            self.callbacks,
            self.system,
            trigger=self.trigger,
            use_wandb=use_wandb,
        )

    @staticmethod
    def get_L(n_monomers: int, density: float) -> int:
        """Extimate box length L"""
        return int((n_monomers / density) ** (1/3))

    def get_init_coords(self, init: str = 'cubic', **kwargs):
        """Get starting conformation"""

        # Only update N if not present
        kwargs.setdefault('N', self.n_monomers)

        match init:
            case 'cubic':
                grow = getattr(generator, 'grow_cubic')
                kwargs.setdefault('boxSize', int(self.L - 2))
                coords = grow(**kwargs)
            case 'random_walk':
                grow = getattr(generator, 'create_random_walk')
                kwargs.setdefault('step_size', 1)
                coords = grow(**kwargs)
                coords_pos = coords - coords.min()
                coords = coords_pos * (self.L - 2) / coords_pos.max()
            case 'spiral':
                grow = getattr(generator, 'create_spiral')
                kwargs.setdefault('r1', 10)
                kwargs.setdefault('r2', 13)
                coords = grow(**kwargs)
                coords_pos = coords - coords.min()
                coords = coords_pos * (self.L - 2) / coords_pos.max()
            case _:
                raise ValueError(
                    f"Could not recognize initialization method {init}")
        return coords

    @staticmethod
    def get_device(notice_level: int = 3, **kwargs) -> hoomd.device.Device:
        """Return GPU if there is one."""
        try:
            device = hoomd.device.GPU(notice_level=notice_level, **kwargs)
            logger.info("HOOMD is funning on the following GPU(s)")
            logger.info("\n".join(device.get_available_devices()))
        except RuntimeError:
            device = hoomd.device.CPU(notice_level=notice_level, **kwargs)
            logger.info("HOOMD is funning on the CPU")
        return device

    def get_forces(self, neighbors_list) -> List:
        repulsion_forces = forces.get_repulsion_forces(neighbors_list, **self.force_dict)
        bonded_forces = forces.get_bonded_forces(**self.force_dict)
        angular_forces = forces.get_angular_forces(**self.force_dict)
        return repulsion_forces + bonded_forces + angular_forces

    def get_force_field(
        self,
        *,
        add_dpd_forces: bool = False,
        add_attraction_forces: bool = False,
    ) -> List:
        neighbors_list = hoomd.md.nlist.Cell(buffer=0.4)
        force_field = self.get_forces(neighbors_list)
        if add_dpd_forces:
            dpd_forces = forces.get_dpd_forces(neighbors_list, **self.force_dict)
            force_field += dpd_forces
        if add_attraction_forces:
            attraction_forces = forces.get_attraction_forces(
                neighbors_list, **self.force_dict)
            force_field += attraction_forces
        return force_field

    def create_simulation(
        self,
        monomer_ids,
        random_state: int,
        monomer_kwargs: Dict[str, Any] = {},
    ) -> hoomd.Simulation:
        system = hoomd.Simulation(device=self.device, seed=random_state)
        snapshot = build.get_simulation_box(box_length=self.L)

        if monomer_ids is None:  # only one type
            monomer_type_list = ['A']
        elif isinstance(monomer_ids, str):
            monomer_ids = self.get_monomer_ids(monomer_ids, **monomer_kwargs)
            monomer_type_list = list(map(lambda x: chr(x + ord('A')), np.unique(monomer_ids)))
        else:
            monomer_type_list = list(map(lambda x: chr(x + ord('A')), np.unique(monomer_ids)))

        build.set_chains(
            snapshot,
            self.init_coords,
            self.chromosome_sizes,
            monomer_type_list=monomer_type_list,
        )
        if monomer_ids is not None:
            snapshot.particles.typeid[:] = monomer_ids

        self.monomer_ids = monomer_ids
        system.create_state_from_snapshot(snapshot)
        return system

    def get_monomer_ids(self, monomer_ids: Literal['tile'], **kwargs) -> np.ndarray:
        """Generate monomer IDs using strategy."""
        if monomer_ids == 'tile':
            domain_size = kwargs.get('domain_size', self.n_monomers // 10)
            motif = np.zeros(3 * domain_size, dtype=int)
            motif[domain_size:2 * domain_size] = 1
            monomer_ids = np.tile(
                motif,
                int(np.ceil(self.n_monomers / motif.shape[0]))
            )[:self.n_monomers]
        else:
            raise NotImplementedError(
                f"Monomer assignment strategy `{monomer_ids}` not implemented."
            )
        return monomer_ids

    @staticmethod
    def add_callbacks(callbacks, system,
                      trigger: int = 10000, use_wandb: bool = True) -> None:
        if 'RGWriter' in callbacks:
            rg = getattr(CB, 'RGWriter')('hoomd', use_wandb=use_wandb)
            system.operations.writers.append(
                hoomd.write.CustomWriter(action=rg, trigger=trigger)
            )

        if 'TableFormatter' in callbacks:
            lg = log.get_logger(system)
            formatter = log.table_formatter(lg, period=1e4)
            system.operations.writers.append(formatter)

    def get_integrator(
        self,
        thermostat_type: str,
        thermostat_kwargs: Dict[str, Any],
        **kwargs,
    ) -> hoomd.md.Integrator:
        thermostat_cls = getattr(hoomd.md.methods, thermostat_type)
        thermostat = thermostat_cls(filter=hoomd.filter.All(), **thermostat_kwargs)
        integrator = hoomd.md.Integrator(
            dt=self.dT,
            methods=[thermostat],
            forces=self.force_field,
            **kwargs,
        )
        return integrator

    def run(self, steps: int = 1e5) -> float:
        """Entry point"""
        start = time()
        self.system.run(steps)
        end = time()
        logger.info(f"Finished running in {end - start}s")
        return end - start

    def draw_snapshot(
        self,
        snapshot: hoomd.Snapshot | None = None,
        cmap: str = 'jet',
        pathtrace: bool = False,
        static_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> polykit.renderers.backends.Fresnel:
        if snapshot is None:
            snapshot = self.system.state.get_snapshot()
        return render.fresnel(snapshot, cmap=cmap, **kwargs).static(
            pathtrace=pathtrace, **static_kwargs,
        )
