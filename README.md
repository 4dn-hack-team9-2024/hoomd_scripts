To run simulation bencharking see `polychrom-hoomd/notebooks/benchmarking.ipynb`

  To simulate langevin integration set the parameters of the `Simulation` class to

    add_dpd_forces=False
    add_attraction_forces=False
    thermostat_type='Langevin'
    thermostat_kwargs={'kT': 1.0}

  To simulate DPD set them to

    add_dpd_forces=True
    add_attraction_forces=True
    thermostat_type='NVE'
    thermostat_kwargs={}
