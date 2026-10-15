#!/usr/bin/env python3
"""Compare ordinary and CPU fits on a generated alloy, without private data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source-offsets', action='store_true',
                        help='Duplicate each geometry across three shifted energy sources')
    args = parser.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import numpy as np
    import pandas as pd
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from pyace.basis import BBasisConfiguration
    from pyace.multispecies_basisextension import create_multispecies_basis_config
    from pyace_cpu.runtime import AVX2, AVX512, select_isa
    from ruamel.yaml import YAML

    # Selection tests do not pretend to execute AVX-512 on unsupported hardware.
    assert select_isa('auto', flags=AVX2) == 'avx2'
    assert select_isa('auto', flags=AVX512) == 'avx512'
    try:
        select_isa('avx512', flags=AVX2)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Unsupported forced ISA was accepted')

    rng = np.random.default_rng(318)
    model = create_multispecies_basis_config(dict(
        elements=['Al', 'Cu', 'Ni'], deltaSplineBins=.001,
        embeddings={'ALL': dict(npot='FinnisSinclairShiftedScaled',
                                fs_parameters=[1, 1, 1, .5], ndensity=2)},
        bonds={'ALL': dict(radbase='SBessel', radparameters=[5.25], rcut=4.0,
                           dcut=.1, NameOfCutoffFunction='cos')},
        functions={'ALL': dict(nradmax_by_orders=[3, 2, 1], lmax_by_orders=[0, 2, 1])}))
    # Retain initial radial mixing, seed only the density expansion coefficients.
    model.set_func_coeffs(rng.normal(scale=.02, size=len(model.get_func_coeffs())))
    model.save(str(root/'initial.yaml'))
    records = []
    for i in range(8):
        atoms = bulk('Cu', 'fcc', a=3.65, cubic=True).repeat((2, 1, 1))
        atoms.set_chemical_symbols(['Cu', 'Al', 'Ni', 'Cu', 'Ni', 'Al', 'Cu', 'Ni'])
        if args.source_offsets and i % 2:
            atoms = atoms.repeat((2, 1, 1))
        atoms.set_cell(atoms.cell*(.97+.01*i), scale_atoms=True)
        atoms.positions += rng.normal(scale=.03, size=atoms.positions.shape)
        atoms.calc = EMT()
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        atoms.calc = None
        records.append(dict(ase_atoms=atoms, energy=energy, energy_corrected=energy,
                            forces=forces, w_energy=1/8,
                            w_forces=np.full((len(atoms), 1), 1/64)))
    if args.source_offsets:
        originals = records
        records = []
        for source, shift in ((0, 0.), (1, 100.), (2, -37.5)):
            for index, record in enumerate(originals):
                shifted = dict(record, ase_atoms=record['ase_atoms'].copy(), source_id=source)
                shifted['energy'] += shift*len(shifted['ase_atoms'])
                shifted['energy_corrected'] = shifted['energy']
                shifted['w_energy'] = (index+1)/108
                shifted['w_forces'] = record['w_forces']/3
                records.append(shifted)
    pd.DataFrame(records).to_pickle(root/'data.pckl.gzip', compression='gzip', protocol=4)
    settings = dict(seed=42, potential=str(root/'initial.yaml'),
                    data=dict(filename=str(root/'data.pckl.gzip')),
                    fit=dict(loss=dict(kappa=.5, L1_coeffs=1e-8, L2_coeffs=1e-8),
                             optimizer='BFGS', maxiter=2),
                    backend=dict(evaluator='tensorpot', batch_size=2, display_step=1))
    if args.source_offsets:
        # Explicit common energy zero keeps automatic reference selection from
        # transforming the synthetic source shifts during data preparation.
        settings['data']['reference_energy'] = {'Al': 0., 'Cu': 0., 'Ni': 0.}
        settings['fit']['source_energy_offsets'] = dict(column='source_id', reference=0)
    metrics = []
    coefficients = []
    for command in ('pacemaker', 'pacemaker-cpu'):
        directory = root/command
        directory.mkdir()
        with (directory/'input.yaml').open('w') as stream:
            YAML().dump(settings, stream)
        env = os.environ.copy()
        env.update(PACEMAKER_CPU_THREADS='2', PACEMAKER_CPU_WORKERS='2',
                   WBC_CPU_THREADS='2', WBC_CPU_WORKERS='2')
        with (directory/'console.log').open('w') as stream:
            result = subprocess.run([str(Path(sys.executable).with_name(command))],
                                    cwd=directory, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=300)
        if result.returncode:
            raise RuntimeError(f'{command} failed, see {directory / "console.log"}')
        metrics.append(pd.read_csv(directory/'metrics.txt', sep=r'\s+'))
        coefficients.append(np.asarray(BBasisConfiguration(
            str(directory/'output_potential.yaml')).get_all_coeffs()))
    columns = ['loss', 'rmse_epa', 'rmse_f_comp']
    np.testing.assert_allclose(metrics[1][columns], metrics[0][columns], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(coefficients[1], coefficients[0], rtol=1e-7, atol=1e-9)
    report = dict(status='passed', isa=select_isa(), frames=len(records),
                  atoms=sum(len(row['ase_atoms']) for row in records), workers=2,
                  fitting_steps=2,
                  max_coefficient_difference=float(np.max(np.abs(coefficients[1]-coefficients[0]))),
                  max_metric_difference=float(np.max(np.abs(
                      metrics[1][columns].values-metrics[0][columns].values))))
    if args.source_offsets:
        predictions = [pd.read_pickle(root/command/'train_pred.pckl.gzip', compression='gzip')
                       for command in ('pacemaker', 'pacemaker-cpu')]
        offsets = []
        for prediction in predictions:
            offsets.append({int(source): float(group['energy_source_offset_fit_per_atom'].iloc[0])
                            for source, group in prediction.groupby('source_id')})
        np.testing.assert_allclose(list(offsets[0].values()), list(offsets[1].values()), atol=1e-9)
        np.testing.assert_allclose(offsets[1][0], 0., atol=1e-12)
        np.testing.assert_allclose(offsets[1][1]-offsets[1][2], 137.5, atol=1e-9)
        report['source_energy_offsets_eV_per_atom'] = offsets[1]
        report['known_relative_offset_eV_per_atom'] = 137.5
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
