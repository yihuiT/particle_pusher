import multiprocessing
import h5py

import constants as cst
from particle import push_particle

if __name__ == "__main__":
    electron = {
        "position": [0, -4 * cst.R_e, 0],
        "charge": cst.q_el,
        "mass": cst.m_el,
        "kinetic_energy": 50e6,
        "theta_deg": 45,
    }

    proton = {
        "position": [0, -1.5 * cst.R_e, 0],
        "charge": cst.q_pr,
        "mass": cst.m_pr,
        "kinetic_energy": 100e6,
        "theta_deg": 45,
    }






    # with multiprocessing.Pool(processes=5) as pool:
    #     results_pr = pool.starmap(push_particle, [
    #         (30, proton, 'boris', 'ana'),
    #         (30, proton, 'hc', 'ana'),
    #         (30, proton, 'rk4', 'ana'),
    #         (30, proton, 'rk8', 'ana'),
    #         (30, proton, 'gca', 'ana')
    #     ])

    # with h5py.File('pr_pusher.h5', 'w') as hf:
    #     for i, method in enumerate(['boris', 'hc', 'rk4', 'rk8', 'gca']):
    #         group = hf.create_group(method)
    #         group.create_dataset('time', data=results_pr[i][0])
    #         group.create_dataset('traj', data=results_pr[i][1])
    #         group.create_dataset('gamma', data=results_pr[i][2])
    #         group.create_dataset('mu', data=results_pr[i][3])
    #         group.create_dataset('p_para', data=results_pr[i][4])







    # with multiprocessing.Pool(processes=5) as pool:
    #     results_el = pool.starmap(push_particle, [
    #         (30, electron, 'boris', 'ana'),
    #         (30, electron, 'hc', 'ana'),
    #         (30, electron, 'rk4', 'ana'),
    #         (30, electron, 'rk8', 'ana'),
    #         (30, electron, 'gca', 'ana')
    #     ])

    # with h5py.File('el_pusher.h5', 'w') as hf:
    #     for i, method in enumerate(['boris', 'hc', 'rk4', 'rk8', 'gca']):
    #         group = hf.create_group(method)
    #         group.create_dataset('time', data=results_el[i][0])
    #         group.create_dataset('traj', data=results_el[i][1])
    #         group.create_dataset('gamma', data=results_el[i][2])
    #         group.create_dataset('mu', data=results_el[i][3])
    #         group.create_dataset('p_para', data=results_el[i][4])






    with multiprocessing.Pool(processes=4) as pool:
        results_pr = pool.starmap(push_particle, [
            (30, electron, 'boris', 'ana'),
            (30, electron, 'boris', 'tri', 0.1, -10, 10),
            (30, electron, 'boris', 'tsc', 0.1, -10, 10),
            (30, electron, 'boris', 'bsp', 0.1, -10, 10)
        ])

    with h5py.File('el_grid.h5', 'w') as hf:
        for i, method in enumerate(['ana', 'tri', 'tsc', 'bsp']):
            group = hf.create_group(method)
            group.create_dataset('time', data=results_pr[i][0])
            group.create_dataset('traj', data=results_pr[i][1])
            group.create_dataset('gamma', data=results_pr[i][2])
            group.create_dataset('mu', data=results_pr[i][3])
            group.create_dataset('p_para', data=results_pr[i][4])