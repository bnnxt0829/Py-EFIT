# ! /usr/bin/python
# coding=utf-8

import sys
import numpy as np
from eq_calculation import *
import eq_calculation
import scipy.io as scio
import h5py
from matplotlib import pyplot as plt

def initial_fuc(machine, shot, time):
    cur = eq_calculation.cur()
    mesh_const = eq_calculation.mesh_const()

    print("Loading data...\n")

    print("Loading Green's functions...\n")
    global green_grid2diag
    global green_pf2diag
    global green_grid2grid
    global green_pf2grid
    green_file = '../green_tables/green_'+machine+'.mat'
    green_in=h5py.File(green_file,'r')
    green_grid2diag=np.array(green_in['grid2mag'],dtype='float64').T
    green_pf2diag=np.array(green_in['pf2mag'],dtype='float64').T
    green_grid2grid=np.array(green_in['grid2grid'],dtype='float64').T
    green_pf2grid=np.array(green_in['pf2grid'],dtype='float64').T

    print("Loading magnetic diagnostic data...\n")
    global Diag
    Diag_file = 'Diag_'+shot+'_t'+time+'.mat'
    Diag=scio.loadmat('../Diag/'+Diag_file)

    print("Loading initial magnetic flux...\n")
    global init_flux
    init_flux_file='../init/init_flux.mat'
    init_flux_in=scio.loadmat(init_flux_file)
    init_flux=init_flux_in['init_flux_out']

    print("Loading limiter position...\n")
    global limiter_rz
    datafile = '../limiter/limiter_2021.dat'
    limiter_rz = read_limiter(datafile)

def gs_calculation():
    cur=eq_calculation.cur()
    mesh_const=eq_calculation.mesh_const()
    print("Equilibrium reconstruction computation starts...\n")
    num_iteration = 15
    flux_grid = np.zeros((129, 129, num_iteration+1))
    convergence_error = np.zeros(num_iteration)
    itera = np.zeros(num_iteration)
    chi2 = np.zeros((74, num_iteration))
    chi2_temp = np.zeros((74,1))
    chi2_all0 = np.zeros(num_iteration)
    Mag_cal=np.zeros((74,1))

    ip_repr_0 = ip_representation_0
    resp_matrix_0 = mag_response_matrix_0
    resp_matri_fit_0 = response_matrix_fitting_0
    flxmp_cal_0 = fluxmap_cal_0
    psi_xp_axis = psi_xp_axis_0
    ip_repre_it_0 = ip_representation_iteration_0
    resp_matri_0 = response_matrix_0
    resp_matri_fit_0 = mag_response_matrix_fitting_0
    flux_cal_0 = fluxmap_cal_0
    drsep_cal = drsep_cal_0
    equili_info_0 = mag_equilibrium_info_0

    print("Setting initial condition for iteration...\n")

    ip_representation = ip_repr_0(mesh_const.mesh,init_flux,mesh_const.rgrid,mesh_const.zgrid,cur.kppcur,cur.kffcur,cur.ppbry,cur.ffbry,cur.dzfit)

    [response,response_fwt,mag_right,uncertanty] = resp_matrix_0(ip_representation, green_grid2diag, green_pf2diag,cur.kppcur, cur.kffcur, cur.dzfit, cur.ppbry, cur.ffbry, mesh_const.nf_diag, mesh_const.nf_coils, cur.capa_p, cur.capa_f,Diag)

    [A, B, X, SV, pf_fit] = resp_matri_fit_0(response_fwt,mag_right,Diag,cur,0)

    [ip_grid, flux_grid_2d, flux_grid_pf, flux_grid_ip] = flxmp_cal_0(mesh_const.mesh, ip_representation, X, green_grid2grid, green_pf2grid)

    [si_bdry, si_axis, xpt] = psi_xp_axis(mesh_const.mesh, flux_grid_2d.T, mesh_const.rgrid, mesh_const.zgrid, limiter_rz)

    flux_grid[:, :, 0] = flux_grid_2d

    iteration = 0
    tol = 1e-3

    print("Iteration starts...\n")

    while 1:
        print('Iteration ',iteration+1,'...\n')
           
        temp_flux_grid = flux_grid[:, :, iteration]

        ip_representation = ip_repre_it_0(mesh_const.mesh, temp_flux_grid, mesh_const.rgrid, mesh_const.zgrid, cur.kppcur, cur.kffcur, cur.ppbry, cur.ffbry, cur.dzfit, si_bdry, si_axis, xpt)

        [response,response_fwt,mag_right,uncertanty] = resp_matrix_0(ip_representation, green_grid2diag, green_pf2diag,cur.kppcur, cur.kffcur, cur.dzfit, cur.ppbry, cur.ffbry, mesh_const.nf_diag, mesh_const.nf_coils, cur.capa_p, cur.capa_f,Diag)

        [A, B, X, SV, pf_fit] = resp_matri_fit_0(response_fwt, mag_right, Diag, cur, 1)

        [ip_grid, flux_grid[:, :, iteration+1], flux_grid_pf, flux_grid_ip] = flxmp_cal_0(mesh_const.mesh, ip_representation, X, green_grid2grid, green_pf2grid)

        [si_bdry, si_axis, xpt] = psi_xp_axis(mesh_const.mesh, flux_grid[:, :, iteration+1].T, mesh_const.rgrid, mesh_const.zgrid, limiter_rz)

        [convergence_error[iteration], chi2_temp, chi2_all0[iteration], Mag_cal] = equili_info_0(si_bdry, si_axis, flux_grid[:, :, iteration], flux_grid[:, :, iteration+1], X, response, Diag, uncertanty)
        chi2[:,iteration]=chi2_temp[:,0]

        itera[iteration]=iteration+1
        print('Iteration error =',convergence_error[iteration],'\n')
        if (convergence_error[iteration]<tol):
            break;
        else:
            iteration = iteration + 1

    plt.figure()
    plt.plot(itera[0:iteration+1],convergence_error[0:iteration+1],'bo-',markersize=8,linewidth=2)
    ax=plt.gca()
    ax.set_xticks([1,2,3,4,5])
    plt.xlabel('iteration step',fontsize=18)
    plt.ylabel('error',fontsize=18)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.savefig('../output/'+'error_'+shot+'_t'+time+'.png')
    plt.show()

    print('Iteration ends...\n')
    print('Searching separatrix...\n')

    [bdry_r,bdry_z]=search_bdry(mesh_const.mesh,flux_grid[:,:,iteration+1],mesh_const.rgrid,mesh_const.zgrid,xpt,si_bdry)

    psi_matlab=scio.loadmat('../efit/'+'psirz_'+shot+'_t'+time+'.mat')
    psi_plot(flux_grid[:,:,iteration+1],mesh_const.rgrid,mesh_const.zgrid,psi_matlab,bdry_r,bdry_z,limiter_rz,shot,time)

    return(1)

def read_limiter(datafile):
    with open(datafile,'r') as f:
        tmp=f.readlines()
    for i in range(len(tmp)):
        tmp[i]=tmp[i].rstrip('\n')
    N=int(tmp[0])
    limiter_rz=np.zeros((N,2))
    j=0
    for i in range(N):
        limiter_rz[i,0]=float(tmp[j*2+1])
        limiter_rz[i,1]=float(tmp[j*2+2])
        j=j+1
    return limiter_rz

if __name__ == "__main__":
    print("Input the tokamak name:\n")
    machine = input()
    shot = 0
    print("Input the shot number:\n")
    shot = input()
    print("Input the time:\n")
    time = input()
    if shot :
        initial_fuc(machine, shot, time)
        gs_calculation()
    else :
        print('end\n')
    print('Complete!\n')
