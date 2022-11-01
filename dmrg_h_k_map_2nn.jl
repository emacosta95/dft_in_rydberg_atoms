# import
using LinearAlgebra
include("src/utils_dmrg.jl")
using ITensors
using Distributions
using Random
using ProgressBars
# fix the number of threads
BLAS.set_num_threads(10)
# parameters
seed=125
linkdims=70
sweep=30
n=16
omega=1.
j_coupling=1.# /(omega^(1. /6.))
hmaxs=LinRange(0,2*omega,20)
#hmaxs=LinRange(0.1,12.,nlinspace) # for studying the phase transition
ndata=1
two_nn=false
omega=0.
#namefile="data/dataset_dmrg/l_{}_h_{}_ndata_{}".format(n,h_max,ndata)
# we need to understand
# how to implement a string
# format

#fix the seed
Random.seed!(seed)

# different sizes
for j=1:20
        #name file
        namefile="data/1d_rydberg/dmrg_h_k_map_rydberg_1nn/h_k_map_check_1nn_$(n)_l_$(hmaxs[j])_h_$(omega)_omega_$(ndata)_n.npz"
        v_tot = zeros(Float64,(ndata,n))
        z_tot= zeros(Float64,(ndata,n))
        zzs=zeros(Float64,(ndata,n,n))
        for i=tqdm(1:ndata)

                # initialize the field
                h=-1.0*hmaxs[j]*ones(n)
                z,zz=dmrg_nn_rydberg_1d(seed,linkdims,sweep,n,j_coupling,hmaxs[j],omega,namefile,h)
                
                # cumulate
                v_tot[i,:]=h
                z_tot[i,:].=z
                zzs[i,:,:].=zz

                #save
                npzwrite(namefile, Dict("potential"=>v_tot,"density"=>z_tot,
                "correlation"=>zzs))
        end
end