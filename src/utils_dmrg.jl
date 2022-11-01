using ITensors
using Distributions
using LinearAlgebra
using Random
using Plots
using NPZ
using ProgressBars

function dmrg_nn_rydberg_1d(seed::Int64,linkdims::Int64,sweep::Int64,n::Int64,j::Float64,h_max::Float64,omega::Float64,namefile::String,hs::Array{Float64})

    #fix the seed
    #Random.seed!(seed)

    #fix the representation
    sites=siteinds("S=1/2",n) #fix the basis representation
    #define the universal part of the hamiltonian
    ham_0=OpSum()
    
    for i=1:n-1
        ham_0+=4*j,"Sz",i,"Sz",i+1 #1 nearest neighbours
    end
    ham_0+=4*j,"Sz",n,"Sz",1 #1 nearest neighbours

    for i=1:n-2
        ham_0+=4*j*((1/2)^6),"Sz",i,"Sz",i+2 #2 nearest neighbours
    end
    ham_0+=4*j*((1/2)^6),"Sz",n-1,"Sz",1 #2 nearest neighbours
    ham_0+=4*j*((1/2)^6),"Sz",n,"Sz",2 #2 nearest neighbours
    for i=1:n-3
        ham_0+=4*j*((1/3)^6),"Sz",i,"Sz",i+3 #3 nearest neighbours
    end
    ham_0+=4*j*((1/3)^6),"Sz",n-2,"Sz",1 #2 nearest neighbours
    ham_0+=4*j*((1/3)^6),"Sz",n-1,"Sz",2 #2 nearest neighbours
    ham_0+=4*j*((1/3)^6),"Sz",n,"Sz",3 #2 nearest neighbours
    for i=1:n-4
        ham_0+=4*j*((1/4)^6),"Sz",i,"Sz",i+4 #4 nearest neighbours
    end
    ham_0+=4*j*((1/4)^6),"Sz",n-3,"Sz",1 #4 nearest neighbours
    ham_0+=4*j*((1/4)^6),"Sz",n-2,"Sz",2 #4 nearest neighbours
    ham_0+=4*j*((1/4)^6),"Sz",n-1,"Sz",3 #4 nearest neighbours
    ham_0+=4*j*((1/4)^6),"Sz",n-4,"Sz",1 #4 nearest neighbours

    
    # there is a problem with 
    # the types
    # e_tot = Vector{Float64}()
    # v_tot = [Vector{Float64}() for _ in 1:ndata]
    # f_tot=Vector{Float64}()
    # dens_f_tot= [Vector{Float64}() for _ in 1:ndata]
    # z_tot= [Vector{Float64}() for _ in 1:ndata]
    
    # alternative
    #create the dataset
    #for i=tqdm(1:ndata)
    #external potential
    ham_ext=OpSum()
    for j=1:n
        ham_ext+=2*hs[j],"Sz",j # external random field
    end

    for j=1:n
        ham_ext+=2*omega,"Sx",j # to
    end 
    #ham_ext+=2*eps_breaking,"Sx",1
    #fix the invariance problem
    #end
    # collect the external
    # field
    #push!(v_tot,potential)

    hamiltonian=ham_0+ham_ext

    # initialize the hamiltonian
    # as matrix product operator
    h=MPO(hamiltonian,sites) #define the hamiltonian
    psi0=randomMPS(sites,linkdims) #initialize the product state


    #fix the sweeps
    sweeps = Sweeps(sweep)
    setmaxdim!(sweeps,5,10,20,linkdims)
    setcutoff!(sweeps, 1E-10)

    # energy values
    energy, psi = dmrg(h,psi0, sweeps,outputlevel=0)

    #compute the transverse magnetization and the density functional per site 
    z=2*expect(psi,"Sz")
    x=2*expect(psi,"Sx")
    
    zz=4*correlation_matrix(psi,"Sz","Sz")
    z_1nn=zeros(Float64,(n))
    z_2nn=zeros(Float64,(n))   
    for j=1:n
        if j==n 
            # push!(x_1nn,xx[j,1])
            # push!(x_,xx[j,2])
            z_1nn[j]=zz[j,1]
            z_2nn[j]=zz[j,2]
        elseif j==n-1
            # push!(x_1nn,zz[j,j+1])
            # push!(x_2nn,zz[j,1])
            z_1nn[j]=zz[j,j+1]
            z_2nn[j]=zz[j,1]
        else
            # push!(z_1nn,zz[j,j+1])
            # push!(z_2nn,zz[j,j+2])
            z_1nn[j]=zz[j,j+1]
            z_2nn[j]=zz[j,j+2]
        end
    end
    if two_nn
        dens_f=j_coupling*(z_1nn+z_2nn)
    else
        dens_f=j_coupling*z_1nn
    end
    f=energy/n-dot(hs,z)/n
    #print(length(f))

    #print("type of energy/n ",typeof(h),"\n")    
    # alternative method
    return energy/n, hs,z,x,dens_f,f,zz
end



function dmrg_nn_ising_check_h_k_map(seed::Int64,linkdims::Int64,sweep::Int64,n::Int64,j::Float64,h_max::Float64,omega::Float64,namefile::String,two_nn::Bool,hs::Array{Float64})

    #fix the seed
    #Random.seed!(seed)

    #fix the representation
    sites=siteinds("S=1/2",n) #fix the basis representation
    #define the universal part of the hamiltonian
    ham_0=OpSum()
    for i=1:n-1
        ham_0+=4*j,"Sz",i,"Sz",i+1 #1 nearest neighbours
    end
    ham_0+=4*j,"Sz",n,"Sz",1 #1 nearest neighbours

    for i=1:n-2
        ham_0+=4*j*((1/2)^6),"Sz",i,"Sz",i+2 #2 nearest neighbours
    end
    ham_0+=4*j*((1/2)^6),"Sz",n-1,"Sz",1 #2 nearest neighbours
    ham_0+=4*j*((1/2)^6),"Sz",n,"Sz",2 #2 nearest neighbours
    for i=1:n-3
        ham_0+=4*j*((1/3)^6),"Sz",i,"Sz",i+3 #3 nearest neighbours
    end
    ham_0+=4*j*((1/3)^6),"Sz",n-2,"Sz",1 #2 nearest neighbours
    ham_0+=4*j*((1/3)^6),"Sz",n-1,"Sz",2 #2 nearest neighbours
    ham_0+=4*j*((1/3)^6),"Sz",n,"Sz",3 #2 nearest neighbours
    for i=1:n-4
        ham_0+=4*j*((1/4)^6),"Sz",i,"Sz",i+4 #4 nearest neighbours
    end
    ham_0+=4*j*((1/4)^6),"Sz",n-3,"Sz",1 #4 nearest neighbours
    ham_0+=4*j*((1/4)^6),"Sz",n-2,"Sz",2 #4 nearest neighbours
    ham_0+=4*j*((1/4)^6),"Sz",n-1,"Sz",3 #4 nearest neighbours
    ham_0+=4*j*((1/4)^6),"Sz",n,"Sz",4 #4 nearest neighbours



    #external potential
    ham_ext=OpSum()
    for j=1:n
        ham_ext+=2*hs[j],"Sz",j # external random field
    end

    for j=1:n
        ham_ext+=2*omega,"Sx",j # to
    end 
    #ham_ext+=2*eps_breaking,"Sx",1
    #fix the invariance problem
    #end
    # collect the external
    # field
    #push!(v_tot,potential)

    hamiltonian=ham_0+ham_ext

    # initialize the hamiltonian
    # as matrix product operator
    h=MPO(hamiltonian,sites) #define the hamiltonian
    psi0=randomMPS(sites,linkdims) #initialize the product state


    #fix the sweeps
    sweeps = Sweeps(sweep)
    setmaxdim!(sweeps,5,10,20,linkdims)
    setcutoff!(sweeps, 1E-10)

    # energy values
    energy, psi = dmrg(h,psi0, sweeps)

    #compute the transverse magnetization and the density functional per site 
    z=2*expect(psi,"Sz")
    # correlation
    zz=4*correlation_matrix(psi,"Sz","Sz")
    return z,zz
end