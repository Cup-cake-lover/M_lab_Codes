import numpy as np
import matplotlib.pyplot as plt ; import matplotlib
from numba import jit
import pandas as pd
from tqdm import tqdm
matplotlib.rcParams['figure.dpi']=100



def init_lattice(L):
  '''
  args : L = lattice size (L)
  returns : ndarray shape L
  '''
  lattice = 2 * np.random.randint(0, 2, L) - 1 ##arrange [-1,+1] spins accordingly
  return lattice

#lattice = init_lattice(5)



def create_target_params(N):
  temp_matrix = np.random.normal(0,1/N,(N,N))
  J = 0.5*(temp_matrix + temp_matrix.T) ; np.fill_diagonal(J,0)
  h = np.random.normal(0,1,size=N)

  return J,h
  
  
def calculate_energy_diff(temp_index,configuration,J,h):
  E = 2*configuration[temp_index] * (np.dot(J[temp_index,:],configuration) + h[temp_index])
  return E
 
 
def metropolis_dynamics(lattice,J,h):
  configuration = lattice.copy() ; L = int(len(configuration))
  for _ in range(len(lattice)):
    random_index = np.random.randint(0,L)
    delta_E = calculate_energy_diff(random_index,configuration,J,h)

    prob = np.exp(-delta_E)
    if np.random.uniform(0,1) < prob:
      configuration[random_index] *= -1

  return configuration
  
 
def markovchain_montecarlo(lattice,J,h,sweeps,burn_in):
  configuration = lattice.copy() ; L = len(lattice)
  C_ij = np.zeros((L,L))
  states = []

  for _ in range(burn_in):
    configuration = metropolis_dynamics(configuration,J,h)

  for sweep in range(sweeps):
    configuration = metropolis_dynamics(configuration,J,h)
    C_ij += np.outer(configuration,configuration)
    states.append(configuration)

  mag = np.mean(states,axis=0)
  C_ij /= sweeps

  return mag,C_ij,states


def calculate_NLL(configs):
    """
    Calculate the negative log-likelihood of the given configurations.

    Args:
    configs : array-like, the sampled configurations of spins

    Returns:
    log_likelihood : float, the negative log-likelihood
    """
    M = len(configs)

    unique_samples, sample_counts = np.unique(configs, axis=0, return_counts=True)

    Prob_distr = sample_counts / M

    log_likelihood = -(1/M) * (np.sum(sample_counts*np.log(Prob_distr)))

    return log_likelihood
    
def h_update(h,mag_model,mag_train,eta):
  h_up = h + eta * (mag_train - mag_model)
  return h_up

def J_update(J,corr_model,corr_train,eta):
  J_up = J + eta * (corr_train - corr_model)
  np.fill_diagonal(J_up,0)
  return J_up


def boltzmann_machine(L,iterations,sweeps,burn_in,mag_train,corr_train,eta):
  J,h = create_target_params(L)
  log_likelihoods = []
  for iter in tqdm(range(iterations)):
    mag_model,corr_model,states = markovchain_montecarlo(init_lattice(L),J,h,sweeps,burn_in)
    h = h_update(h,mag_model,mag_train,eta)
    J = J_update(J,corr_model,corr_train,eta)
    NLL = calculate_NLL(states)
    log_likelihoods.append(NLL)
  return J,h,log_likelihoods


def calculate_salamander_params(data):
  Corrs = np.zeros((data.shape[0],data.shape[0]))
  for _ in range(int(data.shape[1])):
    Corrs += np.outer(data[:,_],data[:,_])

  Corrs /= int(data.shape[1])
  Mags = np.mean(data,axis=1)

  return Corrs,Mags
  

def create_file_salamander(J,h,NLL):
  J_flat = J.flatten()
  df_field = pd.DataFrame(h)
  df_field.to_csv(f'sal_fields',index=False)

  df_interactions = pd.DataFrame(J_flat)
  df_interactions.to_csv(f'sal_interactions',index=False)

  df_NLLs = pd.DataFrame(NLL)
  df_NLLs.to_csv(f'sal_nll',index=False)
  
  return None


  
 
salamander_data = np.loadtxt("bint.txt")

salamander_train,salamander_test = salamander_data[:,:int(salamander_data.shape[1]/2)],salamander_data[:,int(salamander_data.shape[1]/2):]


Corrs_sal,Mags_sal = calculate_salamander_params(salamander_train)

J_inferred_sal,h_inferred_sal,NLL_sal = boltzmann_machine(160,500,10000,2**11,Mags_sal,Corrs_sal,0.05)

create_file_salamander(J_inferred_sal,h_inferred_sal,NLL_sal)
 
 
 
