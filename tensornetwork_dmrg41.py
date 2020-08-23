from tensornetwork import FiniteMPS
from tensornetwork.matrixproductstates.dmrg import FiniteDMRG, BaseDMRG
from tensornetwork.backends import backend_factory
from tensornetwork.backends.jax import jitted_functions
from tensornetwork.matrixproductstates.mpo import FiniteXXZ
from tensornetwork import ncon
import tensornetwork as tn
import pytest
import jax
import jax.numpy as jnp
from jax.config import config; config.update("jax_enable_x64", True)
import time
from scipy import linalg as la
import matplotlib.pyplot as plt
from sys import stdout



from tensornetwork.backend_contextmanager import get_default_backend
from tensornetwork.backends.base_backend import BaseBackend
from typing import List, Union, Text, Optional, Any, Type, Callable, Tuple
Tensor = Any


class BaseMPO:
  """
  Base class for MPOs.
  """

  def __init__(self,
               tensors: List[Tensor],
               backend: Optional[Union[BaseBackend, Text]] = None,
               name: Optional[Text] = None) -> None:
    """
    Initialize a BaseMPO.
    Args:
      tensors: A list of `Tensor` objects.
      backend: The name of the backend that should be used to perform 
        contractions. 
      name: A name for the MPO.
    """
    if backend is None:
      backend = get_default_backend()
    if isinstance(backend, BaseBackend):
      self.backend = backend
    else:
      self.backend = backend_factory.get_backend(backend)
    self.tensors = [self.backend.convert_to_tensor(t) for t in tensors]
    if len(self.tensors) > 0:
      if not all(
          [self.tensors[0].dtype == tensor.dtype for tensor in self.tensors]):
        raise TypeError('not all dtypes in BaseMPO.tensors are the same')

    self.name = name

  def __iter__(self):
    return iter(self.tensors)

  def __len__(self) -> int:
    return len(self.tensors)

  @property
  def dtype(self) -> Type[jnp.number]:
    if not all(
        [self.tensors[0].dtype == tensor.dtype for tensor in self.tensors]):
      raise TypeError('not all dtypes in BaseMPO.tensors are the same')
    return self.tensors[0].dtype

  @property
  def bond_dimensions(self) -> List[int]:
    """Returns a vector of all bond dimensions.
        The vector will have length `N+1`, where `N == num_sites`."""
    return [self.tensors[0].shape[0]
           ] + [tensor.shape[1] for tensor in self.tensors]

class FiniteMPO(BaseMPO):
  """
  Base class for implementation of finite MPOs. Users should implement 
  specific finite MPOs by deriving from FiniteMPO
  """

  def __init__(self,
               tensors: List[Tensor],
               backend: Optional[Union[BaseBackend, Text]] = None,
               name: Optional[Text] = None) -> None:
    """
    Initialize a finite MPO object
    Args:
      tensors: The mpo tensors.
      backend: An optional backend. Defaults to the defaulf backend  
        of TensorNetwork.
      name: An optional name for the MPO.
    """

    super().__init__(tensors=tensors, backend=backend, name=name)
    if (self.bond_dimensions[0] != 1) or (self.bond_dimensions[-1] != 1):
      raise ValueError('left and right MPO ancillary dimensions have to be 1')



def buildMPO():
    
    left = jnp.zeros([1,29],dtype='float64')
    bulk = []
    for i in range(N):
        bulk.append(jnp.zeros([29,29,4,4],dtype='float64'))

    right = jnp.zeros([29,1],dtype='float64')

    sX = jnp.array([[0, 1], [1, 0]],dtype='float64')
    sZ = jnp.array([[1, 0], [0,-1]],dtype='float64')
    sI = jnp.array([[1, 0], [0, 1]],dtype='float64')
    
    #MPO of the Ising model
    MH = jnp.zeros([3,3,2,2],dtype='float64');
    MH = jax.ops.index_update(MH, jax.ops.index[0,0,:,:], sI)
    MH = jax.ops.index_update(MH, jax.ops.index[1,0,:,:], sZ)
    MH = jax.ops.index_update(MH, jax.ops.index[2,0,:,:], h*sZ + g*sX)
    MH = jax.ops.index_update(MH, jax.ops.index[2,1,:,:], -sZ)
    MH = jax.ops.index_update(MH, jax.ops.index[2,2,:,:], sI)
    MLH = jnp.array([0,0,1],dtype='float64').reshape(1,3) #left MPO boundary
    MRH = jnp.array([1,0,0],dtype='float64').reshape(3,1) #right MPO boundary
    

    ML1 = ncon([MLH[0,:],MLH[0,:]],[[-1],[-2]]).reshape(1,9)
    left = jax.ops.index_update(left, jax.ops.index[:,:9], -2*ML1)
    MM1 = ncon([MH,MH],[[-1,-3,-8,-6],[-2,-4,-5,-7]]).reshape(9,9,4,4)
    for i in range(N):
      bulk[i] = jax.ops.index_update(bulk[i], jax.ops.index[:9,:9,:,:], MM1)
    MR1 = ncon([MRH[:,0],MRH[:,0]],[[-1],[-2]]).reshape(9,1)
    right = jax.ops.index_update(right, jax.ops.index[:9,:], MR1)

    iden = jnp.eye(2,dtype='float64')
    
    ML2 = ML1
    left = jax.ops.index_update(left, jax.ops.index[:,9:18], 1*ML2)
    MM2 = ncon([MH,MH,iden],[[-1,-3,-8,1],[-2,-4,1,-6],[-5,-7]]).reshape(9,9,4,4)
    for i in range(N):
      bulk[i] = jax.ops.index_update(bulk[i], jax.ops.index[9:18,9:18,:,:], MM2)
    MR2 = MR1
    right = jax.ops.index_update(right, jax.ops.index[9:18,:], MR2)

    ML3 = ML1
    left = jax.ops.index_update(left, jax.ops.index[:,18:27], 1*ML3)
    MM3 = ncon([MH,MH,iden],[[-1,-3,-5,1],[-2,-4,1,-7],[-6,-8]]).reshape(9,9,4,4)
    for i in range(N):
      bulk[i] = jax.ops.index_update(bulk[i], jax.ops.index[18:27,18:27,:,:], MM3)
    MR3 = MR1
    right = jax.ops.index_update(right, jax.ops.index[18:27,:], MR3)
    
    sigmaz = jnp.zeros((1,1,2,2),dtype='float64')
    sigmaz = jax.ops.index_update(sigmaz, jax.ops.index[0,0,:,:], sZ)

    ML4 = jnp.ones((1,1),dtype='float64')
    left = jax.ops.index_update(left, jax.ops.index[:,27:28], -2*ML4)
    Mbulk4 = jnp.zeros((1,1,4,4),dtype='float64') #identity tensors at all sites except for the last one
    Mbulk4 = jax.ops.index_update(Mbulk4, jax.ops.index[0,0,:,:], jnp.eye(4,dtype='float64'))
    for i in range(N-1):
      bulk[i] = jax.ops.index_update(bulk[i], jax.ops.index[27:28,27:28,:,:], Mbulk4)
    MM4 = ncon([sigmaz,sigmaz],[[-1,-3,-8,-6],[-2,-4,-5,-7]]).reshape(1,1,4,4) #tensor at the last site
    bulk[N-1] = jax.ops.index_update(bulk[N-1], jax.ops.index[27:28,27:28,:,:], MM4)
    MR4 = jnp.ones((1,1),dtype='float64')
    right = jax.ops.index_update(right, jax.ops.index[27:28,:], MR4)


    ML5 = ML4
    left = jax.ops.index_update(left, jax.ops.index[:,28:29], 100*ML5)
    Mbulk5 = ncon([iden,iden],[[-1,-2],[-3,-4]]).reshape(1,1,4,4)
    for i in range(N):
      bulk[i] = jax.ops.index_update(bulk[i], jax.ops.index[28:29,28:29,:,:], Mbulk5)
    MR5 = MR4
    right = jax.ops.index_update(right, jax.ops.index[28:29,:], MR5)


    bulk[0] = ncon([left,bulk[0]],[[-1,1],[1,-2,-3,-4]])
    bulk[N-1] = ncon([bulk[N-1],right],[[-1,1,-3,-4],[1,-2]])

    return bulk


_CACHED_MATVECS = {}
_CACHED_FUNCTIONS = {}


def eigsh_lanczos(
      A: Callable,
      args: Optional[List[Tensor]] = None,
      initial_state: Optional[Tensor] = None,
      shape: Optional[Tuple] = None,
      dtype: Optional[Type[jnp.number]] = None,
      num_krylov_vecs: int = 20,
      numeig: int = 1,
      tol: float = 1E-8,
      delta: float = 1E-8,
      ndiag: int = 10,
      reorthogonalize: Optional[bool] = False) -> Tuple[Tensor, List]:
    """
    Lanczos method for finding the lowest eigenvector-eigenvalue pairs
    of a hermitian linear operator `A`. `A` is a function implementing
    the matrix-vector product.
    WARNING: This routine uses jax.jit to reduce runtimes. jitting is triggered
    at the first invocation of `eigsh_lanczos`, and on any subsequent calls
    if the python `id` of `A` changes, even if the formal definition of `A`
    stays the same.
    Example: the following will jit once at the beginning, and then never again:
    ```python
    import jax
    import numpy as np
    def A(H,x):
      return jax.np.dot(H,x)
    for n in range(100):
      H = jax.np.array(np.random.rand(10,10))
      x = jax.np.array(np.random.rand(10,10))
      res = eigsh_lanczos(A, [H],x) #jitting is triggerd only at `n=0`
    ```
    The following code triggers jitting at every iteration, which
    results in considerably reduced performance
    ```python
    import jax
    import numpy as np
    for n in range(100):
      def A(H,x):
        return jax.np.dot(H,x)
      H = jax.np.array(np.random.rand(10,10))
      x = jax.np.array(np.random.rand(10,10))
      res = eigsh_lanczos(A, [H],x) #jitting is triggerd at every step `n`
    ```
    Args:
      A: A (sparse) implementation of a linear operator.
         Call signature of `A` is `res = A(vector, *args)`, where `vector`
         can be an arbitrary `Tensor`, and `res.shape` has to be `vector.shape`.
      arsg: A list of arguments to `A`.  `A` will be called as
        `res = A(initial_state, *args)`.
      initial_state: An initial vector for the Lanczos algorithm. If `None`,
        a random initial `Tensor` is created using the `backend.randn` method
      shape: The shape of the input-dimension of `A`.
      dtype: The dtype of the input `A`. If no `initial_state` is provided,
        a random initial state with shape `shape` and dtype `dtype` is created.
      num_krylov_vecs: The number of iterations (number of krylov vectors).
      numeig: The number of eigenvector-eigenvalue pairs to be computed.
        If `numeig > 1`, `reorthogonalize` has to be `True`.
      tol: The desired precision of the eigenvalues. For the jax backend
        this has currently no effect, and precision of eigenvalues is not
        guaranteed. This feature may be added at a later point.
        To increase precision the caller can increase `num_krylov_vecs`.
      delta: Stopping criterion for Lanczos iteration.
        If a Krylov vector :math: `x_n` has an L2 norm
        :math:`\\lVert x_n\\rVert < delta`, the iteration
        is stopped. It means that an (approximate) invariant subspace has
        been found.
      ndiag: The tridiagonal Operator is diagonalized every `ndiag` iterations
        to check convergence. This has currently no effect for the jax backend,
        but may be added at a later point.
      reorthogonalize: If `True`, Krylov vectors are kept orthogonal by
        explicit orthogonalization (more costly than `reorthogonalize=False`)
    Returns:
      (eigvals, eigvecs)
       eigvals: A jax-array containing `numeig` lowest eigenvalues
       eigvecs: A list of `numeig` lowest eigenvectors
    """
    if args is None:
      args = []
    if num_krylov_vecs < numeig:
      raise ValueError('`num_krylov_vecs` >= `numeig` required!')

    if numeig > 1 and not reorthogonalize:
      raise ValueError(
          "Got numeig = {} > 1 and `reorthogonalize = False`. "
          "Use `reorthogonalize=True` for `numeig > 1`".format(numeig))
    if initial_state is None:
      if (shape is None) or (dtype is None):
        raise ValueError("if no `initial_state` is passed, then `shape` and"
                         "`dtype` have to be provided")
      initial_state = self.randn(shape, dtype)

    if not isinstance(initial_state, jnp.ndarray):
      raise TypeError("Expected a `jax.array`. Got {}".format(
          type(initial_state)))
    if A not in _CACHED_MATVECS:
      _CACHED_MATVECS[A] = jax.tree_util.Partial(A)
    if "eigsh_lanczos" not in _CACHED_FUNCTIONS:
      eigsh_lanczos = jitted_functions._generate_jitted_eigsh_lanczos(jax)
      _CACHED_FUNCTIONS["eigsh_lanczos"] = eigsh_lanczos
    eigsh_lanczos = _CACHED_FUNCTIONS["eigsh_lanczos"]
    return eigsh_lanczos(_CACHED_MATVECS[A], args, initial_state,
                         num_krylov_vecs, numeig, delta, reorthogonalize)



def _optimize_1s_local(self,
                         sweep_dir,
                         num_krylov_vecs=10,
                         tol=1E-5,
                         delta=1E-6,
                         ndiag=10) -> jnp.number:
    """
    Single-site optimization at the current position of the center site. 
    The method shifts the center position of the mps by one site 
    to the left or to the right, depending on the value of `sweep_dir`.
    Args:
      sweep_dir: Sweep direction; 'left' or 'l' for a sweep from right to left,
        'right' or 'r' for a sweep from left to right.
      num_krylov_vecs: Dimension of the Krylov space used in `eighs_lanczos`.
      tol: The desired precision of the eigenvalues in `eigsh_lanczos'.
      delta: Stopping criterion for Lanczos iteration.
        If a Krylov vector :math: `x_n` has an L2 norm
        :math:`\\lVert x_n\\rVert < delta`, the iteration
        is stopped. 
      ndiag: Inverse frequencey of tridiagonalizations in `eighs_lanczos`.
    Returns:
      float/complex: The local energy after optimization.
    """
    site = self.mps.center_position
    #note: some backends will jit functions
    self.left_envs[site]
    self.right_envs[site]
    energies, states = self.backend.eigsh_lanczos(
        A=self.single_site_matvec,
        args=[
            self.left_envs[site], self.mpo.tensors[site], self.right_envs[site]
        ],
        initial_state=self.mps.tensors[site],
        num_krylov_vecs=num_krylov_vecs,
        numeig=1,
        tol=tol,
        delta=delta,
        ndiag=ndiag,
        reorthogonalize=False)
    local_ground_state = states[0]
    energy = energies[0]
    local_ground_state /= self.backend.norm(local_ground_state)

    if sweep_dir in ('r', 'right'):
      Q, R = self.mps.qr(local_ground_state)
      self.mps.tensors[site] = Q
      if site < len(self.mps.tensors) - 1:
        self.mps.center_position += 1
        self.mps.tensors[site + 1] = ncon([R, self.mps.tensors[site + 1]],
                                          [[-1, 1], [1, -2, -3]],
                                          backend=self.backend.name)
        self.left_envs[site + 1] = self.add_left_layer(self.left_envs[site], Q,
                                                       self.mpo.tensors[site])

    elif sweep_dir in ('l', 'left'):
      R, Q = self.mps.rq(local_ground_state)
      self.mps.tensors[site] = Q
      if site > 0:
        self.mps.center_position -= 1
        self.mps.tensors[site - 1] = ncon([self.mps.tensors[site - 1], R],
                                          [[-1, -2, 1], [1, -3]],
                                          backend=self.backend.name)
        self.right_envs[site - 1] = self.add_right_layer(
            self.right_envs[site], Q, self.mpo.tensors[site])

    return energy


def run_one_site(self,
                   num_sweeps=4,
                   precision=1E-6,
                   num_krylov_vecs=10,
                   verbose=0,
                   delta=1E-6,
                   tol=1E-6,
                   ndiag=10) -> jnp.number:
    """
    Run a single-site DMRG optimization of the MPS.
    Args:
      num_sweeps: Number of DMRG sweeps. A sweep optimizes all sites
        starting at the left side, moving to the right side, and back
        to the left side.
      precision: The desired precision of the energy. If `precision` is
        reached, optimization is terminated.
      num_krylov_vecs: Krylov space dimension used in the iterative 
        eigsh_lanczos method.
      verbose: Verbosity flag. Us`verbose=0` to suppress any output. 
        Larger values produce increasingly more output.
      delta: Convergence parameter of `eigsh_lanczos` to determine if 
        an invariant subspace has been found.
      tol: Tolerance parameter of `eigsh_lanczos`. If eigenvalues in 
        `eigsh_lanczos` have converged within `tol`, `eighs_lanczos` 
        is terminted.
      ndiag: Inverse frequency at which eigenvalues of the 
        tridiagonal Hamiltonian produced by `eigsh_lanczos` are tested 
        for convergence. `ndiag=10` tests at every tenth step.
    Returns:
      float: The energy upon termination of `run_one_site`.
    """
    if num_sweeps == 0:
      return self.compute_energy()

    converged = False
    final_energy = 1E100
    iteration = 1
    initial_site = 0

    self.mps.position(0)  #move center position to the left end
    self.compute_right_envs()

    def print_msg(site):
      #if verbose < 2:
        #stdout.write(f"\rSS-DMRG sweep={iteration}/{num_sweeps}, "
                     #f"site={site}/{len(self.mps)}: optimized E={energy}")
        #stdout.flush()

      if verbose >= 2:
        print(f"SS-DMRG sweep={iteration}/{num_sweeps}, "
              f"site={site}/{len(self.mps)}: optimized E={energy}")

    while not converged:
      if initial_site == 0:
        self.position(0)
        #the part outside the loop covers the len(self)==1 case
        energy = self._optimize_1s_local(
            sweep_dir='right',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag)

        initial_site += 1
        print_msg(site=0)
      while self.mps.center_position < len(self.mps) - 1:
        #_optimize_1site_local shifts the center site internally
        energy = self._optimize_1s_local(
            sweep_dir='right',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag)

        print_msg(site=self.mps.center_position - 1)
      #prepare for left sweep: move center all the way to the right
      self.position(len(self.mps) - 1)
      while self.mps.center_position > 0:
        #_optimize_1site_local shifts the center site internally
        energy = self._optimize_1s_local(
            sweep_dir='left',
            num_krylov_vecs=num_krylov_vecs,
            tol=tol,
            delta=delta,
            ndiag=ndiag)

        print_msg(site=self.mps.center_position + 1)

      if jnp.abs(final_energy - energy) < precision:
        converged = True
        numofitarray.append(iteration)
      final_energy = energy
      iteration += 1
      if iteration > num_sweeps:
        if verbose > 0:
          print()
          print("dmrg did not converge to desired precision {0} "
                "after {1} iterations".format(precision, num_sweeps))
        numofitarray.append(iteration)
        break
    return [self.mps,final_energy]


'''
global num
jax.jit(self._optimize_1s_local,device=jax.devices()[1+(num%3)])
'''


def constructconfigarray(D):
    configarray=[]
    
    bondarray=[]
    for i in range(N-1):
        bondarray.append(D)
    configarray.append(bondarray)
    
    return configarray


def test_finite_DMRG_init(N, config, D):
    
    backend = 'jax'
    dtype = 'float64'
  
    a = buildMPO()
    mpo = FiniteMPO(a,backend=backend)

    mps = FiniteMPS.random([4] * N, config, dtype=dtype, backend=backend)
    
    ans1 = 500
    ans2 = 100
    sumtime = 0
    while abs(2*(ans1-ans2)/(ans1+ans2))>0.1/100:
        
        print("Bond dim = {D}, Configuration = {c}".format(D=D,c=config))
        bondarray.append(D)
        start_time = time.time()
        dmrg = FiniteDMRG(mps, mpo)
        [mps,energy]=run_one_site(dmrg,num_sweeps=1000, num_krylov_vecs=15, precision=prec, delta=prec, tol=prec, verbose=0, ndiag=1)
        timearray.append(jnp.around(float(time.time() - start_time)/60,decimals=2))
        print("{x} minutes".format(x=timearray[-1]))
        sumtime += timearray[-1]
        energy += 2
        minarray.append(energy)
        print("Minimum: {x}".format(x=minarray[-1]))
        ans1=ans2
        ans2=energy
        precisionarray.append(jnp.around(2*(ans1-ans2)/(ans1+ans2)*100, decimals=2))
        print("Precision (%): {x}".format(x=precisionarray[-1]))
        print("Number of iterations: {x}\n".format(x=numofitarray[-1]))


        D*=2
        for i in range(N-1):
            config[i] *= 2
            

        A = mps.tensors
        B = jnp.zeros((1,4,config[0]),dtype='float64')
        B = jax.ops.index_update(B, jax.ops.index[:,:,:A[0].shape[2]], A[0])
        A[0] = B
        for i in range(1,N-1):
            B = jnp.zeros((config[i-1],4,config[i]),dtype='float64')
            B = jax.ops.index_update(B, jax.ops.index[:A[i].shape[0],:,:A[i].shape[2]], A[i])
            A[i] = B
        B = jnp.zeros((config[N-2],4,1),dtype='float64')
        B = jax.ops.index_update(B, jax.ops.index[:A[N-1].shape[0],:,:], A[N-1])
        A[N-1] = B
        mps.tensors = A

    print("")

##### -ZZ+hZ+gX model #############
#######################################

##### Set bond dimensions and simulation options

h = 0.7
g = 1.05

tn.set_default_backend('jax')


print('H = - ZZ + hZ + gX')
print('g = {g}'.format(g=g))
print('h = {h}\n'.format(h=h))

num=0

Narray = [10]

prec = 1E-4

Dinit = 8

for N in Narray:
    print('Length of the chain = {N}\n'.format(N=N))      
    configarray=constructconfigarray(Dinit)
    for config in configarray:
        numofitarray=[]
        minarray=[]
        precisionarray=[]
        timearray=[]
        bondarray=[]
    
        t1 = time.time()
        test_finite_DMRG_init(N,config,Dinit)
        print("{x} minutes".format(x=jnp.around(float(time.time() - t1)/60,decimals=2)))
        print("Bond array: {x}".format(x=bondarray))
        print("Minimum array: {x}".format(x=jnp.float64(minarray)))
        print("Precision array (%): {x}".format(x=jnp.float(precisionarray)))
        print("Time array (minutes): {x}".format(x=jnp.float(timearray)))
        print("Number of iterations (at every bond): {x}".format(x=jnp.int(numofitarray)))
        print("*********************************************************************************\n")
        
        


'''
import jax
print(jax.devices())


def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jax.numpy.where(x > 0, x, alpha * jax.numpy.exp(x) - alpha)

jax.jit(selu,device=jax.devices()[1])
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10,))
print(selu(x))  


from jax import numpy as jnp
print(jnp.ones(3).device_buffer.device())

print(jax.device_put(1, jax.devices()[2]).device_buffer.device())  
'''