:author: Brandon L. Butler
:email: butlerbr@umich.edu
:institution: University of Michigan, Department of Chemical Engineering

:author: Vyas Ramasubramani
:institution: University of Michigan, Department of Chemical Engineering

:author: Joshua A. Anderson
:institution: University of Michigan, Department of Chemical Engineering

:author: Sharon C. Glotzer
:institution: University of Michigan, Department of Chemical Engineering
:institution: University of Michigan, Department of Material Science and Engineering
:institution: University of Michigan, Department of Physics
:institution: University of Michigan, Biointerfaces Institute
:bibliography: references

-----------------------------------------------------------------------------------------------------
HOOMD-blue version 3.0  A Modern, Extensible, Flexible, Object-Oriented API for Molecular Simulations
-----------------------------------------------------------------------------------------------------

.. class:: abstract

    HOOMD-blue is a library for running molecular dynamics and hard particle Monte Carlo
    simulations that uses pybind11 to provide a Python interface to fast C++ internals. One of the
    fastest simulation toolkits available, the package is designed to scale from a single CPU core
    to thousands of NVIDIA or AMD GPUs. In this paper, we discuss the upcoming release of
    HOOMD-blue version 3.0. This focus of this version is to improve the flexibility and
    extensibility of the Python API, with an emphasis on providing simpler and more performant
    entry points to the internal C++ classes and data structures. With these updates, we show how
    HOOMD-blue users will be able to write completely custom simulation methods directly in Python
    and analyze previously inaccessible data. Throughout this paper, we focus on how these goals
    have been achieved and explain design decisions through examples of the newly developed API.

.. class:: keywords

    molecular dynamics, molecular simulations, Monte Carlo simulations, API design

Introduction
------------

Molecular simulation has been an important technique for studying the equilibrium properties of
molecular systems since the 1950s. The two most common methods for this purpose are molecular
dynamics and Monte Carlo simulations :cite:`metropolis.etal1953, alder.wainwright1959`. Molecular
dynamics (MD) is the application of Newton's laws of motion to systems of particles, while Monte
Carlo (MC) simulations statistically sample a system's degrees of freedom to find
equilibrium quantities. Since their inception, these tools have been used to study
systems of colloids :cite:`damasceno.etal2012a`, metallic glasses :cite:`fan.etal2014`, and
proteins :cite:`dignon.etal2018a`, among others.

Today many packages exist to perform these simulations: LAMMPS :cite:`plimpton1993`, GROMACS
:cite:`berendsen.etal1995, abraham.etal2015`, OpenMM :cite:`eastman.etal2017`, and HOOMD-blue **I would use a different example than HOOMD here, maybe CHARMM or Amber**
:cite:`anderson.etal2008, glaser.etal2015, anderson.etal2020` to name a few. Increased computational
power and algorithmic improvements :cite:`niethammer.etal2014` - including the exploitation of GPU
architectures :cite:`spellings.etal2017` - have tremendously increased the length
:cite:`byna.etal2012` and time :cite:`shaw.etal2009` scales of simulations from those conducted in
the mid 1900s. Due to the flexibility and generality of such tools, their usage has dramatically
increased the usage of molecular simulations, further increasing the demand for highly flexible
and customizable software packages that can be tailored to very specific simulation requirements.
Meeting these needs requires a well-designed application programming interface (API) that facilitates
a wide range of custom simulation protocols. Different tools have taken different approaches to
enabling this, such as the text-file scripting in LAMMPS and the command line interface provided by
GROMACS. Although such methods have been widely successfully in balancing flexibility with ease of use
for non-programmers, the unparalleled flexibility of working within a fully-featured programming
language environment motivated HOOMD-blue to be the first simulation engine (**I think, check this with Josh**)
to offer a full-featured Python interface. HOOMD-blue version 3.0 aims to further improve
this interface by providing simpler, more Pythonic ways to write simulations and providing seamless
interfaces for other tools in the Python ecosystem, providing a more usable and flexible API than
ever before.

HOOMD-blue is a Python package written in C++ for MD and hard particle MC simulations. 
First released in 2008 :cite:`anderson.etal2020`, HOOMD-blue was the package to provide
complete support for GPU-based molecular simulations using CUDA and C++. In 2014, HOOMD-blue began
supporting parallelization using domain decomposition (separating a simulation box into local
boxes, one for each rank) through MPI. Recent development on HOOMD-blue enables support for both
NVIDIA and AMD GPUs. At the time of writing, HOOMD-blue's branch for version 3.0 development has
12,510 commits, 1,154 files and 187,382 lines of code excluding blank lines and comments.

Soon after HOOMD-blue's first release, Joshua added an imperative Python interface for writing
simulations scripts. The structure and available commands in the original Python API are largely
inspired by and reminiscent of the structure of other simulation software such as LAMMPS.
This largely remained the same as HOOMD-blue released its version 2.0. The primary goal of the 3.0
release is a complete redesign of the API from the ground up to present a
thoroughly object-oriented and Pythonic interface for users. Where possible we have
sought to provide performant ways to use Python to interface with the HOOMD-blue C++ back-end.
Other Python packages like SciPy :cite:`virtanen.etal2020`, NumPy :cite:`vanderwalt.etal2011`,
scikit-learn :cite:`pedregosa.etal2011`, matplotlib :cite:`hunter2007`, and others have inspired us
in this pursuit (**how so? do you mean that other packages do similar things, or that they would be very useful if you could inject them into inner loops in HOOMD, i.e. are they inspiration or motivation?**).
In this endeavour, we have found ways to make HOOMD-blue more flexible, extensible,
and integrable with the scientific Python community.  Over the next few sections, we will use examples
of HOOMD-blue's version 3.0 API (which is still in development at the time of writing) to highlight
>>>>>>> First round of edits from Vyas.
changes in the package's extensibility, flexibility, and Pythonic interface.
(**After reading through most of the paper, I don't think the object-oriented aspect is really discussed enough here. There should be some discussion of why that's better, what benefits it confers.**)

Example Script
--------------

Here we show a script that simulates a Lennard-Jones fluid using the current implementation of the
version 3.0 API. We also show a rendering of the particle configuration in Figure (:ref:`sim`).

.. code-block:: python

    import hoomd
    import hoomd.md
    import numpy as np

    device = hoomd.device.Auto()
    sim = hoomd.Simulation(device)

    # Place particles on simple cubic lattice
    N_per_side = 14
    N = N_per_side ** 3
    L = 20
    xs = np.linspace(0, 0.9, N_per_side)
    x, y, z = np.meshgrid(xs, xs, xs)
    coords = np.array(
        (x.ravel(), y.ravel(), z.ravel())).T

    snap = hoomd.Snapshot()
    snap.particles.N = N
    snap.configuration.box = hoomd.Box.cube(L)
    snap.particles.position[:] = (coords - 0.5) * L
    snap.particles.types = ['A']

    # Create state
    sim.create_state_from_snapshot(snap)

    # Create integrator and forces
    integrator = hoomd.md.Integrator(dt=0.005)
    langevin = hoomd.md.methods.Langevin(
        hoomd.filter.All(), kT=1., seed=42)

    nlist = md.nlist.Cell()
    lj = md.pair.LJ(nlist, r_cut=2.5)
    lj.params[('A', 'A')] = dict(
        sigma=1., epsilon=1.)

    integrator.methods.append(langevin)
    integrator.forces.append(lj)

    # Setup output
    gsd = hoomd.dump.GSD('dump.gsd', trigger=100)
    log = hoomd.Logger()
    log += lj
    gsd.log = log

    sim.operations.integrator = integrator
    sim.operations.analyzers.append(gsd)
    sim.run(100000)

.. figure:: figures/sim-output.png
    :align: center

    A rendering of the Lennard-Jones fluid simulation script output. Particles are colored by the
    Lennard-Jones potential energy that is logged using the HOOMD-blue :code:`Logger` and
    :code:`GSD` class objects. Figure is rendered in OVITO :cite:`stukowski2009a` using the Tachyon
    :cite:`stone1998` render. :label:`sim`

General API Design
------------------

Simulation, Device, State, Operations
+++++++++++++++++++++++++++++++++++++

Each simulation in HOOMD-blue is now controlled through 3 main objects that are joined together by
the :code:`Simulation` class: the :code:`Device`, :code:`State`, and :code:`Operations` classes. A
simple figure of this relationship with some core attributes/methods for each class is given in
Figure (:ref:`core-objects`). Each :code:`Simulation` object holds the requisite information to run
a full molecular dynamics (MD) or Monte Carlo (MC) simulation.  The :code:`Device` class controls
whether a simulation is run on CPUs or GPUs and the number of cores/GPUS it should run on. In
addition, the device manages other configuration information such as custom memory tracebacks and
the MPI communicator.

.. figure:: figures/object-diagram.pdf
    :align: center

    Diagram of core objects with some attributes and methods. Classes are in bold and orange;
    attributes and methods are blue. Figure is made using Graphviz :cite:`ellson.etal2003,
    gansner.etal1993`. :label:`core-objects`

The :code:`State` class stores the system data (e.g. particle positions, orientations, velocities,
the system box). The :code:`State` class also exposes this data and allows setting it in two
fundamental ways. Through the snapshot API, users interface with a single object exposing many NumPy
arrays of system data. To construct a snapshot, all system data stored across all MPI ranks must be
gathered to and combined on the root rank. Setting the state using the snapshot API requires setting
the snapshot property to an entirely new snapshot. The advantages of this approach come from the
ease of use of working with a single object containing the complete description of the state. The
following snippet showcases hows this approach can be used to set the z position of all particles to
zero.

.. code-block:: python

    snap = sim.state.snapshot
    # snapshot only stores data on rank 0
    if snap.exists:
        # set all z positions to 0
        snap.particles.position[:, 2] = 0
    sim.state.snapshot = snap

The other API for accessing :code:`State` data is via a zero-copy, rank-local access to the
state's data on either the GPU or CPU. On the CPU, we expose the buffers as
:code:`numpy.ndarray`-like objects through provided hooks such as :code:`__array_ufunc__` and
:code:`__array_interface__`. Similarly, on the GPU, we mock much of the
of CuPy's
:cite:`zotero-593` :code:`ndarray` class if it is installed; however, at present the CuPy
:code:`ndarray` class provides fewer hooks, so our integration is more limited. Whether or not CuPy
is installed, we use the
:code:`__cuda_array_interace__` protocol for GPU access. This provides support for libraries such as
numba's :cite:`lam.etal2015` GPU JIT and PyTorch :cite:`paszke.etal2019`. We chose to mock the
interfaces of both NumPy and CuPy rather than just expose :code:`ndarray` objects directly out of
consideration for memory safety. To ensure data integrity, data is only accessible within a specific
context manager that the user must enter to enable zero-copy data access. Using zero-copy and
rank-local access in this way is much faster than using the snapshot API, but it requires the user
to deal with MPI and domain decomposition directly. The example below modifies the previous example
to instead use the zero-copy API.

.. code-block:: python

    # CPU access
    with sim.state.local_snapshot as data:
        data.particles.position[:, 2] = 0

    # GPU access (assumes CuPy is installed)
    with sim.state.gpu_snapshot as data:
        data.particles.position[:, 2] = 0

The final of the three classes, :code:`Operations`, holds the different *operations* that will act
on the simulation state. Broadly these consist of 3 categories: updaters which modify simulation
state, analyzers which observe system state, and tuners which tune the hyperparameters
of other operations for performance. (**Right now the sequence is a little confusing after this.
I think you need a sentence or two to explain that 1) you're going to get back to operations, 2) why
deferred initialization is relevant to operations, and 3) what operations all share in common that
merits having a single infrastructure for all of them. You also should probably explaining how
logging is related to these.**).

Deferred C++ Initialization
+++++++++++++++++++++++++++

Many objects in C++ in HOOMD-blue currently require either a :code:`System` or a :code:`SystemDefinition`
object (both C++ classes) in order to be correctly instantiated. The requirement is foremost due to
the interconnected nature of many things in a simulation (**I would combine this sentence with the previous one and give a more concrete reasoning e.g. you need particle positions or something like that**).
This imposes strict requirements on the order
in which objects must be created. However, having to create a full simulation state just to create, for instance, a
pair potential, limits the composability of the Python interface and makes it harder to write modular simulation protocols. For
example, if a package that wanted to automatically generate a particular force-field in response to
some user inputs, it would require a a previously instantiated :code:`Device` and access to the :code:`State` it was to operate on.
This means that this functionality could
only be invoked after the user had already instantiated a specific simulation state. Moreover, this
requirement makes it more difficult for users to write simulation scripts, because it requires them
be aware of the order in which objects must be created. To circumvent these difficulties, the new API
has moved to a deferred initialization model in which C++ objects are not created until the corresponding
Python objects are "attached" to a :code:`Simulation`.

In addition to ameliorating the difficulties mentioned above, deferred initialization also simplifies
access to the object's internal state (**how so?**) and allows easier duck-typing of parameters. We
take advantage of the accessibility of state by making a complete specification of an object's internal state
(not to be confused with the simulation state) a loggable quantity for
the :code:`Logger` object, and providing a :code:`from_state` factory method for all operations in
HOOMD-blue. This state is sufficient to completely reconstruct the object, greatly increasing the
restartability of simulations since the state of each object can be logged at the end of a given
run and read at the start of the next.

.. code-block:: python


    from hoomd.hpmc.integrate import Sphere

    sphere = Sphere.from_state('example.gsd', frame=-1)

This code block would create a :code:`Sphere` object with the parameters stored from the last frame
of the gsd file :code:`example.gsd`.


The Internal Base Classes
+++++++++++++++++++++++++

To facilitate adding more features to HOOMD-blue, simplify the internal class logic, and provide a
more uniform interface, we wrote the :code:`_Operation` class. This base class, which is inherited
by most other user-facing classes, provides object dependency handling (**what does this mean**),
deferred C++ initialization, and automated synchronization of attributes between Python and C++.

Likewise, to provide a Pythonic interface for interacting with object parameters, robust validation
on setting, and maintaining state between Python and C++ when "attached"(**I don't think we need this in quotes, just define the term the first time then it's good**) we created two solutions:
one for parameters that are type dependent and one for those that were not.  Through the
:code:`ParameterDict` class, we ensure constancy between C++ object members and Python values while
exposing the dictionaries keys as attributes. For type dependent attributes, we use the
:code:`TypeParameter` and :code:`TypeParameterDict` classes. These type dependent quantities are
exposed through dictionary-like attributes with types as keys.

Each class supports validation of their keys, and the :code:`TypeParameterDict` can be used to
define the structure and validation of arbitrarily nested structures of dictionaries, lists, and
tuples. In addition, both classes support a similar level of default specification to their
level of validation. (**I find this paragraph confusing. does the nesting statement not hold true for both? what about defaults?**)
An example object specification and initialization can be seen below.

.. code-block:: python

    TypeParameterDict(
        num=float,
        list_of_str=[str],
        nesting={len_three_vec=(float, float, float)},
        len_keys=2
        )

.. code-block:: python

    from hoomd.hpmc.integrate import Sphere

    sphere = Sphere(seed=42)
    # example using ParameterDict
    sphere.nselect = 2
    # examples using TypeParameter and TypeParameterDict
    sphere.shape['A'] = {'diameter': 1.}
    # sets for 'B', 'C', and 'D'
    sphere.shape[['B', 'C', 'D']] = {'diameter': 0.5}

To store lists that must be synced to C++, the analogous :code:`SyncedList` class transparently
handles synchronization of Python lists and C++ vectors.

.. code-block:: python

    from hoomd import Operations
    from hoomd.dump import GSD

    ops = Operations()
    gsd = GSD('example.gsd')
    # use of SyncedList
    ops.analyzers.append(gsd)

Another improvement to user experience is the error handling for these objects. An
example error message for accidentally trying to set :code:`sigma` for 'A'-'A' interactions in the
Lennard-Jones pair potential to a string (i.e. :code:`lj.params[('A', 'A')] = {'sigma': 'foo',
'epsilon': 1.}` would provide the error message, "TypeConversionError: For types [('A', 'A')], error
In key sigma: Value foo of type <class 'str'> cannot be converted using OnlyType(float).  Raised
error: value foo not convertible into type <class 'float'>.". (**How would this behave in current hoomd? Also maybe bold the error**)

Logging and Accessing Data
--------------------------

The more object-oriented Python API has also led to significant changes and improvements in the
logging infrastructure in version 3. Currently, the primary mode of accessing data in HOOMD is by
logging all data to a file. In the new API, all we directly expose object data through extensive use
of properties. For example, the total potential energy in all pair potentials can now be directly
queried, thereby encouraging users to use such data directly rather than requiring its logging to a
file. To support logging data to files, we have created a Python :code:`Logger` class that uses
these properties to create an intermediate representation of the logged information when called. The
code:`Logger` is quite general and supports logging scalars, strings, arrays, and even general
Python objects. By separating the data collection from the writing to files, and by providing such a
flexible intermediate representation, HOOMD can now support a range of back ends for logging;
moreover, it offers users the flexibility to define their own. For instance, logging data to text
files or standard out is supported out of the box, but other backends like MongoDB, Pandas
:cite:`mckinney2010`, and Python pickles would also be feasible to implement.  Consistent with this
move towards providing numerous output options and thinking of HOOMD as a Python simulation library
first, version 3.0 chooses to make simulation output an opt-in feature even for common simulation
output like performance and thermodynamic quantities (e.g temperature and pressure). In addition to
this improved flexibility in storage possibilities, we have added new properties to objects to
directly expose more of their data than had previously been available. For example, pair potentials
now expose *per-particle* potential energies at any given time (this data is used to color Figure
(:ref:`sim`)).

Logger
++++++

**Some of this sequence is a bit tricky, because objects like the logger are introduced beforehand without definition. Obviously the other objects are more important and probably should come first, but you may at least want to provide a little definition of these classes somewhere up there.**

The :code:`Logger` class aims to provide a simple interface for logging most HOOMD-blue objects and
custom user quantities. Through the :code:`Loggable` metaclass, all subclasses that inherit from
:code:`_Operation` expose their loggable quantities. Adding all loggable quantities of an object to
a logger for logging is as simple as :code:`logger += obj`. The utility of this class lies in its
intermediate representation of the data. Using the HOOMD-blue namespace as the basis for
distinguishing between quantities, we map logged quantities into a nested dictionary. For example,
logging the Lennard-Jones pair potential's total energy would be produce this dictionary by a
:code:`Logger` object :code:`{'md': {'pair': {'LJ': {'energy': (-1.4, 'scalar')}}}}` where 
:code:`'scalar'` is a flag to make processing the logged output easier. In real use cases, the
dictionary would likely be filled with many other quantities. This intermediate form allows
developers and users to more easily create different back ends that a :code:`Logger` object can plug
into for outputting data.

User Customization
------------------

**We don't need to repeat HOOMD version 3.0 everywhere. After the intro it should be clear when we're talking about new and old functionality, you should endeavor to make that obvious in the way you phrase things.**

In HOOMD-blue version 3.0, we provide multiple means of "injecting" Python code into HOOMD-blue's
C++ core. We achieve this through two general means, inheriting from C++ classes through pybind11
:cite:`jakob.etal2017` and wrapping user classes and functions in C++ classes. To guide the
choice between inheritance and composition, we looked at multiple factors such as how simple the class is (only
requires a few methods) and whether or not inheritance would expose internals. Regardless of the
method to add functionality to HOOMD-blue, we have prioritized adding and improving methods for
extending the package as the examples below highlight.

Triggers
++++++++

In HOOMD-blue version 2.x, everything that was not run every timestep had a period and phase
associated with it. The timesteps the operation was run on could then be determined by the
expression, :code:`timestep % period - phase == 0`.  In our refactoring and development, we
recognized that this concept could be made much more general and consequently more flexible. Objects
do not have to be run on a periodic timescale; they just need some indication of when to run. In
other words, the operations needed to be *triggered*. The :code:`Trigger` class encapsulates this
concept  providing a uniform way of specifying when an object should run without limiting options.
Each operation that requires triggering is now associated with a corresponding :code:`Trigger`
instance. This approach enables complex triggering logic through composition of multiple triggers 
such as :code:`Before` and :code:`After` which return :code:`True` before or after a given timestep
with the :code:`And`, :code:`Or`, and :code:`Not` subclasses whose function can be understood by
recognizing that a :code:`Trigger` is essentially a functor that returns a Boolean value.
**I would start with the periodic trigger since that's what people are used to, then come back and discuss boolean operators.**

In addition, to the flexibility the :code:`Trigger` class provides, abstracting out the concept of
triggering an operation, we can provide through pybind11 a way to subclass :code:`Trigger` in
Python. This allows users to create their own triggers in pure Python. An example of such
subclassing that reimplements the functionality of HOOMD-blue version 2.x can be seen in the below
-- this functionality already exists in the :code:`Periodic` class in version 3.0.

.. code-block:: python

    from hoomd.trigger import Trigger

    class CustomTrigger(Trigger):
        def __init__(self, period, phase=0):
            super().__init__()
            self.period = period
            self.phase = phase

        def __call__(self, timestep):
            v = timestep % self.period - self.phase == 0
            return v

While this example is quite simple, user-created subclasses of :code:`Trigger` need not be as seen
**Reference the section by name and maybe number the example (e.g. the third example in section...)**
in an example in a further section. They can implement arbitrarily complex Python code for more
caching, examining the simulation state, etc.

Variants
++++++++

Similar to :code:`Trigger`, we generalized our ability to linearly interpolate values
(:code:`hoomd.variant.liner_interp` in HOOMD-blue version 2.x) across timesteps to a base class
:code:`Variant` which generalizes the concept of functions in the semi-infinite domain of timesteps
:math:`t \in [0,\infty), t \in \mathbb{Z}`. This allows sinusoidal cycling, non-uniform ramps, and
various other behaviors -- as many as there are functions in the non-negative integer domain and
real range. Like :code:`Trigger`, :code:`Variant` is able to be directly subclassed from the C++
class. :code:`Variant` objects are used in HOOMD-blue to specify quantities like temperature,
pressure, and box size for varying objects. An example of a sinusoidal cycled variant is shown
below.

.. code-block:: python

    from math import sin
    from hoomd.variant import Variant

    class SinVariant(Variant):
        def __init__(self, frequency, amplitude,
                    phase=0, center=0):
            super().__init__()
            self.frequency = frequency
            self.amplitude = amplitude
            self.phase = phase
            self.center = center

        def __call__(self, timestep):
            tmp = self.frequency * timestep
            tmp = sin(tmp + self.phase)
            return self.amplitude * tmp + self.center

        def _min(self):
            return -self.amplitude + self.center

        def _max(self):
            return self.amplitude + self.center

ParticleFilters
+++++++++++++++

Unlike :code:`Trigger` or :code:`Variant`, :code:`ParticleFilter` is not a generalization of an
existing concept but the splitting of one class into two. However, this split is also targeted at increasing
flexibility. In HOOMD-blue version 2.x, the :code:`ParticleGroup` class and subclasses served to
provide a subset of particles within a simulation for file output, application of thermodynamic
integrators, and other purposes. The class hosted both the logic for storing the subset of particles
and filtering them out from the system. After the refactoring, the :code:`ParticleGroup` 
is only responsible for the logic to store and preform some basic operations on particle tags (a means
of identifying individual particles), while the new class :code:`ParticleFilter` implements the selection logic.
This choice makes :code:`ParticleFilter` objects much more lightweight and provides a means of
implementing a :code:`State` instance-specific cache of :code:`ParticleFilter` objects. The latter
ensures that we do not create multiple of the same :code:`ParticleGroup`, which can occupy large
amounts of memory. The caching also allows the creation of large numbers of the same
:code:`ParticleFitler` object without needing to worry about memory constraints.

.. TODO Update this section with whatever paradigm we decide to use for user customization.

Finally, thanks to this separation it is now possible for users to define custom filters. Specifically,
unlike the :code:`ParticleGroup`, the :code:`CustomParticleFilter` subclass of
:code:`ParticleFilter` is suitable for user subclassing since its scope is
much more limited. For this reason, :code:`ParticleGroup` is private in version 3.
An example of a :code:`CustomParticleFilter` that selects only particles with positive charges is
given below.

.. code-block:: python

    class PositiveCharge(CustomParticleFilter):
        def __init__(self, state):
            super().__init__(state)

        def __hash__(self):
            return hash(self.__class__.__name__)

        def __eq__(self, other):
            return type(self) == type(other)

        def find_tags(self, state):
            with state.local_snapshot as data:
                mask = data.particles.charge > 0
                return data.particles.tag[mask]

Custom Operations
+++++++++++++++++

**I think this intro of actions is a bit too cursory. I think you need at least one more sentence somewhere explaining what it is, otherwise it's not clear to me what it enables.**
Through composition, HOOMD-blue version 3.0 offers the ability to create custom actions (the object
within HOOMD-blue operations that performs some act with the :code:`Simulation`) in Python that run
in the standard :code:`Simulation` run loop. The feature makes user-created actions behave
indistinguishably from native C++ actions. Through custom actions, users can modify state, tune
hyperparameters for performance, or just observe parts of the simulation. In addition, we are adding
a signal for Actions to send that would stop a :code:`Simulation.run` call. This would allow
actions to run until they are "done" rather than running for a large number of steps to ensure
completion or running for multiple short spurts and checking in between. With respect to
performance, with zero-copy access to the data on CPUs or GPUs, custom actions can also achieve
high performance using standard Python libraries like NumPy, SciPy, numba, CuPy and others.
Furthermore, this performance comes without users having to worry about manual code compilation,
ABI compatibility, or other concerns of compiled languages.

.. TODO need to add example

Larger Examples
---------------

In this section we will provide more substantial applications of features new to HOOMD-blue version 3.0.

Trigger that detects nucleation
+++++++++++++++++++++++++++++++

This example demonstrates a :code:`Trigger` that returns true when a threshold :math:`Q_6`
Steinhardt order parameter :cite:`steinhardt.etal1983` (as calculated by freud
:cite:`ramasubramani.etal2020`) is reached. Such a :code:`Trigger` could be used for BCC nucleation
detection that, depending on the type of simulation, could trigger a decrease in cooling rate, the
more frequent output of simulation trajectories, or any of other desired action. Also, in
this example we showcase the use of the zero-copy rank-local data access. This example also requires
the use of ghost particles, which for each MPI rank are the particles owned by spatially neighboring MPI
whose presence could influence particles in this rank ranks knows about, but is not
directly responsible for updating. In this case, those particles are required for computing
the :math:`Q_6` value for particles near the edges of the current rank's local simulation box.

.. code-block:: python

    import numpy as np
    import freud
    from mpi4py import MPI
    from hoomd.trigger import Trigger

    class Q6Trigger(Trigger):
        def __init__(self, simulation, threshold,
                     mpi_comm=None):
            super().__init__()
            self.threshold = threshold
            self.state = simulation.state
            nr = simulation.device.num_ranks
            if nr > 1 and mpi_comm is None:
                raise RuntimeError()
            elif nr > 1:
                self.comm = mpi_comm
            else:
                self.comm = None
            self.q6 = freud.order.Steinhardt(l=6)

        def __call__(self, timestep):
            with self.state.local_snapshot as data:
                part_data = data.particles
                box = data.box
                aabb_box = freud.locality.AABBQuery(
                    box,
                    part_data.positions_with_ghosts)
                nlist = aabb_box.query(
                    part_data.position,
                    {'num_neighbors': 12,
                     'exclude_ii': True})
                Q6 = np.mean(
                    self.q6.compute(
                        (box, part_data.position),
                        nlist).particle_order)
                if self.comm:
                    return self.comm.allreduce(
                        Q6 >= self.threshold,
                        op=MPI.LOR)
                else:
                    return Q6 >= self.threshold

**You may want to split this into two examples, first without ghosts then just showing how you would adapt to use ghosts (see how long that is).**
Most of the complexity in the logic comes from ensuring that we use as much data as possible and
strive for optimal performance. By using the ghost particles, more particles local to a rank will
have at least 12 neighbors. If we did not care about this, we would not need to construct
:code:`nlist` at all, and could just pass in :code:`(box, data.particle.position)` to the
:code:`compute` method. Another simplification to the :code:`Q6Trigger` class while still using all
the system data would be to use a snapshot instead of rank-local access, but this would be
much slower.

**In this case, if you used freud with snapshots wouldn't you create a separate Q6 instance on each rank when the trigger runs? So actually multithreading in freud doesn't help you much since they'll all be competing for threads (including on processors assigned to other MPI ranks).** 

Pandas Logger Back-end
++++++++++++++++++++++

**Make sure we're consistent on hyphenation of back-end. I think the general best practice is to use "back end" when a noun (e.g. "our new back end does..."), and "back-end" when an adjective (e.g. "our back-end classes such as the trigger...").**
Here we highlight the ability to use the :code:`Logger` class to create novel back-ends
for simulation data. For this example, we will create a Pandas back-end. We will store the scalar
and string quantities in a single :code:`pandas.DataFrame` object while array-like objects will each
be stored in separate :code:`DataFrame` objects. All :code:`DataFrame` objects will be stored in a
single dictionary.

.. code-block:: python

    import pandas as pd
    from hoomd import CustomAction
    from hoomd.util import (
        dict_flatten, dict_filter, dict_map)

    def is_flag(flags):
        def func(v):
            return v[1] in flags
        return func

    def not_none(v):
        return v[0] is not None

    def hnd_2D_arrays(v):
        if v[1] in ['scalar', 'string', 'state']:
            return v
        elif len(v[0].shape) == 2:
            return {
                str(i): col
                for i, col in enumerate(v[0].T)}


    class DataFrameBackEnd(CustomAction):
        def __init__(self, logger):
            self.logger = logger

        def act(self, timestep):
            log_dict = self.logger.log()
            is_scalar = is_flag(['scalar', 'string'])
            sc = dict_flatten(dict_map(dict_filter(
                log_dict,
                lambda x: not_none(x) and is_scalar(x)),
                lambda x: x[0]))
            rem = dict_flatten(dict_map(dict_filter(
                log_dict,
                lambda x: not_none(x) \
                    and not is_scalar(x)),
                hnd_2D_arrays))

            if not hasattr(self, 'data'):
                self.data = {
                    'scalar': pd.DataFrame(
                        columns=[
                            '.'.join(k) for k in sc]),
                    'array': {
                        '.'.join(k): pd.DataFrame()
                        for k in rem}}

            sdf = pd.DataFrame(
                {'.'.join(k): v for k, v in sc.items()},
                index=[timestep])
            rdf = {'.'.join(k): pd.DataFrame(
                        v, columns=[timestep]).T
                for k,v in rem.items()}
            data = self.data
            data['scalar'] = data['scalar'].append(sdf)
            data['array'] = {
                k: v.append(rdf[k])
                for k, v in data['array'].items()}

Conclusion
----------

HOOMD-blue version 3.0 presents a Pythonic API that encourages experimentation and customization.
Through subclassing C++ classes, providing wrappers for custom actions, and exposing data in
zero-copy arrays/buffers, we allow HOOMD-blue to utilize the full potential of Python and the
scientific Python community. Our examples have shown that often this customization is easy to
implement, and only requires greater verbosity or complexity when the desired method requires
a complex or highly performant implemention.
