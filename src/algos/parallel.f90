!================================================================================!
! This file is part of crest.
!
! Copyright (C) 2022-2023  Philipp Pracht
!
! crest is free software: you can redistribute it and/or modify it under
! the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! crest is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Lesser General Public License for more details.
!
! You should have received a copy of the GNU Lesser General Public License
! along with crest.  If not, see <https://www.gnu.org/licenses/>.
!================================================================================!

!> A collection of routines to set up OMP-parallel runs of MDs and optimizations.

!========================================================================================!
!========================================================================================!
!> Interfaces to handle optional arguments
!========================================================================================!
!========================================================================================!
module parallel_interface
!*******************************************************
!* module to load an interface to the parallel routines
!* mandatory to handle any optional input arguments
!*******************************************************
  implicit none
  interface
    subroutine crest_sploop(env,nall,structures,eread,silent)
      use crest_parameters,only:wp,stdout,sep
      use crest_calculator
      use omp_lib
      use crest_data
      use strucrd
      implicit none
      type(systemdata),intent(inout) :: env
      integer,intent(in) :: nall
      type(coord),intent(inout) :: structures(nall)
      real(wp),intent(inout),optional :: eread(nall)
      logical,intent(in),optional :: silent
    end subroutine crest_sploop
  end interface

  !> crest_oloop is generic: the coord-list form is canonical (PBC-capable),
  !> the flat (nat,nall,at,xyz) form is a legacy adapter onto it.
  interface crest_oloop
    subroutine crest_oloop_struc(env,nall,structures,dump,customcalc,eread,silent)
      use crest_parameters,only:wp,stdout,sep
      use crest_calculator
      use omp_lib
      use crest_data
      use strucrd
      use optimize_module
      use iomod,only:makedir,directory_exist,remove
      implicit none
      type(systemdata),target,intent(inout) :: env
      integer,intent(in) :: nall
      type(coord),intent(inout) :: structures(nall)
      logical,intent(in) :: dump
      type(calcdata),intent(in),target,optional :: customcalc
      real(wp),intent(inout),optional :: eread(nall)
      logical,intent(in),optional :: silent
    end subroutine crest_oloop_struc

    subroutine crest_oloop_xyz(env,nat,nall,at,xyz,eread,dump,customcalc)
      use crest_parameters,only:wp
      use crest_calculator
      use crest_data
      use strucrd
      implicit none
      type(systemdata),target,intent(inout) :: env
      integer,intent(in) :: nat,nall
      integer,intent(in) :: at(nat)
      real(wp),intent(inout) :: xyz(3,nat,nall)
      real(wp),intent(inout) :: eread(nall)
      logical,intent(in) :: dump
      type(calcdata),intent(in),target,optional :: customcalc
    end subroutine crest_oloop_xyz
  end interface crest_oloop

  interface
    subroutine crest_hessloop(env,nat,nall,at,xyz,eread,gt_out,stot_out)
      use crest_parameters,only:wp,stdout,sep
      use crest_calculator
      use omp_lib
      use crest_data
      use strucrd
      use thermochem_module
      use iomod,only:makedir,directory_exist,remove
      implicit none
      type(systemdata),intent(inout) :: env
      real(wp),intent(inout) :: xyz(3,nat,nall)
      integer,intent(in)  :: at(nat)
      real(wp),intent(inout) :: eread(nall)
      integer,intent(in) :: nat,nall
      real(wp),optional,intent(out) :: gt_out(:,:)    !> (nall, nt_full)
      real(wp),optional,intent(out) :: stot_out(:,:)  !> (nall, nt_full)
    end subroutine crest_hessloop
  end interface

end module parallel_interface

!========================================================================================!
!========================================================================================!
!> Routines for concurrent singlepoint evaluations
!========================================================================================!
!========================================================================================!
subroutine crest_sploop(env,nall,structures,eread,silent)
!****************************************************************
!* subroutine crest_sploop
!* Concurrent singlepoint evaluations for a list of structures
!* passed as an array of coord objects. Each coord carries its
!* own %lat/%chrg/%uhf, so periodic and heterogeneous systems
!* are handled. xyz must be in Bohr.
!* Energies are stored in structures(i)%energy; the optional
!* eread array, if present, additionally receives them.
!* silent - suppress the progress bar (optional, default .false.)
!****************************************************************
  use crest_parameters,only:wp,stdout,sep
  use crest_calculator
  use omp_lib
  use crest_data
  use strucrd
  use iomod,only:makedir,directory_exist,remove
  use term_ui,only:progress_init,progress_update,progress_finish
  implicit none
  type(systemdata),intent(inout) :: env
  integer,intent(in) :: nall
  type(coord),intent(inout) :: structures(nall)
  real(wp),intent(inout),optional :: eread(nall)
  logical,intent(in),optional :: silent

  type(coord),allocatable :: mols(:)
  integer :: i,j,io,c,k,z,zcopy
  logical :: ex,quiet
  type(calcdata),allocatable :: calculations(:)
  real(wp) :: energy
  real(wp),allocatable :: grad(:,:)
  integer :: thread_id,vz,job
  character(len=80) :: atmp
  real(wp) :: percent,runtime
  type(timer) :: profiler
  integer :: T,Tn
  logical :: nested

!>--- check if we have any calculation settings allocated
  if (env%calc%ncalculations < 1) then
    write (stdout,*) 'no calculations allocated'
    return
  end if

!>--- silent mode? (suppress progress bar + summary printout)
  quiet = .false.
  if (present(silent)) quiet = silent

!>--- prepare calculation objects for parallelization (one per thread)
  call new_ompautoset(env,'auto_nested',nall,T,Tn)
  nested = env%omp_allow_nested
  if (.not.quiet) call ompautoset_summary(env,'singlepoints',T,Tn)

!>--- prepare objects for parallelization
  allocate (calculations(T))
  allocate (mols(T))
  do i = 1,T
    call calculations(i)%copy(env%calc)
    do j = 1,env%calc%ncalculations
      !>--- directories and io preparation
      ex = directory_exist(env%calc%calcs(j)%calcspace)
      if (.not.ex) then
        io = makedir(trim(env%calc%calcs(j)%calcspace))
      end if
      write (atmp,'(a,"_",i0)') sep,i
      calculations(i)%calcs(j)%calcspace = env%calc%calcs(j)%calcspace//trim(atmp)
      if (allocated(calculations(i)%calcs(j)%calcfile)) deallocate (calculations(i)%calcs(j)%calcfile)
      if (allocated(calculations(i)%calcs(j)%systemcall)) deallocate (calculations(i)%calcs(j)%systemcall)
      call calculations(i)%calcs(j)%printid(i,j)
    end do
    calculations(i)%pr_energies = .false.
  end do

!>--- timer initialization
  call profiler%init(1)
  call profiler%start(1)

!>--- initialize progress bar
  if (.not.quiet) then
    call progress_init(env%ps,nall,width=50,prefix=" ↳ ", &
      &                suffix="",show_time=.true.,show_eta=.false.)
    call progress_update(env%ps,0,nall)
  end if

!>--- shared variables
  c = 0  !> counter of successful evaluations
  k = 0  !> counter of total evaluations (fail+success)
  z = 0  !> counter to process structures in order (1...nall)
!>--- pre-start server-based calculators before forking OMP threads
  call preinit_mlip_parallel(calculations,T)
!>=========================================================================
!> Native MLIP (libtorch) GPU batched fast path
!>=========================================================================
!> When a single libtorch level runs on a GPU, the standard OpenMP task
!> loop is inefficient: each thread calls engrad() one structure at a time
!> and the forward passes are serialized by the C++ forward_mutex,
!> underutilizing a GPU which excels at processing many structures at once.
!>
!> Instead we pack ALL structures into one contiguous buffer and hand it
!> to the C++ bridge, which processes it in GPU batches (pipelined,
!> optionally across multiple GPUs). This bypasses the OpenMP loop.
!>
!> Trigger conditions (all required):
!>   - exactly one calculation level,
!>   - that level is jobtype%libtorch with a CUDA device selected,
!>   - all structures share nat and atomic numbers (true for TTConf
!>     candidate batches: same ligand, different torsions).
!>
!> If the conditions are unmet, execution falls through to the standard
!> per-thread OpenMP path below (also the only path for libtorch on CPU).
!>=========================================================================
  block
    use iso_c_binding, only: c_ptr, c_null_ptr
    logical :: use_batch_gpu, same_nat, same_at, all_alloc
    integer :: natb, iat
    integer :: batch_sz, ngpus, ig
    real(wp), allocatable :: all_pos(:), all_grad(:), benergies(:)
    type(c_ptr), allocatable :: gpu_handles(:)

    use_batch_gpu = .false.
    batch_sz = 0
    ngpus = 0
    if (env%calc%ncalculations == 1 .and. &
        env%calc%calcs(1)%id == jobtype%libtorch .and. &
        (env%calc%calcs(1)%libtorch_device_id > 0 .or. env%calc%mlip_batch_opt) .and. &
        nall > 0) then

      natb = structures(1)%nat
      all_alloc = .true.
      same_nat = .true.
      do i = 1,nall
        if (.not.allocated(structures(i)%xyz).or..not.allocated(structures(i)%at)) &
        &  all_alloc = .false.
        if (structures(i)%nat /= natb) same_nat = .false.
      end do
      same_at = .true.
      if (same_nat) then
        do i = 2,nall
          do iat = 1,natb
            if (structures(i)%at(iat) /= structures(1)%at(iat)) then
              same_at = .false.
              exit
            end if
          end do
          if (.not.same_at) exit
        end do
      end if

      use_batch_gpu = all_alloc .and. same_nat .and. same_at
      if (.not.use_batch_gpu .and. .not.quiet) then
        write (stdout,'(a)') ' [libtorch] non-uniform structure list (nat/atomic numbers): '// &
        & 'falling back to the per-thread path'
      end if
    end if

    if (use_batch_gpu) then
      !> determine batch size and number of GPUs
      batch_sz = env%calc%calcs(1)%mlip_batch_size
      if (batch_sz <= 0) batch_sz = mlip_auto_batch_size(natb)
      ngpus = env%calc%calcs(1)%mlip_ngpus
      if (ngpus <= 0) then
        ngpus = libtorch_get_cuda_device_count_f()
        if (ngpus > 2) ngpus = 2  !> cap at 2 for safety
      end if
      if (ngpus < 1) ngpus = 1

      !> configure the ATen thread count for the shared model
      if (env%calc%calcs(1)%mlip_aten_threads > 0) then
        call libtorch_set_threads(env%calc%calcs(1)%mlip_aten_threads)
      else
        call libtorch_set_threads(1)  !> GPU handles parallelism internally
      end if

      if (env%calc%calcs(1)%libtorch_debug) then
        write (stdout,'(a,i0,a,i0,a,i0,a)') &
          ' [libtorch] GPU pipelined mode: ', nall, &
          ' structures, batch_size=', batch_sz, ', ngpus=', ngpus, ''
      end if

      !> pack ALL positions into a contiguous buffer for the C++ bridge
      !> (Bohr; the bridge converts to Angstrom for the MACE-LAMMPS format)
      allocate (all_pos(3*natb*nall))
      allocate (all_grad(3*natb*nall),source=0.0_wp)
      allocate (benergies(nall),source=0.0_wp)
      do i = 1,nall
        all_pos((i-1)*3*natb+1:i*3*natb) = reshape(structures(i)%xyz, [3*natb])
      end do

      if (ngpus == 1) then
        !> --- single GPU: pipelined batch inference ---
        call libtorch_init_shared(env%calc%calcs(1), io)
        if (io == 0) then
          call libtorch_engrad_batch_pipeline_f(env%calc%calcs(1), &
            nall, natb, structures(1)%at, all_pos, benergies, all_grad, &
            batch_sz, io)
        end if
      else
        !> --- multi-GPU: interleaved pipelined batch inference ---
        allocate (gpu_handles(ngpus))
        io = 0
        do ig = 1, ngpus
          call libtorch_load_shared_on_device_f(env%calc%calcs(1), &
            ig-1, gpu_handles(ig), io)
          if (io /= 0) then
            write (stdout,'(a,i0)') '**ERROR** libtorch: failed to load model on CUDA:', ig-1
            exit
          end if
        end do
        if (io == 0) then
          call libtorch_engrad_batch_multigpu_f(gpu_handles, ngpus, &
            nall, natb, structures(1)%at, all_pos, &
            env%calc%calcs(1)%chrg, env%calc%calcs(1)%multiplicity - 1, &
            benergies, all_grad, &
            batch_sz, io)
        end if
        deallocate (gpu_handles)
      end if

      !> write the energies back into the structure list
      c = 0
      if (io == 0) then
        do i = 1,nall
          structures(i)%energy = benergies(i)
          c = c+1
        end do
      else
        write (stdout,'(a)') '**ERROR** libtorch GPU batched evaluation failed'
      end if
      if (present(eread)) eread(:) = benergies(:)

      deallocate (all_pos, all_grad, benergies)

      !> progress: the batch path is one monolithic call, the bar jumps
      !> from 0 to 100% once the C++ pipeline has finished
      if (.not.quiet) call progress_update(env%ps,nall,nall)

      !> release the shared model unless the user wants to keep it loaded
      !> for subsequent calls (e.g. repeated TTConf batches); in that case
      !> the model stays in the C++ registry and is released at program exit
      if (.not.env%calc%mlip_keep_loaded) then
        call libtorch_shared_cleanup()
        env%calc%calcs(1)%libtorch_handle = c_null_ptr
        env%calc%calcs(1)%libtorch_is_shared = .false.
      end if

      !> finalize progress display
      if (.not.quiet) call progress_finish(env%ps)

      !> stop timer and print summary
      call profiler%stop(1)
      if (.not.quiet) then
        percent = float(c)/float(nall)*100.0_wp
        write (atmp,'(f5.1,a)') percent,'% success)'
        write (stdout,'(">",1x,i0,a,i0,a,a)') c,' of ',nall,' structures successfully evaluated (', &
        &     trim(adjustl(atmp))
        write (atmp,'(">",1x,a,i0,a)') 'Total runtime for ',nall,' singlepoint calculations:'
        call profiler%write_timing(stdout,1,trim(atmp),.true.)
        runtime = profiler%get(1)
        write (atmp,'(f16.3,a)') runtime/real(nall,wp),' sec'
        write (stdout,'(a,a,a)') '> Corresponding to approximately ',trim(adjustl(atmp)), &
        &                       ' per processed structure'
      end if

      call profiler%clear()
      deallocate (calculations)
      if (allocated(mols)) deallocate (mols)
      return
    end if
  end block

!>--- libtorch on the standard per-thread path: informational notes
  do j = 1,env%calc%ncalculations
    if (env%calc%calcs(j)%id == jobtype%libtorch .and. &
        env%calc%calcs(j)%libtorch_device_id == 0) then
      write (stdout,'(a)') ' [libtorch] NOTE: Running MLIP on CPU. For production '// &
      & 'throughput, set device="cuda". For CPU-only work, consider method="gfn2".'
      exit
    end if
    if (env%calc%calcs(j)%id == jobtype%libtorch .and. &
        env%calc%calcs(j)%libtorch_device_id > 0 .and. T > 1) then
      write (stdout,'(a)') ' [libtorch] WARNING: GPU MLIP fell through to the '// &
      & 'per-thread OpenMP path. All forward passes are serialized by '// &
      & 'forward_mutex - this is SLOWER than the batched GPU path.'
      write (stdout,'(a)') '   (the batched path needs a single calculation '// &
      & 'level, uniform nat/atomic numbers and nall > 0)'
      exit
    end if
  end do
!>--- loop over the structures
  !$omp parallel &
  !$omp shared(env,calculations,nall,structures,c,k,z,mols,nested,Tn)
  !$omp single
  do i = 1,nall

    call initsignal()
    vz = i
    !$omp task firstprivate( vz ) private(i,j,job,energy,grad,io,thread_id,zcopy)
    call initsignal()

    !>--- OpenMP nested region threads
    if (nested) call ompmklset(Tn)

    thread_id = OMP_GET_THREAD_NUM()
    job = thread_id+1
    !>--- deep-copy this structure into the thread-local working mol
    !$omp critical
    z = z+1
    zcopy = z
    call mols(job)%copy(structures(zcopy))
    !$omp end critical

    allocate (grad(3,mols(job)%nat),source=0.0_wp)

    !>-- energy+gradient call
    call engrad(mols(job),calculations(job),energy,grad,io)

    !$omp critical
    if (io == 0) then
      c = c+1
      structures(zcopy)%energy = energy
    else
      structures(zcopy)%energy = 0.0_wp
    end if
    k = k+1
    !>--- print progress
    if (.not.quiet) call progress_update(env%ps,k,nall)
    !$omp end critical

    deallocate (grad)
    !$omp end task
  end do
  !$omp taskwait
  !$omp end single
  !$omp end parallel

!>--- energies are stored in the structures; optionally also return them
  if (present(eread)) then
    do i = 1,nall
      eread(i) = structures(i)%energy
    end do
  end if

!>--- finalize progress printout
  if (.not.quiet) call progress_finish(env%ps)

!>--- stop timer
  call profiler%stop(1)

!>--- prepare some summary printout
  if (.not.quiet) then
    percent = float(c)/float(nall)*100.0_wp
    write (atmp,'(f5.1,a)') percent,'% success)'
    write (stdout,'(">",1x,i0,a,i0,a,a)') c,' of ',nall,' structures successfully evaluated (', &
    &     trim(adjustl(atmp))
    write (atmp,'(">",1x,a,i0,a)') 'Total runtime for ',nall,' singlepoint calculations:'
    call profiler%write_timing(stdout,1,trim(atmp),.true.)
    runtime = profiler%get(1)
    write (atmp,'(f16.3,a)') runtime/real(nall,wp),' sec'
    write (stdout,'(a,a,a)') '> Corresponding to approximately ',trim(adjustl(atmp)), &
    &                       ' per processed structure'
  end if

  call profiler%clear()
  deallocate (calculations)
  if (allocated(mols)) deallocate (mols)
  return
end subroutine crest_sploop

!========================================================================================!
!========================================================================================!
!> Routines for concurrent singlepoint evaluations
!========================================================================================!
!========================================================================================!
subroutine crest_hessloop(env,nat,nall,at,xyz,eread,gt_out,stot_out)
!***************************************************************
!* subroutine crest_hessloop
!* Concurrent numerical Hessian evaluations for an ensemble.
!* Input eread is overwritten with Gibbs free energies.
!* xyz must be in Bohrs.
!* eread contains only the gt@RT on output!
!* Optional gt_out/stot_out return G and S at all temperatures
!* from env%thermo; requires pre-allocated (nall,nt) arrays.
!*
!* Parallelization is enabled using numhess1 (OpenMP-compatible).
!*
!***************************************************************
  use crest_parameters,only:wp,stdout,sep
  use crest_calculator
  use omp_lib
  use crest_data
  use strucrd
  use optimize_module
  use thermochem_module
  use iomod,only:makedir,directory_exist,remove
  use term_ui,only:progress_init,progress_update,progress_finish
  implicit none
  type(systemdata),intent(inout) :: env
  real(wp),intent(inout) :: xyz(3,nat,nall)
  integer,intent(in)  :: at(nat)
  real(wp),intent(inout) :: eread(nall)
  integer,intent(in) :: nat,nall
  real(wp),optional,intent(out) :: gt_out(:,:)    !> (nall, nt_full)
  real(wp),optional,intent(out) :: stot_out(:,:)  !> (nall, nt_full)

  type(coord),allocatable :: mols(:)
  integer :: i,j,k,l,io,ich,ich2,c,z,job_id,zcopy,nat3
  logical :: pr,wr,ex
  type(calcdata),allocatable :: calculations(:)
  real(wp) :: energy,gnorm
  real(wp),allocatable :: grad(:,:),grads(:,:,:)
  real(wp),allocatable :: freqs(:,:),hess(:,:,:)
  integer :: thread_id,vz,job
  character(len=80) :: atmp
  real(wp) :: percent,runtime

  integer :: nt,nrt
  real(wp),allocatable :: temps(:,:),et(:,:),ht(:,:),gt(:,:),stot(:,:)
  real(wp) :: ithr,sthr,fscal,rt
  character(len=:),allocatable :: emodel

  type(timer) :: profiler
  integer :: T,Tn  !> threads and threads per core
  logical :: nested
  real(wp),parameter :: big = 10e10

!>--- check if we have any calculation settings allocated
  if (env%calc%ncalculations < 1) then
    write (stdout,*) 'no calculations allocated'
    return
  end if

!>--- prepare calculation objects for parallelization (one per thread)
  call new_ompautoset(env,'auto_nested',nall,T,Tn)
  nested = env%omp_allow_nested
  call ompautoset_summary(env,'Hessians',T,Tn)

!>--- prepare objects for parallelization (one working copy per parallel job)
  allocate (calculations(T))!,source=env%calc)
  allocate (mols(T))
  nat3 = nat*3
  allocate (freqs(nat3,T),source=0.0_wp)
  allocate (hess(nat3,nat3,T),source=0.0_wp)
  do i = 1,T
    call calculations(i)%copy(env%calc)
    do j = 1,env%calc%ncalculations
      !calculations(i)%calcs(j) = env%calc%calcs(j)
      !>--- directories and io preparation
      ex = directory_exist(env%calc%calcs(j)%calcspace)
      if (.not.ex) then
        io = makedir(trim(env%calc%calcs(j)%calcspace))
      end if
      write (atmp,'(a,"_",i0)') sep,i
      calculations(i)%calcs(j)%calcspace = env%calc%calcs(j)%calcspace//trim(atmp)
      if (allocated(calculations(i)%calcs(j)%calcfile)) deallocate (calculations(i)%calcs(j)%calcfile)
      if (allocated(calculations(i)%calcs(j)%systemcall)) deallocate (calculations(i)%calcs(j)%systemcall)
      call calculations(i)%calcs(j)%printid(i,j)
    end do
    calculations(i)%pr_energies = .false.
    allocate (mols(i)%at(nat),mols(i)%xyz(3,nat))
  end do

!>--- thermo settings
  !> inversion threshold
  ithr = env%thermo%ithr
  !> frequency scaling factor
  fscal = env%thermo%fscal
  !> RR-HO interpolation (or cut-off)
  sthr = env%thermo%sthr
  !> Svib model
  emodel = env%thermo%emodel
  if (.not.allocated(env%thermo%temps)) then
    call env%thermo%get_temps()
  end if
  if (present(gt_out)) then
    nt = env%thermo%ntemps
    allocate (temps(nt,T),et(nt,T),ht(nt,T),gt(nt,T),stot(nt,T),source=0.0_wp)
    do i = 1,T
      temps(:,i) = env%thermo%temps(:)
    end do
    rt = env%thermo%get_close_rt(nrt)
  else
    nt = 1
    allocate (temps(nt,T),et(nt,T),ht(nt,T),gt(nt,T),stot(nt,T),source=0.0_wp)
    rt = env%thermo%get_close_rt(nrt)
    temps = rt
    nrt = 1
  end if

!>--- printout directions and timer initialization
  pr = .false. !> stdout printout
  wr = .false. !> write crestopt.log.xyz
  call profiler%init(1)
  call profiler%start(1)

!>--- initialize progress bar
  call progress_init(env%ps,nall,width=50,prefix=" ↳ ", &
    &                suffix="",show_time=.true.,show_eta=.false.)
  call progress_update(env%ps,0,nall)

!>--- shared variables
  allocate (grads(3,nat,T),source=0.0_wp)
  c = 0  !> counter of successfull optimizations
  k = 0  !> counter of total optimization (fail+success)
  z = 0  !> counter to perform optimization in right order (1...nall)
  eread(:) = 0.0_wp
  grads(:,:,:) = 0.0_wp
!>--- pre-start server-based calculators before forking OMP threads
  call preinit_mlip_parallel(calculations,T)
!>--- loop over ensemble
  !$omp parallel &
  !$omp shared(env,calculations,nat,nall,at,xyz,eread,grads,c,k,z,pr,wr,nrt) &
  !$omp shared(mols,nested,Tn,freqs,hess,temps,et,ht,gt,stot,nat3,ithr,fscal,sthr,nt,emodel)
  !$omp single
  do i = 1,nall

    call initsignal()
    vz = i
    !$omp task firstprivate( vz ) private(i,j,job,energy,io,thread_id,zcopy)
    call initsignal()

    !>--- OpenMP nested region threads
    if (nested) call ompmklset(Tn)

    thread_id = OMP_GET_THREAD_NUM()
    job = thread_id+1
    !>--- modify calculation spaces
    !$omp critical
    z = z+1
    zcopy = z
    mols(job)%nat = nat
    mols(job)%at(:) = at(:)
    mols(job)%xyz(:,:) = xyz(:,:,z)
    !$omp end critical

    !>-- engery+gradient call first, for setup
    call engrad(mols(job),calculations(job),energy,grads(:,:,job),io)
    !>-- then, numerical hessian
    call numhess1(mols(job)%nat,mols(job)%at,mols(job)%xyz, &
                  calculations(job),hess(:,:,job),io)
    !!$omp critical
    if (io .eq. 0) then
      call prj_mw_hess(mols(job)%nat,mols(job)%at,nat3,mols(job)%xyz,hess(:,:,job))
      !>-- Computes the Frequencies
      call frequencies(mols(job)%nat,mols(job)%at,mols(job)%xyz, &
                       nat3,hess(:,:,job),freqs(:,job),io)
    end if

    if (io .eq. 0) then
      call calcthermo(mols(job)%nat,mols(job)%at,mols(job)%xyz,   &
                      freqs(:,job),.false.,ithr,fscal,sthr,nt,    &
                      temps(:,job),et(:,job),ht(:,job),gt(:,job), &
                      stot(:,job),emodel=emodel)
    end if
    !!$omp end critical

    !$omp critical
    if (io == 0) then
      c = c+1
      eread(zcopy) = gt(nrt,job)
      if (present(gt_out))   gt_out(zcopy,:)   = gt(:,job)
      if (present(stot_out)) stot_out(zcopy,:) = stot(:,job)
    else
      eread(zcopy) = big
    end if
    k = k+1
    call progress_update(env%ps,k,nall)
    !$omp end critical
    !$omp end task
  end do
  !$omp taskwait
  !$omp end single
  !$omp end parallel

!>--- finalize progress printout
  call progress_finish(env%ps)

!>--- stop timer
  call profiler%stop(1)

!>--- prepare some summary printout
  percent = float(c)/float(nall)*100.0_wp
  write (atmp,'(f5.1,a)') percent,'% success)'
  write (stdout,'(">",1x,i0,a,i0,a,a)') c,' of ',nall,' structures successfully evaluated (', &
  &     trim(adjustl(atmp))
  write (atmp,'(">",1x,a,i0,a)') 'Total runtime for ',nall,' frequency calculations:'
  call profiler%write_timing(stdout,1,trim(atmp),.true.)
  runtime = profiler%get(1)
  write (atmp,'(f16.3,a)') runtime/real(nall,wp),' sec'
  write (stdout,'(a,a,a)') '> Corresponding to approximately ',trim(adjustl(atmp)), &
  &                       ' per processed structure'

  deallocate (grads)
  call profiler%clear()
  deallocate (calculations)
  if (allocated(mols)) deallocate (mols)
  if (allocated(freqs)) deallocate (freqs)
  if (allocated(hess)) deallocate (hess)
  return
end subroutine crest_hessloop

!========================================================================================!
!========================================================================================!
!> Routines for concurrent geometry optimization
!========================================================================================!
!========================================================================================!
subroutine mlip_batch_oloop(env,mycalc,calculations,nall,structures,dump,ich,ich2,quiet)
!*******************************************************************************
!* subroutine mlip_batch_oloop
!* Phase C: batched native MLIP (libtorch) geometry optimization for a uniform
!* list of structures.
!*
!* Keeps N independent L-BFGS optimizer states (one per structure) and issues
!* exactly ONE batched energy+gradient evaluation per outer iteration; the
!* C++ bridge processes the packed buffer in GPU batches (pipelined, optionally
!* across several GPUs). On CPU the batched call degrades to sequential
!* single-structure forwards of one shared model instance.
!*
!* The per-structure algorithm mirrors lbfgs_module: fixed base step 0.2 with
!* 0.25 backtracking on energy rise, limited-memory BFGS history of length
!* lbfgs_histsize, and the E/G convergence thresholds of get_optthr. The
!* line search is "amortized": a rejected trial step is retried in the NEXT
!* batched call (per-structure state machine), so every outer iteration costs
!* exactly one batched E+G call no matter how many structures are active.
!*
!* Conventions are identical to crest_oloop_struc: converged geometries and
!* energies are written back into structures; failed optimizations receive an
!* energy of +1.0 (unless anopt is set). xyz is in Bohr(!).
!******************************************************************************
  use crest_parameters,only:wp,stdout
  use crest_calculator
  use crest_data
  use strucrd
  use optimize_type,only:optimizer
  use optimize_utils,only:get_optthr
  use term_ui,only:progress_init,progress_update,progress_finish
  use iso_c_binding,only:c_ptr,c_null_ptr
  implicit none
  type(systemdata),intent(inout) :: env
  type(calcdata),intent(inout) :: mycalc
  type(calcdata),intent(inout) :: calculations(:)
  integer,intent(in) :: nall
  type(coord),intent(inout) :: structures(nall)
  logical,intent(in) :: dump
  integer,intent(in) :: ich,ich2
  logical,intent(in) :: quiet

  !> per-structure L-BFGS state
  type(optimizer),allocatable :: OPT(:)
  real(wp),allocatable :: xacc(:,:),gcand(:,:),xnew(:,:),dirc(:,:),gacc(:,:),gnew(:)
  real(wp),allocatable :: eacc(:),stepz(:),qtmp(:)
  integer,allocatable :: khist(:),iterc(:),retry(:)
  logical,allocatable :: active(:),pending(:)
  !> batch buffers
  real(wp),allocatable :: all_pos(:),all_grad(:),benergies(:)
  type(c_ptr),allocatable :: gpu_handles(:)
  !> settings / bookkeeping
  integer :: natb,nvarb,mhist,maxcycle_i,batch_sz,ngpus,ig,io
  integer :: outer,nact,p,done_count,i,j,c,k,tight
  integer(8) :: tc0,tc1,tcrate
  real(wp) :: ethr,gthr,maxerise,deltaE,gnorm,gamm,yy,ss
  logical :: econv,gconv
  character(len=80) :: atmp
  real(wp) :: percent,runtime
  type(timer) :: profiler
  type(coord) :: molnew

  natb = structures(1)%nat
  nvarb = 3*natb
  mhist = mycalc%lbfgs_histsize

!>--- convergence thresholds (same as lbfgs_module; tight is a scratch copy
!>--- because get_optthr may rewrite it and auto-set maxcycle on mycalc)
  tight = mycalc%optlev
  call get_optthr(natb,tight,mycalc,ethr,gthr)
  maxcycle_i = mycalc%maxcycle
  maxerise = mycalc%maxerise

!>--- allocate per-structure optimizer state
  allocate (OPT(nall))
  allocate (xacc(nvarb,nall),gcand(nvarb,nall),xnew(nvarb,nall),dirc(nvarb,nall))
  allocate (gacc(nvarb,nall),gnew(nvarb))
  allocate (eacc(nall),stepz(nall),qtmp(nvarb))
  allocate (khist(nall),iterc(nall),retry(nall))
  allocate (active(nall),pending(nall))

  do i = 1,nall
    call OPT(i)%allocatelbfgs(nvarb,mhist)
    OPT(i)%S = 0.0_wp
    OPT(i)%Y = 0.0_wp
    OPT(i)%rho = 0.0_wp
    xacc(:,i) = reshape(structures(i)%xyz,[nvarb])
    gcand(:,i) = xacc(:,i)  !> first batch call evaluates the start geometry
    eacc(i) = 0.0_wp
    stepz(i) = 0.2_wp
    khist(i) = 0
    iterc(i) = 0
    retry(i) = 0
    pending(i) = .false.
    active(i) = .true.
  end do

!>--- batch settings (same logic as the sploop GPU fast path)
  batch_sz = mycalc%calcs(1)%mlip_batch_size
  if (batch_sz <= 0) batch_sz = mlip_auto_batch_size(natb)
  ngpus = mycalc%calcs(1)%mlip_ngpus
  if (ngpus <= 0) then
    ngpus = libtorch_get_cuda_device_count_f()
    if (ngpus > 2) ngpus = 2  !> cap at 2 for safety
  end if
  if (ngpus < 1) ngpus = 1

  if (mycalc%calcs(1)%mlip_aten_threads > 0) then
    call libtorch_set_threads(mycalc%calcs(1)%mlip_aten_threads)
  else
    call libtorch_set_threads(1)  !> the batched driver is single-threaded here
  end if

  if (mycalc%calcs(1)%libtorch_debug) then
    write (stdout,'(a,i0,a,i0,a,i0,a)') &
      ' [libtorch] batched optimizer: ', nall, &
      ' structures, batch_size=', batch_sz, ', ngpus=', ngpus, ''
    flush (stdout)
  end if

!>--- batch buffers (sized for the full list; the active subset is repacked
!>--- each iteration)
  allocate (all_pos(3*natb*nall))
  allocate (all_grad(3*natb*nall),source=0.0_wp)
  allocate (benergies(nall),source=0.0_wp)

!>--- load the shared model ONCE; it is reused across all outer iterations
  io = 0
  if (mycalc%calcs(1)%libtorch_device_id > 0 .and. ngpus > 1) then
    allocate (gpu_handles(ngpus))
    do ig = 1,ngpus
      call libtorch_load_shared_on_device_f(mycalc%calcs(1),ig-1,gpu_handles(ig),io)
      if (io /= 0) then
        write (stdout,'(a,i0)') '**ERROR** libtorch: failed to load model on CUDA:',ig-1
        exit
      end if
    end do
  else
    call libtorch_init_shared(mycalc%calcs(1),io)
    if (io /= 0) then
      write (stdout,'(a)') '**ERROR** libtorch: failed to load shared model for the batched optimizer'
    end if
  end if
  if (io /= 0) then
    do i = 1,nall
      structures(i)%energy = 1.0_wp
    end do
    return
  end if

!>--- ensure etmp is allocated so calc_eprint can be used for the dump files
  if (.not.allocated(calculations(1)%etmp)) &
    allocate (calculations(1)%etmp(mycalc%ncalculations),source=0.0_wp)

  call profiler%init(1)
  call profiler%start(1)
  if (.not.quiet) then
    call progress_init(env%ps,nall,width=50,prefix=" ↳ ", &
      &                suffix="",show_time=.true.,show_eta=.false.)
    call progress_update(env%ps,0,nall)
  end if

!>--- main loop: exactly ONE batched E+G evaluation per outer iteration
  c = 0
  k = 0
  outer = 0
  do while (count(active) > 0 .and. outer < maxcycle_i*13 .and. io == 0)
    outer = outer+1
    nact = count(active)

!>--- pack the current candidate geometries of all active structures (Bohr)
    p = 0
    do i = 1,nall
      if (active(i)) then
        p = p+1
        all_pos((p-1)*nvarb+1:p*nvarb) = gcand(:,i)
      end if
    end do

!>--- the single batched E+G evaluation of this iteration
    if (mycalc%calcs(1)%libtorch_debug) call system_clock(tc0,tcrate)
    if (ngpus > 1 .and. mycalc%calcs(1)%libtorch_device_id > 0) then
      call libtorch_engrad_batch_multigpu_f(gpu_handles,ngpus,nact,natb, &
        structures(1)%at,all_pos, &
        mycalc%calcs(1)%chrg, mycalc%calcs(1)%multiplicity - 1, &
        benergies,all_grad,batch_sz,io)
    else
      call libtorch_engrad_batch_pipeline_f(mycalc%calcs(1),nact,natb, &
        structures(1)%at,all_pos,benergies,all_grad,batch_sz,io)
    end if
    if (mycalc%calcs(1)%libtorch_debug) then
      call system_clock(tc1,tcrate)
      write (stdout,'(a,i0,a,i0,a,f8.2,a)') ' [libtorch batch] iter=',outer, &
        ' nact=',nact,'  t=',real(tc1-tc0,wp)/real(tcrate,wp)*1000.0_wp,'ms'
      flush (stdout)
    end if
    if (io /= 0) then
      write (stdout,'(a)') '**ERROR** libtorch batched optimizer: batched E+G call failed'
      exit
    end if

!>--- per-structure L-BFGS state update
    p = 0
    do i = 1,nall
      if (.not.active(i)) cycle
      p = p+1
      gnew = all_grad((p-1)*nvarb+1:p*nvarb)
      gnorm = sqrt(dot_product(gnew,gnew))

      if (.not.pending(i)) then
!>--- the evaluation was at the current (accepted) point: compute the search
!>--- direction and propose the next trial step (to be evaluated in the
!>--- following batch)
        gacc(:,i) = gnew
        eacc(i) = benergies(p)
        iterc(i) = iterc(i)+1
        if (khist(i) == 0) then
          dirc(:,i) = -gnew
        else
          yy = dot_product(OPT(i)%Y(1:nvarb,khist(i)),OPT(i)%Y(1:nvarb,khist(i)))
          if (yy > 1.0d-30) then
            gamm = dot_product(OPT(i)%S(1:nvarb,khist(i)),OPT(i)%Y(1:nvarb,khist(i)))/yy
          else
            gamm = 1.0_wp
          end if
          qtmp = gnew
          do j = khist(i),1,-1
            OPT(i)%alpha(j) = OPT(i)%rho(j)*dot_product(OPT(i)%S(1:nvarb,j),qtmp)
            qtmp = qtmp-OPT(i)%alpha(j)*OPT(i)%Y(1:nvarb,j)
          end do
          xnew(:,i) = gamm*qtmp
          do j = 1,khist(i)
            xnew(:,i) = xnew(:,i)+OPT(i)%S(1:nvarb,j)*(OPT(i)%alpha(j)- &
              & OPT(i)%rho(j)*dot_product(OPT(i)%Y(1:nvarb,j),xnew(:,i)))
          end do
          dirc(:,i) = -xnew(:,i)
        end if
        stepz(i) = 0.2_wp
        retry(i) = 0
        gcand(:,i) = xacc(:,i)+stepz(i)*dirc(:,i)
        pending(i) = .true.
        cycle
      end if

!>--- the evaluation was at a trial point gcand = xacc + stepz*dirc
      deltaE = benergies(p)-eacc(i)
      if (deltaE > maxerise) then
!>--- energy rise: shrink the step, retry in the next batched call
        retry(i) = retry(i)+1
        iterc(i) = iterc(i)+1  !> count rejected evaluations toward the
                                !> per-structure budget; otherwise a stuck
                                !> structure burns the whole global cap
                                !> (13*maxcycle) and may exit the loop with
                                !> xyz/energy never updated
        if (iterc(i) >= maxcycle_i) then
          active(i) = .false.
          if (mycalc%anopt) then
            c = c+1
            structures(i)%xyz = reshape(xacc(:,i),[3,natb])
            structures(i)%energy = eacc(i)
          else
            structures(i)%energy = 1.0_wp
          end if
          cycle
        end if
        if (retry(i) > 12) then
!>--- stuck: restart from the steepest descent direction with a full step
          dirc(:,i) = -gnew
          khist(i) = 0
          retry(i) = 0
          stepz(i) = 0.2_wp
        end if
        stepz(i) = stepz(i)*0.25_wp
        gcand(:,i) = xacc(:,i)+stepz(i)*dirc(:,i)
        cycle
      end if

!>--- accept the step and update the L-BFGS history
      xnew(:,i) = gcand(:,i)-xacc(:,i)
      ss = dot_product(gnew-gacc(:,i),xnew(:,i))
      if (abs(ss) > 1.0d-30) then
        if (khist(i) < mhist) then
          khist(i) = khist(i)+1
          OPT(i)%S(1:nvarb,khist(i)) = xnew(:,i)
          OPT(i)%Y(1:nvarb,khist(i)) = gnew-gacc(:,i)
        else
          OPT(i)%S(1:nvarb,1:mhist-1) = OPT(i)%S(1:nvarb,2:mhist)
          OPT(i)%Y(1:nvarb,1:mhist-1) = OPT(i)%Y(1:nvarb,2:mhist)
          OPT(i)%S(1:nvarb,mhist) = xnew(:,i)
          OPT(i)%Y(1:nvarb,mhist) = gnew-gacc(:,i)
        end if
        OPT(i)%rho(khist(i)) = 1.0_wp/ss
      end if
      xacc(:,i) = gcand(:,i)
      eacc(i) = benergies(p)
      pending(i) = .false.

!>--- convergence bookkeeping (identical criteria to lbfgs_module)
      econv = abs(deltaE) .lt. ethr
      gconv = gnorm .lt. gthr
      if (econv .and. gconv) then
        active(i) = .false.
        c = c+1
        structures(i)%xyz = reshape(xacc(:,i),[3,natb])
        structures(i)%energy = eacc(i)
        if (dump) then
          molnew%nat = natb
          molnew%at = structures(i)%at
          molnew%xyz = structures(i)%xyz
          molnew%energy = eacc(i)  !> appendcoord prefixes " energy= <self%energy>";
                                    !> keep it the real value so grepenergy (CREGEN
                                    !> sorting) does not read a stale/zero energy
          write (atmp,'(1x,"energy=",f16.10,1x,"g norm=",f12.8)') eacc(i),gnorm
          molnew%comment = trim(atmp)
          call molnew%append(ich)
          calculations(1)%etmp(1) = eacc(i)
          call calc_eprint(calculations(1),eacc(i),calculations(1)%etmp,gnorm,ich2)
        end if
      else if (iterc(i) >= maxcycle_i) then
        active(i) = .false.
        if (mycalc%anopt) then
          c = c+1
          structures(i)%xyz = reshape(xacc(:,i),[3,natb])
          structures(i)%energy = eacc(i)
        else
          structures(i)%energy = 1.0_wp
        end if
      end if
    end do

    k = k+1
    if (.not.quiet) then
      done_count = nall-count(active)
      call progress_update(env%ps,done_count,nall)
    end if
  end do

!>--- structures still active after a failed batch call: failure energy
  if (io /= 0) then
    do i = 1,nall
      if (active(i)) structures(i)%energy = 1.0_wp
    end do
  end if

!>--- finalize progress display
  if (.not.quiet) call progress_finish(env%ps)

!>--- stop timer and print summary (mirrors crest_oloop_struc)
  call profiler%stop(1)
  if (.not.quiet) then
    percent = float(c)/float(nall)*100.0_wp
    write (atmp,'(f5.1,a)') percent,'% success)'
    write (stdout,'(">",1x,i0,a,i0,a,a)') c,' of ',nall,' structures successfully optimized (', &
    &     trim(adjustl(atmp))
    write (atmp,'(">",1x,a,i0,a)') 'Total runtime for ',nall,' optimizations:'
    call profiler%write_timing(stdout,1,trim(atmp),.true.)
    runtime = profiler%get(1)
    write (atmp,'(f16.3,a)') runtime/real(nall,wp),' sec'
    write (stdout,'(a,a,a)') '> Corresponding to approximately ',trim(adjustl(atmp)), &
    &                       ' per processed structure'
  end if
  call profiler%clear()

!>--- release the shared model unless the user wants to keep it loaded for
!>--- subsequent calls (e.g. repeated TTConf batches); otherwise it stays in
!>--- the C++ registry until program exit (cleanup.f90 releases it)
  if (.not.mycalc%mlip_keep_loaded) then
    call libtorch_shared_cleanup()
    mycalc%calcs(1)%libtorch_handle = c_null_ptr
    mycalc%calcs(1)%libtorch_is_shared = .false.
  end if

!>--- deallocate
  deallocate (OPT,xacc,gcand,xnew,dirc,gacc,gnew,eacc,stepz,qtmp)
  deallocate (khist,iterc,retry,active,pending)
  deallocate (all_pos,all_grad,benergies)
  if (allocated(gpu_handles)) deallocate (gpu_handles)
  return
end subroutine mlip_batch_oloop

!========================================================================================!
subroutine crest_oloop_struc(env,nall,structures,dump,customcalc,eread,silent)
!*******************************************************************************
!* subroutine crest_oloop_struc
!* Concurrent geometry optimizations for a list of coord objects.
!* Optimized geometries and energies are written back into structures;
!* the optional eread array, if present, additionally receives the energies.
!* Each coord carries its own %lat/%chrg/%uhf, so periodic and heterogeneous
!* systems are handled.
!*
!* dump       - dump an ensemble file (NOT in the input order)
!* customcalc - customized (optional) calculation level data
!* silent     - suppress the progress bar and summary printout (optional,
!*              default .false.); used when the caller drives many small
!*              batches and prints its own progress (e.g. TTConf-light)
!*
!* IMPORTANT: structures xyz must be in Bohr(!)
!******************************************************************************
  use crest_parameters,only:wp,stdout,sep
  use crest_calculator
  use omp_lib
  use crest_data
  use strucrd
  use optimize_module
  use iomod,only:makedir,directory_exist,remove
  use term_ui,only:progress_init,progress_update,progress_finish
  implicit none
  type(systemdata),target,intent(inout) :: env
  integer,intent(in) :: nall
  type(coord),intent(inout) :: structures(nall)
  logical,intent(in) :: dump
  type(calcdata),intent(in),target,optional :: customcalc
  real(wp),intent(inout),optional :: eread(nall)
  logical,intent(in),optional :: silent
  logical :: quiet

  type(coord),allocatable :: mols(:)
  type(coord),allocatable :: molsnew(:)
  integer :: i,j,io,ich,ich2,c,k,z,zcopy
  logical :: pr,wr,ex
  type(calcdata),allocatable :: calculations(:)
  real(wp) :: energy,gnorm
  real(wp),allocatable :: grad(:,:)
  integer :: thread_id,vz,job
  character(len=80) :: atmp
  real(wp) :: percent,runtime
  type(calcdata),pointer :: mycalc
  type(timer) :: profiler
  integer :: T,Tn  !> threads and threads per core
  logical :: nested
  interface
    subroutine mlip_batch_oloop(env,mycalc,calculations,nall,structures,dump,ich,ich2,quiet)
      use crest_parameters,only:wp,stdout
      use crest_calculator
      use crest_data
      use strucrd
      implicit none
      type(systemdata),intent(inout) :: env
      type(calcdata),intent(inout) :: mycalc
      type(calcdata),intent(inout) :: calculations(:)
      integer,intent(in) :: nall
      type(coord),intent(inout) :: structures(nall)
      logical,intent(in) :: dump
      integer,intent(in) :: ich,ich2
      logical,intent(in) :: quiet
    end subroutine mlip_batch_oloop
  end interface

!>--- check which calc to use
  if (present(customcalc)) then
    mycalc => customcalc
  else
    mycalc => env%calc
  end if

!>--- silent mode? (suppress progress bar + summary printout)
  quiet = .false.
  if (present(silent)) quiet = silent

!>--- check if we have any calculation settings allocated
  if (mycalc%ncalculations < 1) then
    write (stdout,*) 'no calculations allocated'
    return
  end if

!>--- prepare calculation objects for parallelization (one per thread)
  call new_ompautoset(env,'auto_nested',nall,T,Tn)
  nested = env%omp_allow_nested
  if (.not.quiet) call ompautoset_summary(env,'optimizations',T,Tn)

!>--- prepare objects for parallelization
  allocate (calculations(T))
  allocate (mols(T),molsnew(T))
  do i = 1,T
    call calculations(i)%copy(mycalc)
    do j = 1,mycalc%ncalculations
      !>--- directories and io preparation
      ex = directory_exist(mycalc%calcs(j)%calcspace)
      if (.not.ex) then
        io = makedir(trim(mycalc%calcs(j)%calcspace))
      end if
      if (calculations(i)%calcs(j)%id == jobtype%tblite) then
        calculations(i)%optnewinit = .true.
      end if
      write (atmp,'(a,"_",i0)') sep,i
      calculations(i)%calcs(j)%calcspace = mycalc%calcs(j)%calcspace//trim(atmp)
      if (allocated(calculations(i)%calcs(j)%calcfile)) deallocate (calculations(i)%calcs(j)%calcfile)
      if (allocated(calculations(i)%calcs(j)%systemcall)) deallocate (calculations(i)%calcs(j)%systemcall)
      call calculations(i)%calcs(j)%printid(i,j)
    end do
    calculations(i)%pr_energies = .false.
  end do

!>--- printout directions and timer initialization
  pr = .false. !> stdout printout
  wr = .false. !> write crestopt.log.xyz
  if (dump) then
    open (newunit=ich,file=ensemblefile)
    open (newunit=ich2,file=ensembleelog)
  end if
  call profiler%init(1)
  call profiler%start(1)

!>--- initialize progress bar
  if (.not.quiet) then
    call progress_init(env%ps,nall,width=50,prefix=" ↳ ", &
      &                suffix="",show_time=.true.,show_eta=.false.)
    call progress_update(env%ps,0,nall)
  end if

!>--- shared variables
  c = 0  !> counter of successfull optimizations
  k = 0  !> counter of total optimization (fail+success)
  z = 0  !> counter to perform optimization in right order (1...nall)
!>--- pre-start server-based calculators before forking OMP threads
  call preinit_mlip_parallel(calculations,T)
!>=========================================================================
!> Phase C: native MLIP (libtorch) batched optimizer fast path
!>=========================================================================
!> For a uniform structure list on a single native MLIP level, the batched
!> L-BFGS driver keeps one optimizer state per structure and issues exactly
!> ONE batched E+G call per outer iteration (GPU-batched and pipelined in
!> the C++ bridge). Trigger: one libtorch level, uniform nat/atomic numbers,
!> and a GPU device OR an explicit mlip_batch_opt request. Otherwise
!> execution falls through to the standard per-thread OpenMP path below.
!>=========================================================================
  block
    logical :: use_batch, same_nat, same_at, all_alloc
    integer :: natb2, iat
    use_batch = .false.
    if (mycalc%ncalculations == 1 .and. &
        mycalc%calcs(1)%id == jobtype%libtorch .and. nall > 0) then
      natb2 = structures(1)%nat
      all_alloc = .true.
      same_nat = .true.
      do i = 1,nall
        if (.not.allocated(structures(i)%xyz).or..not.allocated(structures(i)%at)) &
        &  all_alloc = .false.
        if (structures(i)%nat /= natb2) same_nat = .false.
      end do
      same_at = .true.
      if (same_nat) then
        do i = 2,nall
          do iat = 1,natb2
            if (structures(i)%at(iat) /= structures(1)%at(iat)) then
              same_at = .false.
              exit
            end if
          end do
          if (.not.same_at) exit
        end do
      end if
      use_batch = all_alloc .and. same_nat .and. same_at .and. &
        & (mycalc%calcs(1)%libtorch_device_id > 0 .or. mycalc%mlip_batch_opt)
      if (.not.use_batch .and. .not.quiet) then
        write (stdout,'(a)') ' [libtorch] non-uniform structure list (nat/atomic numbers): '// &
        & 'falling back to the per-thread optimization path'
      end if
    end if
    if (use_batch) then
      call mlip_batch_oloop(env,mycalc,calculations,nall,structures,dump,ich,ich2,quiet)
      if (present(eread)) then
        do i = 1,nall
          eread(i) = structures(i)%energy
        end do
      end if
!>--- close the dump units before the early return; otherwise the Fortran
!>--- buffers are never flushed and the ensemble file is truncated on disk
!>--- (fd leak: units stay open until program exit)
      if (dump) then
        close (ich)
        close (ich2)
      end if
      return
    end if
  end block
!>--- loop over ensemble
  !$omp parallel &
  !$omp shared(env,calculations,nall,structures,c,k,z,pr,wr,dump) &
  !$omp shared(ich,ich2,mols,molsnew,nested,Tn)
  !$omp single
  do i = 1,nall

    call initsignal()
    vz = i
    !$omp task firstprivate( vz ) private(j,job,energy,grad,io,atmp,gnorm,thread_id,zcopy)
    call initsignal()

    !>--- OpenMP nested region threads
    if (nested) call ompmklset(Tn)

    thread_id = OMP_GET_THREAD_NUM()
    job = thread_id+1
    !>--- deep-copy this structure into the thread-local working mol
    !$omp critical
    z = z+1
    zcopy = z
    call mols(job)%copy(structures(zcopy))
    !$omp end critical

    allocate (grad(3,mols(job)%nat),source=0.0_wp)

    !>-- geometry optimization
    call optimize_geometry(mols(job),molsnew(job),calculations(job),energy,grad,pr,wr,io)

    !$omp critical
    if (io == 0) then
      !>--- successful optimization (io==0)
      c = c+1
      structures(zcopy)%xyz = molsnew(job)%xyz
      structures(zcopy)%energy = energy
      if (dump) then
        gnorm = norm2(grad)
        write (atmp,'(1x,"energy=",f16.10,1x,"g norm=",f12.8)') energy,gnorm
        molsnew(job)%comment = trim(atmp)
        call molsnew(job)%append(ich)
        call calc_eprint(calculations(job),energy,calculations(job)%etmp,gnorm,ich2)
      end if
    else if (io == calculations(job)%maxcycle.and.calculations(job)%anopt) then
      !>--- allow partial optimization?
      c = c+1
      structures(zcopy)%xyz = molsnew(job)%xyz
      structures(zcopy)%energy = energy
    else
      structures(zcopy)%energy = 1.0_wp
    end if
    k = k+1
    !>--- print progress
    if (.not.quiet) call progress_update(env%ps,k,nall)
    !$omp end critical

    deallocate (grad)
    !$omp end task
  end do
  !$omp taskwait
  !$omp end single
  !$omp end parallel

!>--- finalize progress printout
  if (.not.quiet) call progress_finish(env%ps)

!>--- stop timer
  call profiler%stop(1)

!>--- energies are stored in the structures; optionally also return them
  if (present(eread)) then
    do i = 1,nall
      eread(i) = structures(i)%energy
    end do
  end if

!>--- prepare some summary printout
  if (.not.quiet) then
    percent = float(c)/float(nall)*100.0_wp
    write (atmp,'(f5.1,a)') percent,'% success)'
    write (stdout,'(">",1x,i0,a,i0,a,a)') c,' of ',nall,' structures successfully optimized (', &
    &     trim(adjustl(atmp))
    write (atmp,'(">",1x,a,i0,a)') 'Total runtime for ',nall,' optimizations:'
    call profiler%write_timing(stdout,1,trim(atmp),.true.)
    runtime = profiler%get(1)
    write (atmp,'(f16.3,a)') runtime/real(nall,wp),' sec'
    write (stdout,'(a,a,a)') '> Corresponding to approximately ',trim(adjustl(atmp)), &
    &                       ' per processed structure'
  end if

!>--- close files (if they are open)
  if (dump) then
    close (ich)
    close (ich2)
  end if

  call profiler%clear()
  deallocate (calculations)
  if (allocated(mols)) deallocate (mols)
  if (allocated(molsnew)) deallocate (molsnew)
  return
end subroutine crest_oloop_struc

!========================================================================================!
!> Flat-array adapter for crest_oloop. Marshals the (nat,nall,at,xyz) ensemble
!> into a coord list, runs the optimization loop, and copies the optimized
!> coordinates and energies back. Kept for the legacy (non-periodic) callers;
!> new code should use the coord-list crest_oloop directly.
!========================================================================================!
subroutine crest_oloop_xyz(env,nat,nall,at,xyz,eread,dump,customcalc)
  use crest_parameters,only:wp
  use crest_calculator
  use crest_data
  use strucrd
  use parallel_interface,only:crest_oloop
  implicit none
  type(systemdata),target,intent(inout) :: env
  integer,intent(in) :: nat,nall
  integer,intent(in) :: at(nat)
  real(wp),intent(inout) :: xyz(3,nat,nall)
  real(wp),intent(inout) :: eread(nall)
  logical,intent(in) :: dump
  type(calcdata),intent(in),target,optional :: customcalc

  type(coord),allocatable :: structures(:)
  integer :: i

  allocate (structures(nall))
  do i = 1,nall
    structures(i)%nat = nat
    structures(i)%at = at
    structures(i)%xyz = xyz(1:3,1:nat,i)
  end do

  if (present(customcalc)) then
    call crest_oloop(env,nall,structures,dump,customcalc,eread)
  else
    call crest_oloop(env,nall,structures,dump,eread=eread)
  end if

  do i = 1,nall
    xyz(1:3,1:nat,i) = structures(i)%xyz(1:3,1:nat)
  end do
  deallocate (structures)
  return
end subroutine crest_oloop_xyz

!========================================================================================!
!========================================================================================!
!> Routines for parallel MDs
!========================================================================================!
!========================================================================================!

subroutine crest_search_multimd(env,mol,mddats,nsim)
!*****************************************************
!* subroutine crest_search_multimd
!* this runs #nsim MDs on the same structure (mol)
!*****************************************************
  use crest_parameters,only:wp,stdout,sep
  use crest_data
  use crest_calculator
  use strucrd
  use dynamics_module
  use iomod,only:makedir,directory_exist,remove
  use omp_lib
  implicit none
  type(systemdata),intent(inout) :: env
  type(mddata) :: mddats(nsim)
  integer :: nsim
  type(coord) :: mol
  type(coord),allocatable :: moltmps(:)
  integer :: i,j,io,ich
  logical :: pr,ex,nested
  integer :: T,Tn
  real(wp) :: percent
  character(len=80) :: atmp
  character(len=*),parameter :: mdir = 'MDFILES'

  type(calcdata),allocatable :: calculations(:)
  integer :: vz,job,thread_id
  real(wp) :: etmp
  real(wp),allocatable :: grdtmp(:,:)
  type(timer) :: profiler
!===========================================================!
!>--- check if we have any MD & calculation settings allocated
  if (.not.env%mddat%requested) then
    write (stdout,*) 'MD requested, but no MD settings present.'
    return
  else if (env%calc%ncalculations < 1) then
    write (stdout,*) 'MD requested, but no calculation settings present.'
    return
  end if

!>--- prepare calculation containers for parallelization (one per thread)
  call new_ompautoset(env,'auto_nested',nsim,T,Tn)
  nested = env%omp_allow_nested
  call ompautoset_summary(env,'MTD/MD runs',T,Tn)

  allocate (calculations(T),source=env%calc)
  allocate (moltmps(T),source=mol)
  allocate (grdtmp(3,mol%nat),source=0.0_wp)
  do i = 1,T
    moltmps(i)%nat = mol%nat
    moltmps(i)%at = mol%at
    moltmps(i)%xyz = mol%xyz
    do j = 1,env%calc%ncalculations
      calculations(i)%calcs(j) = env%calc%calcs(j)
      !>--- directories and io preparation
      ex = directory_exist(env%calc%calcs(j)%calcspace)
      if (.not.ex) then
        io = makedir(trim(env%calc%calcs(j)%calcspace))
      end if
      write (atmp,'(a,"_",i0)') sep,i
      calculations(i)%calcs(j)%calcspace = env%calc%calcs(j)%calcspace//trim(atmp)
      if (allocated(calculations(i)%calcs(j)%calcfile)) deallocate (calculations(i)%calcs(j)%calcfile)
      if (allocated(calculations(i)%calcs(j)%systemcall)) deallocate (calculations(i)%calcs(j)%systemcall)
      call calculations(i)%calcs(j)%printid(i,j)
    end do
    calculations(i)%pr_energies = .false.
    !>--- initialize the calculations
    call engrad(moltmps(i),calculations(i),etmp,grdtmp,io)
  end do

  !>--- other settings
  pr = .false.
  call profiler%init(nsim)

!>--- pre-start server-based calculators before forking OMP threads
  call preinit_mlip_parallel(calculations,T)
  !>--- run the MDs
  !$omp parallel &
  !$omp shared(env,calculations,mddats,mol,pr,percent,ich, nsim, moltmps, nested,Tn) &
  !!$omp single
  !$omp private(vz,i,job,thread_id,io,ex)
  !$omp do
  do i = 1,nsim

    call initsignal()
    vz = i

    !>--- OpenMP nested region threads
    if (nested) call ompmklset(Tn)

    !!$omp task firstprivate( vz ) private( job,thread_id,io,ex )
    call initsignal()

    thread_id = OMP_GET_THREAD_NUM()
    job = thread_id+1
    !$omp critical
    moltmps(job)%nat = mol%nat
    moltmps(job)%at = mol%at
    moltmps(job)%xyz = mol%xyz
    !>--- carry the lattice (PBC) through to the per-thread MD copy
    if (allocated(mol%lat)) moltmps(job)%lat = mol%lat
    !$omp end critical
    !>--- startup printout (thread safe)
    call parallel_md_block_printout(mddats(vz),vz)

    !>--- the acutal MD call with timing
    call profiler%start(vz)
    call dynamics(moltmps(job),mddats(vz),calculations(job),pr,io)
    call profiler%stop(vz)

    !>--- finish printout (thread safe)
    call parallel_md_finish_printout(mddats(vz),vz,io,profiler)
    !!$omp end task
  end do
  !!$omp taskwait
  !$omp end parallel

  !>--- collect trajectories into one
  call collect(nsim,mddats)

  call profiler%clear()
  deallocate (calculations)
  if (allocated(moltmps)) deallocate (moltmps)
  return
contains
  subroutine collect(n,mddats)
    implicit none
    integer :: n
    type(mddata) :: mddats(n)
    logical :: ex
    integer :: i,io,ich,ich2
    character(len=:),allocatable :: atmp
    character(len=256) :: btmp
    open (newunit=ich,file='crest_dynamics.trj.xyz')
    do i = 1,n
      atmp = mddats(i)%trajectoryfile
      inquire (file=atmp,exist=ex)
      if (ex) then
        open (newunit=ich2,file=atmp)
        io = 0
        do while (io == 0)
          read (ich2,'(a)',iostat=io) btmp
          if (io == 0) then
            write (ich,'(a)') trim(btmp)
          end if
        end do
        close (ich2)
      end if
    end do
    close (ich)
    return
  end subroutine collect
end subroutine crest_search_multimd

!========================================================================================!
subroutine crest_search_multimd_init(env,mol,mddat,nsim)
!*******************************************************
!* subroutine crest_search_multimd_init
!* This routine will initialize a copy of env%mddat
!* and save it to the local mddat. If we are about to
!* run RMSD metadynamics, the required number of
!* simulations (#nsim) is returned
!*******************************************************
  use crest_parameters,only:wp,stdout
  use crest_data
  use crest_calculator
  use strucrd
  use dynamics_module
  use iomod,only:makedir,directory_exist,remove
  use omp_lib
  implicit none
  type(systemdata),intent(inout) :: env
  type(mddata) :: mddat
  type(coord) :: mol
  integer,intent(inout) :: nsim
  integer :: i,io
  logical :: pr
!=======================================================!
  type(calcdata),target :: calc
  type(shakedata) :: shk

  real(wp) :: energy
  real(wp),allocatable :: grad(:,:)
  character(len=*),parameter :: mdir = 'MDFILES'
!======================================================!

  !>--- check if we have any MD & calculation settings allocated
  mddat = env%mddat
  if (.not.mddat%requested) then
    write (stdout,*) 'MD requested, but no MD settings present.'
    return
  else if (env%calc%ncalculations < 1) then
    write (stdout,*) 'MD requested, but no calculation settings present.'
    return
  end if

  !>--- init SHAKE?
  if (mddat%shake) then
    if (allocated(env%ref%wbo)) then
      shk%wbo = env%ref%wbo
    else
      calc = env%calc
      calc%calcs(1)%rdwbo = .true.
      allocate (grad(3,mol%nat),source=0.0_wp)
      call engrad(mol,calc,energy,grad,io)
      deallocate (grad)
      calc%calcs(1)%rdwbo = .false.

      shk%shake_mode = env%mddat%shk%shake_mode
      call move_alloc(calc%calcs(1)%wbo,shk%wbo)
    end if

    if (calc%nfreeze > 0) then
      shk%freezeptr => calc%freezelist
    else
      nullify (shk%freezeptr)
    end if

    shk%shake_mode = env%shake
    mddat%shk = shk
    call init_shake(mol%nat,mol%at,mol%xyz,mddat%shk,pr)
    mddat%nshake = mddat%shk%ncons
  end if
  !>--- complete real-time settings to steps
  call mdautoset(mddat,io)

  !>--- (optional)  MTD initialization
  if (nsim < 0) then
    mddat%simtype = type_mtd  !>-- set runtype to MTD

    call defaultGF(env)
    write (stdout,*) 'list of applied metadynamics Vbias parameters:'
    do i = 1,env%nmetadyn
      write (stdout,'(''$metadyn '',f10.5,f8.3,i5)') env%metadfac(i),env%metadexp(i)
    end do
    write (stdout,*)

    !>--- how many simulations
    nsim = env%nmetadyn
  end if

  return
end subroutine crest_search_multimd_init

!========================================================================================!
subroutine crest_search_multimd_init2(env,mddats,nsim)
  use crest_parameters,only:wp,stdout,sep
  use crest_data
  use crest_calculator
  use strucrd
  use dynamics_module
  use iomod,only:makedir,directory_exist,remove
  use omp_lib
  implicit none
  type(systemdata),intent(inout) :: env
  type(mddata) :: mddats(nsim)
  integer :: nsim
  integer :: i,io,j
  logical :: ex
!========================================================!
  type(mtdpot),allocatable :: mtds(:)

  character(len=80) :: atmp
  character(len=*),parameter :: mdir = 'MDFILES'

  !>--- parallel MD setup
  ex = directory_exist(mdir)
  if (ex) then
    call rmrf(mdir)
  end if
  io = makedir(mdir)
  do i = 1,nsim
    mddats(i)%md_index = i
    write (atmp,'(a,i0,a)') 'crest_',i,'.trj'
    mddats(i)%trajectoryfile = mdir//sep//trim(atmp)
    write (atmp,'(a,i0,a)') 'crest_',i,'.mdrestart'
    mddats(i)%restartfile = mdir//sep//trim(atmp)
  end do

  allocate (mtds(nsim))
  do i = 1,nsim
    if (mddats(i)%simtype == type_mtd) then
      mtds(i)%kpush = env%metadfac(i)
      mtds(i)%alpha = env%metadexp(i)
      mtds(i)%cvdump_fs = float(env%mddump)
      mtds(i)%mtdtype = cv_rmsd

      mddats(i)%npot = 1
      allocate (mddats(i)%mtd(1),source=mtds(i))
      allocate (mddats(i)%cvtype(1),source=cv_rmsd)
      !> if necessary exclude atoms from RMSD bias
      if (sum(env%includeRMSD) /= env%ref%nat) then
        if (.not.allocated(mddats(i)%mtd(1)%atinclude)) &
        & allocate (mddats(i)%mtd(1)%atinclude(env%ref%nat),source=.true.)
        do j = 1,env%ref%nat
          if (env%includeRMSD(j) .ne. 1) mddats(i)%mtd(1)%atinclude(j) = .false.
        end do
      end if
    end if
  end do
  if (allocated(mtds)) deallocate (mtds)

  return
end subroutine crest_search_multimd_init2

!========================================================================================!
subroutine crest_search_multimd2(env,mols,mddats,nsim)
!*******************************************************************
!* subroutine crest_search_multimd2
!* this runs #nsim MDs on #nsim selected different structures (mols)
!*******************************************************************
  use crest_parameters,only:wp,stdout,sep
  use crest_data
  use crest_calculator
  use strucrd
  use dynamics_module
  use shake_module
  use iomod,only:makedir,directory_exist,remove
  use omp_lib
  implicit none
  !> INPUT
  type(systemdata),intent(inout) :: env
  type(mddata) :: mddats(nsim)
  integer :: nsim
  type(coord) :: mols(nsim)
  type(coord),allocatable :: moltmps(:)
  integer :: i,j,io,ich
  logical :: pr,ex,nested
  integer :: T,Tn
  real(wp) :: percent
  character(len=80) :: atmp
  character(len=*),parameter :: mdir = 'MDFILES'

  type(calcdata),allocatable :: calculations(:)
  integer :: vz,job,thread_id
  type(timer) :: profiler
!===========================================================!
!>--- check if we have any MD & calculation settings allocated
  if (.not.env%mddat%requested) then
    write (stdout,*) 'MD requested, but no MD settings present.'
    return
  else if (env%calc%ncalculations < 1) then
    write (stdout,*) 'MD requested, but no calculation settings present.'
    return
  end if

!>--- prepare calculation objects for parallelization (one per thread)
  call new_ompautoset(env,'auto_nested',nsim,T,Tn)
  nested = env%omp_allow_nested
  call ompautoset_summary(env,'MTD/MD runs',T,Tn)

  allocate (calculations(T),source=env%calc)
  allocate (moltmps(T),source=mols(1))
  do i = 1,T
    do j = 1,env%calc%ncalculations
      calculations(i)%calcs(j) = env%calc%calcs(j)
      !>--- directories and io preparation
      ex = directory_exist(env%calc%calcs(j)%calcspace)
      if (.not.ex) then
        io = makedir(trim(env%calc%calcs(j)%calcspace))
      end if
      write (atmp,'(a,"_",i0)') sep,i
      calculations(i)%calcs(j)%calcspace = env%calc%calcs(j)%calcspace//trim(atmp)
      if (allocated(calculations(i)%calcs(j)%calcfile)) deallocate (calculations(i)%calcs(j)%calcfile)
      if (allocated(calculations(i)%calcs(j)%systemcall)) deallocate (calculations(i)%calcs(j)%systemcall)
      call calculations(i)%calcs(j)%printid(i,j)
    end do
    calculations(i)%pr_energies = .false.
  end do

!>--- other settings
  pr = .false.
  call profiler%init(nsim)

!>--- pre-start server-based calculators before forking OMP threads
  call preinit_mlip_parallel(calculations,T)
!>--- run the MDs
  !$omp parallel &
  !$omp shared(env,calculations,mddats,mols,pr,percent,ich, moltmps,profiler, nested,Tn)
  !$omp single
  do i = 1,nsim

    call initsignal()
    vz = i

    !>--- OpenMP nested region threads
    if (nested) call ompmklset(Tn)

    !$omp task firstprivate( vz ) private( job,thread_id,io,ex )
    call initsignal()

    thread_id = OMP_GET_THREAD_NUM()
    job = thread_id+1
    !$omp critical
    moltmps(job)%nat = mols(vz)%nat
    moltmps(job)%at = mols(vz)%at
    moltmps(job)%xyz = mols(vz)%xyz
    !>--- carry the per-structure lattice (PBC); moltmps was sourced from mols(1)
    if (allocated(mols(vz)%lat)) moltmps(job)%lat = mols(vz)%lat
    !$omp end critical
    !>--- startup printout (thread safe)
    call parallel_md_block_printout(mddats(vz),vz)

    !>--- the acutal MD call with timing
    call profiler%start(vz)
    call dynamics(moltmps(job),mddats(vz),calculations(job),pr,io)
    call profiler%stop(vz)

    !>--- finish printout (thread safe)
    call parallel_md_finish_printout(mddats(vz),vz,io,profiler)
    !$omp end task
  end do
  !$omp taskwait
  !$omp end single
  !$omp end parallel

!>--- collect trajectories into one
  call collect(nsim,mddats)

  call profiler%clear()
  deallocate (calculations)
  if (allocated(moltmps)) deallocate (moltmps)
  return
contains
  subroutine collect(n,mddats)
    implicit none
    integer :: n
    type(mddata) :: mddats(n)
    logical :: ex
    integer :: i,io,ich,ich2
    character(len=:),allocatable :: atmp
    character(len=256) :: btmp
    open (newunit=ich,file='crest_dynamics.trj.xyz')
    do i = 1,n
      atmp = mddats(i)%trajectoryfile
      inquire (file=atmp,exist=ex)
      if (ex) then
        open (newunit=ich2,file=atmp)
        io = 0
        do while (io == 0)
          read (ich2,'(a)',iostat=io) btmp
          if (io == 0) then
            write (ich,'(a)') trim(btmp)
          end if
        end do
        close (ich2)
      end if
    end do
    close (ich)
    return
  end subroutine collect
end subroutine crest_search_multimd2

!========================================================================================!
subroutine parallel_md_block_printout(MD,vz)
!***********************************************
!* subroutine parallel_md_block_printout
!* This will print information about the MD/MTD
!* simulation. The execution is omp threadsave
!***********************************************
  use crest_parameters,only:wp,stdout,sep
  use crest_data
  use crest_calculator
  use strucrd
  use dynamics_module
  use shake_module
  use iomod,only:to_str,drawbox
  implicit none
  type(mddata),intent(in) :: MD
  integer,intent(in) :: vz
  character(len=60) :: atmp
  integer,parameter :: bw = 54
  !$omp critical

  ! ── title ────────────────────────────────────────────────────
  if (MD%simtype == type_md) then
    write (atmp,'(a,1x,i3)') 'starting MD',vz
  else if (MD%simtype == type_mtd) then
    if (MD%cvtype(1) == cv_rmsd_static) then
      write (atmp,'(a,1x,i3)') 'starting static MTD',vz
    else
      write (atmp,'(a,1x,i4)') 'starting MTD',vz
    end if
  end if
  call drawbox(stdout,trim(atmp),width=bw,charset=4,ltab=1,procedual=0)
  call drawbox(stdout,trim(atmp),width=bw,charset=4,ltab=1,procedual=1)
  call drawbox(stdout,trim(atmp),width=bw,charset=4,ltab=1,procedual=3)

  ! ── simulation parameters ────────────────────────────────────
  write (stdout,'(1x,"│   MD simulation time   :",f8.1," ps",16x,"│")') MD%length_ps
  write (stdout,'(1x,"│   target T             :",f8.1," K",17x,"│")') MD%tsoll
  write (stdout,'(1x,"│   timestep dt          :",f8.1," fs",16x,"│")') MD%tstep
  write (stdout,'(1x,"│   dump interval(trj)   :",f8.1," fs",16x,"│")') MD%dumpstep
  if (MD%shake.and.MD%shk%shake_mode > 0) then
    if (MD%shk%shake_mode == 2) then
      write (stdout,'(1x,"│   SHAKE algorithm      :",a5," (all bonds)",10x,"│")') to_str(MD%shake)
    else
      write (stdout,'(1x,"│   SHAKE algorithm      :",a5," (H only)",13x,"│")') to_str(MD%shake)
    end if
  end if
  if (allocated(MD%active_potentials)) then
    write (stdout,'(1x,"│   active potentials    :",i4," potential(s)",10x,"│")') size(MD%active_potentials,1)
  end if
  if (MD%simtype == type_mtd) then
    if (MD%cvtype(1) == cv_rmsd) then
      write (stdout,'(1x,"│   dump interval(Vbias) :",f8.2," ps",16x,"│")') &
          & MD%mtd(1)%cvdump_fs/1000.0_wp
    end if
    write (stdout,'(1x,"│   Vbias prefactor (k)  :",f8.4," Eh",16x,"│")') MD%mtd(1)%kpush
    if (MD%cvtype(1) == cv_rmsd.or.MD%cvtype(1) == cv_rmsd_static) then
      write (stdout,'(1x,"│   Vbias exponent (α)   :",f8.4," bohr⁻²",12x,"│")') MD%mtd(1)%alpha
    else
      write (stdout,'(1x,"│   Vbias exponent (α)   :",f8.4,19x,"│")') MD%mtd(1)%alpha
    end if
    if (allocated(MD%mtd(1)%atinclude)) then
      write (stdout,'(1x,"│   # active atoms       :",i9," atoms",12x,"│")') count(MD%mtd(1)%atinclude,1)
    end if
  end if

  call drawbox(stdout,'',width=bw,charset=4,ltab=1,procedual=2)

  !$omp end critical

end subroutine parallel_md_block_printout

subroutine parallel_md_finish_printout(MD,vz,io,profiler)
!*******************************************
!* subroutine parallel_md_finish_printout
!* This will print information termination
!* info about the MD/MTD simulation
!*******************************************
  use crest_parameters,only:wp,stdout,sep
  use crest_data
  use crest_calculator
  use strucrd
  use dynamics_module
  use shake_module
  implicit none
  type(mddata),intent(in) :: MD
  integer,intent(in) :: vz,io
  type(timer),intent(inout) :: profiler
  character(len=40) :: atmp
  character(len=80) :: btmp

  !$omp critical

  if (MD%simtype == type_mtd) then
    if (MD%cvtype(1) == cv_rmsd_static) then
      write (atmp,'(a)') '*sMTD'
    else
      write (atmp,'(a)') '*MTD'
    end if
  else
    write (atmp,'(a)') '*MD'
  end if
  if (io == 0) then
    write (btmp,'(a,1x,i3,a)') trim(atmp),vz,' completed successfully'
  else
    write (btmp,'(a,1x,i3,a)') trim(atmp),vz,' terminated EARLY'
  end if
  call profiler%write_timing(stdout,vz,trim(btmp))

  !$omp end critical

end subroutine parallel_md_finish_printout
!========================================================================================!
!========================================================================================!
