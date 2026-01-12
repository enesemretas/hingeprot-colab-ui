	program useblz
cc************************************************************************** ****
c  -- Version I: date: March,2003.						*
c  -- Author: Yongmei Wang, email: yongmei@ncat.edu				*
c    -- Double precision version.						*
c - Purpose:								     	*
c       This is double precision mode, I attached MA47D suites at the end
c       to compile: using lib -I ~yongmei/mylib/libblzpackd.a 

c	This program will solve the Ax=eigx using a slightly different approach	*
c	A translation will be applied, Ax=eig Bx with B set to identy and 	*
c	one can apply a translation sigma during the solver			*
c	The method relies on the kernel of BLZPACK.				*
c	The real equation to solve is then: (Ax-sigmaBx)=0, therefore		*
c	GRNZD=ISTOR(9)=1, a generalized eigenproblem. 
c	sigma need not be specified, using EIGL,EIGR nonzero values, the program *
c	will automatically shifts

c	Blzpack also uses BLAS kernerl and LAPACK kernel. These can be loaded 	* 
c	using -lscs option during compile.				      	*
c - Compile: 
c   f77 SAtrblzpackd.f -64 -O -I "~yongmei/mylib/libblzpackd.a" -lscs -o SAblzpack *
c       for Linux. 
c       g77 -funroll-all-loops -fno-f2c -O3 -I../ -c SAtrblzpackd.f
c       g77 SAtrblzpackd.o -o SAtrblzpackd.x ../lib/libblzpackd_d.a /usr/lib/libblas.a
c       You need to include the BLAS kernel library that is in the 
c       /usr/lib/libblas.a. Somehow you need to explicitly tell to include it.
c       See the Makefile for the easiest.
c       finally The array size should not be too large 
c - Input file									*
c	File: sample input file is: SAtrinput.dat	 		*
c	Becareful that ISTOR(9)=1 must be set.
c*	the input will refer a file that contains the matrix to be diagonolized.*
c*	The matrix is in the sparse matrix format, 
c	first number is the total nonzero elements of the matrix na		*
c*	followed by (irn(k),jcn(k),sa(k),k=1,na) in na lines		        *
c	Only the upper triangular nonzero elements are stored.			*

c - To Run this program								*
c	SAtrblzpack <input >output							*

c* - TO use for specific problems, check the following parameters		*
c	nmax: largest dimension of the matrix					*
c	namax:largest nonzero element of the matrix
c       namax often needs to be increased for larger matrix.
c	NCUV and LEIG should be larger than NREIG=ISTOR(3) specified in the 	*
c	input file. I have set LEIG=NMAX, LN=NMAX, LRSTOR=NMAX*360		*
c	for most cases, I believe these are sufficient				*

c *******************************************************************************

	implicit none
c	working parameters and parameters for MA47 subprograms
	integer namax, nmax,LN
	integer n,i,j,k,na
	parameter(namax=16000000,nmax=12000,LN=nmax)
	integer MAXIW1
	parameter (MAXIW1 = namax*3+6*nmax+10)
	double precision SA(namax),W(LN),CNTL(2),RINFO(4),B(LN),C(NAMAX)
	integer irn(namax),jcn(namax),ICNTL(7),IW1(MAXIW1),IW2(LN*2+2)
	integer INFO(24),KEEP(namax+LN*5+2)

c	The W here is not the eigenvalue. It is used temporarily during
c	the solver Ax=b.

c	parameters for BLZPACK
	integer LISTOR, LEIG, LRSTOR, NCUV,LFLAG,NREIG
	parameter (LISTOR=2300, LEIG=nmax, NCUV=1000, LRSTOR=NMAX*360)
	integer ISTOR(LISTOR), IFLAG, NNEIG, NVOPU
	double precision EIG(LEIG,2),RSTOR(LRSTOR), SIGMA, U(LN,NCUV),
     $  V(LN,NCUV),X(LN,LEIG)
	double precision one, zero
	parameter (one = 1.0D0, zero=0.0D0)
	character*20 matrixfile
	real tarray(2),CPU,etime
	integer NTEIG,ISTORR
	
c	LEIG is needed to for EIG that needs to be greater than NREIG,
c	NCUV similarly also be greater than 3.
c	NREIG is number of required eigenvalues, LEIG should be greater than NREIG
c	NNEIG is not used if GNRZD=0

C******************************************************************
        matrixfile='upperhessian'

C      LN is the first dimension of matrix U and V
       istor(2) = LN
       istor(3) = 36
C      LEIG should be greater than NREIG
       istor(4) = LEIG
C      number of vectors in a block
       istor(5) = 2
C      maximum number of steps per run
       istor(6) = 0
C      number of starting vectors as input
       istor(7) = 0
C      number of eigenpairs given as input
       istor(8) = 0
C      problem type
       istor(9) = 1
C      spectrum slicing
       istor(10) = 0
C      solutions refinement
       istor(11) = 0
C      level of printing
       istor(12) = 6
C      file unit for output
       istor(13) = 44
C       this relates to parrallel version ignored for sequential mode*
       istor(14)= 0
C	Listor declared before compilation
       istor(15)=Listor
       rstor(1) = 0.
C      interval superior limit (EIGR)
       rstor(2) = 0.
C      threshold for convergence
       rstor(3) = 0
C      number of required eigenpairs
       rstor(4) = LRSTOR
C******************************************************************

	i=index(matrixfile,'.')-1
	open(unit=10,err=2,status='old',file=matrixfile)
	open(unit=12,file=matrixfile(1:i)//'.vwmatrixd')

c	read matrix
	read(10,*) na
	if(na.gt.namax) stop '*na>namax, increaes namax and recompile*'
	n=0
	do k=1,namax
	read(10,ERR=2, END=40,FMT=*) irn(k),jcn(k),sa(k)
	n=max(irn(k),jcn(k),n)
	end do
40	close(unit=10,err=2)
	write(*,*) 'you may want to check to see if it is reading correct'
	write(*,*) 'first matrix',sa(1),'10th',sa(10)
	if(n.gt.nmax) stop 'n>nmax increase nmax and recomplie'
	if(n.gt.LN) stop '* error: N>LN set LN same as N*'
	if(istor(5).gt.NCUV) STOP '* Error: NVBset> NCUV'
c*************************************************************************
	if(istor(9).ne.1) STOP '* istor(9) should be 1 wrong inputfile'
        istor(1) = N
c	NVBset should be smaller than NREIG, NCUV should be larger than NREIG

	do I=1,N
	B(I)=one
	end do

	NREIG=istor(3)

c	*main loop begins *
	
c	these loops can be called if you want to sovle for A^-1 eigenvalues
	call MA47ID(CNTL, ICNTL)
	call MA47AD(N,NA,IRN,JCN,IW1,MAXIW1,KEEP,ICNTL,RINFO,INFO)
	if(info(1).ne.0) STOP 'MA47I info(1) is nonzero'

c	first time calling set LFLAG=0
	LFLAG=0


50	continue   

	call BLZDRD(istor, RSTOR, SIGMA, NNEIG, U,V,LFLAG,NVOPU,EIG,X)
	
	if(LFLAG.LT.0) then
	write(*, *)'execution finished:abnormal exit'
	print*,istor(16),istor(1),istor(3),LFLAG
	
	else if(LFLAG.eq.0) then
	write(*, *) 'execution finished: standard exit'
	
	else if(LFLAG.eq.1) then

c	given U, compute V=AU , A is sparse matrix
c	SA only contains the upper triangular nonzero elements
c	do I=1,NVOPU
c	do J=1,N
c	V(J,I)= 0.0
c	end do	
c	do J=1,NA
c	if(IRN(J).eq.JCN(J)) then
c	V(IRN(J),I)=V(IRN(J),I)+SA(J)*U(JCN(J),I)
c	else
c	V(IRN(J),I)=V(IRN(J),I)+SA(J)*U(JCN(J),I)
c	V(JCN(J),I)=V(JCN(J),I)+SA(J)*U(IRN(J),I)
c	end if
c	end do	
c	end do

c	given U, solve CV=U, U dimension is LNxNVOPU

	do I=1,nvopu	
	call DCOPY(N,U(1,I),1,V(1,I),1)
	call MA47CD(N,C,NAMAX,IW1,MAXIW1,W,V(1,I),IW2,ICNTL)
	end do

	goto 50

	else if(LFLAG.eq.2) then
c	given U, compute V=B*U

	do I=1,NVOPU
	Do j=1,N
	V(J,I)=B(J)*U(J,I)
	end do
	end do

	goto 50

	else if(LFLAG.eq.3) then
c	given sigama, form C=A-sigma*B
c	sigma should not be specified, trying using EIGL and EIGR
	do 80 I=1,NA
	C(I)=SA(I)
	if(IRN(I).eq.JCN(I)) C(I)=C(I)-B(IRN(I))*SIGMA
80	continue
c	factor C=LDL'
	call MA47BD(N,NA,JCN,C,NAMAX,IW1,MAXIW1,KEEP,CNTL,ICNTL,IW2,RINFO,INFO)
	if(info(1).ne.0) STOP 'MA47B info(1) is nonzero'
	if(info(24).gt.0) STOP 'ERROR, MA47B, info(24)>0'
	NNEIG=info(23)
	goto 50

	end if

	if(LFLAG.eq.0) then
	NTEIG=istorR(istor,'NTEIG')
	write(12,*) nteig,N
	do i=1,nTeig
	write(12,*) eig(i,1),eig(i,2)
	end do
	write(12,*) ((x(i,j),i=1,N),j=1,nteig)
	end if

	CPU=etime(tarray)
	write(*,*) 'total CPU used',CPU/60.0,' minutes '
	stop

1	stop '* IO error: file does not exist *'
2	stop '* IO error: file matrixa format error *'

	end

C *******************************************************************
C COPYRIGHT (c) 1993 Council for the Central Laboratory
*                    of the Research Councils
C All rights reserved.
C
C None of the comments in this Copyright notice between the lines
C of asterisks shall be removed or altered in any way.
C
C This Package is intended for compilation without modification,
C so most of the embedded comments have been removed.
C
C ALL USE IS SUBJECT TO LICENCE. For full details of an HSL ARCHIVE
C Licence, see http://hsl.rl.ac.uk/archive/cou.html
C
C Please note that for an HSL ARCHIVE Licence:
C
C 1. The Package must not be copied for use by any other person.
C    Supply of any part of the library by the Licensee to a third party
C    shall be subject to prior written agreement between AEA
C    Hyprotech UK Limited and the Licensee on suitable terms and
C    conditions, which will include financial conditions.
C 2. All information on the Package is provided to the Licensee on the
C    understanding that the details thereof are confidential.
C 3. All publications issued by the Licensee that include results obtained
C    with the help of one or more of the Packages shall acknowledge the
C    use of the Packages. The Licensee will notify the Numerical Analysis
C    Group at Rutherford Appleton Laboratory of any such publication.
C 4. The Packages may be modified by or on behalf of the Licensee
C    for such use in research applications but at no time shall such
C    Packages or modifications thereof become the property of the
C    Licensee. The Licensee shall make available free of charge to the
C    copyright holder for any purpose all information relating to
C    any modification.
C 5. Neither CCLRC nor Hyprotech UK Limited shall be liable for any
C    direct or consequential loss or damage whatsoever arising out of
C    the use of Packages by the Licensee.
C *******************************************************************
C
C######DATE 24 May 1993
C 4 Feb 1994. Modified to limit searches for structured pivots under
C control of ICNTL(4).
C July 1994. Code added to avoid thrashing when there are no
C nondefective rows of minimal row count.
C 7/12/94. Some defaults changed following further testing. Minor
C     defects (mostly in comments) remedied.
C 27/3/01. In MA47OD, test for stability of a tile pivot as a pair
C   of 1x1 pivots modified to allow for the off-diagonal entry of
C   the pivot block.
	
      SUBROUTINE MA47AD(N,NE,IRN,JCN,IW,LIW,KEEP,ICNTL,RINFO,INFO)
      INTEGER N,NE,IRN(NE),JCN(NE),LIW,IW(LIW),KEEP(NE+5*N+2),ICNTL(7)
      DOUBLE PRECISION RINFO(4)
      INTEGER INFO(24)
      INTRINSIC MIN
      EXTERNAL MA47GD,MA47HD,MA47JD,MA47KD,MA47LD,MA47MD,MA47ND
      INTEGER FATHER,I,IPE,COUNT,IWFR,K,LDIAG,LEAF,LP,LW,LROW,MAP,MARK,
     +        MP,NODE,NODES,NV,PERM,SIZES
      LP = ICNTL(1)
      MP = ICNTL(2)
      LDIAG = ICNTL(3)
      DO 10 I = 1,14
        INFO(I) = 0

   10 CONTINUE
      IF (LDIAG.GE.3 .AND. MP.GT.0) THEN
        WRITE (MP,'(//A,I6,A,I7,A,I7)') ' Entering MA47AD with N =',N,
     +    '  NE =',NE,'  LIW =',LIW
        WRITE (MP,'(A,6I6)') ' ICNTL(1:6) =', (ICNTL(I),I=1,6)
        K = MIN(9,NE)
        IF (LDIAG.EQ.4) K = NE
        IF (K.GT.0) THEN
          WRITE (MP,'(A/(3(I6,A,2I6,A)))') ' Matrix entries:',
     +      (I,': (',IRN(I),JCN(I),')',I=1,K)
          IF (K.LT.NE) WRITE (MP,'(A)') '     . . .'
        END IF
        K = MIN(10,N)
        IF (LDIAG.EQ.4) K = N
        IF (ICNTL(4).EQ.1 .AND. K.GT.0) THEN
          WRITE (MP,'(A,10I6:/(7X,10I6))') ' KEEP =', (KEEP(I),I=1,K)
          IF (K.LT.N) WRITE (MP,'(7X,A)') '     . . .'
        END IF
      END IF
      IF (N.LT.1 .OR. (N+N+N)/3.NE.N) GO TO 20
      IF (NE.LT.1) GO TO 30
      LW = LIW - 4*N - 4
      IPE = LW + 1
      COUNT = IPE + N + 1
      NV = COUNT + N + 1
      LEAF = NV + N + 1
      PERM = 1
      LROW = PERM + N
      NODE = LROW + N
      MARK = NODE + N
      FATHER = MARK + N
      SIZES = FATHER + N
      MAP = SIZES + 2
      IF (ICNTL(4).NE.1) THEN
        IF (LW.LT.NE*2+N) GO TO 60
        CALL MA47GD(N,NE,IRN,JCN,IW,IW(IPE),IW(COUNT),IW(NV),KEEP(MARK),
     +              IWFR,ICNTL,INFO)
        KEEP(SIZES) = IW(COUNT)
        CALL MA47HD(N,IW(IPE+1),IW,LW,IWFR,IW(COUNT),IW(NV),
     +              KEEP(FATHER),KEEP(LROW),KEEP(NODE),KEEP(MARK),
     +              IW(LEAF),KEEP(PERM),ICNTL,INFO)
        IF (INFO(1).EQ.-3) GO TO 50
      ELSE
        IF (LW.LT.NE+N) GO TO 40
        CALL MA47JD(N,NE,IRN,JCN,KEEP,IW,IW(IPE),IW(COUNT),IW(NV),
     +              KEEP(MARK),IWFR,ICNTL,INFO)
        DO 12 I = 1,N
          KEEP(FATHER+I-1) = IW(NV+I)
   12   CONTINUE
        IF (INFO(1).LT.0) GO TO 80
        CALL MA47KD(N,IW(IPE+1),IW,LW,IWFR,KEEP,IW(COUNT),KEEP(FATHER),
     +              KEEP(LROW),IW(NV),KEEP(NODE),KEEP(MARK),IW(LEAF),
     +              ICNTL,INFO)
        IF (INFO(1).EQ.-3) GO TO 50
        IF (INFO(1).LT.0) GO TO 90
      END IF
      CALL MA47LD(N,KEEP(FATHER),KEEP(LROW),KEEP(NODE),IW(COUNT),IW(NV),
     +            IW(LEAF),KEEP(MARK),KEEP(PERM),NODES,ICNTL)
      KEEP(SIZES) = INFO(3)
      KEEP(SIZES+1) = NODES
      CALL MA47ND(N,NE,IRN,JCN,KEEP(MAP),KEEP(LROW),KEEP(PERM),
     +            IW(COUNT),IW(IPE))
      DO 15 K = 1,NE
        IW(KEEP(MAP+K-1)+LIW-NE) = JCN(K)
   15 CONTINUE
      CALL MA47MD(N,NE-KEEP(SIZES),IW(NODES+N+2),LIW-NODES-N-1,
     +            KEEP(LROW),KEEP(PERM),NODES,IW,KEEP(NODE),IW(NODES+1),
     +            KEEP(FATHER),ICNTL(5),INFO,RINFO)
      GO TO 100
   20 INFO(1) = -1
      IF (LDIAG.GT.0 .AND. LP.GT.0) WRITE (LP,'(A,I3/A,I8)')
     +    ' **** Error return from MA47AD ****  INFO(1) =',INFO(1),
     +    ' N has value',N
      GO TO 100
   30 INFO(1) = -2
      IF (LDIAG.GT.0 .AND. LP.GT.0) WRITE (LP,'(A,I3/A,I10)')
     +    ' **** Error return from MA47AD ****  INFO(1) =',INFO(1),
     +    ' NE has value',NE
      GO TO 100
   40 INFO(2) = NE + 5*N + 4
      GO TO 70
   50 INFO(2) = LIW + INFO(2)
      GO TO 70
   60 INFO(2) = 2*NE + 5*N + 4
   70 INFO(1) = -3
      IF (LDIAG.GT.0 .AND. LP.GT.0) WRITE (LP,'(A,I3/A,I10,A,I10)')
     +    ' **** Error return from MA47AD ****  INFO(1) =',INFO(1),
     +    ' LIW is too small. It must be increased from',LIW,
     +    ' to at least',INFO(2)
      RETURN
   80 IF (LDIAG.GT.0 .AND. LP.GT.0) WRITE (LP,'(A,I3/A/I10,A)')
     +    ' **** Error return from MA47AD ****  INFO(1) =',INFO(1),
     +    ' Invalid permutation supplied in KEEP. Component',INFO(2),
     +    ' is faulty.'
      RETURN
   90 IF (LDIAG.GT.0 .AND. LP.GT.0) WRITE (LP,'(A,I3/A/I10,A)')
     +    ' **** Error return from MA47AD ****  INFO(1) =',INFO(1),
     +    ' Invalid pivot sequence supplied in KEEP. Component',INFO(2),
     +    ' is negative but the next component is not negative.'
      RETURN
  100 IF (LDIAG.GE.3 .AND. MP.GT.0) THEN
        WRITE (MP,'(/A,I7,A,2F7.0/A,5I8/(14X,5(I8)))')
     +    ' Leaving MA47AD with  INFO(1) =',INFO(1),'  RINFO(1:2) =',
     +    RINFO(1),RINFO(2),' INFO(2:14) = ', (INFO(I),I=2,14)
        IF (INFO(1).GE.0) THEN
          K = MIN(10,NE)
          IF (LDIAG.EQ.4) K = NE
          WRITE (MP,'(A/(10I6))') ' Column indices:', (JCN(K),K=1,K)
          IF (K.LT.NE) WRITE (MP,'(A)') '     . . .'
          K = MIN(9,N)
          IF (LDIAG.EQ.4) K = N
          WRITE (MP,9000) ' KEEP(1:',N,') =',' (Permutation)',
     +      (KEEP(I),I=1,K)
          IF (K.LT.N) WRITE (MP,'(16X,A)') ' . . .'
 9000     FORMAT (A,I7,A/A/ (12X,10I6))
          WRITE (MP,9000) ' KEEP(N+1:N+',N,') =',
     +      ' (No. of entries in permuted rows)', (KEEP(I),I=N+1,N+K)
          IF (K.LT.N) WRITE (MP,'(16X,A)') ' . . .'
          WRITE (MP,9000) ' KEEP(2N+1:2N+',N,') =',
     +      ' (Tree nodes at which variables eliminated)',
     +      (KEEP(I),I=2*N+1,2*N+K)
          IF (K.LT.N) WRITE (MP,'(16X,A)') ' . . .'
          K = MIN(10,KEEP(5*N+2))
          WRITE (MP,9000) ' KEEP(3N+1:3N+',KEEP(5*N+2),') =',
     +      ' (Markowitz costs at tree nodes)', (KEEP(I),I=3*N+1,3*N+K)
          IF (K.LT.KEEP(5*N+2)) WRITE (MP,'(16X,A)') ' . . .'
          WRITE (MP,9000) ' KEEP(4N+1:4N+',KEEP(5*N+2),') =',
     +      ' (Fathers of nodes in tree)', (KEEP(I),I=4*N+1,4*N+K)
          IF (K.LT.KEEP(5*N+2)) WRITE (MP,'(16X,A)') ' . . .'
          WRITE (MP,9010) ' KEEP(5N+1:5N+2) =',
     +      ' (Nos. of faulty entries and tree nodes)',KEEP(5*N+1),
     +      KEEP(5*N+2)
 9010     FORMAT (A/A/ (12X,2I6))
          K = MIN(10,NE)
          IF (LDIAG.EQ.4) K = NE
          WRITE (MP,9000) ' KEEP(5N+3:5N+2+',NE,') =',' (Map array)',
     +      (KEEP(I),I=5*N+3,5*N+2+K)
          IF (K.LT.NE) WRITE (MP,'(16X,A)') ' . . .'
        END IF
      END IF
      END
      SUBROUTINE MA47BD(N,NE,JCN,A,LA,IW,LIW,KEEP,CNTL,ICNTL,IW1,RINFO,
     +                  INFO)
      INTEGER N,NE,JCN(NE),LA,ICNTL(7)
      DOUBLE PRECISION A(LA),CNTL(2),RINFO(4)
      INTEGER LIW,IW(LIW),KEEP(NE+5*N+2),IW1(2*N+2),INFO(24)
      INTRINSIC MIN
      EXTERNAL MA47OD,MA47UD
      INTEGER FATHER,I,K,KE,LDIAG,LP,LROW,MAP,MARK,MP,NODE,PERM,SIZES
      LP = ICNTL(1)
      MP = ICNTL(2)
      LDIAG = ICNTL(3)
      IF (N.LT.1) GO TO 30
      IF ((N+N+N)/3.NE.N) GO TO 30
      IF (NE.LT.1) GO TO 40
      IF (LIW.LT.NE*2) GO TO 50
      IF (LA.LT.NE*2) GO TO 70
      INFO(1) = 0
      INFO(2) = 0
      DO 10 I = 15,24
        INFO(I) = 0
   10 CONTINUE
      IF (LDIAG.GT.2 .AND. MP.GT.0) THEN
        WRITE (MP,'(//A,I6,3(A,I7)/A,3I7/A,I7/A,1P2D12.4)')
     +    ' Entering MA47BD with N =',N,'  NE =',NE,'  LA =',LA,
     +    '  LIW =',LIW,' ICNTL(1:3)  =', (ICNTL(I),I=1,3),
     +    ' ICNTL(5)    =',ICNTL(5),' CNTL(1:2)   =',CNTL(1),CNTL(2)
        KE = MIN(6,NE)
        IF (LDIAG.EQ.4) KE = NE
        IF (KE.GT.0) THEN
          WRITE (MP,'(A/( 3(I6,'':'',1PD12.4) ) )')
     +      ' Matrix entries:', (K,A(K),K=1,KE)
          IF (K.LT.NE) WRITE (MP,'(A)') '     . . .'
          WRITE (MP,'(A/(10I6))') ' Column indices:', (JCN(K),K=1,KE)
          IF (K.LT.NE) WRITE (MP,'(A)') '     . . .'
        END IF
        K = MIN(10,N)
        IF (LDIAG.EQ.4) K = N
        WRITE (MP,9000) ' KEEP(1:',N,') =',' (Permutation)',
     +    (KEEP(I),I=1,K)
        IF (K.LT.N) WRITE (MP,'(16X,A)') ' . . .'
 9000   FORMAT (A,I7,A/A/ (12X,10I6))
        WRITE (MP,9000) ' KEEP(N+1:N+',N,') =',
     +    ' (No. of entries in permuted rows)', (KEEP(I),I=N+1,N+K)
        IF (K.LT.N) WRITE (MP,'(16X,A)') ' . . .'
        WRITE (MP,9000) ' KEEP(2N+1:2N+',N,') =',
     +    ' (Tree nodes at which variables eliminated)',
     +    (KEEP(I),I=2*N+1,2*N+K)
        K = MIN(K,KEEP(5*N+2))
        IF (K.LT.KEEP(5*N+2)) WRITE (MP,'(16X,A)') ' . . .'
        WRITE (MP,9000) ' KEEP(3N+1:3N+',KEEP(5*N+2),') =',
     +    ' (Markowitz costs at tree nodes)', (KEEP(I),I=3*N+1,3*N+K)
        IF (K.LT.KEEP(5*N+2)) WRITE (MP,'(16X,A)') ' . . .'
        WRITE (MP,9000) ' KEEP(4N+1:4N+',KEEP(5*N+2),') =',
     +    ' (Fathers of nodes in tree)', (KEEP(I),I=4*N+1,4*N+K)
        IF (K.LT.KEEP(5*N+2)) WRITE (MP,'(16X,A)') ' . . .'
        WRITE (MP,9010) ' KEEP(5N+1:5N+2) =',
     +    ' (Nos. of faulty entries and tree nodes)',KEEP(5*N+1),
     +    KEEP(5*N+2)
 9010   FORMAT (A/A/ (12X,2I6))
        WRITE (MP,9000) ' KEEP(5N+3:5N+2+',NE,') =',' (Map array)',
     +    (KEEP(I),I=5*N+3,5*N+2+KE)
        IF (KE.LT.NE) WRITE (MP,'(16X,A)') ' . . .'
      END IF
      PERM = 1
      LROW = PERM + N
      NODE = LROW + N
      MARK = NODE + N
      FATHER = MARK + N
      SIZES = FATHER + N
      MAP = SIZES + 2
CDIR@ IVDEP
      DO 20 K = 1,NE
        A(KEEP(MAP+K-1)+LA-NE) = A(K)
        IW(KEEP(MAP+K-1)+LIW-NE) = JCN(K)
   20 CONTINUE
      CALL MA47OD(N,NE-KEEP(SIZES),A,LA,IW,LIW,KEEP(LROW),KEEP(PERM),
     +            KEEP(SIZES+1),IW1,KEEP(MARK),KEEP(NODE),IW1(N+1),
     +            KEEP(FATHER),CNTL,ICNTL(5),INFO,RINFO)
      IF (INFO(1).EQ.-3) GO TO 60
      IF (INFO(1).EQ.-4) GO TO 80
      GO TO 90
   30 INFO(1) = -1
      IF (LP.GT.0 .AND. LDIAG.GT.0) WRITE (LP,9020) INFO(1)
 9020 FORMAT (' **** Error return from MA47BD ****  INFO(1)=',I3)
      IF (LP.GT.0 .AND. LDIAG.GT.0)
     +    WRITE (LP,'(A,I10)') ' N has value',N
      GO TO 100
   40 INFO(1) = -2
      IF (LP.GT.0 .AND. LDIAG.GT.0) WRITE (LP,9020) INFO(1)
      IF (LP.GT.0 .AND. LDIAG.GT.0) WRITE (LP,
     +    '(A,I10)') ' NE has value',NE
      GO TO 100
   50 INFO(1) = -3
      INFO(2) = NE*2
   60 IF (LP.GT.0 .AND. LDIAG.GT.0) WRITE (LP,9020) INFO(1)
      IF (LP.GT.0 .AND. LDIAG.GT.0) WRITE (LP,'(A,I10,A,I10)')
     +    ' LIW is too small. It must be increased from',LIW,
     +    ' to at least',INFO(2)
      GO TO 100
   70 INFO(1) = -4
      INFO(2) = NE*2
   80 IF (LP.GT.0 .AND. LDIAG.GT.0) WRITE (LP,9020) INFO(1)
      IF (LP.GT.0 .AND. LDIAG.GT.0) WRITE (LP,'(A,I10,A,I10)')
     +    ' LA is too small. It must be increased from',LA,
     +    ' to at least',INFO(2)
      GO TO 100
   90 IF (LDIAG.GT.2 .AND. MP.GT.0) THEN
        WRITE (MP,'(/A,I7,A,I7/A,2F9.0/A,2I8/A,5I8/(15X,5I8))')
     +    ' Leaving MA47BD with  INFO(1) =',INFO(1),'  IERROR =',
     +    INFO(2),' RINFO(3:4)  = ',RINFO(3),RINFO(4),' INFO(6:7)   = ',
     +     (INFO(I),I=6,7),' INFO(15:24) = ', (INFO(I),I=15,24)
        CALL MA47UD(A,LA,IW,LIW,ICNTL)
      END IF
  100 RETURN
      END
      SUBROUTINE MA47CD(N,A,LA,IW,LIW,W,RHS,IW1,ICNTL)
      INTEGER N,LA
      DOUBLE PRECISION A(LA)
      INTEGER LIW,IW(LIW)
      DOUBLE PRECISION W(N),RHS(N)
      INTEGER IW1(N),ICNTL(7)
      INTRINSIC MIN
      EXTERNAL MA47QD,MA47RD,MA47UD
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0.0D0)
      INTEGER I,K,LDIAG,MP
      MP = ICNTL(2)
      LDIAG = ICNTL(3)
      IF (LDIAG.GE.3 .AND. MP.GT.0) THEN
        WRITE (MP,'(//A,I6,3(A,I7))') ' Entering MA47CD with N =',N,
     +    '  LA =',LA,'  LIW =',LIW
        CALL MA47UD(A,LA,IW,LIW,ICNTL)
        K = MIN(10,N)
        IF (LDIAG.EQ.4) K = N
        WRITE (MP,'(A, 1P,5D13.3/(4X, 1P,5D13.3))') ' RHS',
     +    (RHS(I),I=1,K)
        IF (K.LT.N) WRITE (MP,'(A)') '     . . .'
      END IF
      IF (IW(2).EQ.0) THEN
      ELSE
        CALL MA47QD(N,A,LA,IW,LIW,W,RHS,IW1,ICNTL)
        CALL MA47RD(N,A,LA,IW,LIW,W,RHS,IW1,ICNTL)
      END IF
      IF (LDIAG.GE.3 .AND. MP.GT.0) THEN
        WRITE (MP,'(//A)') ' Leaving MA47CD with RHS:'
        WRITE (MP,'(1P,5D13.3)') (RHS(I),I=1,K)
        IF (K.LT.N) WRITE (MP,'(A)') '     . . .'
      END IF
      END
      SUBROUTINE MA47FD(N,IPE,FLAG,IW,LW,IWFR,NCMPA)
      INTEGER N,IPE(N),FLAG(N),LW,IW(LW),IWFR,NCMPA
      INTEGER I,IR,K,L,LEN1,LEN2,LEN3,LWFR
      NCMPA = NCMPA + 1
      DO 10 I = 1,N
        L = IPE(I)
        IF (L.LE.0 .OR. FLAG(I).EQ.-1) GO TO 10
        IPE(I) = IW(L)
        IW(L) = -I
   10 CONTINUE
      IWFR = 1
      LWFR = 1
      DO 80 IR = 1,N
        DO 20 K = LWFR,LW
          IF (IW(K).LT.0) GO TO 30
   20   CONTINUE
        GO TO 90
   30   I = -IW(K)
        IW(IWFR) = IPE(I)
        IPE(I) = IWFR
        IWFR = IWFR + 1
        IF (FLAG(I).LE.-2) THEN
          L = IWFR - 1
          LEN1 = IW(K+1)
          IW(L+1) = LEN1
          LEN3 = IW(K+2)
          IW(L+2) = LEN3
          LEN2 = IW(L) - LEN1 - LEN3 - 2
          IWFR = L + 3
          DO 40 LWFR = K + 3,K + 2 + LEN1
            IF (FLAG(IW(LWFR)).LT.0) THEN
              IW(L+1) = IW(L+1) - 1
              IW(L) = IW(L) - 1
            ELSE
              IW(IWFR) = IW(LWFR)
              IWFR = IWFR + 1
            END IF
   40     CONTINUE
          K = LWFR
          DO 50 LWFR = K,K - 1 + LEN2
            IF (FLAG(IW(LWFR)).LT.0) THEN
              IW(L) = IW(L) - 1
            ELSE
              IW(IWFR) = IW(LWFR)
              IWFR = IWFR + 1
            END IF
   50     CONTINUE
          K = LWFR
          DO 60 LWFR = K,K - 1 + LEN3
            IF (FLAG(IW(LWFR)).LT.0) THEN
              IW(L+2) = IW(L+2) - 1
              IW(L) = IW(L) - 1
            ELSE
              IW(IWFR) = IW(LWFR)
              IWFR = IWFR + 1
            END IF
   60     CONTINUE
        ELSE
          DO 70 LWFR = K + 1,K + IW(IWFR-1)
            IW(IWFR) = IW(LWFR)
            IWFR = IWFR + 1
   70     CONTINUE
        END IF
   80 CONTINUE
   90 CONTINUE
      END
      SUBROUTINE MA47GD(N,NE,IRN,JCN,IW,IPE,COUNT,NV,FLAG,IWFR,ICNTL,
     +                  INFO)
      INTEGER N,NE,IRN(NE),JCN(NE),IW(NE*2+N),IPE(0:N),COUNT(0:N),NV(N),
     +        FLAG(N),IWFR,ICNTL(3),INFO(4)
      INTRINSIC MAX,MIN
      INTEGER I,J,K,L,LDIAG,MP
      MP = ICNTL(2)
      LDIAG = ICNTL(3)
      IF (MP.LE.0) LDIAG = 0
      INFO(1) = 0
      COUNT(0) = 0
      DO 10 I = 1,N
        FLAG(I) = 0
        COUNT(I) = 0
        NV(I) = -1
   10 CONTINUE
      DO 20 K = 1,NE
        I = IRN(K)
        J = JCN(K)
        IF (MIN(I,J).LT.1 .OR. MAX(I,J).GT.N) THEN
          IRN(K) = 0
          JCN(K) = 0
          COUNT(0) = COUNT(0) + 1
          INFO(1) = 1
          IF (COUNT(0).LE.1 .AND. LDIAG.GT.1) WRITE (MP,'(2A,I2)')
     +        ' *** Warning message from subroutine MA47AD ***',
     +        '  INFO(1) =',INFO(1)
          IF (COUNT(0).LE.10 .AND. LDIAG.GT.1) WRITE (MP,'(3(I6,A))') K,
     +        'th entry (in row',I,' and column',J,') ignored'
        ELSE IF (I.NE.J) THEN
          COUNT(I) = COUNT(I) + 1
          COUNT(J) = COUNT(J) + 1
        ELSE
          COUNT(I) = COUNT(I) + 1
          NV(I) = 1
        END IF
   20 CONTINUE
      IPE(0) = COUNT(0)
      DO 30 I = 1,N
        IPE(I) = IPE(I-1) + COUNT(I) + 1
   30 CONTINUE
      DO 40 K = 1,NE
        I = IRN(K)
        J = JCN(K)
        IW(IPE(I)) = J
        IPE(I) = IPE(I) - 1
        IF (I.NE.J) THEN
          IW(IPE(J)) = I
          IPE(J) = IPE(J) - 1
        END IF
   40 CONTINUE
      INFO(3) = COUNT(0)
      INFO(4) = 0
      IWFR = 1
      DO 60 I = 1,N
        L = IPE(I)
        IPE(I) = IWFR
        DO 50 K = L + 1,L + COUNT(I)
          J = IW(K)
          IF (FLAG(J).NE.I) THEN
            FLAG(J) = I
            IWFR = IWFR + 1
            IW(IWFR) = J
          ELSE
            IF (I.LE.J) INFO(4) = INFO(4) + 1
          END IF
   50   CONTINUE
        IW(IPE(I)) = IWFR - IPE(I)
        IWFR = IWFR + 1
   60 CONTINUE
      IF (INFO(4).GT.0) THEN
        INFO(1) = INFO(1) + 2
        IF (LDIAG.GT.1) WRITE (MP,'(A/I6,A)')
     +      ' *** Warning message from subroutine MA47AD ***',INFO(4),
     +      ' duplicate entries found.'
      END IF
      END
      SUBROUTINE MA47HD(N,IPE,IW,LW,IWFR,COUNT,NV,NEXT,LAST,IPR,FLAG,
     +                  LEAF,SVARS,ICNTL,INFO)
      INTEGER N,IPE(N),LW,IW(LW),IWFR,COUNT(N),NV(N),NEXT(N),LAST(N),
     +        IPR(N),FLAG(N),LEAF(N),SVARS(N),ICNTL(7),INFO(24)
      INTRINSIC ABS,MAX,MIN,REAL
      EXTERNAL MA47FD,MA47TD,MA47VD,MA47ZD
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0D0)
      LOGICAL BOUND
      DOUBLE PRECISION CMIN,COST
      INTEGER FLG,FLG1,I,IE,IEL,IP,IR,IRL,IS,ITHR,JP,JP1,JP2,JP3,JP4,JS,
     +        K,KE,KIND,KP,KP1,KP2,KS,K1,K2,LDIAG,LIST,LOOP,LS,ME,MINF,
     +        MINR,ML,MP,MROWS,MS,NCMP,NEL,NFLG,NP,NP0,NR(3),NROWS,NS,
     +        NSC(3),NSVARS,NVC,NVPIV,PART,PIV(2),PIVOT(2),PIVT,RANK
      INTEGER SHIFT,THRESH
      MP = ICNTL(2)
      LDIAG = ICNTL(3)
      IF (MP.LE.0) LDIAG = 0
      MROWS = N
      IF (ICNTL(4).GT.1)MROWS = ICNTL(4)
      CALL MA47TD(N,IPE,IW,LW,NV,NEXT,LAST,LEAF,FLAG,COUNT,SVARS)
      IF (LDIAG.GT.4) THEN
        WRITE (MP,'(/A)') '    Row   NV   List'
        DO 10 I = 1,N
          KP = IPE(I)
          IF (KP.EQ.0) THEN
            WRITE (MP,'(I7,I5)') I,NV(I)
          ELSE
            WRITE (MP,'(I7,I5,10I6:/ (12X,10I6))') I,NV(I),
     +        (IW(K),K=KP+1,KP+IW(KP))
          END IF
   10   CONTINUE
      END IF
      THRESH = MAX(SQRT(REAL(N)),3.)
      RANK = N
      MINF = 1
      MINR = 1
      NCMP = 0
      NFLG = N*3
      NEL = 0
      DO 20 IS = 1,N
        IPR(IS) = 0
        FLAG(IS) = NFLG
   20 CONTINUE
      DO 30 IS = 1,N
        K = IPE(IS)
        IF (K.EQ.0) THEN
          FLAG(IS) = -1
        ELSE
          IR = IW(K)
          IF (IR.EQ.0) THEN
            RANK = RANK + NV(IS)
            NV(IS) = -NV(IS)
            IR = 1
          END IF
          COUNT(IS) = IR
          NS = IPR(IR)
          IF (NS.GT.0) LAST(NS) = IS
          NEXT(IS) = NS
          IPR(IR) = IS
          LAST(IS) = 0
        END IF
   30 CONTINUE
      DO 670 ML = 1,N
        IF (NEL.GE.N) GO TO 680
        IR = MINR
        DO 40 MINR = IR,N
          IF (IPR(MINR).NE.0) GO TO 50
   40   CONTINUE
   50   DO 170 ITHR = 1,N
          NROWS = 0
          CMIN = REAL(N)**2
          DO 140 IR = MINR,N
            IF (MINF.GT.IR) GO TO 70
            MS = IPR(IR)
            DO 60 LIST = 1,N
              IF (MS.EQ.0) THEN
                 MINF = IR + 1
                 GO TO 70
              END IF
              IF (NV(MS).GT.0) THEN
                IF (IR.LE.THRESH) GO TO 180
                GO TO 160
              END IF
              MS = NEXT(MS)
   60       CONTINUE
   70       MS = IPR(IR)
            DO 120 LIST = 1,N
              IF (MS.EQ.0) GO TO 130
              IF (NFLG.LE.4) CALL MA47ZD(N,FLAG,NFLG)
              NROWS = NROWS + 1
              NFLG = NFLG - 1
              KP = IPE(MS)
              KP1 = KP + 1
              KP2 = KP + IW(KP)
              DO 100 KP = KP1,KP2
                PART = (IW(KP)-1)/N
                KE = IW(KP) - PART*N
                IF (FLAG(KE).EQ.-1) GO TO 100
                IF (FLAG(KE).LE.-2) THEN
                  JP = IPE(KE)
                  IF (PART.EQ.2) THEN
                    JP1 = JP + 3 + IW(JP+1)
                    JP2 = JP + IW(JP)
                  ELSE
                    JP1 = JP + 3
                    JP2 = JP + IW(JP) - IW(JP+2)
                  END IF
                ELSE
                  JP1 = KP
                  JP2 = KP2
                END IF
                DO 90 JP = JP1,JP2
                  IS = IW(JP)
                  IF (FLAG(IS).GT.NFLG) THEN
                    FLAG(IS) = NFLG
                    IF (NV(IS).LT.0) THEN
                      COST = REAL(COUNT(IS)-1)*REAL(IR-1)
                    ELSE
                      COST = REAL(IR+COUNT(IS)-3)*REAL(IR-1)
                    END IF
                    IF (COST.LT.CMIN) THEN
                      CMIN = COST
                      PIVOT(1) = IS
                      PIVOT(2) = MS
                      IF (CMIN.LE.REAL(IR-1)**2) GO TO 150
                    END IF
                  END IF
   90           CONTINUE
                IF (JP2.EQ.KP2) GO TO 110
  100         CONTINUE
  110         IF (NROWS.GE.MROWS) GO TO 150
              MS = NEXT(MS)
  120       CONTINUE
  130       IF (CMIN.LE.REAL(IR)**2) GO TO 150
  140     CONTINUE
  150     IR = MAX(COUNT(PIVOT(1)),COUNT(PIVOT(2)))
          IF (IR.LE.THRESH) GO TO 190
  160     CALL MA47VD(THRESH,IR+N/10,N,IPE,IW,LW,COUNT,NV,NEXT,LAST,IPR,
     +                FLAG,NFLG)
  170   CONTINUE
  180   KIND = 1
        ME = MS
        PIVOT(1) = MS
        PIVOT(2) = MS
        PIVT = 0
        NVPIV = NV(MS)
        CMIN = REAL(IR-1)**2
        FLAG(MS) = -1
        IF (LDIAG.GT.4) WRITE (MP,'(A,I5,A,2I5,A,I5,A,F7.0)')
     +      ' Pivot',NEL,':',PIVOT(1),NV(PIVOT(1)),' Row count:',
     +      COUNT(PIVOT(1)),' M-cost=',CMIN
        NEL = NEL + NVPIV
        GO TO 210
  190   KIND = 2
        IF (NV(PIVOT(1)).LT.0) KIND = 3
        FLAG(PIVOT(1)) = -1
        FLAG(PIVOT(2)) = -1
        PIV(1) = PIVOT(1)
        PIV(2) = PIVOT(2)
        NVPIV = ABS(NV(PIVOT(1)))
        IF (NVPIV.EQ.ABS(NV(PIVOT(2)))) THEN
          PIVT = 0
        ELSE
          IF (NVPIV.GT.ABS(NV(PIVOT(2)))) THEN
            PIVT = PIVOT(1)
            NVPIV = ABS(NV(PIVOT(2)))
          ELSE
            PIVT = PIVOT(2)
          END IF
          MS = LEAF(PIVT)
          DO 200 K = 2,NVPIV
            MS = -NEXT(MS)
  200     CONTINUE
          LEAF(MS) = LEAF(PIVT)
          LEAF(PIVT) = -NEXT(MS)
          FLAG(PIVT) = NFLG
          NEXT(MS) = 0
          NV(PIVT) = SIGN(ABS(NV(PIVT))-NVPIV,NV(PIVT))
          NV(MS) = SIGN(NVPIV,NV(PIVT))
          COUNT(MS) = COUNT(PIVT)
          IF (PIVT.EQ.PIVOT(1)) THEN
            PIV(1) = MS
          ELSE
            PIV(2) = MS
          END IF
        END IF
        ME = PIV(1)
        IF (LDIAG.GT.4) WRITE (*,'(A,I5,A,3I5,A,2I5,A,F7.0)')
     +      ' Pivot',NEL,':',PIV(1),PIV(2),NV(ME),' Row counts:',
     +      COUNT(PIVOT(1)),COUNT(PIVOT(2)),' M-cost=',CMIN
        NSC(2) = 0
        NVC = 0
        NEL = NEL + NVPIV*2
  210   NSVARS = 0
        IR = 0
        DO 280 LOOP = MIN(KIND,2),1,-1
          MS = PIVOT(LOOP)
          NSC(1) = NSVARS
          NR(1) = IR
          IF (MS.NE.PIVT) THEN
            NS = NEXT(MS)
            LS = LAST(MS)
            NEXT(MS) = 0
            IF (NS.GT.0) LAST(NS) = LS
            IF (LS.GT.0) THEN
              NEXT(LS) = NS
            ELSE
              IPR(COUNT(MS)) = NS
            END IF
          END IF
          KP = IPE(MS)
          KP1 = KP + 1
          KP2 = KP + IW(KP)
          DO 270 KP = KP1,KP2
            PART = (IW(KP)-1)/N
            KE = IW(KP) - PART*N
            IF (FLAG(KE).EQ.-1) GO TO 270
            IF (FLAG(KE).LE.-2) THEN
              IE = KE
              DO 220 LIST = 1,N
                IF (NEXT(IE).EQ.0) GO TO 230
                IEL = IE
                IE = -LAST(IE)
                IF (IE.EQ.ME) GO TO 240
                LAST(IEL) = -ME
  220         CONTINUE
  230         NEXT(IE) = -ME
              LAST(IE) = -ME
  240         JP = IPE(KE)
              JP1 = JP + 3
              JP2 = JP + IW(JP)
              IF (PART.EQ.0) THEN
              ELSE IF (PART.EQ.2) THEN
                JP1 = JP1 + IW(JP+1)
              ELSE
                JP2 = JP2 - IW(JP+2)
              END IF
            ELSE
              JP1 = KP
              JP2 = KP2
            END IF
            DO 250 JP = JP1,JP2
              IS = IW(JP)
              IF (FLAG(IS).LE.LOOP) THEN
              ELSE IF (FLAG(IS).EQ.2) THEN
                NVC = NVC + ABS(NV(IS))
                NSC(2) = NSC(2) + 1
                FLAG(IS) = 0
                COUNT(IS) = COUNT(IS) - NVPIV
              ELSE
                IR = IR + ABS(NV(IS))
                FLAG(IS) = LOOP
                NSVARS = NSVARS + 1
                SVARS(NSVARS) = IS
                LS = LAST(IS)
                LAST(IS) = 0
                NS = NEXT(IS)
                NEXT(IS) = 0
                IF (NS.GT.0) LAST(NS) = LS
                IF (LS.GT.0) THEN
                  NEXT(LS) = NS
                ELSE
                  IPR(COUNT(IS)) = NS
                END IF
                COUNT(IS) = COUNT(IS) - NVPIV
              END IF
  250       CONTINUE
            IF (JP2.NE.KP2 .AND. LOOP.EQ.1) THEN
              IF (KIND.EQ.1) THEN
                IF (PART.EQ.0) FLAG(KE) = -1
              ELSE
                DO 260 JP = JP1,JP2
                  IF (IW(JP).EQ.PIVOT(2)) THEN
                    FLAG(KE) = -1
                    GO TO 270
                  END IF
  260           CONTINUE
              END IF
            END IF
  270     CONTINUE
  280   CONTINUE
        IF (KIND.EQ.1) THEN
          NSC(2) = NSVARS
          NR(1) = IR
          NSC(3) = 0
          NR(3) = 0
        ELSE
          NEXT(PIV(2)) = -ME
          LAST(PIV(2)) = -ME
          COUNT(PIV(2)) = -COUNT(PIV(2))
          IF (KIND.EQ.2) THEN
            NSC(3) = NSVARS - NSC(1)
            NR(3) = IR - NR(1)
            NSC(2) = NSC(1)
            NR(2) = NR(1)
            NSC(1) = 0
            NR(1) = IR
          ELSE
            NSC(3) = NSVARS - NSC(1)
            NR(3) = IR - NR(1) + NVC
            NSC(1) = NSC(1) - NSC(2)
            NR(2) = NR(1)
            NR(1) = IR
            K1 = 1
            DO 310 K = 1,NSC(1) + NSC(2)
              IF (FLAG(SVARS(K)).EQ.2) THEN
                KS = SVARS(K)
                SVARS(K) = SVARS(K1)
                SVARS(K1) = KS
                K1 = K1 + 1
              END IF
  310       CONTINUE
          END IF
        END IF
        IF (NSVARS.EQ.0) THEN
          IPE(ME) = 0
          GO TO 670
        END IF
        IF (NFLG.LE.4) CALL MA47ZD(N,FLAG,NFLG)
        NFLG = NFLG - 1
        DO 450 K = 1,NSVARS
          IS = SVARS(K)
          KP = IPE(IS)
          KP1 = KP + 1
          KP2 = IW(KP) + KP
          DO 440 KP = KP1,KP2
            PART = (IW(KP)-1)/N
            KE = IW(KP) - N*PART
            IF (FLAG(KE).GE.0) GO TO 450
            IF (FLAG(KE).EQ.-1) GO TO 440
            IF (FLAG(KE).EQ.-NFLG) GO TO 440
            FLAG(KE) = -NFLG
            JP = IPE(KE)
            JP1 = JP + 3
            JP2 = JP1 + IW(JP+1)
            JP4 = JP + IW(JP)
            JP3 = JP4 - IW(JP+2)
            IF (KIND.EQ.1) THEN
              DO 340 JP = JP1,JP4
                IF (FLAG(IW(JP)).GT.2) GO TO 440
  340         CONTINUE
            ELSE IF (KIND.EQ.2) THEN
              DO 350 JP = JP2,JP3
                IF (FLAG(IW(JP)).GT.2) GO TO 440
                IF (FLAG(IW(JP)).EQ.1) GO TO 440
  350         CONTINUE
              FLG1 = 0
              DO 360 JP = JP3 + 1,JP4
                FLG = FLAG(IW(JP))
                IF (FLG.GT.2) GO TO 440
                IF (FLG.LE.0) GO TO 360
                FLG1 = FLG
  360         CONTINUE
              DO 370 JP = JP1,JP2 - 1
                FLG = FLAG(IW(JP))
                IF (FLG.GT.2) GO TO 440
                IF (FLG.LE.0) GO TO 370
                IF (FLG1.NE.0) GO TO 440
  370         CONTINUE
            ELSE
              DO 380 JP = JP2,JP3
                IF (FLAG(IW(JP)).GT.0) GO TO 440
  380         CONTINUE
              FLG1 = 0
              DO 390 JP = JP3 + 1,JP4
                FLG = FLAG(IW(JP))
                IF (FLG.GT.2) GO TO 440
                IF (FLG.LE.0) GO TO 390
                IF (FLG1.EQ.0) FLG1 = FLG
                IF (FLG.NE.FLG1) GO TO 440
  390         CONTINUE
              FLG1 = 3 - FLG1
              DO 400 JP = JP1,JP2 - 1
                FLG = FLAG(IW(JP))
                IF (FLG.GT.2) GO TO 440
                IF (FLG.LE.0) GO TO 400
                IF (FLG1.EQ.3) FLG1 = FLG
                IF (FLG.NE.FLG1) GO TO 440
  400         CONTINUE
            END IF
            FLAG(KE) = -1
            IE = KE
            DO 420 LIST = 1,N
              IF (NEXT(IE).EQ.0) GO TO 430
              IEL = IE
              IE = -LAST(IE)
              IF (IE.EQ.ME) GO TO 440
              LAST(IEL) = -ME
  420       CONTINUE
  430       NEXT(IE) = -ME
            LAST(IE) = -ME
  440     CONTINUE
  450   CONTINUE
        DO 630 LOOP = 1,KIND
          IF (LOOP.EQ.1) THEN
            K1 = 1 + NSC(1)
            K2 = K1 + NSC(2) - 1
          ELSE IF (LOOP.EQ.2) THEN
            K1 = K2 + 1
            K2 = NSVARS
            DO 460 K = K1,K2
              IS = SVARS(K)
              IF (NV(IS).NE.0) FLAG(IS) = NFLG
  460       CONTINUE
          ELSE
            DO 470 K = K1,K2
              IS = SVARS(K)
              IF (NV(IS).NE.0) FLAG(IS) = 1
  470       CONTINUE
            K1 = 1
            K2 = NSC(1)
            DO 480 K = K1,K2
              IS = SVARS(K)
              IF (NV(IS).NE.0) FLAG(IS) = NFLG
  480       CONTINUE
          END IF
          DO 620 K = K1,K2
            IS = SVARS(K)
            BOUND = COUNT(IS) .GT. THRESH
            IF (LOOP.EQ.1) NV(IS) = ABS(NV(IS))
            IF (NFLG.LE.4) CALL MA47ZD(N,FLAG,NFLG)
            NFLG = NFLG - 1
            IR = NR(LOOP)
            KP = IPE(IS)
            NP = KP + 1
            KP1 = KP + 1
            KP2 = IW(KP) + KP
            DO 520 KP = KP1,KP2
              PART = (IW(KP)-1)/N
              KE = IW(KP) - N*PART
              IF (FLAG(KE).LE.-2) THEN
                FLAG(KE) = -NFLG
                IF (BOUND) GO TO 510
                LEAF(KE) = PART
                JP = IPE(KE)
                JP1 = JP + 3
                JP2 = JP + IW(JP)
                IF (PART.EQ.0) THEN
                ELSE IF (PART.EQ.2) THEN
                  JP1 = JP1 + IW(JP+1)
                ELSE
                  JP2 = JP2 - IW(JP+2)
                END IF
                IRL = IR
                DO 500 JP = JP1,JP2
                  JS = IW(JP)
                  IF (FLAG(JS).LE.NFLG) GO TO 500
                  IR = IR + ABS(NV(JS))
                  FLAG(JS) = NFLG
  500           CONTINUE
                IF (IR.EQ.IRL .AND. LAST(KE).EQ.-ME) GO TO 520
  510           IW(NP) = IW(KP)
                NP = NP + 1
              ELSE IF (FLAG(KE).GE.0) THEN
                GO TO 530
              END IF
  520       CONTINUE
            NP0 = NP
            GO TO 550
  530       NP0 = NP
            KP1 = KP
            DO 540 KP = KP1,KP2
              KS = IW(KP)
              IF (FLAG(KS).LE.NFLG) GO TO 540
              IR = IR + ABS(NV(KS))
              FLAG(KS) = NFLG
              IW(NP) = KS
              NP = NP + 1
  540       CONTINUE
  550       IF (BOUND) IR = COUNT(IS)
            IF (NP.GT.KP2) THEN
              KP = IPE(IS)
              IF (NP+IWFR-KP.GE.LW) THEN
                CALL MA47FD(N,IPE,FLAG,IW,IWFR-1,IWFR,NCMP)
                IF (NP+IWFR-KP.GE.LW) THEN
                  INFO(1) = -3
                  INFO(2) = NP + IWFR - KP + 1 - LW
                END IF
                SHIFT = IPE(IS) - KP
                KP = KP + SHIFT
                KP2 = KP2 + SHIFT
                NP = NP + SHIFT
                NP0 = NP0 + SHIFT
              END IF
              NP = NP + IWFR - KP
              NP0 = NP0 + IWFR - KP
              IPE(IS) = IWFR
              KP1 = KP
              DO 560 KP = KP1,KP2
                IW(IWFR) = IW(KP)
                IWFR = IWFR + 1
  560         CONTINUE
              IW(IWFR) = 0
              IWFR = IWFR + 1
            END IF
            IW(NP) = IW(NP0)
            KP = IPE(IS)
            IW(NP0) = IW(KP+1)
            IW(KP+1) = ME + (LOOP-1)*N
            IW(KP) = NP - KP
            IF (IR.EQ.0) THEN
              IR = -NV(IS)
              NV(IS) = IR
              RANK = RANK - IR
              IR = 1
            END IF
            IF (CMIN.EQ.ZERO) GO TO 610
            IF (IR.GT.THRESH) GO TO 610
            JS = IPR(IR)
            DO 590 LIST = 1,N
              IF (JS.LE.0) GO TO 610
              KP = IPE(JS)
              IF (IW(KP+1).NE.ME+ (LOOP-1)*N) GO TO 610
              IF (SIGN(1,NV(JS)).NE.SIGN(1,NV(IS))) GO TO 580
              KP1 = KP
              DO 570 KP = KP1 + 2,KP1 + IW(KP1)
                PART = (IW(KP)-1)/N
                IE = IW(KP) - PART*N
                IF (ABS(FLAG(IE)).GT.NFLG) GO TO 580
                IF (FLAG(IE).EQ.-NFLG) THEN
                  IF (PART.NE.LEAF(IE)) GO TO 580
                END IF
  570         CONTINUE
              GO TO 600
  580         JS = NEXT(JS)
  590       CONTINUE
  600       IPE(JS) = 0
            NV(IS) = NV(IS) + NV(JS)
            NV(JS) = 0
            FLAG(JS) = -1
            NS = NEXT(JS)
            LS = LAST(JS)
            IF (NS.GT.0) LAST(NS) = IS
            IF (LS.GT.0) NEXT(LS) = IS
            LAST(IS) = LS
            NEXT(IS) = NS
            IF (IPR(IR).EQ.JS) IPR(IR) = IS
            COUNT(IS) = IR
            NEXT(JS) = -LEAF(IS)
            LEAF(IS) = LEAF(JS)
            LAST(JS) = -IS
            GO TO 620
  610       NS = IPR(IR)
            IF (NS.GT.0) LAST(NS) = IS
            NEXT(IS) = NS
            IPR(IR) = IS
            LAST(IS) = 0
            MINR = MIN(MINR,IR)
            IF(NV(IS).GT.0)MINF = MIN(MINF,IR)
            COUNT(IS) = IR
  620     CONTINUE
  630   CONTINUE
        IF (IWFR+NSVARS+3.GE.LW) THEN
          CALL MA47FD(N,IPE,FLAG,IW,IWFR-1,IWFR,NCMP)
          IF (IWFR+NSVARS+3.GE.LW) THEN
            INFO(1) = -3
            INFO(2) = IWFR + NSVARS + 4 - LW
            RETURN
          END IF
        END IF
        IP = IWFR
        IWFR = IWFR + 3
        K2 = 0
        DO 650 LOOP = 1,3
          K1 = K2 + 1
          K2 = K1 + NSC(LOOP) - 1
          DO 640 K = K1,K2
            IS = SVARS(K)
            IF (NV(IS).EQ.0) THEN
              NSC(LOOP) = NSC(LOOP) - 1
            ELSE
              FLAG(IS) = NFLG
              IW(IWFR) = IS
              IWFR = IWFR + 1
            END IF
  640     CONTINUE
  650   CONTINUE
        IW(IP) = IWFR - IP - 1
        IW(IP+1) = NSC(1)
        IW(IP+2) = NSC(3)
        FLAG(ME) = -NFLG
        IPE(ME) = IP
  670 CONTINUE
  680 IF (RANK.LT.N) INFO(1) = INFO(1) + 4
      INFO(8) = N - RANK
      INFO(12) = NCMP
      END
      SUBROUTINE MA47ID(CNTL,ICNTL)
      INTEGER ICNTL(7)
      DOUBLE PRECISION CNTL(2)
      CNTL(1) = 0.001D0
      CNTL(2) = 0.0D0
      ICNTL(1) = 6
      ICNTL(2) = 6
      ICNTL(3) = 1
      ICNTL(4) = 0
      ICNTL(5) = 5
      ICNTL(6) = 5
      ICNTL(7) = 4
      RETURN
      END
      SUBROUTINE MA47JD(N,NE,IRN,JCN,KEEP,IW,IPE,COUNT,PERM,FLAG,IWFR,
     +                  ICNTL,INFO)
      INTEGER N,NE,IRN(NE),JCN(NE),KEEP(N),IW(NE+N),IPE(0:N),COUNT(0:N),
     +        PERM(0:N),FLAG(N),IWFR,ICNTL(3),INFO(4)
      INTRINSIC MAX,MIN
      INTEGER I,J,K,L,LDIAG,MP
      MP = ICNTL(2)
      LDIAG = ICNTL(3)
      IF (MP.LE.0) LDIAG = 0
      INFO(1) = 0
      DO 10 I = 1,N
        PERM(I) = 0
        FLAG(I) = 0
        COUNT(I) = 0
   10 CONTINUE
      COUNT(0) = 0
      PERM(0) = 0
      DO 20 I = 1,N
        J = ABS(KEEP(I))
        IF (J.LT.1 .OR. J.GT.N) GO TO 80
        IF (PERM(J).NE.0) GO TO 80
        PERM(J) = I
   20 CONTINUE
      DO 30 K = 1,NE
        I = IRN(K)
        J = JCN(K)
        IF (MIN(I,J).LT.1 .OR. MAX(I,J).GT.N) THEN
          IRN(K) = 0
          JCN(K) = 0
          COUNT(0) = COUNT(0) + 1
          INFO(1) = 1
          IF (COUNT(0).LE.1 .AND. LDIAG.GT.1) WRITE (MP,'(2A,I2)')
     +        ' *** Warning message from subroutine MA47AD ***',
     +        '  INFO(1) =',INFO(1)
          IF (COUNT(0).LE.10 .AND. LDIAG.GT.1) WRITE (MP,'(3(I6,A))') K,
     +        'th entry (in row',I,' and column',J,') ignored'
        ELSE IF (PERM(I).LE.PERM(J)) THEN
          COUNT(I) = COUNT(I) + 1
        ELSE
          COUNT(J) = COUNT(J) + 1
        END IF
   30 CONTINUE
      IPE(0) = COUNT(0)
      DO 40 I = 1,N
        IPE(I) = IPE(I-1) + COUNT(I) + 1
   40 CONTINUE
      DO 50 K = 1,NE
        I = IRN(K)
        J = JCN(K)
        IF (PERM(I).LE.PERM(J)) THEN
          IW(IPE(I)) = J
          IPE(I) = IPE(I) - 1
        ELSE
          IW(IPE(J)) = I
          IPE(J) = IPE(J) - 1
        END IF
   50 CONTINUE
      IWFR = 1
      INFO(4) = 0
      DO 70 I = 1,N
        L = IPE(I)
        IPE(I) = IWFR
        DO 60 K = L + 1,L + COUNT(I)
          J = IW(K)
          IF (FLAG(J).NE.I) THEN
            FLAG(J) = I
            IWFR = IWFR + 1
            IW(IWFR) = J
          ELSE
            INFO(4) = INFO(4) + 1
          END IF
   60   CONTINUE
        IF (IWFR.GT.IPE(I)) THEN
          IW(IPE(I)) = IWFR - IPE(I)
          IWFR = IWFR + 1
        ELSE
          IPE(I) = 0
        END IF
   70 CONTINUE
      IF (INFO(4).GT.0) THEN
        INFO(1) = INFO(1) + 2
        IF (LDIAG.GT.1) WRITE (MP,'(A/I6,A)')
     +      ' *** Warning message from subroutine MA47AD ***',
     +      INFO(4),' duplicate entries found.'
      END IF
      INFO(3) = COUNT(0)
      RETURN
   80 INFO(1) = -5
      INFO(2) = I
      END
      SUBROUTINE MA47KD(N,IPE,IW,LW,IWFR,KEEP,COUNT,NEXT,LAST,IPR,FLAG,
     +                  NEXTE,VARS,ICNTL,INFO)
      INTEGER N,IPE(N),LW,IW(LW),IWFR,KEEP(N),COUNT(N),NEXT(N),LAST(N),
     +        IPR(N),FLAG(N),NEXTE(N),VARS(N),ICNTL(7),INFO(24)
      INTRINSIC MIN
      EXTERNAL MA47FD
      LOGICAL A11,A22,A21,DIAG
      INTEGER IE,IEL,IP,IS,JP,JP1,JP2,JP3,JP4,JS,K,KE,KIND,KP,KP1,KP2,
     +        KS,K1,LDIAG,LIST,LOOP,ME,ML,MP,MS,NCMP,NE,NEL,NSC(3),
     +        NVARS,NVC,NV1,NV2,PIVE(2),PIVOT(2),RANK
      MP = ICNTL(2)
      LDIAG = ICNTL(3)
      IF (MP.LE.0) LDIAG = 0
      IF (LDIAG.GT.4) THEN
        WRITE (MP,'(/A)') '    Row   List'
        DO 1 IS = 1,N
          KP = IPE(IS)
          IF (KP.EQ.0) THEN
            WRITE (MP,'(I7)') IS
          ELSE
            WRITE (MP,'(I7,10I6:/ (7X,10I6))') IS,
     +        (IW(K),K=KP+1,KP+IW(KP))
          END IF
    1   CONTINUE
      END IF
      RANK = N
      NCMP = 0
      NEL = 0
      DO 10 IS = 1,N
        IPR(IS) = 0
        FLAG(IS) = 4
   10 CONTINUE
      DO 210 ML = 1,N
        IF (NEL.GE.N) GO TO 220
        MS = KEEP(NEL+1)
        IF (MS.GT.0) THEN
          KIND = 1
          PIVOT(1) = MS
          PIVOT(2) = MS
          NEL = NEL + 1
        ELSE
          KIND = 2
          MS = -MS
          IF (NEL+2.GT.N) GO TO 230
          IS = -KEEP(NEL+2)
          IF (IS.LT.0) GO TO 230
          PIVOT(1) = MS
          PIVOT(2) = IS
          NVC = 0
          NEL = NEL + 2
        END IF
        NVARS = 0
        A21 = .FALSE.
        DO 180 LOOP = 1,KIND
          DIAG = .FALSE.
          MS = PIVOT(LOOP)
          KE = IPR(MS)
          PIVE(LOOP) = KE
          IPR(MS) = 1
          DO 110 KP = 1,N + 1
            IF (KE.NE.0) THEN
              JP = IPE(KE)
              JP1 = JP + 3
              JP4 = JP + IW(JP)
              JP2 = JP1 + IW(JP+1)
              JP3 = JP4 - IW(JP+2)
              KP1 = JP1
              KP2 = JP4
              DO 20 JP = JP1,JP2 - 1
                IF (IW(JP).EQ.MS) GO TO 40
   20         CONTINUE
              DO 30 JP = JP3 + 1,JP4
                IF (IW(JP).EQ.MS) GO TO 50
   30         CONTINUE
              JP2 = JP4
              GO TO 60
   40         JP1 = JP2
              JP2 = JP4
              GO TO 60
   50         JP2 = JP3
            ELSE
              JP = IPE(MS)
              IF (JP.EQ.0) GO TO 120
              JP1 = JP + 1
              JP2 = JP + IW(JP)
            END IF
   60       DO 70 JP = JP1,JP2
              IS = IW(JP)
              IF (FLAG(IS).LE.3-LOOP) THEN
              ELSE IF (FLAG(IS).EQ.2) THEN
                FLAG(IS) = 0
                IF (IS.EQ.MS) THEN
                  DIAG = .TRUE.
                ELSE
                  NVC = NVC + 1
                END IF
              ELSE
                FLAG(IS) = 3 - LOOP
                IF (IS.EQ.MS) THEN
                  DIAG = .TRUE.
                ELSE IF (IS.EQ.PIVOT(2)) THEN
                  A21 = .TRUE.
                ELSE
                  NVARS = NVARS + 1
                  VARS(NVARS) = IS
                END IF
              END IF
   70       CONTINUE
            IF (KE.EQ.0) GO TO 120
            IF (LOOP.EQ.1) THEN
              IF (KIND.EQ.1) THEN
                IF (JP1.EQ.KP1 .AND. JP2.EQ.KP2) FLAG(KE) = -2
              ELSE
                DO 80 JP = JP1,JP2
                  IF (IW(JP).EQ.PIVOT(2)) THEN
                    FLAG(KE) = -2
                    GO TO 90
                  END IF
   80           CONTINUE
   90         END IF
            END IF
            KE = NEXTE(KE)
  110     CONTINUE
  120     IF (LOOP.EQ.1) THEN
            A11 = DIAG
            ME = PIVOT(2)
            IF (DIAG) ME = PIVOT(1)
            NV1 = NVARS
          ELSE
            A22 = DIAG
          END IF
          KE = PIVE(LOOP)
          DO 170 KP = 1,N + 1
            IF (KE.EQ.0) GO TO 180
            IF (FLAG(KE).LE.-2) THEN
              IE = KE
              DO 130 LIST = 1,N
                IF (NEXT(IE).EQ.0) GO TO 140
                IEL = IE
                IE = -LAST(IE)
                IF (IE.EQ.ME) GO TO 150
                LAST(IEL) = -ME
  130         CONTINUE
  140         NEXT(IE) = -ME
              LAST(IE) = -ME
              IF (FLAG(KE).EQ.-2 .AND. LOOP.EQ.KIND) FLAG(KE) = -1
            END IF
  150       NE = NEXTE(KE)
            FLAG(MS) = -1
            IF (FLAG(KE).LE.-2) THEN
              JP = IPE(KE)
              JP1 = JP + 3
              JP2 = JP + IW(JP)
              JS = N + 1
              DO 160 JP = JP1,JP2
                IS = IW(JP)
                IF (FLAG(IS).GE.0) JS = MIN(JS,NEXT(IS))
  160         CONTINUE
              IF (JS.LE.N) THEN
                JS = ABS(KEEP(JS))
                NEXTE(KE) = IPR(JS)
                IPR(JS) = KE
              ELSE
                FLAG(KE) = -1
              END IF
            END IF
            KE = NE
  170     CONTINUE
  180   CONTINUE
        NSC(1) = 0
        NSC(3) = 0
        IF (KIND.EQ.1) THEN
          COUNT(MS) = NVARS + 1
          FLAG(MS) = -1
        ELSE
          NV2 = NVARS - NV1 + NVC
          COUNT(PIVOT(1)) = NV1
          COUNT(PIVOT(2)) = NV2
          IF (A11) COUNT(PIVOT(1)) = NV1 + 1
          IF (A22) COUNT(PIVOT(2)) = NV2 + 1
          IF (A21) THEN
            COUNT(PIVOT(1)) = COUNT(PIVOT(1)) + 1
            COUNT(PIVOT(2)) = COUNT(PIVOT(2)) + 1
          END IF
          FLAG(PIVOT(1)) = -1
          FLAG(PIVOT(2)) = -1
          KIND = 4
          IF (A11) THEN
            IF (A21 .AND. .NOT.A22) THEN
              KIND = 2
              NSC(1) = NVARS - NV2
              IPR(PIVOT(2)) = -1
            END IF
          ELSE
            IF (A21) THEN
              IF (A22) THEN
                KIND = 2
                NSC(3) = NV2 - NVC
                IPR(PIVOT(1)) = -1
              ELSE
                KIND = 3
                IPR(PIVOT(1)) = -1
                IPR(PIVOT(2)) = -1
                NSC(1) = NVARS - NV2
                NSC(3) = NV2 - NVC
              END IF
            END IF
          END IF
          IF (.NOT.A22) THEN
            K1 = 1
            DO 190 K = 1,NV1
              IF (FLAG(VARS(K)).EQ.2) THEN
                KS = VARS(K)
                VARS(K) = VARS(K1)
                VARS(K1) = KS
                K1 = K1 + 1
              END IF
  190       CONTINUE
          END IF
          IF (ME.EQ.PIVOT(1)) THEN
            NEXT(PIVOT(2)) = -ME
            LAST(PIVOT(2)) = -ME
            COUNT(PIVOT(2)) = -COUNT(PIVOT(2))
          ELSE
            NEXT(PIVOT(1)) = -ME
            LAST(PIVOT(1)) = -ME
            COUNT(PIVOT(1)) = -COUNT(PIVOT(1))
          END IF
          IF (KIND.EQ.4) THEN
            COUNT(PIVOT(1)) = NVARS + 2
            COUNT(PIVOT(2)) = NVARS + 2
            IPR(PIVOT(1)) = 0
            IPR(PIVOT(2)) = 0
            IPR(ME) = 2
          END IF
        END IF
        NEXT(ME) = 0
        IF (NVARS.EQ.0) THEN
          IF (KIND.EQ.1) THEN
            IF (.NOT.A11) RANK = RANK - 1
          ELSE
            IF (.NOT.A21) THEN
              IF (.NOT.A11) RANK = RANK - 1
              IF (.NOT.A22) RANK = RANK - 1
            END IF
          END IF
          IPE(ME) = 0
          GO TO 210
        END IF
        IF (IWFR+NVARS+3.GE.LW) THEN
          CALL MA47FD(N,IPE,FLAG,IW,IWFR-1,IWFR,NCMP)
          IF (IWFR+NVARS+3.GE.LW) THEN
            INFO(1) = -3
            INFO(2) = IWFR + NVARS + 4 - LW
            RETURN
          END IF
        END IF
        IP = IWFR
        IWFR = IWFR + 3
        MS = N + 1
        DO 200 K = 1,NVARS
          IS = VARS(K)
          MS = MIN(MS,NEXT(IS))
          FLAG(IS) = 4
          IW(IWFR) = IS
          IWFR = IWFR + 1
  200   CONTINUE
        IW(IP) = IWFR - IP - 1
        IW(IP+1) = NSC(1)
        IW(IP+2) = NSC(3)
        FLAG(ME) = -4
        IPE(ME) = IP
        MS = ABS(KEEP(MS))
        NEXTE(ME) = IPR(MS)
        IPR(MS) = ME
  210 CONTINUE
  220 IF (RANK.LT.N) INFO(1) = INFO(1) + 4
      INFO(8) = N - RANK
      INFO(12) = NCMP
      RETURN
  230 INFO(2) = NEL + 1
      INFO(1) = -6
      END
      SUBROUTINE MA47LD(N,FATHER,SON,NODE,COUNT,NE,NA,MARK,PERM,NODES,
     +                  ICNTL)
      INTEGER N,FATHER(N),SON(N),NODE(N),COUNT(N),NE(N),NA(N),MARK(N),
     +        PERM(N),NODES,ICNTL(7)
      INTEGER COUNT2,I,IBRTHR,IFATHR,ISON,J,K,L,LDIAG,LEVEL,MP,NDE,
     +        NEMIN,NR,NRL,NST,NVPIV,STAGE
      MP = ICNTL(2)
      LDIAG = ICNTL(3)
      IF (MP.LE.0) LDIAG = 0
      NEMIN = ICNTL(6)
      DO 10 I = 1,N
        SON(I) = 0
        NODE(I) = 0
   10 CONTINUE
      NRL = N + 1
      DO 20 I = 1,N
        IFATHR = -FATHER(I)
        NA(I) = -IFATHR
        IF (IFATHR.EQ.0) THEN
          NRL = NRL - 1
          MARK(NRL) = I
        ELSE
          IBRTHR = SON(IFATHR)
          IF (IBRTHR.GT.0) NA(I) = IBRTHR
          SON(IFATHR) = I
          IF (COUNT(I).LT.0) THEN
            NE(IFATHR) = NE(IFATHR)*2
            NE(I) = 0
            NODE(IFATHR) = -COUNT(I)
          END IF
        END IF
   20 CONTINUE
      NR = NRL
      I = 0
      DO 70 K = 1,N
        IF (I.LE.0) THEN
          I = MARK(NR)
          NR = NR + 1
          LEVEL = N
          NST = 0
          PERM(LEVEL) = I
        END IF
        DO 30 L = 1,N
          IF (SON(I).LE.0) GO TO 40
          ISON = SON(I)
          SON(I) = 0
          I = ISON
          IF (NODE(I).NE.0) NST = NST + 1
          LEVEL = LEVEL - 1
          PERM(LEVEL) = I
   30   CONTINUE
   40   NVPIV = NE(I)
        IF (NVPIV.EQ.0) GO TO 60
        IF (LEVEL.EQ.N) GO TO 60
        IFATHR = PERM(LEVEL+1)
        IF (NODE(I).EQ.0 .AND. NODE(IFATHR).EQ.0) THEN
          IF (COUNT(I)-NVPIV.EQ.COUNT(IFATHR)) GO TO 50
          IF (NST.GT.0) GO TO 60
          IF (NVPIV.GE.NEMIN) GO TO 60
          IF (NE(IFATHR).GE.NEMIN) GO TO 60
        ELSE
          IF (NODE(I).EQ.0 .OR. NODE(IFATHR).EQ.0) GO TO 60
          IF (NVPIV.GT.0 .AND. NE(IFATHR).GT.0) THEN
            IF (NODE(I)-NVPIV/2.NE.NODE(IFATHR)) GO TO 60
            IF (COUNT(I)-NVPIV.NE.COUNT(IFATHR)) GO TO 60
          ELSE IF (NVPIV.LT.0 .AND. NE(IFATHR).LT.0) THEN
            NVPIV = -NVPIV/2
            IF (NODE(I)-NVPIV.NE.NODE(IFATHR)) GO TO 60
            IF (COUNT(I)-NVPIV.NE.COUNT(IFATHR)) GO TO 60
            IF (COUNT(I).EQ.NODE(I)) GO TO 60
          ELSE
            GO TO 60
          END IF
          NODE(IFATHR) = NODE(I)
          NST = NST - 1
        END IF
   50   NE(IFATHR) = NE(IFATHR) + NE(I)
        NE(I) = 0
        IF (LDIAG.GT.4) WRITE (MP,'(A,2I5)') ' Merging nodes',I,IFATHR
        COUNT(IFATHR) = COUNT(IFATHR) + NVPIV
        NODE(I) = -1
   60   IBRTHR = NA(I)
        IF (NODE(I).GT.0) NST = NST - 1
        IF (IBRTHR.GT.0) THEN
          PERM(LEVEL) = IBRTHR
          I = IBRTHR
          IF (NODE(I).GT.0) NST = NST + 1
        ELSE
          LEVEL = LEVEL + 1
          I = -IBRTHR
        END IF
   70 CONTINUE
      DO 80 I = 1,N
        SON(I) = 0
   80 CONTINUE
      DO 90 I = 1,N
        IFATHR = -FATHER(I)
        NA(I) = -IFATHR
        IF (IFATHR.NE.0) THEN
          IBRTHR = SON(IFATHR)
          IF (IBRTHR.GT.0) NA(I) = IBRTHR
          SON(IFATHR) = I
        END IF
   90 CONTINUE
      I = 0
      NR = NRL
      DO 120 K = 1,N
        IF (I.LE.0) THEN
          I = MARK(NR)
          NR = NR + 1
          LEVEL = N
          PERM(N) = I
        END IF
        DO 100 L = 1,N
          IF (SON(I).LE.0) GO TO 110
          ISON = SON(I)
          SON(I) = 0
          I = ISON
          FATHER(I) = -PERM(LEVEL)
          IF (NODE(I).GE.0) THEN
            LEVEL = LEVEL - 1
            PERM(LEVEL) = I
          END IF
  100   CONTINUE
  110   IBRTHR = NA(I)
        IF (IBRTHR.GT.0) THEN
          IF (NODE(I).LT.0) LEVEL = LEVEL - 1
          I = IBRTHR
          PERM(LEVEL) = I
          FATHER(I) = -PERM(LEVEL+1)
          IF (NODE(I).LT.0) LEVEL = LEVEL + 1
        ELSE
          IF (NODE(I).GE.0) LEVEL = LEVEL + 1
          I = -IBRTHR
        END IF
  120 CONTINUE
      DO 130 I = 1,N
        SON(I) = 0
  130 CONTINUE
      DO 140 I = 1,N
        IF (NE(I).EQ.0 .AND. COUNT(I).GE.0) THEN
          IFATHR = -FATHER(I)
          IBRTHR = SON(IFATHR)
          IF (IBRTHR.GT.0) FATHER(I) = IBRTHR
          SON(IFATHR) = I
        END IF
  140 CONTINUE
      DO 150 I = 1,N
        IF (COUNT(I).LT.0) THEN
          IFATHR = -FATHER(I)
          IBRTHR = SON(IFATHR)
          IF (IBRTHR.GT.0) FATHER(I) = IBRTHR
          SON(IFATHR) = I
        END IF
  150 CONTINUE
      DO 160 I = 1,N
        IF (NE(I).NE.0) THEN
          IFATHR = -FATHER(I)
          IF (IFATHR.NE.0) THEN
            IBRTHR = SON(IFATHR)
            IF (IBRTHR.GT.0) FATHER(I) = IBRTHR
            SON(IFATHR) = I
          END IF
        END IF
  160 CONTINUE
      STAGE = 1
      I = 0
      NR = NRL
      DO 220 K = 1,N
        IF (I.LE.0) THEN
          I = MARK(NR)
          NR = NR + 1
          LEVEL = N
          NA(N) = 0
        END IF
        L = LEVEL
        DO 170 LEVEL = L,1,-1
          IF (SON(I).LE.0) GO TO 180
          ISON = SON(I)
          SON(I) = 0
          I = ISON
          NA(LEVEL-1) = 0
  170   CONTINUE
  180   PERM(K) = I
        COUNT2 = NODE(I)
        NODE(I) = STAGE
        IF (NE(I).EQ.0) GO TO 210
        IF (LEVEL.LT.N) NA(LEVEL+1) = NA(LEVEL+1) + 1
        NA(STAGE) = NA(LEVEL)
        IF (COUNT2.EQ.0) THEN
          MARK(STAGE) = (COUNT(I)-1)**2
        ELSE IF (NE(I).GT.0) THEN
          DO 190 J = K - NE(I) + 1,K - NE(I)/2
            NODE(PERM(J)) = -NODE(PERM(J))
  190     CONTINUE
          MARK(STAGE) = (COUNT(I)+COUNT2-3)* (COUNT2-1)
        ELSE
          DO 195 J = K + NE(I) + 1,K
            NODE(PERM(J)) = -NODE(PERM(J))
  195     CONTINUE
          MARK(STAGE) = (COUNT(I)-1)* (COUNT2-1)
        END IF
        STAGE = STAGE + 1
  210   IBRTHR = FATHER(I)
        IF (IBRTHR.GT.0) THEN
          NA(LEVEL) = 0
          I = IBRTHR
        ELSE
          LEVEL = LEVEL + 1
          ISON = I
          I = -IBRTHR
        END IF
  220 CONTINUE
      NODES = STAGE - 1
      K = 0
      L = 1
      DO 240 NDE = 1,NODES
        FATHER(NDE) = NODES + 1
        DO 230 I = 1,NA(NDE)
          FATHER(NA(K)) = NDE
          K = K - 1
  230   CONTINUE
        K = K + 1
        NA(K) = NDE
  240 CONTINUE
      END
      SUBROUTINE MA47MD(N,NE,IW,LIW,LROW,PERM,NODES,ELEMS,NODE,PPOS,
     +                  FATHER,NBLOC,INFO,RINFO)
      INTEGER N,NE
      INTEGER LIW,IW(LIW),LROW(N),PERM(N),NODES,ELEMS(NODES),NODE(N),
     +        PPOS(0:N),FATHER(NODES),NBLOC,INFO(24)
      DOUBLE PRECISION RINFO(4)
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0.0D0)
      INTEGER APOS,APOS1,APOS3,APOS4,ELT,FLAG
      DOUBLE PRECISION FLOPSB,FLOPSX
      INTEGER I,IASS,IELL,IELLN,IINPUT,INTSPA,IORG,ISTK,ISW,IWNFS,IWPOS,
     +        J,J1,J2,JJ,JP1,JP2,JP3,JP4,KIND,LAELL,LEFT,LIELL,LOOP,
     +        MAXFRT,N1,N2,N3,NCMP,NELL,NFRONT
      LOGICAL NOSURE
      INTEGER NOTIN1,NOXO,NPIV,NPOTPV,NSC1,NSC2,NSCHUR,NSTKAC(2),NTILE,
     +        NTOTPV,NULL,NUM1,NUM2,NUM3,NUMM1,NUMM2,NUMM3,NUMORG,NORG1,
     +        PTRIRN,RLSPA,RSTKAC
      INTRINSIC MIN,MAX,ABS
      EXTERNAL MA47PD
      NCMP = 0
      NTILE = 0
      NOXO = 0
      FLOPSB = ZERO
      FLOPSX = ZERO
      MAXFRT = 0
      NSTKAC(1) = NE
      NSTKAC(2) = NE
      RLSPA = NE
      INTSPA = NE
      PTRIRN = LIW - NE + 1
      ISTK = PTRIRN - 1
      IINPUT = 0
      NTOTPV = 0
      NPOTPV = 0
      APOS = 1
      IWPOS = 8
      DO 10 I = 0,N
        PPOS(I) = N + 1
   10 CONTINUE
      DO 20 I = 1,NODES
        ELEMS(I) = 0
   20 CONTINUE
      DO 380 IASS = 1,NODES
        INTSPA = MAX(INTSPA,IWPOS+2* (N-NTOTPV)+NSTKAC(2))
        IF (1+2* (N-NTOTPV).GT.ISTK) THEN
          CALL MA47PD(IW,ISTK,PTRIRN,IINPUT,NCMP)
          IF (1+2* (N-NTOTPV).GT.ISTK) THEN
            INFO(2) = NSTKAC(2) + 1 + 2* (N-NTOTPV)
            INFO(1) = -3
            GO TO 390
          END IF
        END IF
        FLAG = ISTK - (N-NTOTPV)
        NUMORG = 0
        DO 30 I = NPOTPV + 1,N
          J = PERM(I)
          IF (ABS(NODE(J)).GT.IASS) GO TO 40
          NUMORG = NUMORG + 1
          IW(NUMORG) = J
          PPOS(J) = NUMORG
          IW(FLAG+PPOS(J)) = 0
   30   CONTINUE
   40   NULL = 0
        DO 50 I = 1,NUMORG
          NULL = NULL + LROW(NPOTPV+I)
   50   CONTINUE
        IF (NULL.EQ.0 .AND. ELEMS(IASS).EQ.0) THEN
          NPOTPV = NPOTPV + NUMORG
          GO TO 380
        END IF
        KIND = 1
        IF (NUMORG.GE.2) THEN
          IF (NODE(PERM(NPOTPV+NUMORG/2)).LT.0) THEN
            KIND = 2
            IF (NODE(PERM(NPOTPV+NUMORG)).LT.0) KIND = 3
          END IF
        END IF
        IWNFS = NUMORG + 1
        NORG1 = NUMORG
        IF (KIND.GT.1) NORG1 = NUMORG/2
        RSTKAC = 0
        DO 170 LOOP = 1,MIN(2,KIND)
          NUM1 = IWNFS - NUMORG - 1
          NUM2 = 0
          IF (NULL.EQ.0) GO TO 80
          J1 = PTRIRN
          DO 70 IORG = (LOOP-1)*NORG1 + 1,LOOP*NORG1
            J2 = J1 + LROW(NPOTPV+IORG) - 1
            DO 60 JJ = J1,J2
              J = IW(JJ)
              IF (PPOS(J).LE.N) THEN
                IF (LOOP.EQ.2) THEN
                  IF (IW(FLAG+PPOS(J)).EQ.1) THEN
                    IW(FLAG+PPOS(J)) = 3
                    NUM2 = NUM2 + 1
                  END IF
                END IF
              ELSE
                IW(IWNFS) = J
                IWNFS = IWNFS + 1
                PPOS(J) = IWNFS - 1
                IW(FLAG+PPOS(J)) = LOOP
              END IF
   60       CONTINUE
            J1 = J2 + 1
   70     CONTINUE
          NSTKAC(2) = NSTKAC(2) - J1 + PTRIRN
          RSTKAC = RSTKAC + J1 - PTRIRN
          IINPUT = IINPUT + J1 - PTRIRN
          PTRIRN = J1
   80     NELL = ELEMS(IASS)
          IELL = ISTK + 1
          PPOS(0) = 0
          DO 160 ELT = 1,NELL
            IF (IW(IELL).LT.0) IELL = IELL - IW(IELL)
            JP1 = IELL + 3
            JP2 = JP1 + IW(IELL+1)
            JP3 = JP2 + IW(IELL+2)
            JP4 = IELL + IW(IELL) - 2
            J1 = JP1
            J2 = JP4
            DO 90 JJ = JP2,JP3 - 1
              J = IW(JJ)
              IF (PPOS(J).GE. (LOOP-1)*NORG1+1 .AND.
     +            PPOS(J).LE.LOOP*NORG1) GO TO 130
   90       CONTINUE
            NOTIN1 = 0
            DO 100 JJ = JP1,JP2 - 1
              J = IW(JJ)
              IF (PPOS(J).GE. (LOOP-1)*NORG1+1 .AND.
     +            PPOS(J).LE.LOOP*NORG1) GO TO 110
  100       CONTINUE
            NOTIN1 = 1
            J2 = JP3 - 1
  110       DO 120 JJ = JP3,JP4
              J = IW(JJ)
              IF (PPOS(J).GE. (LOOP-1)*NORG1+1 .AND.
     +            PPOS(J).LE.LOOP*NORG1) GO TO 130
  120       CONTINUE
            IF (NOTIN1.EQ.1) GO TO 150
            J1 = JP2
  130       DO 140 JJ = J1,J2
              J = IW(JJ)
              IF (J.EQ.0) GO TO 140
              IF (PPOS(J).LE.N) THEN
                NULL = 1
                IF (LOOP.EQ.2) THEN
                  IF (IW(FLAG+PPOS(J)).EQ.1) THEN
                    IW(FLAG+PPOS(J)) = 3
                    NUM2 = NUM2 + 1
                  END IF
                END IF
              ELSE
                IW(IWNFS) = J
                IWNFS = IWNFS + 1
                PPOS(J) = IWNFS - 1
                IW(FLAG+PPOS(J)) = LOOP
              END IF
  140       CONTINUE
  150       IELL = IELL + IW(IELL)
  160     CONTINUE
  170   CONTINUE
        PPOS(0) = N + 1
        IF (NULL.EQ.0) GO TO 190
        NFRONT = IWNFS - 1
        MAXFRT = MAX(MAXFRT,NFRONT)
        IF (KIND.EQ.1) THEN
          NUM1 = 0
          NUM2 = NFRONT - NUMORG
          NUM3 = 0
        ELSE
          IF (KIND.EQ.3) THEN
            JJ = NUMORG
            J1 = JJ
            DO 180 J = 1,NUM1
              JJ = JJ + 1
              IF (IW(FLAG+JJ).EQ.1) THEN
                J1 = J1 + 1
                ISW = IW(JJ)
                IW(JJ) = IW(J1)
                IW(J1) = ISW
                PPOS(IW(JJ)) = JJ
                PPOS(IW(J1)) = J1
              END IF
  180       CONTINUE
            NUM3 = NFRONT - NUMORG - NUM1
            NUM1 = NUM1 - NUM2
          ELSE
            NUM2 = NUM1
            NUM1 = 0
            NUM3 = NFRONT - NUMORG - NUM2
          END IF
        END IF
        RLSPA = MAX(RLSPA,APOS+ (NUMORG* (NUMORG+1))/2+NFRONT*NUMORG+
     +          NSTKAC(1))
  190   IF (NULL.NE.0) NTOTPV = NTOTPV + NUMORG
        NPOTPV = NPOTPV + NUMORG
        IELL = ISTK + 1
        DO 250 ELT = 1,NELL
          IF (IW(IELL).LT.0) IELL = IELL - IW(IELL)
          LEFT = 0
          JP1 = IELL + 3
          JP4 = IELL + IW(IELL) - 2
          JP2 = JP1 + IW(IELL+1)
          NUMM1 = IW(IELL+1)
          NUMM2 = IW(IELL+2)
          JP3 = JP2 + NUMM2
          NUMM3 = IW(IELL) - NUMM1 - NUMM2 - 4
          DO 200 JJ = JP1,JP2 - 1
            J = IW(JJ)
            IF (J.EQ.0) GO TO 200
            IF (PPOS(J).LE.NUMORG) THEN
              IW(JJ) = 0
            ELSE
              LEFT = LEFT + 1
            END IF
  200     CONTINUE
          DO 210 JJ = JP2,JP3 - 1
            J = IW(JJ)
            IF (J.EQ.0) GO TO 210
            IF (PPOS(J).LE.NUMORG) THEN
              IW(JJ) = 0
            ELSE
              LEFT = LEFT + 1
            END IF
  210     CONTINUE
          IF (LEFT.EQ.0) GO TO 230
          DO 220 JJ = JP3,JP4
            J = IW(JJ)
            IF (J.EQ.0) GO TO 220
            IF (PPOS(J).LE.NUMORG) THEN
              IW(JJ) = 0
            ELSE
              LEFT = LEFT + 1
            END IF
  220     CONTINUE
  230     PPOS(0) = N + 1
          IELLN = IELL + IW(IELL)
          IF (LEFT.EQ.0) THEN
            NELL = NELL - 1
            ELEMS(IASS) = ELEMS(IASS) - 1
            LAELL = (NUMM2* (NUMM2+1))/2 + NUMM1* (NUMM2+NUMM3) +
     +              NUMM2*NUMM3 + 2
            NSTKAC(1) = NSTKAC(1) - LAELL
            LIELL = IW(IELL)
            NSTKAC(2) = NSTKAC(2) - LIELL
            J1 = IELL
            J2 = IELLN - 1
            IF (IELLN.LT.PTRIRN-IINPUT) THEN
              IF (IW(IELLN).LT.0) THEN
                LIELL = LIELL - IW(IELLN)
                J2 = J2 - IW(IELLN)
              END IF
            END IF
            IF ((IELL-1).EQ.ISTK) THEN
              ISTK = ISTK + LIELL
            ELSE
              IF (IW(IELL-1).LT.0) THEN
                LIELL = LIELL - IW(IELL-1)
                J1 = IELL + IW(IELL-1)
              END IF
              IW(J1) = -LIELL
              IW(J2) = -LIELL
            END IF
          END IF
          IELL = IELLN
  250   CONTINUE
        NSTKAC(1) = NSTKAC(1) - RSTKAC
        IF (NULL.EQ.0) GO TO 380
        NPIV = NUMORG
        IF (KIND.EQ.2) NTILE = NTILE + NPIV/2
        IF (KIND.EQ.3) NOXO = NOXO + NPIV/2
        IF (KIND.EQ.1) THEN
          DO 270 J = 1,NPIV
            FLOPSB = FLOPSB + 1
            J2 = NPIV - J
            DO 260 J1 = 1,NPIV - J
              FLOPSB = FLOPSB + 1 + 2* (NFRONT-NPIV+J2)
              J2 = J2 - 1
  260       CONTINUE
            FLOPSB = FLOPSB + NUM2
  270     CONTINUE
        ELSE
          DO 290 J = 1,NORG1
            FLOPSB = FLOPSB + 3 + (NORG1-J)* (2* (NFRONT-NPIV/2-J)+1)
            J2 = NORG1 - J
            DO 280 J1 = 1,NORG1 - J
              FLOPSB = FLOPSB + 4* (NFRONT-NPIV+J2+1)
              J2 = J2 - 1
  280       CONTINUE
            FLOPSB = FLOPSB + NUM1 + 4*NUM2 + NUM3
  290     CONTINUE
        END IF
        NOSURE = (NBLOC.GE. (NFRONT-NUMORG)) .AND. (KIND.EQ.1)
        IF (NPIV.EQ.NFRONT) GO TO 360
        IF (KIND.GT.1) THEN
          NSC1 = NUM1 + NUM2
          NSC2 = NUM2 + NUM3
        ELSE
          NSC1 = NFRONT - NUMORG
          NSC2 = NFRONT - NUMORG
        END IF
        IF (NSC1*NSC2.EQ.0) GO TO 360
        APOS3 = APOS + (NUMORG* (NUMORG+1))/2 + NFRONT*NPIV
        IF (NOSURE) THEN
          RLSPA = MAX(RLSPA,APOS3+ ((NSC1+1)*NSC1)/2+2+NSTKAC(1))
        ELSE
          APOS4 = APOS3 + NPIV*NSC2
          IF (KIND.GT.1) THEN
            NSCHUR = MAX(NSC1*NSC2+1,NUM1* (NUM2+NUM3)+NUM2*NUM3+
     +               (NUM2* (NUM2+1))/2+2)
          ELSE
            NSCHUR = MAX(NSC1*NSC2+1, (NSC1* (NSC1+1))/2+2)
          END IF
          RLSPA = MAX(RLSPA,APOS4+NSCHUR+NSTKAC(1))
        END IF
        IF (KIND.GT.1) THEN
          FLOPSB = FLOPSB + 2*NPIV*NUM1* (NUM2+NUM3)
          FLOPSB = FLOPSB + NPIV* (2*NUM3+NUM2+1)*NUM2
          FLOPSX = FLOPSX + NPIV*NUM2*NUM3 +
     +             NBLOC* (NBLOC-1)*NPIV* (NUM2/NBLOC)
        ELSE
          FLOPSB = FLOPSB + NPIV* (NFRONT-NUMORG)* (NFRONT-NUMORG+1)
          IF (NPIV.GE.NBLOC)
     +        FLOPSX = FLOPSX + NBLOC* (NBLOC-1)*NPIV* (NUM2/NBLOC)
        END IF
        IF (KIND.GT.1) THEN
          NUMM1 = NUM1
          NUMM2 = NUM2
        ELSE
          NUMM1 = 0
          NUMM2 = NFRONT - NUMORG
        END IF
        NELL = ELEMS(IASS)
        IELL = ISTK + 1
        PPOS(0) = 0
        DO 340 ELT = 1,NELL
          IF (IW(IELL).LT.0) IELL = IELL - IW(IELL)
          IELLN = IELL + IW(IELL)
          JP1 = IELL + 3
          JP2 = JP1 + IW(IELL+1)
          JP3 = JP2 + IW(IELL+2)
          JP4 = IELL + IW(IELL) - 2
          DO 300 JJ = JP2,JP3 - 1
            IF (IW(JJ).EQ.0) GO TO 300
            IF (PPOS(IW(JJ))-NUMORG.LE.NUMM1 .OR.
     +          PPOS(IW(JJ))-NUMORG.GT.NUMM1+NUMM2) GO TO 330
  300     CONTINUE
          APOS1 = 0
          DO 310 JJ = JP1,JP2 - 1
            IF (IW(JJ).EQ.0) GO TO 310
            IF (PPOS(IW(JJ))-NUMORG.LE.NUMM1) THEN
              IF (APOS1.EQ.2) GO TO 330
              APOS1 = 1
            END IF
            IF (PPOS(IW(JJ))-NUMORG.GT.NUMM1+NUMM2) THEN
              IF (PPOS(IW(JJ)).EQ.N+1) GO TO 330
              IF (APOS1.EQ.1) GO TO 330
              APOS1 = 2
            END IF
  310     CONTINUE
          DO 320 JJ = JP3,JP4
            IF (IW(JJ).EQ.0) GO TO 320
            IF (PPOS(IW(JJ))-NUMORG.LE.NUMM1) THEN
              IF (APOS1.EQ.1) GO TO 330
            END IF
            IF (PPOS(IW(JJ))-NUMORG.GT.NUMM1+NUMM2) THEN
              IF (PPOS(IW(JJ)).EQ.N+1) GO TO 330
              IF (APOS1.EQ.2) GO TO 330
            END IF
  320     CONTINUE
          NELL = NELL - 1
          ELEMS(IASS) = ELEMS(IASS) - 1
          N1 = IW(IELL+1)
          N2 = IW(IELL+2)
          N3 = IW(IELL) - 4 - N1 - N2
          LAELL = (N2* (N2+1))/2 + N1* (N2+N3) + N2*N3 + 2
          NSTKAC(1) = NSTKAC(1) - LAELL
          LIELL = IW(IELL)
          NSTKAC(2) = NSTKAC(2) - LIELL
          J1 = IELL
          J2 = IELLN - 1
          IF (IELLN.LT.PTRIRN-IINPUT) THEN
            IF (IW(IELLN).LT.0) THEN
              LIELL = LIELL - IW(IELLN)
              J2 = J2 - IW(IELLN)
            END IF
          END IF
          IF ((IELL-1).EQ.ISTK) THEN
            ISTK = ISTK + LIELL
          ELSE
            IF (IW(IELL-1).LT.0) THEN
              LIELL = LIELL - IW(IELL-1)
              J1 = IELL + IW(IELL-1)
            END IF
            IW(J1) = -LIELL
            IW(J2) = -LIELL
          END IF
  330     IELL = IELLN
  340   CONTINUE
        PPOS(0) = N + 1
        ELEMS(FATHER(IASS)) = ELEMS(FATHER(IASS)) + 1
        IF (KIND.GT.1) THEN
          LAELL = NUM1* (NUM2+NUM3) + (NUM2* (NUM2+1))/2 + NUM2*NUM3 + 2
        ELSE
          LAELL = ((NSC1+1)* (NSC1))/2 + 2
        END IF
        NSTKAC(1) = NSTKAC(1) + LAELL
        LIELL = NFRONT - NUMORG + 4
        NSTKAC(2) = NSTKAC(2) + LIELL
        INTSPA = MAX(INTSPA,IWPOS+2*NFRONT+LIELL+NSTKAC(2))
        IF (1+2*NFRONT+LIELL.GT.ISTK) THEN
          CALL MA47PD(IW,ISTK,PTRIRN,IINPUT,NCMP)
          IF (1+2*NFRONT+LIELL.GT.ISTK) THEN
            INFO(2) = NSTKAC(2) + 1 + 2*NFRONT + LIELL
            INFO(1) = -3
            GO TO 390
          END IF
        END IF
        IW(ISTK) = LIELL
        ISTK = ISTK - 1
        DO 350 I = 1,NFRONT - NUMORG
          IW(ISTK) = IW(NFRONT+1-I)
          ISTK = ISTK - 1
  350   CONTINUE
        IF (KIND.GT.1) THEN
          IW(ISTK) = NUM2
          IW(ISTK-1) = NUM1
        ELSE
          IW(ISTK) = NFRONT - NUMORG
          IW(ISTK-1) = 0
        END IF
        IW(ISTK-2) = LIELL
        ISTK = ISTK - 3
  360   DO 370 JJ = NPIV + 1,NFRONT
          J = ABS(IW(JJ))
          PPOS(J) = N + 1
  370   CONTINUE
        IF (FATHER(IASS).LE.NODES) THEN
          ELEMS(FATHER(IASS)) = ELEMS(FATHER(IASS)) + NELL
        END IF
C********************************
C********************************
        IF (KIND.EQ.1) THEN
          IWPOS = IWPOS - 2
          APOS = APOS + (NPIV* (NPIV+1))/2 + NPIV* (NFRONT-NPIV)
        ELSE
          IF (KIND.EQ.2) THEN
            IWPOS = IWPOS - 1
            APOS = APOS + 3* (NORG1* (NORG1+1))/2 +
     +             NORG1* (NUM1+2*NUM2+NUM3)
          ELSE
            APOS = APOS + NORG1* (NORG1+2) + NORG1* (NUM1+2*NUM2+NUM3)
          END IF
        END IF
        IWPOS = IWPOS + NFRONT + 4
  380 CONTINUE
      INFO(5) = NODES
      INFO(6) = MAX(2*NE,RLSPA)
      INFO(7) = MAX(2*NE,INTSPA)
      INFO(8) = N - NTOTPV
      INFO(9) = MAXFRT
      INFO(10) = APOS - 1
      INFO(11) = IWPOS - 5
      INFO(12) = INFO(12) + NCMP
      INFO(13) = NTILE
      INFO(14) = NOXO
      RINFO(1) = FLOPSB
      RINFO(2) = FLOPSX
  390 RETURN
      END
      SUBROUTINE MA47ND(N,NE,IRN,JCN,MAP,LROW,PERM,COUNT,PERM0)
      INTEGER N,NE
      INTEGER IRN(NE),JCN(NE),MAP(NE),LROW(N),PERM(N),COUNT(0:N),
     +        PERM0(0:N)
      INTEGER I,J,K
      COUNT(0) = 0
      PERM0(0) = 0
      DO 10 I = 1,N
        COUNT(I) = 0
        PERM0(PERM(I)) = I
   10 CONTINUE
      DO 20 K = 1,NE
        I = IRN(K)
        J = JCN(K)
        IF (PERM0(I).LE.PERM0(J)) THEN
          I = PERM0(I)
          COUNT(I) = COUNT(I) + 1
        ELSE
          JCN(K) = I
          IRN(K) = J
          J = PERM0(J)
          COUNT(J) = COUNT(J) + 1
        END IF
   20 CONTINUE
      DO 30 I = 1,N
        LROW(I) = COUNT(I)
        COUNT(I) = COUNT(I-1) + LROW(I)
   30 CONTINUE
      DO 40 K = 1,NE
        I = PERM0(IRN(K))
        MAP(K) = COUNT(I)
        COUNT(I) = COUNT(I) - 1
   40 CONTINUE
      RETURN
      END
      SUBROUTINE MA47OD(N,NE,A,LA,IW,LIW,LROW,PERM,NODES,ELEMS,MARK,
     +                  NODE,PPOS,FATHER,CNTL,NBLOC,INFO,RINFO)
      INTEGER N,NE,LA
      DOUBLE PRECISION A(LA),CNTL(2),RINFO(4)
      INTEGER LIW,IW(LIW),LROW(N),PERM(N),NODES,ELEMS(NODES),
     +        MARK(NODES),NODE(N),PPOS(0:N),FATHER(NODES),NBLOC,INFO(24)
      DOUBLE PRECISION ZERO,HALF,ONE
      PARAMETER (ZERO=0.0D0,HALF=0.5D0,ONE=1.0D0)
      INTEGER AINPUT
      DOUBLE PRECISION AMAX,A1MAX,AMULT1,AMULT2
      INTEGER APOS,APOSA,APOSB,APOSBB,APOSC,APOSH,APOSI,APOSJ,APOS1,
     +        APOS2,APOS3,APOS4,ASTK,ATRASH,CNTNZ1,CNTNZ2
      DOUBLE PRECISION DETPIV
      INTEGER ELT
      DOUBLE PRECISION FLOPSB,FLOPSX
      INTEGER I,I1,IASS,IBEG,IELL,IELLN,IEND,IEXCH,IINPUT,INC,INTSPA,
     +        IORG,IPIV,IPOS,IROW,ISTK,ISWOP,IWNFS,IWPOS,J,JA1,JA2,JAY,
     +        J1,J2,JCOL,JJ,JJJ,JMAX,J1MAX,JP1,JP2,JP3,JP4,JPIV,K,K1,K2,
     +        KBLK,KIND,KR,KROW,KSWEEP,L,LAELL,LDUMMY,LEFT,LIELL,LNPIV,
     +        LPIV
      LOGICAL LTWO
      INTEGER MA1,MAXFRT
      DOUBLE PRECISION MAXPIV
      INTEGER NASS,NBLK,NCMPBI,NCMPBR,NCOL,NEIG,NELL,NFRONT,NFS,NFULLB,
     +        NIRBDU,NNELL,NNULL
      LOGICAL NOSURE
      INTEGER NOTIN1,NOXOB,NPIV,NPIVD2,NPOTPV,NRLBDU,NSC1,NSC2,NSCHUR,
     +        NSTKAC(2),NSTRUC,NTILEB,NTOTPV,NUM1,NUM2,NUM3,NUMM1,NUMM2,
     +        NUMORG,NUMPVT,OFFDAG
      DOUBLE PRECISION PIVOT
      INTEGER PIVSIZ,POSELT,POSPV1,POSPV2,PTRA,PTRELT,PTRIRN,RLSPA
      DOUBLE PRECISION RMAX,SWOP,TMAX,TOL,UU
      INTEGER ZCOL
      DOUBLE PRECISION ZONE
C?? We could refine this count and have NCOLST, NCOLP etc.
      INTRINSIC MIN,MAX,ABS,NINT
      INTEGER IDAMAX
      EXTERNAL MA47PD,MA47SD,MA47YD,MA47WD,MA47XD,IDAMAX,DGEMM
      TOL = CNTL(2)
      NBLK = 0
      NTILEB = 0
      NOXOB = 0
      NFULLB = 0
      NCMPBI = 0
      NCMPBR = 0
      FLOPSB = ZERO
      FLOPSX = ZERO
      NEIG = 0
      MAXFRT = 0
      UU = MIN(CNTL(1),HALF)
      UU = MAX(UU,ZERO)
      DO 10 I = 0,N
        PPOS(I) = N + 1
   10 CONTINUE
      DO 20 I = 1,NODES
        ELEMS(I) = 0
   20 CONTINUE
      IWPOS = 8
      APOSBB = 1
      NSTKAC(1) = NE
      NSTKAC(2) = NE
      INTSPA = NE
      RLSPA = NE
      PTRIRN = LIW - NE + 1
      PTRA = LA - NE + 1
      ISTK = PTRIRN - 1
      ASTK = PTRA - 1
      AINPUT = 0
      IINPUT = 0
      NTOTPV = 0
      NPOTPV = 0
      DO 2160 IASS = 1,NODES
        NNULL = 0
        INTSPA = MAX(INTSPA,IWPOS+2* (N-NTOTPV)+NSTKAC(2))
        IF (IWPOS+2* (N-NTOTPV).GT.ISTK) THEN
          CALL MA47PD(IW,ISTK,PTRIRN,IINPUT,NCMPBI)
          IF (IWPOS+2* (N-NTOTPV).GT.ISTK) THEN
            INFO(2) = INTSPA
            INFO(1) = -3
            GO TO 2170
          END IF
        END IF
        NUMORG = 0
        DO 30 I = NPOTPV + 1,N
          J = PERM(I)
          IF (ABS(NODE(J)).GT.IASS) GO TO 40
          IW(IWPOS+NUMORG) = J
          NUMORG = NUMORG + 1
          PPOS(J) = NUMORG
   30   CONTINUE
   40   KIND = 1
        IF (NUMORG.GE.2) THEN
          IF (NODE(PERM(NPOTPV+NUMORG/2)).LT.0) THEN
            KIND = 2
            NUMPVT = NUMORG/2
            IF (NODE(PERM(NPOTPV+NUMORG)).LT.0) KIND = 3
          END IF
        END IF
        NASS = NUMORG
        NELL = ELEMS(IASS)
        NNELL = NELL
        IELL = ISTK + 1
        DO 70 ELT = 1,NELL
          IF (IW(IELL).LT.0) IELL = IELL - IW(IELL)
          IF (IW(IELL+1).NE.-1) GO TO 60
          DO 50 JJ = IELL + 3,IELL + 2 + IW(IELL+2)
            J = IW(JJ)
            IW(IWPOS+NASS) = J
            NASS = NASS + 1
            PPOS(J) = NASS
   50     CONTINUE
   60     IELL = IELL + IW(IELL)
   70   CONTINUE
        IWNFS = IWPOS + NASS
        J1 = PTRIRN
        DO 90 IORG = 1,NUMORG
          J2 = J1 + LROW(NPOTPV+IORG) - 1
          DO 80 JJ = J1,J2
            J = IW(JJ)
            IF (PPOS(J).LE.N) GO TO 80
            IW(IWNFS) = J
            IWNFS = IWNFS + 1
            PPOS(J) = IWNFS - IWPOS
   80     CONTINUE
          J1 = J2 + 1
   90   CONTINUE
        IELL = ISTK + 1
        DO 170 ELT = 1,NELL
          IF (IW(IELL).LT.0) IELL = IELL - IW(IELL)
          IF (IW(IELL+1).EQ.-1) GO TO 160
          JP1 = IELL + 3
          JP2 = JP1 + IW(IELL+1)
          JP3 = JP2 + IW(IELL+2)
          JP4 = IELL + IW(IELL) - 2
          J1 = JP1
          J2 = JP4
          DO 100 JJ = JP2,JP3 - 1
            J = IW(JJ)
            IF (PPOS(J).LE.NASS) GO TO 140
  100     CONTINUE
          NOTIN1 = 0
          DO 110 JJ = JP1,JP2 - 1
            J = IW(JJ)
            IF (PPOS(J).LE.NASS) GO TO 120
  110     CONTINUE
          NOTIN1 = 1
          J2 = JP3 - 1
  120     DO 130 JJ = JP3,JP4
            J = IW(JJ)
            IF (PPOS(J).LE.NASS) GO TO 140
  130     CONTINUE
          IF (NOTIN1.EQ.1) GO TO 160
          J1 = JP2
  140     PPOS(0) = 0
          DO 150 JJ = J1,J2
            J = IW(JJ)
            IF (PPOS(J).LE.N) GO TO 150
            IW(IWNFS) = J
            IWNFS = IWNFS + 1
            PPOS(J) = IWNFS - IWPOS
  150     CONTINUE
          PPOS(0) = N + 1
  160     IELL = IELL + IW(IELL)
  170   CONTINUE
        NCOL = IWNFS - IWPOS
        IELL = ISTK + 1
        DO 200 ELT = 1,NELL
          IF (IW(IELL).LT.0) IELL = IELL - IW(IELL)
          IF (IW(IELL+1).NE.-1) GO TO 190
          PPOS(0) = 0
          DO 180 JJ = IELL + 3 + IW(IELL+2),IELL + IW(IELL) - 2
            J = IW(JJ)
            IF (PPOS(J).LE.N) GO TO 180
            IW(IWNFS) = J
            IWNFS = IWNFS + 1
            PPOS(J) = IWNFS - IWPOS
  180     CONTINUE
          PPOS(0) = N + 1
  190     IELL = IELL + IW(IELL)
  200   CONTINUE
        NFRONT = IWNFS - IWPOS
        MAXFRT = MAX(MAXFRT,NFRONT)
        IF (INFO(1).NE.-4) THEN
          APOS = APOSBB + (NASS* (NASS+1))/2
        ELSE
          APOS = 1
        END IF
        RLSPA = MAX(RLSPA,INFO(2)+APOS+NFRONT*NASS+NSTKAC(1))
        IF (APOS+NFRONT*NASS.GT.ASTK) THEN
          CALL MA47SD(A,ASTK,PTRA,AINPUT,NCMPBR)
          IF (APOS+NFRONT*NASS.GT.ASTK) THEN
            INFO(2) = INFO(2) + APOS - 1
            APOS = 1
            APOSBB = 1
            INFO(1) = -4
            IF (NFRONT*NASS.GT.ASTK) THEN
              INFO(2) = RLSPA
              GO TO 2170
            END IF
          END IF
        END IF
        ATRASH = APOS + NFRONT*NASS
        DO 210 JJ = APOS,ATRASH
          A(JJ) = ZERO
  210   CONTINUE
        J1 = PTRIRN
        DO 230 IORG = 1,NUMORG
          J = PERM(NPOTPV+IORG)
          APOSI = APOS + (PPOS(J)-1)*NFRONT - 1
          J2 = J1 + LROW(NPOTPV+IORG) - 1
          DO 220 JJ = J1,J2
            APOS2 = APOSI + PPOS(IW(JJ))
            A(APOS2) = A(APOS2) + A(PTRA)
            PTRA = PTRA + 1
  220     CONTINUE
          AINPUT = AINPUT + J2 - J1 + 1
          NSTKAC(1) = NSTKAC(1) - J2 + J1 - 1
          J1 = J2 + 1
  230   CONTINUE
        IINPUT = IINPUT + J1 - PTRIRN
        NSTKAC(2) = NSTKAC(2) - J1 + PTRIRN
        PTRIRN = J1
        NPOTPV = NPOTPV + NUMORG
        NFS = NUMORG
        IELL = ISTK + 1
        PTRELT = ASTK + 1
        DO 380 ELT = 1,NELL
          IF (NINT(A(PTRELT)).LT.0) PTRELT = PTRELT - NINT(A(PTRELT))
          IF (IW(IELL).LT.0) IELL = IELL - IW(IELL)
          LEFT = 0
          POSELT = PTRELT + 1
          JP1 = IELL + 3
          JP4 = IELL + IW(IELL) - 2
          IF (IW(IELL+1).NE.-1) GO TO 260
          LIELL = JP4 - JP1 + 1
          I1 = 1
          DO 250 JJ = JP1,JP4
            J = IW(JJ)
            APOS2 = APOS + NFS*NFRONT - 1 + PPOS(J)
            APOS1 = POSELT
            DO 240 I = 1,MIN(I1,IW(IELL+2))
              A(APOS2) = A(APOS1)
              APOS1 = APOS1 + LIELL
              APOS2 = APOS2 + NFRONT
  240       CONTINUE
            I1 = I1 + 1
            POSELT = POSELT + 1
  250     CONTINUE
          NFS = NFS + IW(IELL+2)
          GO TO 370
  260     JP2 = JP1 + IW(IELL+1)
          NUM1 = IW(IELL+1)
          NUM2 = IW(IELL+2)
          JP3 = JP2 + NUM2
          NUM3 = IW(IELL) - NUM1 - NUM2 - 4
          DO 280 JJ = JP1,JP2 - 1
            J = IW(JJ)
            IF (J.EQ.0) GO TO 280
            IF (PPOS(J).LE.NASS) THEN
              POSELT = PTRELT + 1 + (JJ-JP1)* (NUM2+NUM3)
              APOSI = APOS + (PPOS(J)-1)*NFRONT - 1
              PPOS(0) = ATRASH - APOSI
              DO 270 JJJ = JP2,JP4
                APOS2 = APOSI + PPOS(IW(JJJ))
                A(APOS2) = A(APOS2) + A(POSELT)
                POSELT = POSELT + 1
  270         CONTINUE
              IW(JJ) = 0
            ELSE
              LEFT = LEFT + 1
            END IF
  280     CONTINUE
          DO 320 JJ = JP2,JP3 - 1
            J = IW(JJ)
            IF (J.EQ.0) GO TO 320
            IF (PPOS(J).LE.NASS) THEN
              POSELT = PTRELT + 1 + JJ - JP2
              INC = NUM2 + NUM3
              APOSI = APOS + (PPOS(J)-1)*NFRONT - 1
              PPOS(0) = ATRASH - APOSI
              DO 290 JJJ = JP1,JP2 - 1
                APOS2 = APOSI + PPOS(IW(JJJ))
                A(APOS2) = A(APOS2) + A(POSELT)
                POSELT = POSELT + INC
  290         CONTINUE
              DO 300 JJJ = JP2,JJ - 1
                APOS2 = APOSI + PPOS(IW(JJJ))
                A(APOS2) = A(APOS2) + A(POSELT)
                INC = INC - 1
                POSELT = POSELT + INC
  300         CONTINUE
              DO 310 JJJ = JJ,JP4
                APOS2 = APOSI + PPOS(IW(JJJ))
                A(APOS2) = A(APOS2) + A(POSELT)
                POSELT = POSELT + 1
  310         CONTINUE
              IW(JJ) = 0
            ELSE
              LEFT = LEFT + 1
            END IF
  320     CONTINUE
          IF (LEFT.EQ.0) GO TO 360
          DO 350 JJ = JP3,JP4
            J = IW(JJ)
            IF (J.EQ.0) GO TO 350
            IF (PPOS(J).LE.NASS) THEN
              POSELT = PTRELT + 1 + NUM2 + JJ - JP3
              APOSI = APOS + (PPOS(J)-1)*NFRONT - 1
              PPOS(0) = ATRASH - APOSI
              INC = NUM2 + NUM3
              DO 330 JJJ = JP1,JP2 - 1
                APOS2 = APOSI + PPOS(IW(JJJ))
                A(APOS2) = A(APOS2) + A(POSELT)
                POSELT = POSELT + INC
  330         CONTINUE
              DO 340 JJJ = JP2,JP3 - 1
                APOS2 = APOSI + PPOS(IW(JJJ))
                A(APOS2) = A(APOS2) + A(POSELT)
                INC = INC - 1
                POSELT = POSELT + INC
  340         CONTINUE
              IW(JJ) = 0
            ELSE
              LEFT = LEFT + 1
            END IF
  350     CONTINUE
  360     PPOS(0) = N + 1
  370     IELLN = IELL + IW(IELL)
          POSELT = PTRELT + NINT(A(PTRELT))
          IF (LEFT.EQ.0) THEN
            NELL = NELL - 1
            NNELL = NELL
            ELEMS(IASS) = ELEMS(IASS) - 1
            LIELL = IW(IELL)
            NSTKAC(2) = NSTKAC(2) - LIELL
            J1 = IELL
            J2 = IELLN - 1
            IF (IELLN.LT.PTRIRN-IINPUT) THEN
              IF (IW(IELLN).LT.0) THEN
                LIELL = LIELL - IW(IELLN)
                J2 = J2 - IW(IELLN)
              END IF
            END IF
            IF ((IELL-1).EQ.ISTK) THEN
              ISTK = ISTK + LIELL
            ELSE
              IF (IW(IELL-1).LT.0) THEN
                LIELL = LIELL - IW(IELL-1)
                J1 = IELL + IW(IELL-1)
              END IF
              IW(J1) = -LIELL
              IW(J2) = -LIELL
            END IF
            LAELL = NINT(A(PTRELT))
            NSTKAC(1) = NSTKAC(1) - LAELL
            JA1 = PTRELT
            JA2 = POSELT - 1
            IF (POSELT.LT.PTRA-AINPUT) THEN
              IF (NINT(A(POSELT)).LT.0) THEN
                LAELL = LAELL - NINT(A(POSELT))
                JA2 = JA2 - NINT(A(POSELT))
              END IF
            END IF
            IF (PTRELT-1.EQ.ASTK) THEN
              ASTK = ASTK + LAELL
            ELSE
              IF (NINT(A(PTRELT-1)).LT.0) THEN
                LAELL = LAELL - NINT(A(PTRELT-1))
                JA1 = PTRELT + NINT(A(PTRELT-1))
              END IF
              A(JA1) = -LAELL
              A(JA2) = -LAELL
            END IF
          END IF
          IELL = IELLN
          PTRELT = POSELT
  380   CONTINUE
        DO 410 I = 1,NUMORG
          APOS2 = APOS + (NFRONT+1)* (I-1)
          DO 390 J = 1,NUMORG - I
            A(APOS2+J*NFRONT) = A(APOS2+J*NFRONT) + A(APOS2+J)
  390     CONTINUE
          DO 400 J = 1,NASS - I
            A(APOS2+J) = A(APOS2+J*NFRONT)
  400     CONTINUE
  410   CONTINUE
        NPIV = 0
        NSTRUC = 0
        IF (KIND.EQ.1) GO TO 760
        DO 440 K = 1,KIND - 1
          POSELT = APOS + (K-1)* (NUMORG/2* (NFRONT+1)) - 1
          DO 430 I = 1,NUMORG/2
            DO 420 J = I,NUMORG/2
              IF (A(POSELT+J).NE.ZERO) GO TO 760
  420       CONTINUE
            POSELT = POSELT + NFRONT
  430     CONTINUE
  440   CONTINUE
        CNTNZ1 = NCOL - NASS
        CNTNZ2 = NCOL - NASS
        DO 480 J = NASS + 1,NCOL
          JA1 = APOS + J - 1
          DO 450 I = 1,NUMPVT
            IF (A(JA1).NE.ZERO) GO TO 460
            JA1 = JA1 + NFRONT
  450     CONTINUE
          CNTNZ1 = CNTNZ1 - 1
  460     IF (KIND.EQ.3) THEN
            JA2 = APOS + NFRONT*NUMPVT + J - 1
            DO 470 I = 1,NUMPVT
              IF (A(JA2).NE.ZERO) GO TO 480
              JA2 = JA2 + NFRONT
  470       CONTINUE
            CNTNZ2 = CNTNZ2 - 1
          END IF
  480   CONTINUE
        MA1 = (CNTNZ1+NUMPVT-1)* (CNTNZ2+ NUMPVT-1)
        IF (FATHER(IASS).NE.NODES+1) THEN
        END IF
        DO 650 KSWEEP = 1, (NUMORG-NPIV)/2
          NNULL = 0
          LNPIV = NPIV
          NPIVD2 = NPIV/2
          DO 640 IPIV = NPIVD2 + 1,NUMPVT
            APOSI = APOS + NUMPVT + NPIVD2 + (IPIV-1)*NFRONT
            JMAX = IDAMAX(NUMPVT-NPIVD2,A(APOSI),1)
            OFFDAG = APOSI + JMAX - 1
            JMAX = JMAX + NUMPVT + NPIVD2
            PIVOT = A(OFFDAG)
            A(OFFDAG) = ZERO
            JJ = IDAMAX(NCOL- (NUMPVT+NPIVD2),A(APOSI),1)
            RMAX = ABS(A(APOSI+JJ-1))
            IF (MAX(RMAX,ABS(PIVOT)).LE.TOL) THEN
              NNULL = NNULL + 1
              GO TO 640
            END IF
            IF (ABS(PIVOT).LE.TOL) GO TO 640
            POSPV1 = APOS + (IPIV-1)* (NFRONT+1)
            POSPV2 = APOS + (JMAX-1)* (NFRONT+1)
            IF (MA1.EQ.0) THEN
              DETPIV = -PIVOT*PIVOT
              IF (ABS(DETPIV/PIVOT).LE.TOL) GO TO 640
              A(OFFDAG) = PIVOT
              GO TO 530
            END IF
            APOSJ = APOS + (JMAX-1)*NFRONT
            AMULT2 = A(POSPV2)
            A(POSPV2) = ZERO
            J1MAX = NPIVD2 + IDAMAX(NUMPVT-NPIVD2,
     +              A(APOS+NPIVD2*NFRONT+JMAX-1),NFRONT)
            TMAX = ABS(A(APOS+JMAX-1+ (J1MAX-1)*NFRONT))
            IF ( JMAX-1- (NUMPVT+NPIVD2) .GT. 0)THEN
               J1MAX = NUMPVT + NPIVD2 + IDAMAX(JMAX-1- (NUMPVT+NPIVD2),
     +              A(APOS+ (NUMPVT+NPIVD2)*NFRONT+JMAX-1),NFRONT)
               TMAX = MAX(TMAX,ABS(A(APOS+JMAX-1+ (J1MAX-1)*NFRONT)))
            END IF
            J1MAX = JMAX - 1 + IDAMAX(NCOL-JMAX+1,A(POSPV2),1)
            TMAX = MAX(TMAX,ABS(A(APOSJ+J1MAX-1)))
            A(POSPV2) = AMULT2
            A(OFFDAG) = PIVOT
            DETPIV = -PIVOT*PIVOT
            IF (ABS(DETPIV/PIVOT).LE.TOL) GO TO 640
            IF ((ABS(A(POSPV2))*RMAX+ABS(PIVOT)*TMAX)*UU.GT.
     +          ABS(DETPIV)) GO TO 490
            IF ((ABS(PIVOT)*RMAX)*UU.GT.ABS(DETPIV)) GO TO 490
            GO TO 530
  490       IF (KIND.EQ.3) GO TO 640
            IF (ABS(A(POSPV2)).LT.UU*MAX(ABS(PIVOT),TMAX)) GO TO 640
            TMAX = ZERO
            AMULT2 = - (A(OFFDAG)/A(POSPV2))
            JA1 = APOS + NPIVD2*NFRONT + (JMAX-1)
            DO 500 J = NPIVD2 + 1,NUMPVT
              TMAX = MAX(TMAX,ABS(A(JA1)*AMULT2))
              JA1 = JA1 + NFRONT
  500       CONTINUE
            JA1 = APOS + (NUMPVT+NPIVD2)*NFRONT + (JMAX-1)
            JA2 = APOS + (IPIV-1)*NFRONT + NUMPVT + NPIVD2
            DO 510 J = NUMPVT + NPIVD2 + 1,JMAX - 1
              TMAX = MAX(TMAX,ABS(A(JA2)+A(JA1)*AMULT2))
              JA1 = JA1 + NFRONT
              JA2 = JA2 + 1
  510       CONTINUE
            JA1 = APOS + (JMAX-1)*NFRONT + JMAX
            JA2 = APOS + (IPIV-1)*NFRONT + JMAX
            DO 520 J = JMAX + 1,NCOL
              TMAX = MAX(TMAX,ABS(A(JA2)+A(JA1)*AMULT2))
              JA1 = JA1 + 1
              JA2 = JA2 + 1
  520       CONTINUE
            IF (ABS(DETPIV/A(POSPV2)).LT.UU*TMAX) GO TO 640
  530       IF (IPIV.EQ.NPIVD2+1) GO TO 550
            JA1 = APOS + (IPIV-1)*NFRONT + NUMPVT
            J1 = APOS + NPIVD2*NFRONT + NUMPVT
            DO 540 JJ = 1,NCOL - NUMPVT
              SWOP = A(JA1)
              A(JA1) = A(J1)
              A(J1) = SWOP
              JA1 = JA1 + 1
              J1 = J1 + 1
  540       CONTINUE
            IPOS = IWPOS + NPIVD2
            IEXCH = IWPOS + IPIV - 1
            ISWOP = IW(IPOS)
            IW(IPOS) = IW(IEXCH)
            IW(IEXCH) = ISWOP
  550       IF (JMAX.EQ.NUMPVT+NPIVD2+1) GO TO 590
            JA1 = APOS + JMAX - 1
            J1 = APOS + NUMPVT + NPIVD2
            DO 560 JJ = 1,NUMPVT + NPIVD2
              SWOP = A(JA1)
              A(JA1) = A(J1)
              A(J1) = SWOP
              JA1 = JA1 + NFRONT
              J1 = J1 + NFRONT
  560       CONTINUE
            JA1 = APOS + (NUMPVT+NPIVD2+1)*NFRONT + JMAX - 1
            J1 = APOS + (NUMPVT+NPIVD2)* (NFRONT+1) + 1
            DO 570 JJ = NUMPVT + NPIVD2 + 2,JMAX - 1
              SWOP = A(JA1)
              A(JA1) = A(J1)
              A(J1) = SWOP
              JA1 = JA1 + NFRONT
              J1 = J1 + 1
  570       CONTINUE
            DO 580 JJ = 1,NCOL - JMAX
              JA1 = JA1 + 1
              J1 = J1 + 1
              SWOP = A(JA1)
              A(JA1) = A(J1)
              A(J1) = SWOP
  580       CONTINUE
            SWOP = A(APOS+ (NUMPVT+NPIVD2)* (NFRONT+1))
            A(APOS+ (NUMPVT+NPIVD2)* (NFRONT+1)) = A(APOS+
     +        (JMAX-1)* (NFRONT+1))
            A(APOS+ (JMAX-1)* (NFRONT+1)) = SWOP
            IPOS = IWPOS + NUMPVT + NPIVD2
            IEXCH = IWPOS + JMAX - 1
            ISWOP = IW(IPOS)
            IW(IPOS) = IW(IEXCH)
            IW(IEXCH) = ISWOP
  590       POSPV1 = APOS + NPIVD2* (NFRONT+1)
            POSPV2 = APOS + (NUMPVT+NPIVD2)* (NFRONT+1)
            OFFDAG = POSPV1 + NUMPVT
            FLOPSB = FLOPSB + 3.0
            NEIG = NEIG + 1
C?? We could decide not to swop these over.
C?? Also could save if pivot was oxo.
            A(POSPV1) = A(POSPV2)/DETPIV
            A(POSPV2) = ZERO
            A(OFFDAG) = -A(OFFDAG)/DETPIV
            J1 = OFFDAG + NFRONT
            J2 = OFFDAG + (NUMPVT-NPIVD2-1)*NFRONT
            IBEG = J1 + 1
            IEND = APOS + (NPIVD2+1)*NFRONT + NCOL - 1
            DO 610 JJ = J1,J2,NFRONT
              AMULT1 = - (A(JJ)*A(OFFDAG))
              FLOPSB = FLOPSB + (IEND-IBEG+1)*2 + 1
              K1 = OFFDAG + 1
CDIR$            IVDEP
              DO 600 IROW = IBEG,IEND
                A(IROW) = A(IROW) + AMULT1*A(K1)
                K1 = K1 + 1
  600         CONTINUE
              A(JJ) = -AMULT1
              IBEG = IBEG + NFRONT
              IEND = IEND + NFRONT
  610       CONTINUE
            J1 = POSPV2 + 1
            J2 = POSPV2 + NASS - NUMPVT - NPIVD2 - 1
            IBEG = J1 + NFRONT
            IEND = APOS + (NUMPVT+NPIVD2+2)*NFRONT - 1
            DO 630 JJ = J1,J2
              AMULT1 = - (A(JJ-NUMPVT*NFRONT)*A(POSPV1)+A(JJ)*A(OFFDAG))
              AMULT2 = - (A(JJ-NUMPVT*NFRONT)*A(OFFDAG))
              FLOPSB = FLOPSB + (IEND-IBEG+2)*4
              K1 = JJ - NFRONT*NUMPVT
              K2 = JJ
CDIR$            IVDEP
              DO 620 IROW = IBEG,IEND
                A(IROW) = A(IROW) + AMULT1*A(K1) + AMULT2*A(K2)
                K1 = K1 + 1
                K2 = K2 + 1
  620         CONTINUE
              IF (JJ.LE.POSPV2+NUMPVT-NPIVD2-1) THEN
                A(JJ-NUMPVT*NFRONT) = -AMULT2
                A(JJ) = AMULT1
              ELSE
                A(JJ-NUMPVT*NFRONT) = AMULT1
                A(JJ) = AMULT2
              END IF
              IBEG = IBEG + NFRONT + 1
              IEND = IEND + NFRONT
  630       CONTINUE
            IW(IWPOS+NUMPVT+NPIVD2) = -IW(IWPOS+NUMPVT+NPIVD2)
            IW(IWPOS+NPIVD2) = -IW(IWPOS+NPIVD2)
            NPIV = NPIV + 2
            NPIVD2 = NPIVD2 + 1
            NTOTPV = NTOTPV + 2
            IF (KIND.EQ.3) NOXOB = NOXOB + 1
            IF (KIND.EQ.2) NTILEB = NTILEB + 1
            NSTRUC = NSTRUC + 2
  640     CONTINUE
          IF (LNPIV.EQ.NPIV) GO TO 660
  650   CONTINUE
  660   IF (NPIV.EQ.NASS) GO TO 930
        IF (NPIV.EQ.NUMORG .OR. NPIV.EQ.0) GO TO 750
        DO 680 I = 1,NPIVD2
          JA1 = APOS + NUMPVT + NPIVD2 + (I-1)*NFRONT
          DO 670 J = 1,NUMPVT - NPIVD2
            A(JA1) = -A(JA1)
            JA1 = JA1 + 1
  670     CONTINUE
  680   CONTINUE
        DO 700 I = 1,NUMPVT - NPIVD2
          JA1 = APOS + NPIVD2*NFRONT + NUMPVT + (I-1)*NFRONT
          DO 690 J = 1,NPIVD2
            A(JA1) = -A(JA1)
            JA1 = JA1 + 1
  690     CONTINUE
  700   CONTINUE
        CALL MA47YD(A(APOS+NPIVD2),NFRONT,NPIVD2,NUMPVT-NPIVD2,
     +              A(APOS+NUMPVT*NFRONT),NFRONT,.FALSE.,.FALSE.)
        CALL MA47YD(A(APOS+NUMPVT),NFRONT,NPIVD2,NPIVD2,A(APOS+NPIVD2),
     +              NFRONT,.FALSE.,.FALSE.)
        CALL MA47YD(A(APOS+NUMPVT*NFRONT),NFRONT,NPIVD2,NUMPVT-NPIVD2,
     +              A(APOS+NPIV),NFRONT,.FALSE.,.FALSE.)
        CALL MA47YD(A(APOS+NPIVD2*NFRONT+NUMPVT),NFRONT,NUMPVT-NPIVD2,
     +              NPIVD2,A(APOS+NUMPVT*NFRONT),NFRONT,.FALSE.,.FALSE.)
        CALL MA47YD(A(APOS+NUMPVT* (NFRONT+1)),NFRONT,NPIVD2,NPIVD2,
     +              A(APOS+NPIVD2* (NFRONT+1)),NFRONT,.TRUE.,.FALSE.)
        CALL MA47XD(A(APOS+NPIV* (NFRONT+1)),NFRONT,NUMPVT-NPIVD2,
     +              NUMPVT-NPIVD2)
        CALL MA47YD(A(APOS+NUMPVT*NFRONT),NFRONT,NUMPVT-NPIVD2,NPIVD2,
     +              A(APOS+NPIVD2*NFRONT+NPIV),NFRONT,.FALSE.,.TRUE.)
        ATRASH = (NUMPVT-NPIVD2)* (NFRONT-NUMPVT-NPIVD2)
        RLSPA = MAX(RLSPA,INFO(2)+APOS+NFRONT*NASS+ATRASH+NSTKAC(1))
        IF (APOS+NASS*NFRONT+ATRASH.GT.ASTK) THEN
          CALL MA47SD(A,ASTK,PTRA,AINPUT,NCMPBR)
          IF (APOS+NASS*NFRONT+ATRASH.GT.ASTK) THEN
            INFO(2) = INFO(2) + APOS - 1
            DO 710 I = 1,NASS*NFRONT
              A(I) = A(APOS+I-1)
  710       CONTINUE
            APOS = 1
            APOSBB = 1
            INFO(1) = -4
            IF (NFRONT*NASS+ATRASH.GT.ASTK) THEN
              INFO(2) = RLSPA
              GO TO 2170
            END IF
          END IF
        END IF
        ASTK = ASTK - ATRASH
        CALL MA47YD(A(APOS+NPIVD2*NFRONT+NPIVD2+NUMPVT),NFRONT,
     +              NUMPVT-NPIVD2,NFRONT-NUMPVT-NPIVD2,A(ASTK),
     +              NFRONT-NUMPVT-NPIVD2,.FALSE.,.FALSE.)
        CALL MA47YD(A(APOS+NUMPVT*NFRONT+NPIVD2+NUMPVT),NFRONT,NPIVD2,
     +              NFRONT-NUMPVT-NPIVD2,A(APOS+NPIVD2*NFRONT+NPIVD2+
     +              NUMPVT),NFRONT,.FALSE.,.FALSE.)
        CALL MA47YD(A(ASTK),NFRONT-NUMPVT-NPIVD2,NUMPVT-NPIVD2,
     +              NFRONT-NUMPVT-NPIVD2,A(APOS+NPIV*NFRONT+NPIVD2+
     +              NUMPVT),NFRONT,.FALSE.,.FALSE.)
        ASTK = ASTK + ATRASH
        CALL MA47YD(A(APOS+NPIV),NFRONT,NPIVD2,2* (NUMPVT-NPIVD2),
     +              A(APOS+NPIV*NFRONT),NFRONT,.FALSE.,.TRUE.)
        CALL MA47YD(A(APOS+NPIVD2*NFRONT+NPIV),NFRONT,NPIVD2,
     +              2* (NUMPVT-NPIVD2),A(APOS+NPIV),NFRONT,.FALSE.,
     +              .FALSE.)
        CALL MA47YD(A(APOS+NPIV*NFRONT),NFRONT,2* (NUMPVT-NPIVD2),
     +              NPIVD2,A(APOS+NPIVD2*NFRONT+NPIV),NFRONT,.FALSE.,
     +              .TRUE.)
        J1 = IWPOS + N - NTOTPV + NPIV
        DO 720 JJ = NPIVD2,NUMPVT - 1
          IW(J1) = IW(IWPOS+JJ)
          J1 = J1 + 1
  720   CONTINUE
        J1 = IWPOS + NPIVD2
        DO 730 JJ = NUMPVT,NUMPVT + NPIVD2 - 1
          IW(J1) = IW(IWPOS+JJ)
          J1 = J1 + 1
  730   CONTINUE
        DO 740 JJ = 1,NUMPVT - NPIVD2
          IW(J1) = IW(IWPOS+N-NTOTPV+NPIV+JJ-1)
          J1 = J1 + 1
  740   CONTINUE
  750   NCOL = NFRONT
  760   MA1 = (NCOL-NPIV-1)**2
        IF (FATHER(IASS).NE.NODES+1) THEN
        END IF
        DO 920 KSWEEP = 1,NASS - NPIV
          NNULL = 0
          LNPIV = NPIV
          JPIV = 1
          DO 910 IPIV = NPIV + 1,NASS
            JPIV = JPIV - 1
            IF (JPIV.EQ.1) GO TO 910
            IF (IPIV.GT.NUMORG) NCOL = NFRONT
            APOSI = APOS + (IPIV-1)*NFRONT
            POSPV1 = APOSI + IPIV - 1
            PIVOT = A(POSPV1)
            IF (MA1.EQ.0) THEN
              IF (ABS(PIVOT).GT.TOL) THEN
                PIVSIZ = 1
                JMAX = IPIV
                GO TO 810
              END IF
            END IF
            A(POSPV1) = ZERO
            J1MAX = NPIV + IDAMAX(IPIV-NPIV,A(APOS+NPIV*NFRONT+IPIV-1),
     +              NFRONT)
            A1MAX = ABS(A(APOS+IPIV-1+ (J1MAX-1)*NFRONT))
            JMAX = IPIV - 1 + IDAMAX(NASS-IPIV+1,A(POSPV1),1)
            AMAX = ABS(A(APOSI+JMAX-1))
            IF (A1MAX.GT.AMAX) THEN
              AMAX = A1MAX
              JMAX = J1MAX
            END IF
            IF (NASS.EQ.NFRONT) THEN
              RMAX = ZERO
            ELSE
              JJ = IDAMAX(NFRONT-NASS,A(APOSI+NASS),1)
              RMAX = ABS(A(APOSI+NASS+JJ-1))
            END IF
            IF (MAX(AMAX,RMAX,ABS(PIVOT)).LE.TOL) THEN
              NNULL = NNULL + 1
              GO TO 910
            END IF
            IF (MAX(AMAX,ABS(PIVOT)).LE.TOL) GO TO 910
            PIVSIZ = 0
            IF (ABS(PIVOT).GT.UU*MAX(RMAX,AMAX)) THEN
              PIVSIZ = 1
              A(POSPV1) = PIVOT
              GO TO 810
            END IF
            IF (NPIV+1.EQ.NASS) THEN
              A(POSPV1) = PIVOT
              GO TO 910
            END IF
            IF (RMAX.LT.AMAX) THEN
              AMULT1 = A(APOSI+JMAX-1)
              A(APOSI+JMAX-1) = ZERO
              J1MAX = NPIV + IDAMAX(IPIV-NPIV,
     +                A(APOS+NPIV*NFRONT+IPIV-1),NFRONT)
              A1MAX = ABS(A(APOS+IPIV-1+ (J1MAX-1)*NFRONT))
              J1MAX = IPIV - 1 + IDAMAX(NASS-IPIV+1,A(POSPV1),1)
              RMAX = MAX(RMAX,A1MAX,ABS(A(APOSI+J1MAX-1)))
              A(APOSI+JMAX-1) = AMULT1
            END IF
            APOSJ = APOS + (JMAX-1)*NFRONT
            IF (JMAX.GT.NUMORG) NCOL = NFRONT
            POSPV2 = APOSJ + JMAX - 1
            IF (IPIV.GT.JMAX) THEN
              OFFDAG = APOSJ + IPIV - 1
            ELSE
              OFFDAG = APOSI + JMAX - 1
            END IF
            AMULT1 = A(OFFDAG)
            A(OFFDAG) = ZERO
            AMULT2 = A(POSPV2)
            A(POSPV2) = ZERO
            J1MAX = NPIV + IDAMAX(JMAX-NPIV,A(APOS+NPIV*NFRONT+JMAX-1),
     +              NFRONT)
            TMAX = ABS(A(APOS+JMAX-1+ (J1MAX-1)*NFRONT))
            J1MAX = JMAX - 1 + IDAMAX(NCOL-JMAX+1,A(POSPV2),1)
            TMAX = MAX(TMAX,ABS(A(APOSJ+J1MAX-1)))
            A(OFFDAG) = AMULT1
            A(POSPV2) = AMULT2
            A(POSPV1) = PIVOT
            DETPIV = A(POSPV1)*A(POSPV2) - AMAX*AMAX
            MAXPIV = MAX(ABS(A(POSPV1)),ABS(A(POSPV2)))
            IF (MAXPIV.EQ.ZERO) MAXPIV = ONE
            IF (ABS(DETPIV)/MAXPIV.LE.TOL) GO TO 910
            PIVSIZ = 2
            IF ((ABS(A(POSPV2))*RMAX+AMAX*TMAX)*UU.GT.
     +          ABS(DETPIV)) GO TO 770
            IF ((ABS(A(POSPV1))*TMAX+AMAX*RMAX)*UU.GT.
     +          ABS(DETPIV)) GO TO 770
            GO TO 810
  770       IF (ABS(A(POSPV2)).LT.UU*MAX(AMAX,TMAX)) GO TO 910
            IF (ABS(A(POSPV2)).LE.TOL) GO TO 910
            TMAX = ZERO
            AMULT2 = - (A(OFFDAG)/A(POSPV2))
            JA1 = APOS + (JMAX-1) + NPIV*NFRONT
            JA2 = APOS + (IPIV-1) + NPIV*NFRONT
            DO 780 J = NPIV + 1,MIN(IPIV,JMAX) - 1
              TMAX = MAX(TMAX,ABS(A(JA2)+A(JA1)*AMULT2))
              JA1 = JA1 + NFRONT
              JA2 = JA2 + NFRONT
  780       CONTINUE
            IF (JMAX.GT.IPIV) THEN
              J1 = NFRONT
              J2 = 1
              JA1 = APOS + IPIV*NFRONT + JMAX - 1
              JA2 = APOS + (IPIV-1)*NFRONT + IPIV
            ELSE
              J1 = 1
              J2 = NFRONT
              JA1 = APOS + (JMAX-1)*NFRONT + JMAX
              JA2 = APOS + JMAX*NFRONT + IPIV - 1
            END IF
            DO 790 J = MIN(IPIV,JMAX) + 1,MAX(IPIV,JMAX) - 1
              TMAX = MAX(TMAX,ABS(A(JA2)+A(JA1)*AMULT2))
              JA1 = JA1 + J1
              JA2 = JA2 + J2
  790       CONTINUE
            JA1 = APOS + (JMAX-1)*NFRONT + JMAX
            JA2 = APOS + (IPIV-1)*NFRONT + JMAX
            DO 800 J = MAX(IPIV,JMAX) + 1,NCOL
              TMAX = MAX(TMAX,ABS(A(JA2)+A(JA1)*AMULT2))
              JA1 = JA1 + 1
              JA2 = JA2 + 1
  800       CONTINUE
            IF (ABS(DETPIV/A(POSPV2)).LT.UU*TMAX) GO TO 910
  810       LPIV = IPIV
            IF (PIVSIZ.EQ.2) LPIV = MIN(IPIV,JMAX)
            DO 860 KROW = NPIV,NPIV + PIVSIZ - 1
              IF (LPIV.EQ.KROW+1) GO TO 850
              JA1 = APOS + (LPIV-1)
              J1 = APOS + KROW
              DO 820 JJ = 1,KROW
                SWOP = A(JA1)
                A(JA1) = A(J1)
                A(J1) = SWOP
                JA1 = JA1 + NFRONT
                J1 = J1 + NFRONT
  820         CONTINUE
              JA1 = JA1 + NFRONT
              J1 = J1 + 1
              DO 830 JJ = 1,LPIV - KROW - 2
                SWOP = A(JA1)
                A(JA1) = A(J1)
                A(J1) = SWOP
                JA1 = JA1 + NFRONT
                J1 = J1 + 1
  830         CONTINUE
              SWOP = A(APOS+KROW* (NFRONT+1))
              A(APOS+KROW* (NFRONT+1)) = A(JA1)
              A(JA1) = SWOP
              DO 840 JJ = 1,NCOL - LPIV
                JA1 = JA1 + 1
                J1 = J1 + 1
                SWOP = A(JA1)
                A(JA1) = A(J1)
                A(J1) = SWOP
  840         CONTINUE
              IPOS = IWPOS + KROW
              IEXCH = IWPOS + LPIV - 1
              ISWOP = IW(IPOS)
              IW(IPOS) = IW(IEXCH)
              IW(IEXCH) = ISWOP
  850         LPIV = MAX(IPIV,JMAX)
  860       CONTINUE
            POSPV1 = APOS + NPIV* (NFRONT+1)
            POSPV2 = POSPV1 + NFRONT + 1
            IF (PIVSIZ.EQ.1) THEN
              FLOPSB = FLOPSB + ONE
              A(POSPV1) = ONE/A(POSPV1)
              IF (A(POSPV1).LT.ZERO) NEIG = NEIG + 1
              J1 = POSPV1 + 1
              J2 = POSPV1 + NASS - (NPIV+1)
              IBEG = POSPV1 + NFRONT + 1
              IEND = APOS + (NPIV+1)*NFRONT + NCOL - 1
              DO 880 JJ = J1,J2
                AMULT1 = -A(JJ)*A(POSPV1)
                JCOL = JJ
                FLOPSB = FLOPSB + (IEND-IBEG+1)*2 + 1
CDIR$            IVDEP
                DO 870 IROW = IBEG,IEND
                  A(IROW) = A(IROW) + AMULT1*A(JCOL)
                  JCOL = JCOL + 1
  870           CONTINUE
                A(JJ) = AMULT1
                IBEG = IBEG + NFRONT + 1
                IEND = IEND + NFRONT
  880         CONTINUE
              NPIV = NPIV + 1
              NTOTPV = NTOTPV + 1
              JPIV = 1
            ELSE
              OFFDAG = POSPV1 + 1
              FLOPSB = FLOPSB + 6.0
              SWOP = A(POSPV2)
              IF (DETPIV.LT.ZERO) THEN
                NEIG = NEIG + 1
              ELSE
                IF (SWOP.LT.ZERO) NEIG = NEIG + 2
              END IF
              A(POSPV2) = A(POSPV1)/DETPIV
              A(POSPV1) = SWOP/DETPIV
              A(OFFDAG) = -A(OFFDAG)/DETPIV
              J1 = POSPV1 + 2
              J2 = POSPV1 + NASS - (NPIV+1)
              IBEG = POSPV2 + NFRONT + 1
              IEND = APOS + (NPIV+2)*NFRONT + NCOL - 1
              DO 900 JJ = J1,J2
                K1 = JJ
                K2 = JJ + NFRONT
                AMULT1 = - (A(POSPV1)*A(K1)+A(POSPV1+1)*A(K2))
                AMULT2 = - (A(POSPV1+1)*A(K1)+A(POSPV2)*A(K2))
                FLOPSB = FLOPSB + (IEND-IBEG+1)*4 + 6
CDIR$            IVDEP
                DO 890 IROW = IBEG,IEND
                  A(IROW) = A(IROW) + AMULT1*A(K1) + AMULT2*A(K2)
                  K1 = K1 + 1
                  K2 = K2 + 1
  890           CONTINUE
                A(JJ) = AMULT1
                A(JJ+NFRONT) = AMULT2
                IBEG = IBEG + NFRONT + 1
                IEND = IEND + NFRONT
  900         CONTINUE
              IPOS = IWPOS + NPIV
              IW(IPOS) = -IW(IPOS)
              IW(IPOS+1) = -IW(IPOS+1)
              NPIV = NPIV + 2
              NTOTPV = NTOTPV + 2
              NFULLB = NFULLB + 1
              JPIV = 2
            END IF
  910     CONTINUE
          IF (LNPIV.EQ.NPIV) GO TO 930
  920   CONTINUE
  930   IF (NPIV.EQ.0) GO TO 1730
        NOSURE = (NBLOC.GE. (NCOL-NASS)) .AND. (NSTRUC.EQ.0)
        ZONE = ZERO
        IF (NSTRUC.GT.0 .AND. NPIV.NE.NSTRUC) ZONE = ONE
        NUM1 = 0
        NUM2 = 0
        NUM3 = 0
        IF (NPIV.EQ.NFRONT) GO TO 1830
        IF (NSTRUC.GT.0) THEN
          J2 = NCOL
          DO 1000 J1 = NASS + 1,NCOL
            JA1 = APOS + J1 - 1
            DO 940 K = 1,NSTRUC/2
              IF (A(JA1).NE.ZERO) GO TO 990
              JA1 = JA1 + NFRONT
  940       CONTINUE
            DO 960 JJ = J2,J1 + 1,-1
              JA2 = APOS + JJ - 1
              DO 950 K = 1,NSTRUC/2
                IF (A(JA2).NE.ZERO) THEN
                  J2 = JJ
                  GO TO 970
                END IF
                JA2 = JA2 + NFRONT
  950         CONTINUE
  960       CONTINUE
            J2 = J1 - 1
            GO TO 1010
  970       ISWOP = IW(IWPOS-1+J1)
            IW(IWPOS-1+J1) = IW(IWPOS-1+J2)
            IW(IWPOS-1+J2) = ISWOP
            PPOS(ISWOP) = J2
            PPOS(IW(IWPOS-1+J1)) = J1
            JA1 = APOS + J1 - 1
            JA2 = APOS + J2 - 1
            DO 980 LDUMMY = 1,NASS
              SWOP = A(JA1)
              A(JA1) = A(JA2)
              A(JA2) = SWOP
              JA1 = JA1 + NFRONT
              JA2 = JA2 + NFRONT
  980       CONTINUE
            J2 = J2 - 1
  990       IF (J1.GE.J2) GO TO 1010
 1000     CONTINUE
 1010     NUM1 = 0
          NUM2 = J2 - NASS
          NUM3 = NCOL - J2
          IF (KIND.EQ.3) THEN
            DO 1090 J1 = NASS + 1,NASS + NUM2
              JA1 = APOS + (NSTRUC/2)*NFRONT + J1 - 1
              DO 1020 K = 1,NSTRUC/2
                IF (A(JA1).NE.ZERO) GO TO 1030
                JA1 = JA1 + NFRONT
 1020         CONTINUE
              GO TO 1080
 1030         DO 1050 JJ = J2,J1 + 1,-1
                JA2 = APOS + (NSTRUC/2)*NFRONT + JJ - 1
                DO 1040 K = 1,NSTRUC/2
                  IF (A(JA2).NE.ZERO) GO TO 1050
                  JA2 = JA2 + NFRONT
 1040           CONTINUE
                J2 = JJ
                GO TO 1060
 1050         CONTINUE
              J2 = J1 - 1
              GO TO 1100
 1060         ISWOP = IW(IWPOS-1+J1)
              IW(IWPOS-1+J1) = IW(IWPOS-1+J2)
              IW(IWPOS-1+J2) = ISWOP
              PPOS(ISWOP) = J2
              PPOS(IW(IWPOS-1+J1)) = J1
              JA1 = APOS + J1 - 1
              JA2 = APOS + J2 - 1
              DO 1070 LDUMMY = 1,NASS
                SWOP = A(JA1)
                A(JA1) = A(JA2)
                A(JA2) = SWOP
                JA1 = JA1 + NFRONT
                JA2 = JA2 + NFRONT
 1070         CONTINUE
              J2 = J2 - 1
 1080         IF (J1.GE.J2) GO TO 1100
 1090       CONTINUE
 1100       NUM1 = J2 - NASS
            NUM2 = NUM2 - NUM1
          END IF
        END IF
        ZCOL = 0
        APOS1 = APOS + NCOL - 1
        APOS2 = APOS + NCOL
        DO 1140 J = 1,NCOL - (NUM1+NUM2+NASS)
          DO 1110 JA1 = 1,NPIV
            IF (A(APOS1+ (JA1-1)*NFRONT).NE.ZERO) GO TO 1130
 1110     CONTINUE
          ZCOL = ZCOL + 1
          APOS2 = APOS2 - 1
          IF (APOS1.EQ.APOS2) GO TO 1130
          DO 1120 JA1 = 1,NASS
            SWOP = A(APOS1+ (JA1-1)*NFRONT)
            A(APOS1+ (JA1-1)*NFRONT) = A(APOS2+ (JA1-1)*NFRONT)
            A(APOS2+ (JA1-1)*NFRONT) = SWOP
 1120     CONTINUE
          ISWOP = IW(IWPOS+APOS2-APOS)
          IW(IWPOS+APOS2-APOS) = IW(IWPOS+APOS1-APOS)
          IW(IWPOS+APOS1-APOS) = ISWOP
          PPOS(ISWOP) = APOS1 - APOS + 1
          PPOS(IW(IWPOS+APOS2-APOS)) = APOS2 - APOS + 1
 1130     APOS1 = APOS1 - 1
 1140   CONTINUE
        NCOL = NCOL - ZCOL
        NUM3 = NUM3 - ZCOL
        IF (NCOL.EQ.NASS) GO TO 1730
        APOS3 = APOS + NFRONT*NASS
        IF (NPIV.EQ.NSTRUC) THEN
          NSC1 = NUM1 + NUM2
          NSC2 = NUM2 + NUM3
          IF (NSC1*NSC2.EQ.0) THEN
            APOSI = APOS + NASS
            OFFDAG = APOS + NSTRUC/2
            FLOPSB = FLOPSB + (NUM1+NUM3)*NSTRUC/2
            DO 1170 K = 1,NSTRUC/2
              DO 1150 JJ = APOSI,APOSI + NUM1 - 1
                A(JJ+NFRONT*NSTRUC/2) = -A(OFFDAG)*A(JJ)
                A(JJ) = ZERO
 1150         CONTINUE
              DO 1160 JJ = APOSI + NUM1,APOSI + NUM1 + NUM3 - 1
                A(JJ) = - (A(OFFDAG)*A(JJ+NFRONT*NSTRUC/2))
                A(JJ+NFRONT*NSTRUC/2) = ZERO
 1160         CONTINUE
              APOSI = APOSI + NFRONT
              OFFDAG = OFFDAG + NFRONT + 1
 1170       CONTINUE
            GO TO 1730
          END IF
        ELSE
          NSC1 = NCOL - NASS
          NSC2 = NCOL - NASS
        END IF
        IF (NOSURE) THEN
C
          APOS4 = ASTK - ((NSC1+1)*NSC1)/2 - 1
          RLSPA = MAX(RLSPA,INFO(2)+APOS3+ ((NSC1+1)*NSC1)/2+2+
     +            NSTKAC(1))
          IF (APOS3.GT.APOS4) THEN
            CALL MA47SD(A,ASTK,PTRA,AINPUT,NCMPBR)
C
            APOS4 = ASTK - ((NSC1+1)*NSC1)/2 - 1
            IF (APOS3.GT.APOS4) THEN
              INFO(2) = INFO(2) + APOS - 1
              DO 1180 I = 1,NASS*NFRONT
                A(I) = A(APOS+I-1)
 1180         CONTINUE
              APOSBB = 1
              APOS = 1
              APOS3 = APOS + NFRONT*NASS
              INFO(1) = -4
              IF (APOS3.GT.APOS4) THEN
                INFO(2) = RLSPA
                GO TO 2170
              END IF
            END IF
          END IF
          A(ASTK) = ((NSC1+1)*NSC1)/2 + 2
          A(APOS4) = ((NSC1+1)*NSC1)/2 + 2
          NSTKAC(1) = NSTKAC(1) + ((NSC1+1)*NSC1)/2 + 2
        ELSE
          APOS4 = APOS3 + NPIV*NSC2
          IF (NPIV.EQ.NSTRUC) THEN
            NSCHUR = MAX(NSC1*NSC2+1,NUM1* (NUM2+NUM3)+NUM2*NUM3+
     +               (NUM2* (NUM2+1))/2+2)
          ELSE
            NSCHUR = MAX(NSC1*NSC2+1, (NSC1* (NSC1+1))/2+2)
          END IF
          RLSPA = MAX(RLSPA,INFO(2)+APOS4+NSCHUR+NSTKAC(1))
          IF (APOS4+NSCHUR-1.GT.ASTK) THEN
            CALL MA47SD(A,ASTK,PTRA,AINPUT,NCMPBR)
            IF (APOS4+NSCHUR-1.GT.ASTK) THEN
              INFO(2) = INFO(2) + APOS - 1
              DO 1190 I = 1,NASS*NFRONT
                A(I) = A(APOS+I-1)
 1190         CONTINUE
              APOSBB = 1
              APOS = 1
              APOS3 = APOS + NFRONT*NASS
              INFO(1) = -4
              APOS4 = APOS3 + NPIV*NSC2
              IF (APOS4+NSCHUR-1.GT.ASTK) THEN
                INFO(2) = RLSPA
                GO TO 2170
              END IF
            END IF
          END IF
          IF (ZONE.EQ.ONE) THEN
            DO 1200 K = APOS4,APOS4 + NSC1*NSC2 - 1
              A(K) = ZERO
 1200       CONTINUE
          END IF
        END IF
        IF (NSTRUC.GT.0) THEN
          POSPV1 = APOS
          APOSI = APOS
          OFFDAG = POSPV1 + NSTRUC/2
          FLOPSB = FLOPSB + (NUM1+4*NUM2+NUM3)*NSTRUC/2
          DO 1240 K = 1,NSTRUC/2
            DO 1210 JJ = APOSI + NASS,APOSI + NASS + NUM1 - 1
              A(JJ+NFRONT*NSTRUC/2) = -A(OFFDAG)*A(JJ)
              A(JJ) = ZERO
 1210       CONTINUE
            APOS2 = APOS3 + (K-1)* (NUM2+NUM3)
            DO 1220 JJ = APOSI + NASS + NUM1,
     +             APOSI + NASS + NUM1 + NUM2 - 1
              A(APOS2) = A(JJ)
              A(APOS2+ (NSTRUC/2)* (NUM2+NUM3)) = A(JJ+NFRONT*NSTRUC/2)
              A(JJ) = - (A(POSPV1)*A(JJ)+A(OFFDAG)*
     +                A(JJ+NFRONT*NSTRUC/2))
              A(JJ+NFRONT*NSTRUC/2) = -A(OFFDAG)*A(APOS2)
              APOS2 = APOS2 + 1
 1220       CONTINUE
            APOS2 = APOS3 + (K-1)* (NUM2+NUM3) + NUM2
            DO 1230 JJ = APOSI + NASS + NUM1 + NUM2,APOSI + NCOL - 1
              A(APOS2) = ZERO
              A(APOS2+ (NSTRUC/2)* (NUM2+NUM3)) = A(JJ+NFRONT*NSTRUC/2)
              A(JJ) = - (A(OFFDAG)*A(JJ+NFRONT*NSTRUC/2))
              A(JJ+NFRONT*NSTRUC/2) = ZERO
              APOS2 = APOS2 + 1
 1230       CONTINUE
            POSPV1 = POSPV1 + NFRONT + 1
            APOSI = APOSI + NFRONT
            OFFDAG = OFFDAG + NFRONT + 1
 1240     CONTINUE
          IF (NPIV.EQ.NSTRUC) THEN
            APOS1 = APOS4
          ELSE
            APOS1 = APOS4 + NUM1
          END IF
          FLOPSB = FLOPSB + 2*NSTRUC*NUM1* (NUM2+NUM3)
          IF (NUM1.GE.NBLOC) THEN
            CALL DGEMM('N','T',NUM2+NUM3,NUM1,NSTRUC,ONE,A(APOS3),
     +                 NUM2+NUM3,A(APOS+NASS),NFRONT,ZERO,A(APOS1),NSC2)
          ELSE
            APOSI = APOS
            DO 1280 K = 1,NSTRUC/2
              INC = NSC2
              JA1 = APOS1
              J1 = APOSI + NASS + NFRONT*NSTRUC/2
              DO 1270 JJ = 1,NUM1
                AMULT2 = A(J1)
                J1 = J1 + 1
                K2 = APOS3 + (K+NSTRUC/2-1)* (NUM2+NUM3)
                IF (K.EQ.1) THEN
                  DO 1250 IROW = JA1,JA1 + NUM2 + NUM3 - 1
                    A(IROW) = AMULT2*A(K2)
                    K2 = K2 + 1
 1250             CONTINUE
                ELSE
                  DO 1260 IROW = JA1,JA1 + NUM2 + NUM3 - 1
                    A(IROW) = A(IROW) + AMULT2*A(K2)
                    K2 = K2 + 1
 1260             CONTINUE
                END IF
                JA1 = JA1 + INC
 1270         CONTINUE
              APOSI = APOSI + NFRONT
 1280       CONTINUE
          END IF
          FLOPSB = FLOPSB + NSTRUC* (2*NUM3+NUM2+1)*NUM2
          FLOPSX = FLOPSX + NSTRUC*NUM2*NUM3
          APOS1 = APOS1 + NUM1*NSC2
          JA1 = APOS1
          KBLK = NUM2/NBLOC
          DO 1290 K = 1,KBLK
            JA1 = (K-1)*NBLOC
            FLOPSX = FLOPSX + NBLOC* (NBLOC-1)*NSTRUC
            CALL DGEMM('N','T',NUM2+NUM3-JA1,NBLOC,NSTRUC,ONE,
     +                 A(APOS3+JA1),NUM2+NUM3,A(APOS+NASS+NUM1+JA1),
     +                 NFRONT,ZERO,A(APOS1+JA1* (NSC2+1)),NSC2)
 1290     CONTINUE
          APOSI = APOS
          APOS1 = APOS1 + KBLK*NBLOC* (NSC2+1)
          DO 1350 K = 1,NSTRUC/2
            INC = NSC2
            JA1 = APOS1
            JA2 = JA1 + NUM2 - 1 - KBLK*NBLOC
            J1 = APOSI + NASS + NUM1 + KBLK*NBLOC
            DO 1340 JJ = 1,NUM2 - KBLK*NBLOC
              AMULT1 = A(J1)
              AMULT2 = A(J1+NFRONT*NSTRUC/2)
              J1 = J1 + 1
              K1 = APOS3 + KBLK*NBLOC + (K-1)* (NUM2+NUM3) + JJ - 1
              K2 = K1 + (NSTRUC/2)* (NUM2+NUM3)
              IF (K.EQ.1) THEN
                DO 1300 IROW = JA1,JA2
                  A(IROW) = AMULT1*A(K1) + AMULT2*A(K2)
                  K1 = K1 + 1
                  K2 = K2 + 1
 1300           CONTINUE
              ELSE
                DO 1310 IROW = JA1,JA2
                  A(IROW) = A(IROW) + AMULT1*A(K1) + AMULT2*A(K2)
                  K1 = K1 + 1
                  K2 = K2 + 1
 1310           CONTINUE
              END IF
              IF (K.EQ.1) THEN
                DO 1320 IROW = JA2 + 1,JA2 + NUM3
                  A(IROW) = AMULT2*A(K2)
                  K2 = K2 + 1
 1320           CONTINUE
              ELSE
                DO 1330 IROW = JA2 + 1,JA2 + NUM3
                  A(IROW) = A(IROW) + AMULT2*A(K2)
                  K2 = K2 + 1
 1330           CONTINUE
              END IF
              JA1 = JA1 + INC + 1
              JA2 = JA2 + INC
 1340       CONTINUE
            APOSI = APOSI + NFRONT
 1350     CONTINUE
        END IF
        IF (NSTRUC.EQ.NPIV) GO TO 1560
        J1 = IWPOS + NSTRUC
        LTWO = .FALSE.
        POSPV1 = APOS + NFRONT*NSTRUC + NSTRUC
        IF (NOSURE) THEN
          I = NSTRUC + 2
          APOSI = APOS + NSTRUC*NFRONT + NASS
          J2 = APOS + NFRONT*NSTRUC + NCOL - 1
          APOSC = APOS4 + 1
          IF (IW(J1).GT.0) THEN
            FLOPSB = FLOPSB + (NCOL-NASS) + (NCOL-NASS)* (NCOL-NASS+1)
            DO 1370 JJ = APOSI,J2
              AMULT1 = -A(JJ)*A(POSPV1)
              DO 1360 JJJ = JJ,J2
                A(APOSC) = AMULT1*A(JJJ)
                APOSC = APOSC + 1
 1360         CONTINUE
              A(JJ) = AMULT1
 1370       CONTINUE
            J1 = J1 + 1
          ELSE
            POSPV2 = POSPV1 + NFRONT + 1
            OFFDAG = POSPV1 + 1
            FLOPSB = FLOPSB + 6* (NCOL-NASS) +
     +               2* (NCOL-NASS)* (NCOL-NASS+1)
            DO 1390 JJ = APOSI,J2
              AMULT1 = - (A(POSPV1)*A(JJ)+A(OFFDAG)*A(JJ+NFRONT))
              AMULT2 = -A(POSPV2)*A(JJ+NFRONT) - A(OFFDAG)*A(JJ)
              DO 1380 JJJ = JJ,J2
                A(APOSC) = AMULT1*A(JJJ) + AMULT2*A(JJJ+NFRONT)
                APOSC = APOSC + 1
 1380         CONTINUE
              A(JJ) = AMULT1
              A(JJ+NFRONT) = AMULT2
 1390       CONTINUE
            J1 = J1 + 2
            POSPV1 = POSPV2
            I = I + 1
          END IF
          POSPV1 = POSPV1 + NFRONT + 1
          DO 1450 I1 = I,NPIV
            IF (LTWO) GO TO 1440
            APOSI = APOS + (I1-1)*NFRONT + NASS
            J2 = APOS + NFRONT* (I1-1) + NCOL - 1
            APOSC = APOS4 + 1
            IF (IW(J1).GT.0) THEN
              FLOPSB = FLOPSB + (NCOL-NASS) + (NCOL-NASS)* (NCOL-NASS+1)
              DO 1410 JJ = APOSI,J2
                AMULT1 = -A(JJ)*A(POSPV1)
                DO 1400 JJJ = JJ,J2
                  A(APOSC) = A(APOSC) + AMULT1*A(JJJ)
                  APOSC = APOSC + 1
 1400           CONTINUE
                A(JJ) = AMULT1
 1410         CONTINUE
              J1 = J1 + 1
            ELSE
              POSPV2 = POSPV1 + NFRONT + 1
              OFFDAG = POSPV1 + 1
              FLOPSB = FLOPSB + 6* (NCOL-NASS) +
     +                 2* (NCOL-NASS)* (NCOL-NASS+1)
              DO 1430 JJ = APOSI,J2
                AMULT1 = - (A(POSPV1)*A(JJ)+A(OFFDAG)*A(JJ+NFRONT))
                AMULT2 = -A(POSPV2)*A(JJ+NFRONT) - A(OFFDAG)*A(JJ)
                DO 1420 JJJ = JJ,J2
                  A(APOSC) = A(APOSC) + AMULT1*A(JJJ) +
     +                       AMULT2*A(JJJ+NFRONT)
                  APOSC = APOSC + 1
 1420           CONTINUE
                A(JJ) = AMULT1
                A(JJ+NFRONT) = AMULT2
 1430         CONTINUE
              J1 = J1 + 2
              POSPV1 = POSPV2
              LTWO = .TRUE.
              GO TO 1450
            END IF
 1440       LTWO = .FALSE.
            POSPV1 = POSPV1 + NFRONT + 1
 1450     CONTINUE
        ELSE
          DO 1490 I = NSTRUC + 1,NPIV
            IF (LTWO) GO TO 1480
            APOSI = APOS + (I-1)*NFRONT + NASS
            POSELT = APOS3 + (I-NSTRUC-1)* (NCOL-NASS)
            IF (IW(J1).GT.0) THEN
              FLOPSB = FLOPSB + (NCOL-NASS)
              DO 1460 JJ = APOSI,APOS + NFRONT* (I-1) + NCOL - 1
                A(POSELT) = A(JJ)
                A(JJ) = -A(JJ)*A(POSPV1)
                POSELT = POSELT + 1
 1460         CONTINUE
              J1 = J1 + 1
            ELSE
              POSPV2 = POSPV1 + NFRONT + 1
              OFFDAG = POSPV1 + 1
              FLOPSB = FLOPSB + 6* (NCOL-NASS)
              DO 1470 JJ = APOSI,APOS + NFRONT* (I-1) + NCOL - 1
                A(POSELT) = A(JJ)
                A(POSELT+NCOL-NASS) = A(JJ+NFRONT)
                A(JJ) = - (A(POSPV1)*A(JJ)+A(OFFDAG)*A(JJ+NFRONT))
                A(JJ+NFRONT) = -A(POSPV2)*A(JJ+NFRONT) -
     +                         A(OFFDAG)*A(POSELT)
                POSELT = POSELT + 1
 1470         CONTINUE
              J1 = J1 + 2
              POSPV1 = POSPV2
              LTWO = .TRUE.
              GO TO 1490
            END IF
 1480       LTWO = .FALSE.
            POSPV1 = POSPV1 + NFRONT + 1
 1490     CONTINUE
          FLOPSB = FLOPSB + NPIV* (NCOL-NASS)**2 + NPIV* (NCOL-NASS)
          KBLK = (NCOL-NASS)/NBLOC
          IF (NPIV-NSTRUC.LT.NBLOC) KBLK = 0
          APOSH = APOS + NSTRUC*NFRONT
          L = NCOL - NASS
          DO 1500 KR = 1,KBLK
            FLOPSX = FLOPSX + NBLOC* (NBLOC-1)* (NPIV-NSTRUC)
            CALL DGEMM('N','T',L- (KR-1)*NBLOC,NBLOC,NPIV-NSTRUC,ONE,
     +                 A(APOSH+NASS+NBLOC* (KR-1)),NFRONT,
     +                 A(APOS3+NBLOC* (KR-1)),L,ZONE,
     +                 A(APOS4+NBLOC* (L+1)* (KR-1)),L)
 1500     CONTINUE
          DO 1550 I = 1 + KBLK*NBLOC,L
            APOSA = APOS + NSTRUC*NFRONT + NASS + I - 1
            APOSB = APOS3 - 1
            APOSC = APOS4 + (I-1)*L - 1
            IF (ZONE.EQ.ONE) THEN
              DO 1510 J = I,L
                A(APOSC+J) = A(APOSC+J) + A(APOSA)*A(APOSB+J)
 1510         CONTINUE
            ELSE
              DO 1520 J = I,L
                A(APOSC+J) = A(APOSA)*A(APOSB+J)
 1520         CONTINUE
            END IF
            DO 1540 K = 2,NPIV - NSTRUC
              APOSA = APOSA + NFRONT
              APOSB = APOSB + L
              DO 1530 J = I,L
                A(APOSC+J) = A(APOSC+J) + A(APOSA)*A(APOSB+J)
 1530         CONTINUE
 1540       CONTINUE
 1550     CONTINUE
        END IF
 1560   IF (NCOL.LT.NFRONT) THEN
          DO 1570 JJ = IWPOS + NCOL,IWPOS + NFRONT - 1
            J = IW(JJ)
            PPOS(J) = N + 1
 1570     CONTINUE
        END IF
        IF (NPIV.EQ.NSTRUC) THEN
          NUMM1 = NUM1
          NUMM2 = NUM2
        ELSE
          NUMM1 = 0
          NUMM2 = NCOL - NASS
        END IF
        NELL = ELEMS(IASS)
        IELL = ISTK + 1
        PTRELT = ASTK + 1
        DO 1650 ELT = 1,NELL
          IF (NINT(A(PTRELT)).LT.0) PTRELT = PTRELT - NINT(A(PTRELT))
          IF (IW(IELL).LT.0) IELL = IELL - IW(IELL)
          IELLN = IELL + IW(IELL)
          POSELT = PTRELT + NINT(A(PTRELT))
          JP1 = IELL + 3
          JP2 = JP1 + IW(IELL+1)
          JP3 = JP2 + IW(IELL+2)
          JP4 = IELL + IW(IELL) - 2
          PPOS(0) = 0
          DO 1580 JJ = JP2,JP3 - 1
            IF (IW(JJ).EQ.0) GO TO 1580
            IF (PPOS(IW(JJ))-NASS.LE.NUMM1 .OR.
     +          PPOS(IW(JJ))-NASS.GT.NUMM1+NUMM2) GO TO 1640
 1580     CONTINUE
          APOS1 = 0
          DO 1590 JJ = JP1,JP2 - 1
            IF (IW(JJ).EQ.0) GO TO 1590
            IF (PPOS(IW(JJ))-NASS.LE.NUMM1) THEN
              IF (APOS1.EQ.2) GO TO 1640
              APOS1 = 1
            END IF
            IF (PPOS(IW(JJ))-NASS.GT.NUMM1+NUMM2) THEN
              IF (PPOS(IW(JJ)).EQ.N+1) GO TO 1640
              IF (APOS1.EQ.1) GO TO 1640
              APOS1 = 2
            END IF
 1590     CONTINUE
          DO 1600 JJ = JP3,JP4
            IF (IW(JJ).EQ.0) GO TO 1600
            IF (PPOS(IW(JJ))-NASS.LE.NUMM1) THEN
              IF (APOS1.EQ.1) GO TO 1640
            END IF
            IF (PPOS(IW(JJ))-NASS.GT.NUMM1+NUMM2) THEN
              IF (PPOS(IW(JJ)).EQ.N+1) GO TO 1640
              IF (APOS1.EQ.2) GO TO 1640
            END IF
 1600     CONTINUE
          APOS1 = PTRELT + 1
          JA2 = JP2
          DO 1630 JJ = JP1,JP3 - 1
            J = IW(JJ)
            IF (J.EQ.0) THEN
              IF (JJ.LE.JP2) THEN
                APOS1 = APOS1 + JP4 - JP2 + 1
              ELSE
                APOS1 = APOS1 + JP4 - JJ + 1
              END IF
              GO TO 1630
            END IF
            IF (JJ.GT.JP2) JA2 = JJ
            IF (NOSURE) THEN
              DO 1610 JJJ = JA2,JP4
                JAY = IW(JJJ)
                IF (PPOS(JAY).GE.PPOS(J)) THEN
                  APOS2 = APOS4 + 1 + ((PPOS(J)-NASS-1)*
     +                    (2*NSC2-PPOS(J)+NASS+2))/2 + PPOS(JAY) -
     +                    PPOS(J)
                  A(APOS2) = A(APOS2) + A(APOS1)
                ELSE IF (JAY.GT.0) THEN
                  APOS2 = APOS4 + 1 + ((PPOS(JAY)-NASS-1)*
     +                    (2*NSC2-PPOS(JAY)+NASS+2))/2 + PPOS(J) -
     +                    PPOS(JAY)
                  A(APOS2) = A(APOS2) + A(APOS1)
                END IF
                APOS1 = APOS1 + 1
 1610         CONTINUE
            ELSE
              DO 1620 JJJ = JA2,JP4
                JAY = IW(JJJ)
                IF (PPOS(JAY).GE.PPOS(J)) THEN
                  APOS2 = APOS4 + (PPOS(J)-NASS-1)*NSC2 + PPOS(JAY) -
     +                    NASS - 1
                  IF (NSTRUC.EQ.NPIV) APOS2 = APOS2 - NUM1
                  A(APOS2) = A(APOS2) + A(APOS1)
                ELSE IF (JAY.GT.0) THEN
                  APOS2 = APOS4 + (PPOS(JAY)-NASS-1)*NSC2 + PPOS(J) -
     +                    NASS - 1
                  IF (NSTRUC.EQ.NPIV) APOS2 = APOS2 - NUM1
                  A(APOS2) = A(APOS2) + A(APOS1)
                END IF
                APOS1 = APOS1 + 1
 1620         CONTINUE
            END IF
 1630     CONTINUE
          PPOS(0) = N + 1
          NELL = NELL - 1
          NNELL = NNELL - 1
          ELEMS(IASS) = ELEMS(IASS) - 1
          LIELL = IW(IELL)
          NSTKAC(2) = NSTKAC(2) - LIELL
          J1 = IELL
          J2 = IELLN - 1
          IF (IELLN.LT.PTRIRN-IINPUT) THEN
            IF (IW(IELLN).LT.0) THEN
              LIELL = LIELL - IW(IELLN)
              J2 = J2 - IW(IELLN)
            END IF
          END IF
          IF ((IELL-1).EQ.ISTK) THEN
            ISTK = ISTK + LIELL
          ELSE
            IF (IW(IELL-1).LT.0) THEN
              LIELL = LIELL - IW(IELL-1)
              J1 = IELL + IW(IELL-1)
            END IF
            IW(J1) = -LIELL
            IW(J2) = -LIELL
          END IF
          LAELL = NINT(A(PTRELT))
          NSTKAC(1) = NSTKAC(1) - LAELL
          JA1 = PTRELT
          JA2 = POSELT - 1
          IF (POSELT.LT.PTRA-AINPUT) THEN
            IF (NINT(A(POSELT)).LT.0) THEN
              LAELL = LAELL - NINT(A(POSELT))
              JA2 = JA2 - NINT(A(POSELT))
            END IF
          END IF
          IF ((PTRELT-1.EQ.ASTK) .AND. (.NOT.NOSURE)) THEN
            ASTK = ASTK + LAELL
          ELSE
            IF (NINT(A(PTRELT-1)).LT.0) THEN
              LAELL = LAELL - NINT(A(PTRELT-1))
              JA1 = PTRELT + NINT(A(PTRELT-1))
            END IF
            A(JA1) = -LAELL
            A(JA2) = -LAELL
          END IF
 1640     IELL = IELLN
          PTRELT = POSELT
 1650   CONTINUE
        PPOS(0) = N + 1
        ELEMS(FATHER(IASS)) = ELEMS(FATHER(IASS)) + 1
        IF (NOSURE) THEN
          ASTK = APOS4 - 1
          LAELL = ((NSC1+1)*NSC1)/2 + 2
        ELSE
          IF (NSTRUC.EQ.NPIV) THEN
            LAELL = NUM1* (NUM2+NUM3) + (NUM2* (NUM2+1))/2 + NUM2*NUM3 +
     +              2
            JA1 = APOS4 + (NUM1+NUM2)* (NUM2+NUM3) - 1
          ELSE
            LAELL = ((NSC1+1)* (NSC1))/2 + 2
            JA1 = APOS4 + (NSC1)* (NSC1) - 1
          END IF
          A(ASTK) = LAELL
          NSTKAC(1) = NSTKAC(1) + LAELL
          ASTK = ASTK - 1
          IF (NSTRUC.EQ.NPIV) THEN
            DO 1670 I = 1,NUM2
              DO 1660 JJ = JA1,JA1 - NUM3 - I + 1,-1
                A(ASTK) = A(JJ)
                ASTK = ASTK - 1
 1660         CONTINUE
              JA1 = JA1 - (NUM2+NUM3)
 1670       CONTINUE
            DO 1690 I = 1,NUM1
              DO 1680 JJ = 1,NUM2 + NUM3
                A(ASTK) = A(JA1)
                ASTK = ASTK - 1
                JA1 = JA1 - 1
 1680         CONTINUE
 1690       CONTINUE
          ELSE
            DO 1710 I = 1,NSC1
              DO 1700 JJ = JA1,JA1 - I + 1,-1
                A(ASTK) = A(JJ)
                ASTK = ASTK - 1
 1700         CONTINUE
              JA1 = JA1 - NSC1
 1710       CONTINUE
          END IF
          A(ASTK) = LAELL
          ASTK = ASTK - 1
        END IF
        LIELL = NCOL - NASS + 4
        NSTKAC(2) = NSTKAC(2) + LIELL
        INTSPA = MAX(INTSPA,IWPOS+2*NFRONT+LIELL+NSTKAC(2))
        IF (IWPOS+2*NFRONT+LIELL.GT.ISTK) THEN
          CALL MA47PD(IW,ISTK,PTRIRN,IINPUT,NCMPBI)
          IF (IWPOS+2*NFRONT+LIELL.GT.ISTK) THEN
            INFO(2) = INTSPA
            INFO(1) = -3
            GO TO 2170
          END IF
        END IF
        IW(ISTK) = LIELL
        ISTK = ISTK - 1
        NNELL = NNELL + 1
        DO 1720 I = 1,NCOL - NASS
          IW(ISTK) = IW(IWPOS+NCOL-I)
          ISTK = ISTK - 1
 1720   CONTINUE
        IF (NSTRUC.EQ.NPIV) THEN
          IW(ISTK) = NUM2
          IW(ISTK-1) = NUM1
        ELSE
          IW(ISTK) = NCOL - NASS
          IW(ISTK-1) = 0
        END IF
        IW(ISTK-2) = LIELL
        ISTK = ISTK - 3
 1730   IF (NPIV+NNULL.LT.NASS) THEN
          ZCOL = 0
          APOS1 = APOS + NPIV*NFRONT + NASS - 1
          DO 1750 J = NASS + 1,NFRONT
            APOS1 = APOS1 + 1
            DO 1740 JA1 = 1,NASS - NPIV
              IF (A(APOS1+ (JA1-1)*NFRONT).NE.ZERO) GO TO 1750
 1740       CONTINUE
            IW(IWPOS+J-1) = -IW(IWPOS+J-1)
            ZCOL = ZCOL + 1
 1750     CONTINUE
          ELEMS(FATHER(IASS)) = ELEMS(FATHER(IASS)) + 1
          LAELL = (NASS-NPIV)* (NFRONT-NPIV-ZCOL) + 2
          RLSPA = MAX(RLSPA,INFO(2)+APOS+NASS*NFRONT+LAELL+NSTKAC(1))
          IF (APOS+NASS*NFRONT+LAELL.GT.ASTK) THEN
            CALL MA47SD(A,ASTK,PTRA,AINPUT,NCMPBR)
            IF (APOS+NASS*NFRONT+LAELL.GT.ASTK) THEN
              INFO(2) = INFO(2) + APOS - 1
              DO 1760 I = 1,NASS*NFRONT
                A(I) = A(APOS+I-1)
 1760         CONTINUE
              APOSBB = 1
              APOS = 1
              INFO(1) = -4
              IF (NASS*NFRONT+LAELL.GT.ASTK) THEN
                INFO(2) = RLSPA
                GO TO 2170
              END IF
            END IF
          END IF
          A(ASTK) = LAELL
          NSTKAC(1) = NSTKAC(1) + LAELL
          ASTK = ASTK - LAELL
          A(ASTK+1) = LAELL
          JA1 = ASTK + 2
          JA2 = APOS + NPIV*NFRONT + NPIV
          IF (ZCOL.EQ.0) THEN
            DO 1780 I = 1,NASS - NPIV
              DO 1770 JJ = JA2,JA2 + NFRONT - NPIV - 1
                A(JA1) = A(JJ)
                JA1 = JA1 + 1
 1770         CONTINUE
              JA2 = JA2 + NFRONT
 1780       CONTINUE
          ELSE
            DO 1800 I = 1,NASS - NPIV
              DO 1790 JJ = JA2,JA2 + NFRONT - NPIV - 1
                IF (IW(IWPOS+NPIV+JJ-JA2).GT.0) THEN
                  A(JA1) = A(JJ)
                  JA1 = JA1 + 1
                END IF
 1790         CONTINUE
              JA2 = JA2 + NFRONT
 1800       CONTINUE
          END IF
          LIELL = NFRONT - NPIV - ZCOL + 4
          INTSPA = MAX(INTSPA,IWPOS+2*NFRONT+2+LIELL+NSTKAC(2))
          IF (IWPOS+2*NFRONT+2+LIELL.GT.ISTK) THEN
            CALL MA47PD(IW,ISTK,PTRIRN,IINPUT,NCMPBI)
            IF (IWPOS+2*NFRONT+2+LIELL.GT.ISTK) THEN
              INFO(2) = INTSPA
              INFO(1) = -3
              GO TO 2170
            END IF
          END IF
          NNELL = NNELL + 1
          IW(ISTK) = LIELL
          NSTKAC(2) = NSTKAC(2) + LIELL
          ISTK = ISTK - LIELL
          IW(ISTK+1) = LIELL
          IW(ISTK+2) = -1
          IW(ISTK+3) = NASS - NPIV
          J1 = ISTK + 4
          IF (ZCOL.EQ.0) THEN
            DO 1810 JJ = IWPOS + NPIV,IWPOS + NFRONT - 1
              IW(J1) = IW(JJ)
              J1 = J1 + 1
 1810       CONTINUE
          ELSE
            DO 1820 JJ = IWPOS + NPIV,IWPOS + NFRONT - 1
              IF (IW(JJ).LE.0) THEN
                IW(JJ) = -IW(JJ)
              ELSE
                IW(J1) = IW(JJ)
                J1 = J1 + 1
              END IF
 1820       CONTINUE
          END IF
        END IF
 1830   DO 1840 JJ = IWPOS + NPIV,IWPOS + NFRONT - 1
          J = ABS(IW(JJ))
          PPOS(J) = N + 1
 1840   CONTINUE
        IF (FATHER(IASS).LE.NODES) THEN
          ELEMS(FATHER(IASS)) = ELEMS(FATHER(IASS)) + NELL
        END IF
C********************************
C********************************
        IF (NPIV.EQ.0) GO TO 2160
        NBLK = NBLK + 1
        ZCOL = 0
        APOS1 = APOS + NPIV
        DO 1880 J = 1,NASS - NPIV
          DO 1850 JA1 = 1,NPIV
            IF (A(APOS1+ (JA1-1)*NFRONT).NE.ZERO) GO TO 1870
 1850     CONTINUE
          ZCOL = ZCOL + 1
          APOS2 = APOS + NASS - ZCOL
          DO 1860 JA1 = 1,NPIV
            SWOP = A(APOS1+ (JA1-1)*NFRONT)
            A(APOS1+ (JA1-1)*NFRONT) = A(APOS2+ (JA1-1)*NFRONT)
            A(APOS2+ (JA1-1)*NFRONT) = SWOP
 1860     CONTINUE
          ISWOP = IW(IWPOS+NPIV+J-ZCOL)
          IW(IWPOS+NPIV+J-ZCOL) = IW(IWPOS+NASS-ZCOL)
          IW(IWPOS+NASS-ZCOL) = ISWOP
          GO TO 1880
 1870     APOS1 = APOS1 + 1
 1880   CONTINUE
        IF (ZCOL.GT.0) THEN
          APOS1 = APOS + NASS - ZCOL
          APOS2 = APOS + NASS
          DO 1900 J = 1,NCOL - NASS
            DO 1890 JA1 = 1,NPIV
              A(APOS1+ (JA1-1)*NFRONT) = A(APOS2+ (JA1-1)*NFRONT)
 1890       CONTINUE
            IW(IWPOS+NASS-ZCOL+J-1) = IW(IWPOS+NASS+J-1)
            APOS1 = APOS1 + 1
            APOS2 = APOS2 + 1
 1900     CONTINUE
        END IF
        NCOL = NCOL - ZCOL
        NASS = NASS - ZCOL
        IF (NSTRUC.GT.0) THEN
          IW(IWPOS-4) = -NCOL
          IW(IWPOS-3) = NSTRUC
          IF (KIND.EQ.3) IW(IWPOS-3) = -NSTRUC
          IW(IWPOS-2) = NUM3
          IF (KIND.EQ.2) THEN
            DO 1910 I = 1,NCOL
              IW(IWPOS+I-2) = IW(IWPOS+I-1)
 1910       CONTINUE
            IWPOS = IWPOS + NCOL + 3
            NUM1 = 0
          ELSE
            IW(IWPOS-1) = NUM1
            IWPOS = IWPOS + NCOL + 4
          END IF
          IF (NPIV.NE.NSTRUC) THEN
            NBLK = NBLK + 1
            J1 = IWPOS - NCOL - 4 + NSTRUC
            J2 = IWPOS - 2
            DO 1920 I = NSTRUC + 1,NCOL
              IW(J2) = IW(J1)
              J2 = J2 + 1
              J1 = J1 + 1
 1920       CONTINUE
          END IF
          IF (NSTRUC.LT.NASS) THEN
            IF (NUM1.NE.0) THEN
              J1 = IWPOS - 4 - NCOL + NSTRUC
              J2 = J1
              DO 1930 I = 1,NASS - NSTRUC
                PPOS(I) = IW(J2)
                J2 = J2 + 1
 1930         CONTINUE
              J2 = J1 + NASS - NSTRUC
              DO 1940 I = 1,NUM1
                IW(J1) = IW(J2)
                J1 = J1 + 1
                J2 = J2 + 1
 1940         CONTINUE
              DO 1950 I = 1,NASS - NSTRUC
                IW(J1) = PPOS(I)
                PPOS(I) = N + 1
                J1 = J1 + 1
 1950         CONTINUE
            END IF
          END IF
          IF (NPIV.EQ.NSTRUC) GO TO 1970
          IWPOS = IWPOS - 2
        ELSE
          J1 = IWPOS
          J2 = IWPOS - 2
          DO 1960 I = 1,NCOL
            IW(J2) = IW(J1)
            J2 = J2 + 1
            J1 = J1 + 1
 1960     CONTINUE
          IWPOS = IWPOS - 2
        END IF
        IW(IWPOS-2) = NCOL - NSTRUC
        IW(IWPOS-1) = NPIV - NSTRUC
        IWPOS = IWPOS + NCOL - NSTRUC + 4
 1970   IF (INFO(1).EQ.-4) THEN
          INFO(2) = INFO(2) + ((NPIV-NSTRUC)* (2*NCOL-NPIV+NSTRUC+1))/2
          GO TO 2160
        END IF
        APOS2 = APOSBB
        DO 1990 I = 1,NSTRUC/2
          JA1 = APOS + (I-1)*NFRONT + NSTRUC/2
          DO 1980 J = 1,NSTRUC/2
            A(APOS2) = A(JA1)
            APOS2 = APOS2 + 1
            JA1 = JA1 + 1
 1980     CONTINUE
 1990   CONTINUE
        IF (KIND.EQ.2) THEN
          DO 2010 I = 1,NSTRUC/2
            JA1 = APOS + (NSTRUC/2+I-1)* (NFRONT+1) + 1
            DO 2000 J = I + 1,NSTRUC/2
              A(APOS2) = A(JA1)
              APOS2 = APOS2 + 1
              JA1 = JA1 + 1
 2000       CONTINUE
 2010     CONTINUE
          JA1 = APOS
          DO 2020 I = 1,NSTRUC/2
            A(APOS2) = A(JA1)
            JA1 = JA1 + NFRONT + 1
            APOS2 = APOS2 + 1
 2020     CONTINUE
          DO 2030 I = 1,NSTRUC/2
            A(APOS2) = ZERO
            APOS2 = APOS2 + 1
 2030     CONTINUE
        ELSE
          DO 2040 I = 1,NSTRUC
            A(APOS2) = ZERO
            APOS2 = APOS2 + 1
 2040     CONTINUE
        END IF
        DO 2070 I = 1,NSTRUC/2
          JA1 = APOS + (I-1)*NFRONT + NSTRUC
          DO 2050 J = NSTRUC + 1,NASS
            A(APOS2) = A(JA1)
            APOS2 = APOS2 + 1
            JA1 = JA1 + 1
 2050     CONTINUE
          JA1 = JA1 + NUM1
          DO 2060 J = 1,NCOL - NASS - NUM1
            A(APOS2) = A(JA1)
            APOS2 = APOS2 + 1
            JA1 = JA1 + 1
 2060     CONTINUE
 2070   CONTINUE
        DO 2110 I = NSTRUC/2 + 1,NSTRUC
          JA1 = APOS + (I-1)*NFRONT + NASS
          DO 2080 J = 1,NUM1
            A(APOS2) = A(JA1)
            APOS2 = APOS2 + 1
            JA1 = JA1 + 1
 2080     CONTINUE
          JA1 = APOS + (I-1)*NFRONT + NSTRUC
          DO 2090 J = 1,NASS - NSTRUC
            A(APOS2) = A(JA1)
            APOS2 = APOS2 + 1
            JA1 = JA1 + 1
 2090     CONTINUE
          JA1 = APOS + (I-1)*NFRONT + NASS + NUM1
          DO 2100 J = 1,NCOL - NASS - NUM1 - NUM3
            A(APOS2) = A(JA1)
            APOS2 = APOS2 + 1
            JA1 = JA1 + 1
 2100     CONTINUE
 2110   CONTINUE
        DO 2130 I = 1,NPIV - NSTRUC
          JA1 = APOS + (NSTRUC+I-1)* (NFRONT+1)
          DO 2120 J = I,NPIV - NSTRUC
            A(APOS2) = A(JA1)
            APOS2 = APOS2 + 1
            JA1 = JA1 + 1
 2120     CONTINUE
 2130   CONTINUE
        DO 2150 I = 1,NPIV - NSTRUC
          JA1 = APOS + (NSTRUC+I-1)*NFRONT + NPIV
          DO 2140 J = 1,NCOL - NPIV
            A(APOS2) = A(JA1)
            APOS2 = APOS2 + 1
            JA1 = JA1 + 1
 2140     CONTINUE
 2150   CONTINUE
        APOSBB = APOS2
 2160 CONTINUE
      IF (INFO(1).EQ.-4) THEN
        INFO(2) = RLSPA
        GO TO 2170
      END IF
      NRLBDU = APOSBB - 1
      NIRBDU = IWPOS - 5
      IW(1) = NRLBDU + 1
      IW(2) = NRLBDU + NFULLB
      IW(3) = NBLK
      CALL MA47WD(A,LA,IW,LIW,NRLBDU)
      INFO(6) = MAX(2*NE,RLSPA)
      INFO(7) = MAX(2*NE,INTSPA)
      INFO(15) = MAXFRT
      INFO(16) = NRLBDU
      INFO(17) = NIRBDU
      INFO(18) = NCMPBR
      INFO(19) = NCMPBI
      INFO(20) = NTILEB
      INFO(21) = NOXOB
      INFO(22) = NFULLB
      INFO(23) = NEIG
      INFO(24) = N - NTOTPV
      RINFO(3) = FLOPSB
      RINFO(4) = FLOPSX
 2170 RETURN
      END
      SUBROUTINE MA47PD(IW,BOTTOM,TOP,MOVE,NCMP)
      INTEGER IW(*)
      INTEGER BOTTOM,TOP,MOVE,NCMP
      INTEGER IDUMMY,JJ,K,SIZE
      NCMP = NCMP + 1
      K = TOP - MOVE - 1
      DO 20 IDUMMY = 1,TOP
        IF (K.LE.BOTTOM) GO TO 30
        SIZE = IW(K)
        IF (SIZE.LT.0) THEN
          MOVE = MOVE - SIZE
          K = K + SIZE
        ELSE
          IF (MOVE.GT.0) THEN
            DO 10 JJ = K,K - SIZE + 1,-1
              IW(JJ+MOVE) = IW(JJ)
   10       CONTINUE
          END IF
          K = K - SIZE
        END IF
   20 CONTINUE
   30 BOTTOM = BOTTOM + MOVE
      MOVE = 0
      RETURN
      END
      SUBROUTINE MA47QD(N,A,LA,IW,LIW,W,RHS,IW1,ICNTL)
      INTEGER N,LA
      DOUBLE PRECISION A(LA)
      INTEGER LIW,IW(LIW)
      DOUBLE PRECISION W(N),RHS(N)
      INTEGER IW1(N),ICNTL(7)
      INTRINSIC ABS
      EXTERNAL DGEMV,DTPMV,DTPSV,DTRSV
      DOUBLE PRECISION ONE
      PARAMETER (ONE=1.0D0)
      INTEGER APOS,APOSM,APOSO,I,IBLK,IPIV,IRHS,IWPOS,J,J1,J2,K,KIND,
     +        NCOLS,NROWS,NUM1,NUM3,N2
      DOUBLE PRECISION W1
      APOS = 1
      IWPOS = 4
      DO 270 IBLK = 1,IW(3)
        IW1(IBLK) = IWPOS
        NCOLS = IW(IWPOS)
        NROWS = IW(IWPOS+1)
        IWPOS = IWPOS + 2
        NUM1 = 0
        NUM3 = 0
        IF (NCOLS.GT.0) THEN
          KIND = 1
          N2 = NROWS
        ELSE
          NCOLS = -NCOLS
          NUM3 = IW(IWPOS)
          IWPOS = IWPOS + 1
          IF (NROWS.GT.0) THEN
            KIND = 2
          ELSE
            NROWS = -NROWS
            NUM1 = IW(IWPOS)
            IWPOS = IWPOS + 1
            KIND = 3
          END IF
          N2 = NROWS/2
        END IF
        IF (N2.GT.ICNTL(7)) THEN
          DO 10 I = 1,NCOLS
            W(I) = RHS(ABS(IW(IWPOS+I-1)))
   10     CONTINUE
          IF (KIND.EQ.1) THEN
            CALL DTPSV('L','N','U',NROWS,A(APOS),W,1)
            APOS = APOS + (NROWS* (NROWS+1))/2
            IF (NCOLS.GT.NROWS) CALL DGEMV('N',NCOLS-NROWS,NROWS,ONE,
     +                               A(APOS),NCOLS-NROWS,W,1,ONE,
     +                               W(NROWS+1),1)
            APOS = APOS + NROWS* (NCOLS-NROWS)
            DO 35 I = 1,NCOLS
              RHS(ABS(IW(IWPOS+I-1))) = W(I)
   35       CONTINUE
          ELSE IF (KIND.EQ.2) THEN
            CALL DTRSV('U','T','U',N2,A(APOS),N2,W,1)
            APOSM = APOS + N2*N2
            APOSO = APOSM + (N2* (N2+1))/2 + N2
            K = NCOLS - NROWS
            IF (K.GT.NUM1) CALL DGEMV('N',K-NUM1,N2,ONE,A(APOSO),K-NUM1,
     +                                W,1,ONE,W(NROWS+1+NUM1),1)
            DO 40 I = 1,N2
              RHS(ABS(IW(IWPOS+I-1))) = W(I)
   40       CONTINUE
            CALL DTPMV('L','N','N',N2-1,A(APOSM),W,1)
            DO 50 I = N2 + 2,NROWS
              W(I) = W(I) + W(I-N2-1)
   50       CONTINUE
            CALL DTRSV('L','N','U',N2,A(APOS),N2,W(N2+1),1)
            APOS = APOSO + N2* (K-NUM1)
            IF (K.GT.NUM3) CALL DGEMV('N',K-NUM3,N2,ONE,A(APOS),K-NUM3,
     +                                W(N2+1),1,ONE,W(NROWS+1),1)
            APOS = APOS + N2* (K-NUM3)
            DO 60 I = N2 + 1,NCOLS
              RHS(ABS(IW(IWPOS+I-1))) = W(I)
   60       CONTINUE
          ELSE
            CALL DTRSV('U','T','U',N2,A(APOS),N2,W,1)
            CALL DTRSV('L','N','U',N2,A(APOS),N2,W(N2+1),1)
            APOS = APOS + N2*N2 + 2*N2
            K = NCOLS - NROWS
            IF (K.GT.NUM1) CALL DGEMV('N',K-NUM1,N2,ONE,A(APOS),K-NUM1,
     +                                W,1,ONE,W(NROWS+1+NUM1),1)
            APOS = APOS + N2* (K-NUM1)
            IF (K.GT.NUM3) CALL DGEMV('N',K-NUM3,N2,ONE,A(APOS),K-NUM3,
     +                                W(N2+1),1,ONE,W(NROWS+1),1)
            APOS = APOS + N2* (K-NUM3)
            DO 90 I = 1,NCOLS
              RHS(ABS(IW(IWPOS+I-1))) = W(I)
   90       CONTINUE
          END IF
        ELSE
          J1 = IWPOS
          J2 = IWPOS + NROWS - 1
          IF (KIND.EQ.1) THEN
            DO 130 IPIV = 1,NROWS
              APOS = APOS + 1
              W1 = RHS(ABS(IW(J1)))
              J1 = J1 + 1
              DO 100 J = J1,J2
                IRHS = ABS(IW(J))
                RHS(IRHS) = RHS(IRHS) - A(APOS)*W1
                APOS = APOS + 1
  100         CONTINUE
  130       CONTINUE
            J2 = IWPOS + NCOLS - 1
            DO 136 IPIV = 1,NROWS
              W1 = RHS(ABS(IW(IWPOS+IPIV-1)))
              DO 133 J = J1,J2
                IRHS = ABS(IW(J))
                RHS(IRHS) = RHS(IRHS) + W1*A(APOS)
                APOS = APOS + 1
  133         CONTINUE
  136       CONTINUE
          ELSE
            J2 = IWPOS + N2 - 1
            APOSO = APOS + N2*N2
            DO 145 IPIV = 1,N2 - 1
              K = APOS + (IPIV-1)* (N2+1)
              W1 = RHS(-IW(J1))
              J1 = J1 + 1
              DO 140 J = J1,J2
                IRHS = ABS(IW(J))
                K = K + N2
                RHS(IRHS) = RHS(IRHS) - W1*A(K)
  140         CONTINUE
  145       CONTINUE
            IF (KIND.EQ.2) THEN
              K = APOSO
              APOSO = APOSO + (N2* (N2+1))/2 + N2
              J1 = IWPOS
              DO 155 IPIV = 1,N2 - 1
                W1 = RHS(-IW(J1))
                J1 = J1 + 1
                DO 150 J = J1,J2
                  IRHS = ABS(IW(J+N2))
                  RHS(IRHS) = RHS(IRHS) + W1*A(K)
                  K = K + 1
  150           CONTINUE
  155         CONTINUE
            ELSE
              APOSO = APOSO + 2*N2
            END IF
            J1 = IWPOS + N2
            J2 = J2 + N2
            DO 165 IPIV = 1,N2 - 1
              K = APOS + (IPIV-1)* (N2+1)
              W1 = RHS(-IW(J1))
              J1 = J1 + 1
              DO 160 J = J1,J2
                IRHS = ABS(IW(J))
                K = K + 1
                RHS(IRHS) = RHS(IRHS) - W1*A(K)
  160         CONTINUE
  165       CONTINUE
            APOS = APOSO
            J1 = IWPOS + NROWS
            J2 = IWPOS + NCOLS - 1
            DO 195 IPIV = 1,N2
              W1 = RHS(ABS(IW(IWPOS+IPIV-1)))
              DO 190 J = J1 + NUM1,J2
                IRHS = ABS(IW(J))
                RHS(IRHS) = RHS(IRHS) + W1*A(APOS)
                APOS = APOS + 1
  190         CONTINUE
  195       CONTINUE
            DO 210 IPIV = 1,N2
              W1 = RHS(ABS(IW(IWPOS+IPIV-1+N2)))
              DO 200 J = J1,J2 - NUM3
                IRHS = ABS(IW(J))
                RHS(IRHS) = RHS(IRHS) + W1*A(APOS)
                APOS = APOS + 1
  200         CONTINUE
  210       CONTINUE
          END IF
        END IF
        IWPOS = IWPOS + NCOLS
  270 CONTINUE
      END
      SUBROUTINE MA47RD(N,A,LA,IW,LIW,W,RHS,IW1,ICNTL)
      INTEGER N,LA
      DOUBLE PRECISION A(LA)
      INTEGER LIW,IW(LIW)
      DOUBLE PRECISION W(N),RHS(N)
      INTEGER IW1(N),ICNTL(7)
      INTRINSIC ABS
      EXTERNAL DGEMV,DTPMV,DTPSV,DTRSV
      DOUBLE PRECISION ONE
      PARAMETER (ONE=1.0D0)
      INTEGER APOS,APOSD,APOSE,APOSF,APOSM,APOS2,I,IBLK,IPIV,IRHS,IRHS1,
     +        IRHS2,IWPOS,J,JPIV,J1,J2,K,KIND,LROW,NCOLS,NROWS,NUM1,
     +        NUM3,N2
      DOUBLE PRECISION W1
      APOS = IW(1)
      APOS2 = IW(2)
      DO 380 IBLK = IW(3),1,-1
        IWPOS = IW1(IBLK)
        NCOLS = ABS(IW(IWPOS))
        NROWS = ABS(IW(IWPOS+1))
        NUM1 = 0
        NUM3 = 0
        IF (IW(IWPOS).GT.0) THEN
          KIND = 1
          APOS = APOS - NROWS* (NCOLS-NROWS)
          N2 = NROWS
        ELSE
          N2 = NROWS/2
          APOS = APOS - (NROWS* (NCOLS+NCOLS-NROWS+1))/2
          NUM3 = IW(IWPOS+2)
          APOS = APOS + NUM3*N2 + (N2* (N2+1))/2
          IWPOS = IWPOS + 1
          IF (IW(IWPOS).GT.0) THEN
            APOS = APOS - N2
            KIND = 2
          ELSE
            NUM1 = IW(IWPOS+2)
            APOS = APOS + NUM1*N2 + (N2* (N2+1))/2
            APOS = APOS - 2*N2
            IWPOS = IWPOS + 1
            KIND = 3
          END IF
        END IF
        IWPOS = IWPOS + 2
        IF (N2.GT.ICNTL(7)) THEN
          DO 5 I = NROWS + 1,NCOLS
            W(I) = RHS(ABS(IW(IWPOS+I-1)))
    5     CONTINUE
          IF (KIND.EQ.1) THEN
            DO 10 IPIV = NROWS,1,-1
              IRHS = ABS(IW(IWPOS+IPIV-1))
              APOS = APOS - (NROWS+1-IPIV)
              W(IPIV) = RHS(IRHS)*A(APOS)
   10       CONTINUE
            JPIV = -1
            DO 20 IPIV = NROWS,1,-1
              IRHS = IW(IWPOS+IPIV-1)
              IF (IRHS.LT.0) THEN
                IRHS1 = -IW(IWPOS+IPIV-1+JPIV)
                W(IPIV) = RHS(IRHS1)*A(APOS2) + W(IPIV)
                IF (JPIV.EQ.1) APOS2 = APOS2 - 1
                JPIV = -JPIV
              END IF
   20       CONTINUE
            K = NCOLS - NROWS
            IF (K.GT.0) CALL DGEMV('T',K,NROWS,ONE,
     +                             A(APOS+ (NROWS* (NROWS+1))/2),K,
     +                             W(NROWS+1),1,ONE,W,1)
            CALL DTPSV('L','T','U',NROWS,A(APOS),W,1)
            DO 60 I = 1,NROWS
              RHS(ABS(IW(IWPOS+I-1))) = W(I)
   60       CONTINUE
          ELSE
            APOSD = APOS
            APOSM = APOS + N2*N2
            APOSE = APOSM + (N2* (N2-1))/2
            IF (KIND.EQ.3) APOSE = APOSM
            APOSF = APOSE + N2
            DO 70 IPIV = 1,N2
              IRHS1 = -IW(IWPOS+IPIV-1)
              IRHS2 = -IW(IWPOS+IPIV-1+N2)
              W(IPIV) = RHS(IRHS1)*A(APOSE) + RHS(IRHS2)*A(APOSD)
              W(IPIV+N2) = RHS(IRHS1)*A(APOSD) + RHS(IRHS2)*A(APOSF)
              APOSD = APOSD + N2 + 1
              APOSE = APOSE + 1
              APOSF = APOSF + 1
   70       CONTINUE
            APOSD = APOS
            APOS = APOSF
            K = NCOLS - NROWS
            IF (K.GT.NUM1) CALL DGEMV('T',K-NUM1,N2,ONE,A(APOS),K-NUM1,
     +                                W(NROWS+NUM1+1),1,ONE,W,1)
            IF (K.GT.NUM3) CALL DGEMV('T',K-NUM3,N2,ONE,
     +                                A(APOS+N2* (K-NUM1)),K-NUM3,
     +                                W(NROWS+1),1,ONE,W(N2+1),1)
            IF (KIND.EQ.2) THEN
              APOS = APOSD
              CALL DTRSV('L','T','U',N2,A(APOSD),N2,W(N2+1),1)
              DO 90 I = N2 + 1,NROWS
                RHS(ABS(IW(IWPOS+I-1))) = W(I)
   90         CONTINUE
              CALL DTPMV('L','T','N',N2-1,A(APOSM),W(N2+2),1)
              DO 100 I = 1,N2 - 1
                W(I) = W(I) + W(I+N2+1)
  100         CONTINUE
              CALL DTRSV('U','N','U',N2,A(APOSD),N2,W,1)
              DO 110 I = 1,N2
                RHS(ABS(IW(IWPOS+I-1))) = W(I)
  110         CONTINUE
            ELSE
              APOS = APOSD
              CALL DTRSV('L','T','U',N2,A(APOS),N2,W(N2+1),1)
              CALL DTRSV('U','N','U',N2,A(APOS),N2,W,1)
              DO 180 I = 1,NROWS
                RHS(ABS(IW(IWPOS+I-1))) = W(I)
  180         CONTINUE
            END IF
          END IF
        ELSE
          J1 = IWPOS
          J2 = IWPOS + NCOLS - 1
          K = APOS
          IF (KIND.EQ.1) THEN
            JPIV = -1
            DO 210 IPIV = NROWS,1,-1
              IRHS = IW(IWPOS+IPIV-1)
              LROW = NROWS + 1 - IPIV
              IF (IRHS.GT.0) THEN
                APOS = APOS - LROW
                RHS(IRHS) = RHS(IRHS)*A(APOS)
              ELSE
                IF (JPIV.EQ.-1) THEN
                  IRHS1 = -IW(IWPOS+IPIV-2)
                  IRHS2 = -IRHS
                  APOS = APOS - LROW - LROW - 1
                  W1 = RHS(IRHS1)*A(APOS) + RHS(IRHS2)*A(APOS2)
                  RHS(IRHS2) = RHS(IRHS1)*A(APOS2) +
     +                         RHS(IRHS2)*A(APOS+LROW+1)
                  RHS(IRHS1) = W1
                  APOS2 = APOS2 - 1
                END IF
                JPIV = -JPIV
              END IF
  210       CONTINUE
            APOS = APOS + (NROWS* (NROWS+1))/2
            K = APOS
            J1 = IWPOS + NROWS
            DO 220 IPIV = 1,NROWS
              IRHS = ABS(IW(IWPOS+IPIV-1))
              W1 = RHS(IRHS)
              DO 215 J = J1,J2
                W1 = W1 + A(K)*RHS(ABS(IW(J)))
                K = K + 1
  215         CONTINUE
              RHS(IRHS) = W1
  220       CONTINUE
            J2 = IWPOS + NROWS - 1
            DO 260 IPIV = 1,NROWS
              IRHS = ABS(IW(J1-1))
              APOS = APOS - IPIV
              W1 = RHS(IRHS)
              K = APOS + 1
              DO 230 J = J1,J2
                W1 = W1 - A(K)*RHS(ABS(IW(J)))
                K = K + 1
  230         CONTINUE
              RHS(IRHS) = W1
              J1 = J1 - 1
  260       CONTINUE
          ELSE
            APOSD = APOS
            APOSM = APOS + N2*N2
            APOSE = APOSM + (N2* (N2-1))/2
            IF (KIND.EQ.3) APOSE = APOSM
            APOSF = APOSE + N2
            DO 270 IPIV = N2,1,-1
              IRHS1 = -IW(J1)
              IRHS2 = -IW(J1+N2)
              W1 = RHS(IRHS1)*A(APOSE) + RHS(IRHS2)*A(APOSD)
              RHS(IRHS2) = RHS(IRHS1)*A(APOSD) + RHS(IRHS2)*A(APOSF)
              RHS(IRHS1) = W1
              APOSD = APOSD + N2 + 1
              APOSE = APOSE + 1
              APOSF = APOSF + 1
              J1 = J1 + 1
  270       CONTINUE
            K = APOSF
            J1 = IWPOS + NROWS
            DO 295 IPIV = 1,N2
              IRHS1 = ABS(IW(IWPOS+IPIV-1))
              W1 = RHS(IRHS1)
              DO 290 J = J1 + NUM1,J2
                IRHS = ABS(IW(J))
                W1 = W1 + RHS(IRHS)*A(K)
                K = K + 1
  290         CONTINUE
              RHS(IRHS1) = W1
  295       CONTINUE
            DO 310 IPIV = 1,N2
              IRHS2 = ABS(IW(IWPOS+IPIV-1+N2))
              W1 = RHS(IRHS2)
              DO 300 J = J1,J2 - NUM3
                IRHS = ABS(IW(J))
                W1 = W1 + RHS(IRHS)*A(K)
                K = K + 1
  300         CONTINUE
              RHS(IRHS2) = W1
  310       CONTINUE
            J1 = IWPOS + NROWS - 1
            J2 = J1
            DO 330 IPIV = N2 - 1,1,-1
              IRHS1 = -IW(J1-1)
              K = APOS + (IPIV-1)* (N2+1)
              DO 320 J = J1,J2
                K = K + 1
                RHS(IRHS1) = RHS(IRHS1) - RHS(ABS(IW(J)))*A(K)
  320         CONTINUE
              J1 = J1 - 1
  330       CONTINUE
            IF (KIND.EQ.2) THEN
              K = APOSM
              DO 350 IPIV = 1,N2 - 1
                J1 = J1 + 1
                IRHS2 = -IW(J1-1-N2)
                DO 340 J = J1,J2
                  RHS(IRHS2) = RHS(IRHS2) + RHS(ABS(IW(J)))*A(K)
                  K = K + 1
  340           CONTINUE
  350         CONTINUE
            END IF
            J1 = J2 - N2
            J2 = J1
            DO 370 IPIV = N2 - 1,1,-1
              IRHS2 = -IW(J1-1)
              K = APOS + (IPIV-1)* (N2+1)
              DO 360 J = J1,J2
                K = K + N2
                RHS(IRHS2) = RHS(IRHS2) - RHS(ABS(IW(J)))*A(K)
  360         CONTINUE
              J1 = J1 - 1
  370       CONTINUE
          END IF
        END IF
  380 CONTINUE
      END
      SUBROUTINE MA47SD(A,BOTTOM,TOP,MOVE,NCMP)
      DOUBLE PRECISION A(*)
      INTEGER BOTTOM,TOP,NCMP
      INTEGER IDUMMY,JJ,K,MOVE,SIZE
      NCMP = NCMP + 1
      K = TOP - MOVE - 1
      DO 20 IDUMMY = 1,TOP
        IF (K.LE.BOTTOM) GO TO 30
        SIZE = A(K)
        IF (SIZE.LT.0) THEN
          MOVE = MOVE - SIZE
          K = K + SIZE
        ELSE
          IF (MOVE.GT.0) THEN
            DO 10 JJ = K,K - SIZE + 1,-1
              A(JJ+MOVE) = A(JJ)
   10       CONTINUE
          END IF
          K = K - SIZE
        END IF
   20 CONTINUE
   30 BOTTOM = BOTTOM + MOVE
      MOVE = 0
      RETURN
      END
      SUBROUTINE MA47TD(N,IPE,IW,LW,NV,NEXT,LAST,LEAF,FLAG,VAR,SVAR)
      INTEGER N,IPE(N),LW,IW(LW),NV(N),NEXT(N),LAST(N),LEAF(N),FLAG(N),
     +        VAR(N),SVAR(N)
      INTRINSIC SIGN
      INTEGER FREE,I,IS,J,JS,K,KK,LS,NS
      DO 10 I = 1,N
        SVAR(I) = 1
        LAST(I) = I - 1
        NEXT(I) = I + 1
        FLAG(I) = 0
        VAR(I) = I + 1
   10 CONTINUE
      LAST(1) = N
      NEXT(N) = 1
      FREE = 2
      DO 30 J = 1,N
        KK = IPE(J)
        DO 20 K = KK + 1,KK + IW(KK)
          I = IW(K)
          IS = SVAR(I)
          IF (FLAG(IS).NE.J) THEN
            FLAG(IS) = J
            IF (NEXT(I).EQ.I) THEN
            ELSE
              JS = FREE
              FREE = VAR(JS)
              VAR(IS) = I
              SVAR(I) = JS
              NS = NEXT(I)
              LS = LAST(I)
              NEXT(LS) = NS
              LAST(NS) = LS
              NEXT(I) = I
              LAST(I) = I
            END IF
          ELSE
            IF (NEXT(I).EQ.I) THEN
              NS = VAR(IS)
              VAR(IS) = FREE
              FREE = IS
            ELSE
              NS = NEXT(I)
              LS = LAST(I)
              NEXT(LS) = NS
              LAST(NS) = LS
              NS = VAR(IS)
            END IF
            LS = LAST(NS)
            NEXT(LS) = I
            NEXT(I) = NS
            LAST(NS) = I
            LAST(I) = LS
            SVAR(I) = SVAR(NS)
          END IF
   20   CONTINUE
   30 CONTINUE
      DO 60 IS = 1,N
        LEAF(IS) = IS
        IF (LAST(IS).EQ.IS) GO TO 60
        IF (LAST(IS).LT.0) GO TO 60
        LS = LAST(IS)
        LEAF(IS) = LS
        DO 40 K = 1,N
          I = LS
          IF (I.EQ.IS) GO TO 50
          IPE(I) = 0
          LS = LAST(I)
          NEXT(I) = -LS
          LAST(I) = -LS
          NV(I) = 0
   40   CONTINUE
   50   NV(IS) = SIGN(K,NV(IS))
   60 CONTINUE
      END
      SUBROUTINE MA47UD(A,LA,IW,LIW,ICNTL)
      INTEGER LA
      DOUBLE PRECISION A(LA)
      INTEGER LIW,IW(LIW),ICNTL(7)
      INTRINSIC ABS,INT,MIN,SIGN
      CHARACTER*72 LINE
      INTEGER APOS,APOSM,APOS2,IBLK,ILINE,IROW,IWPOS,J,JPIV,J1,J2,K,
     +        KIND,LDIAG,LEN,MP,NBLK,NCOLS,NROWS,NUM1,NUM3,N2
      CHARACTER*1 NZ,SGN,PM(-2:2)
      DOUBLE PRECISION AK
      DATA PM/'*','-','.','+','.'/
      SGN(K) = PM(SIGN(1,K))
      NZ(AK) = PM(INT(SIGN(2D0,-ABS(AK))))
      MP = ICNTL(2)
      LDIAG = ICNTL(3)
      APOS2 = IW(1)
      NBLK = IW(3)
      IF (LDIAG.LE.2) NBLK = 0
      IF (LDIAG.EQ.3) NBLK = MIN(1,NBLK)
      LEN = 12
      IF (LDIAG.EQ.5) LEN = 1
      IF (LEN.EQ.12) THEN
        IF (NBLK.EQ.IW(3)) THEN
          WRITE (MP,'(A)')
     +      ' For each block, the following information is provided:'
        ELSE
          WRITE (MP,'(A,A)') ' For the first block only,',
     +      ' the following information is provided:'
        END IF
      END IF
      IF (LEN.EQ.12) WRITE (MP,'(A)')
     +    '   1. Block number, number of rows, number of columns',
     +    '   2. Whether the block is full, tile, or oxo',
     +    '   3. List of indices for the pivot, each negated if part of'
     +    ,'      a 2x2 pivot',
     +    '   4. The factorized block pivot (see below)',
     +    '   5. List of indices for the non-pivot columns',
     +    '   6. The non-pivot part, with structural zeros printed as',
     +    '      0000000000',' A factorized full pivot has the form',
     +    '               T','        L  D  L ',
     +    '                          T',
     +    ' and is printed as D and L  packed together.'
      IF (LEN.EQ.12) WRITE (MP,'(A)')
     +    ' A factorized tile or oxo pivot has the form',
     +    '                              T',
     +    '       ( L     ) ( F   D ) ( L   M )',
     +    '       (  T  T ) (       ) (       )',
     +    '       ( M  U  ) ( D   E ) (     U )',
     +    ' where L is unit lower triangular; D, E, and F are diagonal;'
     +    ,
     +   ' M is zero for an oxo pivot and is upper triangular with zero'
     +    ,' diagonal for a tile pivot; and U is upper triangular with',
     +    ' unit diagonal. It is printed as',
     +    '    a. L,D,U  packed together,',
     +    '    b. M (tile case only) in packed form,','    c. F and E.'
      IWPOS = 4
      APOS = 1
      DO 300 IBLK = 1,NBLK
        NCOLS = IW(IWPOS)
        NROWS = IW(IWPOS+1)
        IWPOS = IWPOS + 2
        IF (NCOLS.GT.0) THEN
          KIND = 1
        ELSE
          NCOLS = -NCOLS
          NUM3 = IW(IWPOS)
          IWPOS = IWPOS + 1
          IF (NROWS.GT.0) THEN
            KIND = 2
          ELSE
            NROWS = -NROWS
            NUM1 = IW(IWPOS)
            IWPOS = IWPOS + 1
            KIND = 3
          END IF
        END IF
        WRITE (MP,'(4(A,I6))') ' Block',IBLK,' with',NROWS,' rows and',
     +    NCOLS,' columns'
        IF (KIND.EQ.1) WRITE (MP,'(A)') ' Full pivot'
        IF (KIND.EQ.2) WRITE (MP,'(A)') ' Tile pivot'
        IF (KIND.EQ.3) WRITE (MP,'(A)') ' Oxo pivot'
        IF (LEN.EQ.12) WRITE (MP,'(6I12)') (IW(K),K=IWPOS,IWPOS+NROWS-1)
        IF (LEN.EQ.1) WRITE (MP,'(72A1)') (SGN(IW(K)),K=IWPOS,
     +      IWPOS+NROWS-1)
        IF (KIND.EQ.1) THEN
          JPIV = 0
          DO 30 IROW = 1,NROWS
            IF (JPIV.EQ.1) THEN
              JPIV = 0
            ELSE
              IF (IW(IWPOS+IROW-1).LT.0) JPIV = 1
            END IF
            ILINE = 1
            DO 10 J = 1,IROW - 1
              WRITE (LINE(ILINE:ILINE+LEN-1),'(A)') ' '
              ILINE = ILINE + LEN
              IF (ILINE.GT.72) THEN
                WRITE (MP,'(A)') LINE
                ILINE = 1
              END IF
   10       CONTINUE
            DO 20 J = IROW,NROWS
              IF (LEN.EQ.12) WRITE (LINE(ILINE:ILINE+11),
     +            '(1P,D12.4)') A(APOS)
              IF (LEN.EQ.1) WRITE (LINE(ILINE:ILINE),'(A)') NZ(A(APOS))
              APOS = APOS + 1
              IF (J.EQ.IROW+1) THEN
                IF (JPIV.EQ.1) THEN
                  IF (LEN.EQ.12) WRITE (LINE(ILINE:ILINE+11),
     +                '(1P,D12.4)') A(APOS2)
                  IF (LEN.EQ.1) WRITE (LINE(ILINE:ILINE),
     +                '(A)') NZ(A(APOS2))
                  APOS2 = APOS2 + 1
                END IF
              END IF
              ILINE = ILINE + LEN
              IF (ILINE.GT.72) THEN
                WRITE (MP,'(A)') LINE
                ILINE = 1
              END IF
   20       CONTINUE
            IF (ILINE.GT.1) THEN
              LINE(ILINE:) = ' '
              WRITE (MP,'(A)') LINE
            END IF
   30     CONTINUE
        ELSE
          N2 = NROWS/2
          DO 50 IROW = 1,N2
            ILINE = 1
            DO 40 J = 1,N2
              IF (LEN.EQ.12) WRITE (LINE(ILINE:ILINE+11),
     +            '(1P,D12.4)') A(APOS)
              IF (LEN.EQ.1) WRITE (LINE(ILINE:ILINE),'(A)') NZ(A(APOS))
              APOS = APOS + 1
              ILINE = ILINE + LEN
              IF (ILINE.GT.72) THEN
                WRITE (MP,'(A)') LINE
                ILINE = 1
              END IF
   40       CONTINUE
            IF (ILINE.GT.1) THEN
              LINE(ILINE:) = ' '
              WRITE (MP,'(A)') LINE
            END IF
   50     CONTINUE
          IF (KIND.EQ.2) THEN
            DO 80 IROW = 2,N2
              ILINE = 1
              APOSM = APOS + IROW - 2
              DO 60 J = 1,IROW - 1
                IF (LEN.EQ.12) WRITE (LINE(ILINE:ILINE+11),
     +              '(1P,D12.4)') A(APOSM)
                IF (LEN.EQ.1) WRITE (LINE(ILINE:ILINE),
     +              '(A)') NZ(A(APOSM))
                APOSM = APOSM + N2 - J - 1
                ILINE = ILINE + LEN
                IF (ILINE.GT.72) THEN
                  WRITE (MP,'(A)') LINE
                  ILINE = 1
                END IF
   60         CONTINUE
              DO 70 J = IROW + 1,N2
                WRITE (LINE(ILINE:ILINE+LEN-1),'(A)') ' '
                ILINE = ILINE + LEN
                IF (ILINE.GT.72) THEN
                  WRITE (MP,'(A)') LINE
                  ILINE = 1
                END IF
   70         CONTINUE
              IF (ILINE.GT.1) THEN
                LINE(ILINE:) = ' '
                WRITE (MP,'(A)') LINE
              END IF
   80       CONTINUE
            APOS = APOS + (N2* (N2-1))/2
          END IF
          ILINE = 1
          APOS = APOS + N2
          DO 90 J = 1,NROWS
            IF (LEN.EQ.12) WRITE (LINE(ILINE:ILINE+11),
     +          '(1P,D12.4)') A(APOS)
            IF (LEN.EQ.1) WRITE (LINE(ILINE:ILINE),'(A)') NZ(A(APOS))
            APOS = APOS + 1
            ILINE = ILINE + LEN
            IF (ILINE.GT.72) THEN
              WRITE (MP,'(A)') LINE
              ILINE = 1
            END IF
            IF (J.EQ.N2) APOS = APOS - NROWS
   90     CONTINUE
          APOS = APOS + N2
          IF (ILINE.GT.1) THEN
            LINE(ILINE:) = ' '
            WRITE (MP,'(A)') LINE
          END IF
        END IF
        IWPOS = IWPOS + NROWS
        IF (LEN.EQ.12) WRITE (MP,'(6I12)') (IW(K),K=IWPOS,
     +      IWPOS+NCOLS-NROWS-1)
        IF (LEN.EQ.1) WRITE (MP,'(72A1)') (SGN(IW(K)),K=IWPOS,
     +      IWPOS+NCOLS-NROWS-1)
        IWPOS = IWPOS + NCOLS - NROWS
        DO 280 IROW = 1,NROWS
          J1 = NROWS
          J2 = NCOLS
          IF (KIND.GT.1) THEN
            IF (IROW.GT.N2) THEN
              J2 = NCOLS - NUM3
            ELSE IF (KIND.EQ.3) THEN
              J1 = NROWS + NUM1
            END IF
          END IF
          ILINE = 1
          DO 100 J = NROWS + 1,J1
            IF (LEN.EQ.12) WRITE (LINE(ILINE:ILINE+11),
     +          '(A)') '  0000000000'
            IF (LEN.EQ.1) WRITE (LINE(ILINE:ILINE),'(A)') '0'
            ILINE = ILINE + LEN
            IF (ILINE.GT.72) THEN
              WRITE (MP,'(A)') LINE
              ILINE = 1
            END IF
  100     CONTINUE
          DO 110 J = J1 + 1,J2
            IF (LEN.EQ.12) WRITE (LINE(ILINE:ILINE+11),
     +          '(1P,D12.4)') A(APOS)
            IF (LEN.EQ.1) WRITE (LINE(ILINE:ILINE),'(A)') NZ(A(APOS))
            APOS = APOS + 1
            ILINE = ILINE + LEN
            IF (ILINE.GT.72) THEN
              WRITE (MP,'(A)') LINE
              ILINE = 1
            END IF
  110     CONTINUE
          DO 120 J = J2 + 1,NCOLS
            IF (LEN.EQ.12) WRITE (LINE(ILINE:ILINE+11),
     +          '(A)') '  0000000000'
            IF (LEN.EQ.1) WRITE (LINE(ILINE:ILINE),'(A)') '0'
            ILINE = ILINE + LEN
            IF (ILINE.GT.72) THEN
              WRITE (MP,'(A)') LINE
              ILINE = 1
            END IF
  120     CONTINUE
          IF (ILINE.GT.1) THEN
            LINE(ILINE:) = ' '
            WRITE (MP,'(A)') LINE
          END IF
  280   CONTINUE
  300 CONTINUE
      END
      SUBROUTINE MA47VD(THRESH,NEWTHR,N,IPE,IW,LW,COUNT,NV,NEXT,LAST,
     +                  IPR,FLAG,NFLG)
      INTEGER THRESH,NEWTHR,N,IPE(N),LW,IW(LW),COUNT(N),NV(N),NEXT(N),
     +        LAST(N),IPR(N),FLAG(N),NFLG
      EXTERNAL MA47ZD
      INTEGER IR,IS,JP,JP1,JP2,K,KE,KP,KP2,LS,MS,NS,PART
      DO 50 MS = 1,N
        IF (FLAG(MS).LT.0) GO TO 50
        IF (COUNT(MS).LE.THRESH) GO TO 50
        IF (COUNT(MS).GT.NEWTHR) GO TO 50
        IR = 0
        IF (NFLG.LE.4) CALL MA47ZD(N,FLAG,NFLG)
        NFLG = NFLG - 1
        K = IPE(MS)
        KP2 = K + IW(K)
        DO 30 KP = K + 1,KP2
          PART = (IW(KP)-1)/N
          KE = IW(KP) - PART*N
          IF (FLAG(KE).EQ.-1) GO TO 30
          IF (FLAG(KE).LE.-2) THEN
            JP = IPE(KE)
            JP1 = JP + 3
            JP2 = JP + IW(JP)
            IF (PART.EQ.0) THEN
            ELSE IF (PART.EQ.2) THEN
              JP1 = JP1 + IW(JP+1)
            ELSE
              JP2 = JP2 - IW(JP+2)
            END IF
          ELSE
            JP1 = KP
            JP2 = KP2
          END IF
          DO 20 JP = JP1,JP2
            IS = IW(JP)
            IF (FLAG(IS).GT.NFLG) THEN
              FLAG(IS) = NFLG
              IR = IR + ABS(NV(IS))
            END IF
   20     CONTINUE
          IF (JP2.EQ.KP2 .OR. IR.GT.NEWTHR) GO TO 40
   30   CONTINUE
   40   IF (IR.NE.COUNT(MS)) THEN
          NS = NEXT(MS)
          LS = LAST(MS)
          NEXT(MS) = 0
          IF (NS.GT.0) LAST(NS) = LS
          IF (LS.GT.0) THEN
            NEXT(LS) = NS
          ELSE
            IPR(COUNT(MS)) = NS
          END IF
          NS = IPR(IR)
          IF (NS.GT.0) LAST(NS) = MS
          NEXT(MS) = NS
          IPR(IR) = MS
          LAST(MS) = 0
          COUNT(MS) = IR
        END IF
   50 CONTINUE
      THRESH = NEWTHR
      END
      SUBROUTINE MA47WD(A,LA,IW,LIW,NRLBDU)
      INTEGER LA,LIW
      DOUBLE PRECISION A(LA)
      INTEGER IW(LIW)
      INTEGER NRLBDU
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0.0D0)
      INTEGER APOS,IBLK,IROW,IWPOS,J,JPIV,NCOLS,NROWS,NUM1,NUM3,N2
      APOS = 1
      IWPOS = 6
      DO 40 IBLK = 1,IW(3)
        NCOLS = IW(IWPOS-2)
        NROWS = IW(IWPOS-1)
        N2 = ABS(NROWS)/2
        IF (NCOLS.GT.0) THEN
          JPIV = 1
          DO 30 IROW = 1,NROWS
            JPIV = JPIV - 1
            IF (JPIV.EQ.1) GO TO 10
            IF (IW(IWPOS+IROW-1).LT.0) THEN
              JPIV = 2
              NRLBDU = NRLBDU + 1
              A(NRLBDU) = A(APOS+1)
              A(APOS+1) = ZERO
            END IF
   10       DO 20 J = APOS + 1,APOS + NROWS - IROW
              A(J) = -A(J)
   20       CONTINUE
            APOS = APOS + NROWS - IROW + 1
   30     CONTINUE
          APOS = APOS + NROWS* (NCOLS-NROWS)
        ELSE
          NCOLS = -NCOLS
          NUM3 = IW(IWPOS)
          APOS = APOS - NUM3*N2 - N2* (N2+1)/2 + N2
          IWPOS = IWPOS + 1
          IF (NROWS.LT.0) THEN
            NROWS = -NROWS
            NUM1 = IW(IWPOS)
            APOS = APOS - NUM1*N2 - N2* (N2+1)/2 + N2
            IWPOS = IWPOS + 1
          END IF
          APOS = APOS + NROWS* (NCOLS+NCOLS-NROWS+1)/2
        END IF
        IWPOS = IWPOS + ABS(NCOLS) + 2
   40 CONTINUE
      END
      SUBROUTINE MA47XD(A,LDA,M,N)
      INTEGER LDA,M,N
      DOUBLE PRECISION A(LDA,*)
      INTEGER I,J
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0.0D0)
      DO 20 J = 1,M
        DO 10 I = J,N
          A(I,J) = ZERO
   10   CONTINUE
   20 CONTINUE
      RETURN
      END
      SUBROUTINE MA47YD(A,LDA,M,N,B,LDB,TRIANG,TRANS)
      INTEGER LDA,M,N,LDB
      LOGICAL TRIANG,TRANS
      DOUBLE PRECISION A(LDA,*),B(LDB,*)
      INTEGER I,J
      IF (.NOT.TRIANG) THEN
        IF (.NOT.TRANS) THEN
          DO 20 J = 1,M
            DO 10 I = 1,N
              B(I,J) = A(I,J)
   10       CONTINUE
   20     CONTINUE
        ELSE
          DO 40 J = 1,M
            DO 30 I = 1,N
              B(J,I) = A(I,J)
   30       CONTINUE
   40     CONTINUE
        END IF
      ELSE
        DO 60 J = 1,M
          DO 50 I = J,N
            B(I,J) = A(I,J)
   50     CONTINUE
   60   CONTINUE
      END IF
      RETURN
      END
      SUBROUTINE MA47ZD(N,FLAG,NFLG)
      INTEGER N,FLAG(N),NFLG
      INTEGER I,N3
      N3 = N*3
      DO 80 I = 1,N
        IF (FLAG(I).GT.2) FLAG(I) = N3
        IF (FLAG(I).LE.-2) FLAG(I) = -N3
   80 CONTINUE
      NFLG = N3
      END


