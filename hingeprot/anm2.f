C23456789012345678901234567890123456789012345678901234567890123456789012
      program rbp

      character*3 resnam
      character*3 restyp(5000)
      character*9 fil(2)
      character*4 attyp(5000)
      character*4 atnam
      character*6 dummy
      character*2 chtyp(5000),chnam
      character*1 resex(5000)

      integer resno1(2),reduction
      integer atnum,resnum,resold,res1,resnum1

      real x(5000,1),y(5000,1),z(5000,1),rcanm
      real xx(5000),yy(5000),zz(5000)
      real xxnew(5000),yynew(5000),zznew(5000)
      real r(5000,5000),dn(15000,15000)

      open (38,file ='anmcutoff')
      read(38,*)rcanm

      	fil(1)='alpha.cor'
	

C********************************************************************
C       parameter
      iiii=1
C*********************************************************************

	do 40893 j=1,5000
C		5000is the largest # of amino acids in the pdb files
		xx(j)=0.
		yy(j)=0.
40893		zz(j)=0.

      open(10,file=fil(iiii))

      do 3 i=1,5000
      do 3 j=1,1
         x(i,j)=0.
         y(i,j)=0.
 3       z(i,j)=0.

      resold=0


 50   Format(A6)

 65   FORMAT (A6,1X,I4,1X,A4,1X,A3,A2,I4,A1,3X,3F8.3)


 5    read(10,50)dummy

      if(dummy.ne.'ATOM  ') goto 5
      backspace (10)
      i=1

 1231   read(10,50,end=33)dummy
        if(dummy.ne.'ATOM  ') go to 33
 	backspace (10)

       read(10,65)dummy,atnum,atnam,resnam,chnam,resnum,resex(atnum),
     :x(atnum,i),y(atnum,i),z(atnum,i)

        resnum1=atnum

C*****  write(6,*)'modu gectim'


C*****  write(6,*)'nucleotidi gectim'
	
	if(atnum.eq.1) then
C	write(6,*)'atnum eq 1'
	resno1(iiii)=resnum
	endif
 
      if(atnam.eq.' CA')then

C*****  write(6,*)'bunu da gectim'
	resnum=resold+1

      restyp(atnum)=resnam
      attyp(atnum)=atnam
      chtyp(atnum)=chnam
      resold=atnum
      endif

       goto 1231
33    close(10)
C      natoms(resnum)=i-1

      atnum=resold

C Debye factor from PDB
C_________
C*****	write(6,*) resnum

C____________________________________________

	do  i=1,atnum
	xx(i)=x(i,1)
	yy(i)=y(i,1)
	zz(i)=z(i,1)
        end do

C	Calculation of the position vectors with respect to the centroid

	centx=0.
	centy=0.
	centz=0.

	do  i=1,atnum
	centx=centx+xx(i)
	centy=centy+yy(i)
	centz=centz+zz(i)
	end do
	centx=centx/resnum
	centy=centy/resnum
	centz=centz/resnum
	do i=1,atnum
	xxnew(i)=-centx+xx(i)
	yynew(i)=-centy+yy(i)
	zznew(i)=-centz+zz(i)
	end do




C	BEGINNING OF ANM CALCULATIONS

	res1=3*atnum

C---------------------------------------------------------------------------------------
C---------------------------------------------------------------------------------------




C	GENERATION OF THE HESSIAN (DIVIDED BY GAMMA)


        do j=1,atnum
        do k=1,atnum
	ga=1.
	bx=xxnew(j)-xxnew(k)
	by=yynew(j)-yynew(k)
	bz=zznew(j)-zznew(k)
	r(j,k)=sqrt(bx*bx+by*by+bz*bz)
        
C        if((natoms(j).eq.0).or.(natoms(k).eq.0))then
C         r(j,k)=2.*rcanm
C        endif

	if(j.ne.k.and.r(j,k).le.rcanm)then
C	if(r(j,k).le.5.2) ga=5.0
C	if(r(j,k).le.4.2) ga=10.0

C diagonals (for j)
C	write(*,*)3*j-2,3*j-2,r(j,k),j,k
	dn(3*j-2,3*j-2)=dn(3*j-2,3*j-2)+ga*bx*bx/r(j,k)**2.
	dn(3*j-1,3*j-1)=dn(3*j-1,3*j-1)+ga*by*by/r(j,k)**2.
	dn(3*j,3*j)=dn(3*j,3*j)+ga*bz*bz/r(j,k)**2.
C off-diagonals of diagonal superelements (for j)
	dn(3*j-2,3*j-1)=dn(3*j-2,3*j-1)+ga*bx*by/r(j,k)**2.
	dn(3*j-2,3*j)=dn(3*j-2,3*j)+ga*bx*bz/r(j,k)**2.
	dn(3*j-1,3*j-2)=dn(3*j-1,3*j-2)+ga*by*bx/r(j,k)**2.
	dn(3*j-1,3*j)=dn(3*j-1,3*j)+ga*by*bz/r(j,k)**2.
	dn(3*j,3*j-2)=dn(3*j,3*j-2)+ga*bx*bz/r(j,k)**2.
	dn(3*j,3*j-1)=dn(3*j,3*j-1)+ga*by*bz/r(j,k)**2.
	endif
        enddo
        enddo

C	pause

C	write(*,*)'off'
        do j=1,atnum
        do k=1,atnum
	ga=1.0
	bx=xxnew(j)-xxnew(k)
	by=yynew(j)-yynew(k)
	bz=zznew(j)-zznew(k)
	r(j,k)=sqrt(bx*bx+by*by+bz*bz)

C        if((natoms(j).eq.0).or.(natoms(k).eq.0))then
C         r(j,k)=2.*rcanm
C        endif

	if(j.ne.k.and.r(j,k).le.rcanm) then
C	if(r(j,k).le.5.2) ga=5.0
C	if(r(j,k).le.4.2) ga=10.0
C diagonals (for j&k)
C	write(*,*)j,k,3*j-2,3*k-2,r(j,k)
	dn(3*j-2,3*k-2)=-ga*bx*bx/r(j,k)**2.
	dn(3*j-1,3*k-1)=-ga*by*by/r(j,k)**2.
	dn(3*j,3*k)=-ga*bz*bz/r(j,k)**2.

C off-diagonals (for j&k)
	dn(3*j-2,3*k-1)=-ga*bx*by/r(j,k)**2.
	dn(3*j-2,3*k)=-ga*bx*bz/r(j,k)**2.
	dn(3*j-1,3*k-2)=-ga*by*bx/r(j,k)**2.
	dn(3*j-1,3*k)=-ga*by*bz/r(j,k)**2.
	dn(3*j,3*k-2)=-ga*bx*bz/r(j,k)**2.
	dn(3*j,3*k-1)=-ga*by*bz/r(j,k)**2.
	endif
       enddo
       enddo



C---------------------------------------------------------------------------------------
C---------------------------------------------------------------------------------------

C	INVERSION OF THE HESSIAN AND CORRELATIONS

	jres=res1

      ijk=0
      do ij=1,jres
       do ik=ij,jres
        if (dn(ij,ik).ne.0) ijk=ijk+1
       enddo
      enddo
60    FORMAT(i6,2x,i6,1x,f20.10)
      open (unit=44,file='upperhessian')
      write(44,*) ijk

      do ij=1,jres
       do ik=ij,jres
        if (dn(ij,ik).ne.0) write(44,*) ij,ik,dn(ij,ik)
       end do
      end do
      close (44)
      end

