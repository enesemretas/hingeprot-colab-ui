C23456789012345678901234567890123456789012345678901234567890123456789012
      program rbp

      character*3 restyp(5000)
      character*9 fil
      character*4 attyp(5000)
      character*7 dummy
      character*1 resex(5000)

      integer im,n,mj,fileindex
      integer ind(5000)
      integer resnum,k
      real xx(5000),yy(5000),zz(5000),dumn,mag1,mag2,chnam(5000)
      real xxnew(5000),yynew(5000),zznew(5000)
        dimension w1(5000),v1(5000,5000)

        fil='alpha.cor'


	open (90,file='newcoordinat.mds')
        open(10,file=fil)


	


	do 40893 j=1,5000
C		5000is the largest # of amino acids in the pdb files
		xx(j)=0.
		yy(j)=0.
40893		zz(j)=0.


 65   FORMAT (A6,1X,I4,1X,A4,1X,A3,1x,A1,I4,A1,3X,3F8.3)


C       write(6,*)resold,resnum1,resnum,mod(resnum,reduction)
      
990   read(10,65,end=991)dummy,k,attyp(k),restyp(k),chnam(k),ind(k),
     :resex(k),xx(k),yy(k),zz(k)
      goto 990

C____________________________________________

991   resnum=k

C	Calculation of the position vectors with respect to the centroid

	centx=0.
	centy=0.
	centz=0.

	do  i=1,resnum
	centx=centx+xx(i)
	centy=centy+yy(i)
	centz=centz+zz(i)
	end do
	centx=centx/resnum
	centy=centy/resnum
	centz=centz/resnum
	do i=1,resnum
	xxnew(i)=-centx+xx(i)
	yynew(i)=-centy+yy(i)
	zznew(i)=-centz+zz(i)
	end do



C**********************************************
       open (unit=44)

50     format (A7)
51     format (38x,I5)
       do i=1,12
       read(44,50) dummy
       enddo
       read(44,51) nmax
       jres=nmax/3
1001   read (44,50) dummy
       if (dummy.eq.'vector ') then
       read (44,*) dummy
1002   read (44,*) nn,w1(nn),dumn
       if (nn.lt.36) goto 1002
       else
       goto 1001
       endif
       k=0
1005   k=k+1
1003   read (44,50) dummy
       if (dummy.eq.'vector ') then
       read (44,*) dummy
       n=-1
1004   n=n+1
          if (n*6+3.eq.nmax) then
           read (44,*) (v1(n*6+j,k),j=1,3)
          else
           read (44,*) (v1(n*6+j,k),j=1,6)
          endif
	       if (n*6+6.lt.nmax) goto 1004
       else
       goto 1003
       endif
       if (k.lt.36) goto 1005
       close(44)

C---------------------------------------------------------------------------------------

       open (45,file='eigenanm') 
	do i=1,36
	write(45,'(I4,2X,F8.4)')i,w1(i)
	enddo

C-----------------------------------------------------------------------------


C	contribution of mode k1
	k1=7
	k2=k1+1
	k3=k1+2
	k4=k1+3
	k5=k1+4
	k6=k1+5
	k7=k1+6
	k8=k1+7
	k9=k1+8
	k10=k1+9
	

        open (unit=64,file='1coor')
        open (unit=65,file='2coor')
        open (unit=66,file='3coor')
        open (unit=67,file='4coor')
        open (unit=68,file='5coor')
        open (unit=69,file='6coor')
        open (unit=70,file='7coor')
        open (unit=71,file='8coor')
        open (unit=72,file='9coor')
        open (unit=73,file='10coor')
C	SUEZ ADVA ICIN EKLENDI 28.12.2009
	open (unit=1111,file='11coor')
	open (unit=1112,file='12coor')
	open (unit=1113,file='13coor')
	open (unit=1114,file='14coor')
	open (unit=1115,file='15coor')
	open (unit=1116,file='16coor')
	open (unit=1117,file='17coor')
	open (unit=1118,file='18coor')
	open (unit=1119,file='19coor')
	open (unit=1120,file='20coor')
	open (unit=1121,file='21coor')
	open (unit=1122,file='22coor')
	open (unit=1123,file='23coor')
	open (unit=1124,file='24coor')
	open (unit=1125,file='25coor')
	open (unit=1126,file='26coor')
	open (unit=1127,file='27coor')
	open (unit=1128,file='28coor')
	open (unit=1129,file='29coor')
	open (unit=1130,file='30coor')
	open (unit=1131,file='31coor')
	open (unit=1132,file='32coor')
	open (unit=1133,file='33coor')
	open (unit=1134,file='34coor')
	open (unit=1135,file='35coor')
	open (unit=1136,file='36coor')
C	SUEZ ADVA
        open (unit=74,file='1cross')
        open (unit=75,file='2cross')
        open (unit=76,file='3cross')
        open (unit=77,file='4cross')
        open (unit=78,file='5cross')
        open (unit=79,file='6cross')
        open (unit=80,file='7cross')
        open (unit=81,file='8cross')
        open (unit=82,file='9cross')
        open (unit=83,file='10cross')
9798    format (I4,4(3x,F8.5))
9799    format (I4,I4,3x,F8.5)

        do j=1,jres
C	x component's index is 3j-2, y's is 3j-1, z's is 3j
	ix=3*j-2
	iy=3*j-1
	iz=3*j

        write (64,9798) j,v1(ix,k1),v1(iy,k1),v1(iz,k1)
     :,(v1(ix,k1)**2+v1(iy,k1)**2+v1(iz,k1)**2)**0.5
        write (65,9798) j,v1(ix,k2),v1(iy,k2),v1(iz,k2)
     :,(v1(ix,k2)**2+v1(iy,k2)**2+v1(iz,k2)**2)**0.5
        write (66,9798) j,v1(ix,k3),v1(iy,k3),v1(iz,k3)
     :,(v1(ix,k3)**2+v1(iy,k3)**2+v1(iz,k3)**2)**0.5
        write (67,9798) j,v1(ix,k4),v1(iy,k4),v1(iz,k4)
     :,(v1(ix,k4)**2+v1(iy,k4)**2+v1(iz,k4)**2)**0.5
        write (68,9798) j,v1(ix,k5),v1(iy,k5),v1(iz,k5)
     :,(v1(ix,k5)**2+v1(iy,k5)**2+v1(iz,k5)**2)**0.5
        write (69,9798) j,v1(ix,k6),v1(iy,k6),v1(iz,k6)
     :,(v1(ix,k6)**2+v1(iy,k6)**2+v1(iz,k6)**2)**0.5
        write (70,9798) j,v1(ix,k7),v1(iy,k7),v1(iz,k7)
     :,(v1(ix,k7)**2+v1(iy,k7)**2+v1(iz,k7)**2)**0.5
        write (71,9798) j,v1(ix,k8),v1(iy,k8),v1(iz,k8)
     :,(v1(ix,k8)**2+v1(iy,k8)**2+v1(iz,k8)**2)**0.5
        write (72,9798) j,v1(ix,k9),v1(iy,k9),v1(iz,k9)
     :,(v1(ix,k9)**2+v1(iy,k9)**2+v1(iz,k9)**2)**0.5
        write (73,9798) j,v1(ix,k10),v1(iy,k10),v1(iz,k10)
     :,(v1(ix,k10)**2+v1(iy,k10)**2+v1(iz,k10)**2)**0.5
     	do ss_unit=1111,1136
     	ss = k1 + ss_unit - 1101
        write (ss_unit,9798) j,v1(ix,ss),v1(iy,ss),v1(iz,ss)
     :,(v1(ix,ss)**2+v1(iy,ss)**2+v1(iz,ss)**2)**0.5     	
     	enddo
	enddo
	
      do fileindex=74,83
      k=fileindex-67
        do i=1,jres
          do j=i,jres
        mag1 = ((v1(3*i-2,k))**2+(v1(3*i-1,k))**2+(v1(3*i,k))**2)**0.5
        mag2 = ((v1(3*j-2,k))**2+(v1(3*j-1,k))**2+(v1(3*j,k))**2)**0.5
        dumn = (v1(3*i-2,k)*v1(3*j-2,k)+v1(3*i-1,k)*v1(3*j-1,k)
     :+v1(3*i,k)*v1(3*j,k))/mag1/mag2

        write (fileindex,9799) i,j,dumn

        enddo
        enddo
        enddo
	
	close (64)
	close (65)
	close (66)
	close (67)
	close (68)
	close (69)
	close (70)
	close (71)
	close (72)
        close (73)
        close (74)
        close (75)
        close (76)
        close (77)
        close (78)
        close (79)
        close (80)
        close (81)
        close (82)
        close (83)

C******** read residue no's
      open(10,file=fil)
97    format (7x,I4,11x,I4)
3441  read (10,97,end=3442) im,ind(im)
      goto 3441
3442  close (10)
C**************************
        mj=0
	do  j=1,jres
C	ind(j)=j
C         if (j.ne.1) then
C         if (chnam(j-1).ne.chnam(j)) mj=mj+1000
C         endif
      write(90,31)'ATOM',j,attyp(j),restyp(j),chnam(j)
     :,ind(j),resex(j),xxnew(j),yynew(j),zznew(j),1.00,ind(j)
      write(*,31)'ATOM',j,attyp(j),restyp(j),chnam(j)
     :,ind(j),resex(j),xxnew(j),yynew(j),zznew(j),1.00,ind(j)

	enddo
	
31      format(a4,3x,I4,1x,a4,1x,a3,1x,a1,i4,a1,f11.3,f8.3,f8.3,
     :1x,f5.2,1x,i4,1x,f4.3,1x,f4.3,1x,f4.3)



	stop
	end

C
