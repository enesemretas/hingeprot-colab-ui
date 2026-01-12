c23456789012345678901234567890123456789012345678901234567890123456789012


      program convpdbfi

      character*1 resex(5000),s(5000)
      character*3 restyp(5000)
      character*9 fil
      character*3 atnam
      character*6 dummy
      integer atnum,i,resnum(5000),ires
      real x(5000),y(5000),z(5000)


        fil='pdb'

50      format(A6)
55      format (A6,1X,I4,2X,A3,1X,A3,1X,A1,I4,A1,3X,3F8.3)
56      format(i5,3f9.2,3x,A3,1x,A1)
57      format(1x,I4)

        open(10,file=fil)
5       read(10,50)dummy
        if(dummy.ne.'ATOM  ') goto 5
        backspace(10)
        i=0
12      i=i+1
1231    read(10,50,end=33)dummy
C        if(dummy.eq.'TER   ') goto 33
        if(dummy.ne.'ATOM  ') goto 1231
 	backspace (10)
 
        read(10,55)dummy,atnum,atnam,restyp(i),s(i),resnum(i),resex(i),
     + x(i),y(i),z(i)

        if (atnam.eq.'CA '.and.i.eq.1) goto 12
        if (atnam.eq.'CA '.and.resnum(i).ne.resnum(i-1).and.
     + s(i).eq.s(i-1)) goto 12
        if (resex(i).ne.' '.and.atnam.eq.'CA ') goto 12
        if (atnam.eq.'CA '.and.s(i).ne.s(i-1)) goto 12
        
        goto 1231


33      close(10)
        ires=i-1

30      format(A4,3X,i4,1X,A3,2X,A3,1X,A1,I4,A1,3X,3F8.3)
 	open(unit=15,file='alpha.cor')
        do j=1,ires
  	write(15,30)'ATOM',j,'CA',restyp(j),s(j),resnum(j),resex(j)
     +  ,x(j),y(j),z(j)
        enddo

        open(70,file='coordinates')
        write(70,57) ires
        do j=1,ires
        write(70,56) j,x(j),y(j),z(j),restyp(j),resex(j)
        end do
        
	close(70)


        stop
	end

























