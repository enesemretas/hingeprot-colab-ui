       program hingebend
       integer hinge1(100),hinge2(100),n1,n2

       real anmsum(10),gnmsum(10)
       real x(10,5000),y(10,5000),z(10,5000)
       real e(10,10),f(10,10)
       real gnmboy(10,5000),boy(10,5000),anmboy(10,5000)
       real anmfark(10,5000),gnmfark(10,5000),rescale
       character*4 label(5000),attyp(5000)
       character*3 restyp(5000)
       character*2 chtyp(5000)
       integer ind(5000)
       character*1 resex(5000)
       real xdist(5000),ydist(5000),zdist(5000)
       integer hinge12(100),n12
       integer cano(5000),resno
       character*15 file1,file2,file5
       
       do i=1,100
       hinge1(i)=0
       hinge2(i)=0
       hinge12(i)=0
       end do
        
       n1=0
       n2=0
       n12=0

C13    format (4x,I4,3x,A3)
C14    format (2x,I4,4(3x,F8.5))
15    format (3x,I4,1x,I4,A1,2x,A1)

      open (unit=51,file='rescale')
      read(51,*)rescale
      file1='crosscorrslow1'
      file2='crosscorrslow2'
      file5='slowmodes'
      open (unit=19,file='coordinates')
      read (19,*) resno
      close(19)
      open (unit=20,file='alpha.cor')
      open (unit=21,file='1coor')
      open (unit=22,file='2coor')
      open (unit=23,file='3coor')
      open (unit=24,file='4coor')
      open (unit=25,file='5coor')
      open (unit=26,file='6coor')
      open (unit=27,file='7coor')
      open (unit=28,file='8coor')
      open (unit=29,file='9coor')
      open (unit=30,file='10coor')
      open (unit=31,file='anm_length')
      open (unit=32,file='newcoordinat.mds')
      open (unit=35,file='newcoordinat2.mds')
      open (unit=40,file='mapping.out')
      

97    format (17x,A3,2x,I4)
98    format (4x,4(3x,F8.5))
99    format (I4,2x,I4,10(3x,F8.5))
100   format (a4,8x,a4,1x,a3,1x,a1,i4,a1,f11.3,f8.3,f8.3,1x,f5.2)
105   format(a4,3x,I4,1x,a4,1x,a3,1x,a1,i4,a1,f11.3,f8.3,f8.3,
     :1x,f5.2,1x,i4,1x,f4.3,1x,f4.3,1x,f4.3)
101   format (I4,4(3x,F8.5))
      do j=1,10
        do i=1,resno
          boy(j,i)=0
        end do
      end do
      do i=1,resno
      read (20,97) restyp(i),cano(i)
      read (21,98) x(1,i),y(1,i),z(1,i),boy(1,i)
      read (22,98) x(2,i),y(2,i),z(2,i),boy(2,i)
      read (23,98) x(3,i),y(3,i),z(3,i),boy(3,i)
      read (24,98) x(4,i),y(4,i),z(4,i),boy(4,i)
      read (25,98) x(5,i),y(5,i),z(5,i),boy(5,i)
      read (26,98) x(6,i),y(6,i),z(6,i),boy(6,i)
      read (27,98) x(7,i),y(7,i),z(7,i),boy(7,i)
      read (28,98) x(8,i),y(8,i),z(8,i),boy(8,i)
      read (29,98) x(9,i),y(9,i),z(9,i),boy(9,i)
      read (30,98) x(10,i),y(10,i),z(10,i),boy(10,i)
      read (32,100) label(i),attyp(i),restyp(i),chtyp(i)
     :,ind(i),resex(i),xdist(i),ydist(i),zdist(i)
      write (35,100) label(i),attyp(i),restyp(i),chtyp(i)
     :,ind(i),resex(i),xdist(i),ydist(i),zdist(i)
      write (*,*)i
      write (31,99) i,cano(i),(boy(j,i),j=1,10)
       do j=1,10
       boy(j,i)=boy(j,i)**2
       enddo
      end do
      do i=20,32
      close (i)
      enddo

C*********   mapping ***************************************

16    format(4x,10(4x,F8.5))
      file5='slowmodes'
      open (unit=25,file=file5)
      do i=1,resno
      read (25,16) (gnmboy(j,i),j=1,10)
      end do
      close(25)

       do j=1,10
       anmsum(j)=0
       gnmsum(j)=0
       enddo

C     normalizaton sum
      do j=1,10
         do i=8,resno-8
          gnmsum(j)=gnmsum(j)+gnmboy(j,i)
          anmsum(j)=anmsum(j)+boy(j,i)
        enddo
      enddo

C      normazlize lengths
      do j=1,10
         do i=1,resno
          gnmboy(j,i)=gnmboy(j,i)/gnmsum(j)
          anmboy(j,i)=boy(j,i)/anmsum(j)
        enddo
      enddo


C     error calculation
      do i=1,2
       do j=1,10
        e(i,j)=0
        enddo
      enddo

      do k=1,2
       do j=1,10
        do i=8,resno-8
        e(k,j)=e(k,j)+abs(gnmboy(k,i)-anmboy(j,i))
        enddo
       enddo
      enddo


C      write (40,*) '*************',rescale
C     first mode
      iii=1
      do j=2,10
        if ((e(1,j)/e(1,iii)).lt.0.95) iii=j
      end do
C     second mode
      if (iii.ne.1) jjj=1
      if (iii.eq.1) jjj=2
      do j=2,10
      if (((e(2,j)/e(2,jjj)).lt.0.95).and.(j.ne.iii)) jjj=j
      end do
C     print results
      write (40,*) 'selection for GNM 1 : ',iii
      write (40,*) 'selection for GNM 2 : ',jjj
17    format ('a1=',F9.5,' a2=',F9.5,' a3=',F9.5,' a4=',F9.5,
     &' a5=',F9.5,' a6=',F9.5)

      write (40,*) ' GNM 1st mode normalized parameters '
      write (40,17) e(1,1),e(1,2),e(1,3),e(1,4),e(1,5),e(1,6)
      write (40,*) ' GNM 2nd mode normalized parameters '
      write (40,17) e(2,1),e(2,2),e(2,3),e(2,4),e(2,5),e(2,6)

      open (unit=41,file='coor1.mds12')
      open (unit=43,file='coor2.mds12')
      open (unit=42,file='coor3.mds12')
      open (unit=44,file='coor4.mds12')
      open (unit=45,file='gnm1anmvector')
      open (unit=46,file='gnm2anmvector')
      do i=1,resno
      write(41,100) label(i),attyp(i),restyp(i),chtyp(i)
     :,ind(i),resex(i),xdist(i)+x(iii,i)*rescale,
     :ydist(i)+y(iii,i)*rescale,zdist(i)+z(iii,i)*rescale,1.00
      write(42,100) label(i),attyp(i),restyp(i),chtyp(i)
     :,ind(i),resex(i),xdist(i)+x(jjj,i)*rescale,
     :ydist(i)+y(jjj,i)*rescale,zdist(i)+z(jjj,i)*rescale,1.00
      write(43,100) label(i),attyp(i),restyp(i),chtyp(i)
     :,ind(i),resex(i),xdist(i)-x(iii,i)*rescale,
     :ydist(i)-y(iii,i)*rescale,zdist(i)-z(iii,i)*rescale,1.00
      write(44,100) label(i),attyp(i),restyp(i),chtyp(i)
     :,ind(i),resex(i),xdist(i)-x(jjj,i)*rescale,
     :ydist(i)-y(jjj,i)*rescale,zdist(i)-z(jjj,i)*rescale,1.00
      write(45,101) i,x(iii,i),y(iii,i),z(iii,i),boy(iii,i)
      write(46,101) i,x(jjj,i),y(jjj,i),z(jjj,i),boy(jjj,i)
      enddo

C      write (6,*) ' GNM 1st mode differences '
C      write (6,17) f(1,1),f(1,2),f(1,3),f(1,4),f(1,5),f(1,6)
C      write (6,*) ' GNM 2nd mode differences '
C      write (6,17) f(2,1),f(2,2),f(2,3),f(2,4),f(2,5),f(2,6)

C***********************************************************

      call findhinge(file1,resno,hinge1,n1)
      call findhinge(file2,resno,hinge2,n2)
C      call findhinge(file3,resno,hinge12,n12)
      open (unit=18,file='hinges')

      write (18,*) '----> crosscorrelation : 1st slowest mode'
      do i=1,n1
      write (18,15) hinge1(i),cano(hinge1(i)),resex(hinge1(i)),
     &chtyp(hinge1(i))
      enddo


      write (18,*) '----> crosscorrelation : 2nd slowest mode'
      do i=1,n2
      write (18,15) hinge2(i),cano(hinge2(i)),resex(hinge2(i)),
     &chtyp(hinge2(i))
      enddo

      close(18)
      end
      
      
C*******************************************************************

      subroutine findhinge(infile,resno,hinge,n)

      character*15 infile
      character*13 charlin
      integer i,j,resno,n,hinge(100)
      real mat(5000)
      open (unit=11,file=infile)

       do j=1,resno
         mat(j)=0
       end do
       
21    format (1x,I4,1x,I4,1x,F6.1)
22    format (A13)
C23    format (10(1x,I3))


      read (11,22) charlin
      if (charlin(11:13).eq.'nan') goto 9999
      backspace (11)

2000  read (11,21)  i,j,mat(j)
      if (j.eq.resno) goto 2001
      goto 2000
2001  close(11)

      n=0
       do j=1,resno-1
        if ((mat(j).gt.0.and.mat(j+1).lt.0).or.
     :(mat(j).lt.0.and.mat(j+1).gt.0)) then
            n=n+1
            hinge(n)=j
          end if
        end do
C       print*,'. . . processing file : ',infile
C       write (6,23) (hinge(j),j=1,n)
C       print*,' '
       goto 10000
9999   print*,'!!! nan etnry in file !!! => ',infile
10000  continue
      end
         


