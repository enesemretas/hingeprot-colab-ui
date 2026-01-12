        program main
        character*15 infile(38),outfile(38)
        infile(1)='gnm1anmvector'
        infile(2)='gnm2anmvector'
        infile(3)='1coor'
        infile(4)='2coor'
        infile(5)='3coor'
        infile(6)='4coor'
        infile(7)='5coor'
        infile(8)='6coor'
        infile(9)='7coor'
        infile(10)='8coor'
        infile(11)='9coor'
        infile(12)='10coor'
        infile(13)='11coor'
        infile(14)='12coor'
        infile(15)='13coor'
        infile(16)='14coor'
        infile(17)='15coor'
        infile(18)='16coor'
        infile(19)='17coor'
        infile(20)='18coor'
        infile(21)='19coor'
        infile(22)='20coor'
        infile(23)='21coor'
        infile(24)='22coor'
        infile(25)='23coor'
        infile(26)='24coor'
        infile(27)='25coor'
        infile(28)='26coor'
        infile(29)='27coor'
        infile(30)='28coor'
        infile(31)='29coor'
        infile(32)='30coor'
        infile(33)='31coor'
        infile(34)='32coor'
        infile(35)='33coor'
        infile(36)='34coor'
        infile(37)='35coor'
        infile(38)='36coor'
        outfile(1)='mod1'
        outfile(2)='mod2'
        outfile(3)='1anm.pdb'
        outfile(4)='2anm.pdb'
        outfile(5)='3anm.pdb'
        outfile(6)='4anm.pdb'
        outfile(7)='5anm.pdb'
        outfile(8)='6anm.pdb'
        outfile(9)='7anm.pdb'
        outfile(10)='8anm.pdb'
        outfile(11)='9anm.pdb'
        outfile(12)='10anm.pdb'
        outfile(13)='11anm.pdb'
        outfile(14)='12anm.pdb'
        outfile(15)='13anm.pdb'
        outfile(16)='14anm.pdb'
        outfile(17)='15anm.pdb'
        outfile(18)='16anm.pdb'
        outfile(19)='17anm.pdb'
        outfile(20)='18anm.pdb'
        outfile(21)='19anm.pdb'
        outfile(22)='20anm.pdb'
        outfile(23)='21anm.pdb'
        outfile(24)='22anm.pdb'
        outfile(25)='23anm.pdb'
        outfile(26)='24anm.pdb'
        outfile(27)='25anm.pdb'
        outfile(28)='26anm.pdb'
        outfile(29)='27anm.pdb'
        outfile(30)='28anm.pdb'
        outfile(31)='29anm.pdb'
        outfile(32)='30anm.pdb'
        outfile(33)='31anm.pdb'
        outfile(34)='32anm.pdb'
        outfile(35)='33anm.pdb'
        outfile(36)='34anm.pdb'
        outfile(37)='35anm.pdb'
        outfile(38)='36anm.pdb'
        do i=1,38
        call coor2pdb(infile(i),outfile(i))
        end do
        
        end
        
        

        subroutine coor2pdb(infile,outfile)
! Bu program anm çıktısı olan Xcoor dosyalarını ve orjina pdb dosyasını
! kullanarak içinde animasyon için kullanılacak modelleri içeren yeni
! ve tüm atomları içeren bir pdb dosyası yaratıyoruz.
!				10.4.2006 		Serkan K.
	integer resmax,lnmax
	parameter (resmax=5000,lnmax=100000)
	
	character*1 resex
	character*2 atom,chtyp
	character*3 restyp
	character*4 attyp
	character*6 dummy
        character*15 infile,outfile
	character*80 pdb(lnmax)
	
	integer i,ii	!ii kontrol icin
	integer resnum,atnum,resnm,modnum,modind,lastatomline
	integer pdbln,pdbi,matomline    	! pdbln: pdb dosyasinda kac satir var!
	real x(resmax), y(resmax), z(resmax), mag(resmax),resc(resmax)
	real rescale,magsum,ang,rscaled		! perturbation
	real xx,yy,zz,occup,bfactor		! pdbdosyasindan okunan
	real xx1,yy1,zz1		! yeni pdb dosyasina yazilan
	

!	Coor Dosyası okuma.
	open (unit=1, file=infile)
	 ii=0
2000	 read (1,*, end = 2001) i,x(i),y(i),z(i)
         mag(i)=(x(i)**2+y(i)**2+z(i)**2)**(.5)
	 ii=ii+1
         goto 2000
2001     close(1)

	resnum = i

	i=1
	open(unit=1, file="pdb")
3000	read (1,'(A80)',end = 3001) pdb(i)
        i=i+1
        goto 3000
3001	close(1)
        pdbln=i-1
	

        
	ang = 3. 	! ang, angstrom icin. ortalama hareketi bu seviyeye getirir.
	magsum = 0
	do i = 10, resnum-10
	 magsum = magsum + mag(i)
	enddo
	magsum = magsum / real(resnum)
	
	rescale = ang / magsum
 	do i = 1, resnum
           if (rescale*mag(i).ge.3) then
           resc(i)=0.75/mag(i)
           else
           resc(i)=rescale
           endif
	enddo


	
	modnum=3
	modind = (-1)*modnum


!	PDB dosyasini okumaya ve Modelleri yazmaya basliyoruz.
2	FORMAT(A6,I5,2X,A3,1X,A3,1X,A1,1X,I3,A1,3X,3F8.3,F6.2,F6.2,10X,
     &A2) !Pdb okuma format
3	FORMAT(A6,I5,A2,A3,A1,A3,2A1,I4,A1,A3,3F8.3,2F6.2,"          ",
     &A2) !PBD yazma format

	open (unit=2, file=outfile)
        pdbi=1
5000    read ( pdb(pdbi), '(A6)') dummy
        if (dummy.NE."ATOM  ") then
        write (2,'(A80)') pdb(pdbi)
        pdbi=pdbi+1
        goto 5000
        endif
        matomline=pdbi
	do while (modind .LE. modnum) ! Model Yazma
         pdbi=matomline
	 i=0		! Xcoor dosyasi kacinci residue.
	 ii = 0  	! bir residuedan digerine gecimizi kontrol ediyoruz.
         write (2,'(A6,I2)') "MODEL ",modind+modnum+1
	 read ( pdb(pdbi), '(A6)') dummy
         do while( dummy .NE. "END   ".and.dummy.NE."ENDMDL")
	  read( pdb(pdbi),'(A6)') dummy	

	  if (dummy .EQ. "ATOM  " ) then
           lastatomline=pdbi
           READ( pdb(pdbi) ,2) dummy,atnum,attyp,restyp,chtyp,resnm,
     &resex,xx,yy,zz,occup,bfactor,atom
	   
	   if (resnm .NE.ii ) then ! ilk okumada kesinlikle girecek.
 !	    print *, "ii ",ii," resnm ",resnm
	    ii = resnm
	    i = i+1
	   endif
	   
	   rscaled=resc(i)*(real(modind)/real(modnum))

           xx1=xx+x(i)*rscaled
	   yy1=yy+y(i)*rscaled
	   zz1=zz+z(i)*rscaled

C	   if (attyp.eq.'CA ') then
	   write (2,3)  "ATOM  ",atnum,"  ",attyp," ",restyp," ",chtyp,
     &resnm,resex,"   ",xx1,yy1,zz1,occup,bfactor,atom
C           endif
     	  endif					! ATOM  OLAN SATIRLAR SON.
	  
	  
	  pdbi=pdbi+1	! Bir sonraki satir
	 enddo ! PDB okuma yazma Bitti
	 
	
	 write (2,'(A6)') "ENDMDL"
	 if ( modind .EQ. modnum ) write (2,'(A3)') "END"
	 modind=modind+1
	
	enddo ! Model Yazma Domgusu bitti

        do i=lastatomline+1,pdbln
        if (pdb(i)(1:3).NE.'TER') write (2,'(A80)') pdb(i)
        enddo

	
	close(2)
	end
