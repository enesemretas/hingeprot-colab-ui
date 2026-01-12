       program splitter
       character*8 infile1,infile2
       integer start1,start2
       start1=10
       start2=20
       infile1='modeent1'
       infile2='modeent2'
       call split(start1,infile1)
       call split(start2,infile2)
       end
       
       
       subroutine split(start,infile)
       
       real bfact
       character*1  chain,chainr
       character*21 st1
       character*34 st2
       character*24 st3
       character*4 dummy
       integer ir,resnum,start,bfactr
       character*8 infile

50    format (A21,A1,I4,A34,F6.2,A24)
51    format (A4,17x,A1,I4,34x,I3)

	open(unit=1, file="pdb")
	open(unit=2, file=infile)
5000    read(1,'(A4)') dummy
        if (dummy.ne.'ATOM') goto 5000
        backspace(1)
        read(2,*)
2000    read (2,51) dummy,chainr,ir,bfactr
3000	read (1,50) st1,chain,resnum,st2,bfact,st3
        if (st1(1:3).eq.'TER') goto 3000
        if (dummy.ne.'ATOM') goto 7000
        if (chainr.eq.chain.and.mod(resnum,1000).eq.mod(ir,1000)) then
         write (start+bfactr/5,50) st1,chain,resnum,st2,bfact,st3
           goto 3000
           else
           backspace (1)
           goto 2000
        endif
7000    continue
        close (1)
        close (2)
        end
