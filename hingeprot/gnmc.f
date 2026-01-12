      program gnm
      integer        maxx
      parameter    ( maxx=5000)

      integer        i, j, k,resnum ,mode

      real  xx(maxx),yy(maxx),zz(maxx)
      real  w(maxx),v(maxx,maxx),rcut,rcanm

      real  bx,by,bz,cont(maxx,maxx),ulim,llim
      real  cm(maxx,maxx)
	real x1(maxx,1),y1(maxx,1),z1(maxx,1)
   	 real slow1(5000),slow1c(maxx),s1(5000,5000),s2(5000)
  	 real slow2(5000),slow2c(maxx)
          real sslow2c(maxx),sslow1c(maxx)
          character*256 filename,filenames
	open (24,file ='sloweigenvectors')
	open (37,file ='gnmcutoff')
	read(37,*)rcut
	open (25,file ='slowmodes')

	open (28,file ='slow12avg')
	open (30,file ='crosscorr')

c	 READPDB FILE

 	  
  	open(33,file='coordinates')

         
	READ(33,*)resnum
           do i=1,resnum

           read(33,*)a,x1(i,1),y1(i,1),z1(i,1)
   	
         enddo

	
	do i=1,resnum


	 xx(i)=x1(i,1)
	 yy(i)=y1(i,1)
	 zz(i)=z1(i,1)


	enddo

c------ MAIN SECTION OF GNM


c--- obtain connectivity matrix
  
       do j=1,resnum
       do k=1,resnum
	bx=xx(j)-xx(k)
	by=yy(j)-yy(k)
	bz=zz(j)-zz(k)
	r=sqrt(bx*bx+by*by+bz*bz)
	if ((r.le.rcut).and.(k.ne.j).and.(r.gt.0.0001)) then
	 cont(j,k)=-1.
	else
	 cont(j,k)=0.
	end if 
       enddo
      enddo

      top2=0.
      do 201 j=1,resnum
	top1=0.
	do 200 k=1,resnum
	  top1=top1+cont(j,k)
200     continue
        top2=top2+top1
	cont(j,j)=-top1
201   continue

	
	

c--- call SVD subroutine


        call svdcmp(cont,resnum,resnum,resnum,resnum,w,v)

        do i=1,resnum
        do j=i,resnum
        if(w(i).lt.w(j)) then
         top1=w(i)
         w(i)=w(j)
         w(j)=top1
         do k=1,resnum       
          top1=v(k,i)
          v(k,i)=v(k,j)
          v(k,j)=top1
          top2=cont(k,i)
          cont(k,i)=cont(k,j)
          cont(k,j)=top2
         enddo
         endif
         enddo
         enddo
        open (unit=61,file='sortedeigen')
	do i=1,resnum
	write(61,'(I4,2X,F8.4)')i,w(resnum+1-i)
	enddo
        close(61)



c--- recover connectivity matrix

        do i=1,resnum
         do j=1,resnum
  	  cm(i,j)=0.
   	   do k=1,resnum-1
	     cm(i,j)=cm(i,j)+cont(i,k)*v(j,k)/w(k)
           enddo
         enddo
        enddo


c-- comparison of theoretical values with experimental temperature
c-- factors
       


c-- ten slow eigenmodes

       do i=1,resnum
 	write(24,11) i,(v(i,j),j=resnum-1,resnum-10,-1)
       enddo  
 

      do i=1,resnum
	write(25,11)i,(v(i,j)*v(i,j),j=resnum-1,resnum-10,-1)
	 
	slow1(i)=v(i,resnum-1)*v(i,resnum-1)
	slow2(i)=v(i,resnum-2)*v(i,resnum-2)

	slow1c(i)=slow1c(i)+slow1(i)
	slow2c(i)=slow2c(i)+slow2(i)
	sslow1c(i)=sslow1c(i)+slow1(i)/w(resnum-1)
	sslow2c(i)=sslow2c(i)+slow2(i)/w(resnum-2)
     	enddo  
11    format(I4,300f12.5)



c-- weighted averages two slowest eigenmodes

       do i=1,resnum
       do j=resnum-2,resnum-1
        s1(i,j)=v(i,j)**2./w(j)
       enddo
      enddo
	


       do i=1,resnum
	s2(i)=0.
        do j=resnum-2,resnum-1
         s2(i)=s2(i)+s1(i,j)
        enddo
       enddo
	

       top2=0.
       do i=resnum-2,resnum-1
       top2=top2+1./w(i)
       enddo
111    format(I4,f12.5)
	do i=1,resnum
       write(28,111)i,s2(i)/top2
       enddo


c-- data for cross-correlation graph


	do i=1,resnum
	do j=1,resnum
		a=cm(i,i)*cm(j,j)
 		write(30,113)i,j,cm(i,j)/sqrt(a)
113     format(i4,1x,i4,1x,f7.4)
	enddo
	write(30,*)
	enddo

      filename='crosscorrslow1'
      filenames='crosscorrslow1ext'
      CALL cross(w,v,cont,resnum,1,filename,filenames)

      filename='crosscorrslow2'
      filenames='crosscorrslow2ext'
      CALL cross(w,v,cont,resnum,2,filename,filenames)

      filename='crosscorrslow3'
      filenames='crosscorrslow3ext'
      CALL cross(w,v,cont,resnum,3,filename,filenames)

      filename='crosscorrslow4'
      filenames='crosscorrslow4ext'
      CALL cross(w,v,cont,resnum,4,filename,filenames)

      filename='crosscorrslow5'
      filenames='crosscorrslow5ext'
      CALL cross(w,v,cont,resnum,5,filename,filenames)

      filename='crosscorrslow6'
      filenames='crosscorrslow6ext'
      CALL cross(w,v,cont,resnum,6,filename,filenames)

      filename='crosscorrslow7'
      filenames='crosscorrslow7ext'
      CALL cross(w,v,cont,resnum,7,filename,filenames)

      filename='crosscorrslow8'
      filenames='crosscorrslow8ext'
      CALL cross(w,v,cont,resnum,8,filename,filenames)

      filename='crosscorrslow9'
      filenames='crosscorrslow9ext'
      CALL cross(w,v,cont,resnum,9,filename,filenames)

      filename='crosscorrslow10'
      filenames='crosscorrslow10ext'
      CALL cross(w,v,cont,resnum,10,filename,filenames)

	end 
c----------------SVD SUBROUTINE--------------------------


      SUBROUTINE svdcmp(a,m,n,mp,np,w,v)
      INTEGER m,mp,n,np,NMAX
      REAL a(5000,5000),v(5000,5000),w(5000)
      PARAMETER (NMAX=5000)
C     USES pythag
      INTEGER i,its,j,jj,k,l,nm
      REAL anorm,c,f,g,h,s,scale,x,y,z,rv1(NMAX),pythag
      g=0.0
      scale=0.0
      anorm=0.0
      do 25 i=1,n
c	write(6,*)'hi there loop 25    i=',i
        l=i+1
        rv1(i)=scale*g
        g=0.0
        s=0.0
        scale=0.0
        if(i.le.m)then
          do 11 k=i,m
            scale=scale+abs(a(k,i))
11        continue
          if(scale.ne.0.0)then
            do 12 k=i,m
              a(k,i)=a(k,i)/scale
              s=s+a(k,i)*a(k,i)
12          continue
            f=a(i,i)
            g=-sign(sqrt(s),f)
            h=f*g-s
            a(i,i)=f-g
            do 15 j=l,n
              s=0.0
              do 13 k=i,m
                s=s+a(k,i)*a(k,j)
13            continue
              f=s/h
              do 14 k=i,m
                a(k,j)=a(k,j)+f*a(k,i)
14            continue
15          continue
            do 16 k=i,m
              a(k,i)=scale*a(k,i)
16          continue
          endif
        endif
        w(i)=scale *g
        g=0.0
        s=0.0
        scale=0.0
        if((i.le.m).and.(i.ne.n))then
          do 17 k=l,n
            scale=scale+abs(a(i,k))
17        continue
          if(scale.ne.0.0)then
            do 18 k=l,n
              a(i,k)=a(i,k)/scale
              s=s+a(i,k)*a(i,k)
18          continue
            f=a(i,l)
            g=-sign(sqrt(s),f)
            h=f*g-s
            a(i,l)=f-g
            do 19 k=l,n
              rv1(k)=a(i,k)/h
19          continue
            do 23 j=l,m
              s=0.0
              do 21 k=l,n
                s=s+a(j,k)*a(i,k)
21            continue
              do 22 k=l,n
                a(j,k)=a(j,k)+s*rv1(k)
22            continue
23          continue
            do 24 k=l,n
              a(i,k)=scale*a(i,k)
24          continue
          endif
        endif
        anorm=max(anorm,(abs(w(i))+abs(rv1(i))))
25    continue
      do 32 i=n,1,-1
c	write(6,*)'and now loop 32    i=',i
        if(i.lt.n)then
          if(g.ne.0.0)then
            do 26 j=l,n
              v(j,i)=(a(i,j)/a(i,l))/g
26          continue
            do 29 j=l,n
              s=0.0
              do 27 k=l,n
                s=s+a(i,k)*v(k,j)
27            continue
              do 28 k=l,n
                v(k,j)=v(k,j)+s*v(k,i)
28            continue
29          continue
          endif
          do 31 j=l,n
            v(i,j)=0.0
            v(j,i)=0.0
31        continue
        endif
        v(i,i)=1.0
        g=rv1(i)
        l=i
32    continue
      do 39 i=min(m,n),1,-1
c	write(6,*)'and now we are in loop 39    i=',i
        l=i+1
        g=w(i)
        do 33 j=l,n
          a(i,j)=0.0
33      continue
        if(g.ne.0.0)then
          g=1.0/g
          do 36 j=l,n
            s=0.0
            do 34 k=l,m
              s=s+a(k,i)*a(k,j)
34          continue
            f=(s/a(i,i))*g
            do 35 k=i,m
              a(k,j)=a(k,j)+f*a(k,i)
35          continue
36        continue
          do 37 j=i,m
            a(j,i)=a(j,i)*g
37        continue
        else
          do 38 j= i,m
            a(j,i)=0.0
38        continue
        endif
        a(i,i)=a(i,i)+1.0
39    continue
      do 49 k=n,1,-1
c	write(6,*)'in loop 49 k=',k
        do 48 its=1,50
          do 41 l=k,1,-1
            nm=l-1
            if((abs(rv1(l))+anorm).eq.anorm)  goto 2
            if((abs(w(nm))+anorm).eq.anorm)  goto 1
41        continue
1         c=0.0
          s=1.0
          do 43 i=l,k
            f=s*rv1(i)
            rv1(i)=c*rv1(i)
            if((abs(f)+anorm).eq.anorm) goto 2
            g=w(i)
            h=pythag(f,g)
            w(i)=h
            h=1.0/h
            c= (g*h)
            s=-(f*h)
            do 42 j=1,m
              y=a(j,nm)
              z=a(j,i)
              a(j,nm)=(y*c)+(z*s)
              a(j,i)=-(y*s)+(z*c)
42          continue
43        continue
2         z=w(k)
          if(l.eq.k)then
            if(z.lt.0.0)then
              w(k)=-z
              do 44 j=1,n
                v(j,k)=-v(j,k)
44            continue
            endif
            goto 3
          endif
          if(its.eq.50) write(6,*) 'its=',its
          x=w(l)
          nm=k-1
          y=w(nm)
          g=rv1(nm)
          h=rv1(k)
          f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
          g=pythag(f,1.0)
          f=((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x
          c=1.0
          s=1.0
          do 47 j=l,nm
            i=j+1
            g=rv1(i)
            y=w(i)
            h=s*g
            g=c*g
            z=pythag(f,h)
            rv1(j)=z
            c=f/z
            s=h/z
            f= (x*c)+(g*s)
            g=-(x*s)+(g*c)
            h=y*s
            y=y*c
            do 45 jj=1,n
              x=v(jj,j)
              z=v(jj,i)
              v(jj,j)= (x*c)+(z*s)
              v(jj,i)=-(x*s)+(z*c)
45          continue
            z=pythag(f,h)
            w(j)=z
            if(z.ne.0.0)then
              z=1.0/z
              c=f*z
              s=h*z
            endif
            f= (c*g)+(s*y)
            x=-(s*g)+(c*y)
            do 46 jj=1,m
              y=a(jj,j)
              z=a(jj,i)
              a(jj,j)= (y*c)+(z*s)
              a(jj,i)=-(y*s)+(z*c)
46          continue
47        continue
          rv1(l)=0.0
          rv1(k)=f
          w(k)=x
48      continue
3       continue
49    continue
      return
      END

      FUNCTION pythag(a,b)
      REAL a,b,pythag
      REAL absa,absb
      absa=abs(a)
      absb=abs(b)
      if(absa.gt.absb)then
        pythag=absa*sqrt(1.+(absb/absa)**2)
      else
        if(absb.eq.0.)then
          pythag=0.
        else
          pythag=absb*sqrt(1.+(absa/absb)**2)
        endif
      endif
      return
      END
C***************************************************************

      subroutine cross(w,v,cont,jres,mode,outfile,outfile2)

      integer jres,mode
      real w(5000),v(5000,5000),cont(5000,5000)
      real cm(5000,5000),a,rl
      character*256  outfile,outfile2

        open (30,file = outfile)
        open (40,file = outfile2)
      
        do i=1,jres
         do j=1,jres
          cm(i,j)=cont(i,jres-mode)*v(j,jres-mode)/w(jres-mode)
         enddo
        enddo


c-- data for cross-correlation graph

	do i=1,jres
	do j=1,jres
		a=cm(i,i)*cm(j,j)
		write(40,112)i,j,cm(i,j)/sqrt(a)
	enddo
	write(40,*)
	enddo

	do i=1,1
	do j=1,jres
               rl=0.
               if (cm(i,j).lt.0) rl=-1.
               if (cm(i,j).gt.0) rl=1.
C               a=cm(i,i)*cm(j,j)
 	       write(30,112)i,j,rl
	enddo
        write(30,*)
	enddo

112    format (1x,I4,1x,I4,3x,F4.1)

  	close(30)
  	close(40)

       end




  
