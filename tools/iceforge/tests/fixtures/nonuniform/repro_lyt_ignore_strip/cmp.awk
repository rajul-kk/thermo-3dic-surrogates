function load(f, a,   n,line,i,m,t){n=0; while((getline line < f)>0){ if(line ~ /^%/||line ~ /^ *$/) continue; m=split(line,t," "); for(i=1;i<=m;i++) a[++n]=t[i]} close(f); return n}
BEGIN{n1=load(F1,A); n2=load(F2,B); if(n1!=n2){print "count mismatch",n1,n2; exit} for(i=1;i<=n1;i++){d=A[i]-B[i]; if(d<0)d=-d; if(d>mx)mx=d; if(A[i]>ma)ma=A[i]} printf "n=%d max|diff|=%.6g maxT=%.4f\n",n1,mx,ma}
