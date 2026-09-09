data sample;
    input group $ value;
    datalines;
A 5
A 7
A 8
A 11
B 8
B 10
B 14
B 17
;
run;
proc npar1way data=sample wilcoxon fp;
    class group;
    var value;
run;
