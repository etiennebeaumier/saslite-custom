/* Synthetic summaries and one-sided analyses within sites. */
data sample;
    input site $ group $ age visits value;
    datalines;
A X 18 1 2
A X 21 2 4
A X 25 2 5
A Y 31 3 3
A Y 35 4 6
A Y 40 5 9
B X 20 1 1
B X 27 2 3
B X 32 4 6
B Y 43 5 4
B Y 52 8 8
B Y 60 9 10
;
run;

proc sort data=sample; by site; run;
proc means data=sample n nmiss mean std nway;
    by site;
    class group;
    var age visits;
    output out=summary mean= n= nmiss= / autoname;
run;
proc print data=summary; run;

proc ttest data=sample sides=u h0=-1;
    by site;
    class group;
    var value;
run;
proc npar1way data=sample wilcoxon fp;
    by site;
    class group;
    var value;
run;
