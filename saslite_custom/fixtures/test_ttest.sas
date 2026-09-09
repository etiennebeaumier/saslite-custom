data test;
    input groupe offre;
    datalines;
1 100
1 110
1 120
2 150
2 160
2 170
;
run;

proc ttest data=test;
    class groupe;
    var offre;
run;
