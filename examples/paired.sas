data paired_data;
    input before after;
    datalines;
10 12
12 13
9 12
15 16
11 15
;
run;
proc ttest data=paired_data;
    paired before*after;
run;
