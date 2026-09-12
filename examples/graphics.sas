/* Synthetic coursework example. No private data or SAS installation needed. */
data intro_ex1;
    input age n_visite;
    label age="Âge" n_visite="Nombre de visites";
    datalines;
18 1
21 2
23 1
25 3
31 4
35 2
42 5
50 7
63 8
78 10
;
run;

/* Partie 4 : Graphiques — writes three PNGs beside this script in graphs/. */
ods graphics on;
*ods rtf;

/* Histogramme de l'âge */
proc sgplot data=intro_ex1;
    histogram age;
run;

/* Box plot de l'âge */
proc sgplot data=intro_ex1;
    hbox age;
run;

/* Nombre de visites en fonction de l'âge */
proc sgplot data=intro_ex1;
    scatter x=age y=n_visite;
run;

ods graphics off;
*ods graphics off;
