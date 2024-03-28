# Interpretation of the metrics

## SARI

## SAMSA

## FKGL
`` (0.39 * (#word/#sentences)) + (11.8 * (#syllables/#words)) - 15.59 ``
Comparing the scores and passages of https://stars.library.ucf.edu/cgi/viewcontent.cgi?article=1055&context=istlibrary, EASSE produce different scores.

| FKGL 	    | Description 	|
|-----------|---	|
| 5 	       | very easy 	|
| 6 	       | easy 	|
| 7 	       | fairly easy 	|
| 8 - 9 	   | standard 	|
| 10 - 12 	 | fairly difficult 	|
| 13 - 16 	 | difficult 	|
| > 16 	    | very difficult 	|

## BLEU

## FLESCH READING EASE (FRE)
Officially designed for English readability measurement by Flesch (1948):

``FRE = 206.835 - (0.846 * (#word/#sentences)) - (1.015 * (#syllables/#words))``

| FRE 	      | Description 	| Typical Magazine 	|
|------------|---	|---	|
| 90 - 100 	 | very easy 	| comics 	|
| 80 - 90 	  | easy 	| pulp-fiction 	|
| 70 - 80 	  | fairly easy 	| slick-fiction 	|
| 60 - 70 	  | standard 	| digest 	|
| 50 - 60 	  | fairly difficult 	| quality 	|
| 30 - 50 	  | difficult 	| academic 	|
| 0 - 30 	   | very difficult 	| scientific 	|


Amstad (1978) adapted the formula for German texts:

``FRE_DE = 180 - (#word/#sentences) - (58.5 * (#syllables/#words))``

| FRE 	     | Interpretation 	|
|-----------|---	|
| \> 70 	   | ease 	|
| 60 - 70 	 | simple 	|
| 50 - 60 	 | average 	|
| 40 - 50 	 | average 	|
| 30 - 40 	 | sophisticated 	|
| 20 - 30 	 | difficult 	|
| \< 20 	   | very difficult 	|


## Wiener Sachtextformel 1-4

| Wiener Sachtextformel | Required years of school | Interpretation |
|-----------------------|--------------------------|----------------|
| <=4                   | 4                        | very easy      |
| 5                     | 5                        |                |
| 6                     | 6                        |                |
| 7                     | 7                        |                |
| 8                     | 8                        |                |
| 9                     | 9                        |                |
| 10                    | 10                       |                |
| 11                    | 11                       |                |
| >12                   | >12                      | very difficult |

