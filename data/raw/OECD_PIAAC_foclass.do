	// FO 70-occupation classification for ISCO 08 4-digit
	gen fo = .
	replace fo = 0 if isco08_c == 2211		// 1. Generalist medical practitioners
	replace fo = 0 if isco08_c == 2212		// 1. Specialist medical practitioners
	replace fo = 0 if isco08_c == 2261		// 2. dentists	
	replace fo = 0 if isco08_c == 1344		// 3. Social welfare managers
	replace fo = 0 if isco08_c == 2342 		// 4. Early childhood educators
	replace fo = 0 if isco08_c == 2636		// 5.1. Religious professionals
*	replace fo = 0 if isco08_c == 3413		// 5.2. Religious associate professionals
	replace fo = 0 if isco08_c == 2221		// 6.1 Nursing and midwifery professionals
*	replace fo = 0 if isco08_c == 3221		// 6.2 Nursing associate professionals
	replace fo = 0 if isco08_c == 2635		// 12. Social work and counselling professionals (for Marriage and family therapists)
	replace fo = 0 if isco08_c == 1120		// 8. Directors and chief executives
	replace fo = 0 if isco08_c == 1345 		// 9. Education managers
	replace fo = 0 if isco08_c == 2142		// 10. Civil engineers
	replace fo = 0 if isco08_c == 2163		// 11. Product and garment designers 		
	replace fo = 0 if isco08_c == 2635		// 12. Social work and counselling professionals (note: same as 12)
	replace fo = 0 if isco08_c == 2611		// 13. Lawyers	
	replace fo = 0 if isco08_c == 3332		// 14. Conference and event planners
	replace fo = 0 if isco08_c == 2162 		// 15. Landscape architects
*	replace fo = 0 if isco08_c == 2269 		// 16.1 Health professionals, other
	replace fo = 0 if isco08_c == 3259 		// 16.2 Health associate professionals, other
*	replace fo = 0 if isco08_c == 3139 		// 17. Process control technicians, other	
	replace fo = 0 if isco08_c == 5311 		// 18. Child-care workers
	replace fo = 0 if isco08_c == 3434 		// 19. Chefs
	replace fo = 0 if isco08_c == 2151 		// 20. Electrical engineers
	replace fo = 0 if isco08_c == 2111 		// 21. Physicists and astronomers
	replace fo = 0 if isco08_c == 5141 		// 22.1 Hairdressers
	replace fo = 0 if isco08_c == 5142 		// 22.2 Beauticians and related workers"
	replace fo = 0 if isco08_c == 4224 		// 23. Hotel receptionists (instead of Concierges)
	replace fo = 0 if isco08_c == 3421 		// 24. Athletes, sportspersons and related associate professionals	
	replace fo = 0 if isco08_c == 2131 		// 25. Biologists
	replace fo = 0 if isco08_c == 7126 		// 26. Plumbers and pipe fitters
	replace fo = 0 if isco08_c == 5111 		// 27. Travel attendants and travel stewards (instead of flight attendant)		
	replace fo = 1 if isco08_c == 2165 		// 28. Cartographers and surveyors
	replace fo = 0 if isco08_c == 2612 		// 29. Judges
*	replace fo = 1 if isco08_c == 3342 		// 30.1 Legal secretaries (checked: https://www.onetonline.org/link/summary/23-1012.00
	replace fo = 1 if isco08_c == 3411 		// 30.2 Legal and related associate professionals (checked: https://www.onetonline.org/link/summary/23-1012.00
	replace fo = 0 if isco08_c == 2631 		// 31. Economists
	* silenced because of lack of correspondence between SOC and ISCO for this occ group and a previous coding mistake
*	replace fo = 0 if isco08_c == 3315 		// 32. Valuers and loss assessors (instead of cost estimators) (compare: http://www.tucareers.com/iscocareers/3315 and https://www.onetonline.org/link/summary/13-1051.00)
	replace fo = 0 if isco08_c == 1324 		// 33. Supply, distribution and related managers 
	* changed from 0 to 1 to correct previous coding mistake
	replace fo = 1 if isco08_c == 2431		// 34. Advertising and marketing professionals
*	replace fo = 1 if isco08_c == 3152		// 35. Ships' deck officers and pilots (for motorboat operators) (check: https://www.onetonline.org/link/summary/53-5022.00)
	replace fo = 1 if isco08_c == 8331 		// 36. Bus and tram drivers
*	replace fo = 1 if isco08_c == 8332 		// 37. Heavy-truck and lorry drivers (instead of  Light truck or delivery services drivers)
*	replace fo = 0 if isco08_c == 5152 		// 38.1 Domestic housekeepers
	replace fo = 0 if isco08_c == 9111 		// 38.2 Domestic helpers and cleaners
	replace fo = 1 if isco08_c == 3112 		// 39. Civil engineering technicians
											// 40. no match: dishwashers
	replace fo = 0 if isco08_c == 6224 		// 41. Hunters and trappers
	replace fo = 1 if isco08_c == 5120 		// 42. Cooks (for Chefs and head cooks)		
*	replace fo = 1 if isco08_c == 3113 		// 43.1 Electrical engineering technicians
*	replace fo = 1 if isco08_c == 3114 		// 43.2 Electronics and telecommunications engineering technicians
	replace fo = 1 if isco08_c == 7213 		// 44. Sheet metal workers
	replace fo = 1 if isco08_c == 9623 		// 45. Meter readers
*	replace fo = 1 if isco08_c == 8189 		// 46. Stationary plant and machine operators, other
											// 47. no match: for parking lot attendants
*	replace fo = 1 if isco08_c == 3256 		// 48.1 Medical assistants (for Medical transcriptionists)
	replace fo = 1 if isco08_c == 3344 		// 48.2 Medical secretaries (for Medical transcriptionists)
											// 49. no match: Technical writers
	replace fo = 1 if isco08_c == 8153 		// 50. Sewing-machine operators
	replace fo = 1 if isco08_c == 8322 		// 51. Car, taxi and van drivers
	replace fo = 1 if isco08_c == 4416 		// 52. Personnel clerks
	replace fo = 1 if isco08_c == 3352 		// 53. Government tax and excise officials (Tax examiners and collectors)
*	replace fo = 1 if isco08_c == 8341 		// 54 Mobile farm and forestry plant operators
	replace fo = 1 if isco08_c == 2411 		// 55.1 Accountants
	replace fo = 0 if isco08_c == 5131 		// 56. Waiters
	replace fo = 1 if isco08_c == 9621 		// 57. Messengers, Package Deliverers And Luggage Porters
											// 58. no match:  Paralegals and legal assistants (check: https://www.onetonline.org/link/summary/23-2011.00	
	replace fo = 1 if isco08_c == 8212 		// 59. Electrical and electronic equipment assemblers
	replace fo = 1 if isco08_c == 4223 		// 60. Telephone switchboard operators
											// 61. no match: gaming dealers
											// 62. no match: Farm labour contractors
	replace fo = 1 if isco08_c == 5230 		// 63. Cashiers and ticket clerks		
	replace fo = 1 if isco08_c == 4415 		// 64. Filing and copying clerks
*	replace fo = 1 if isco08_c == 3312		// 65. No match for Credit authorizers, checkers, and clerks
	replace fo = 1 if isco08_c == 3315		// 66. Valuers and loss assessors (for claims adjusters)
	replace fo = 1 if isco08_c == 2413 		// 67. Financial analyst (for credit analysts)
	replace fo = 1 if isco08_c == 3312		// 68. Credit and loans officers
	replace fo = 1 if isco08_c == 4132 		// 69. Data entry clerks
	replace fo = 1 if isco08_c == 3321 		// 70. Insurance representatives (for insurance underwriters)
