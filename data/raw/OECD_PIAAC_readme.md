# Risk-of-automation
Stata code to recreate Nedelkoska and Quintini (2018) estimates of the risk of automation across OECD countries from publicly available PIAAC data. PIAAC PUF data with added estimated risk of automation (pr_fo). 

To run these files:
1. Download them into a single folder.
2. Unzip the folder if it is compressed. 
3. Open 0_piaac_PUF.do in STATA. 
4. Change the path of the working directory to the single folder by modifying line: loc pwd ""
5. Run the do file. The other do files in the folder will be called by this do file.
6. The final dataset (~16MB) will be saved in the same single folder. 
7. Variable pr_fo recreates the risk of automation as estimated in Nedelkoska and Quintini (2018).

IMPORTANT: This code uses the PIAAC Public Use Files (PUF), as available on https://webfs.oecd.org/piaac/puf-data.
OECD updates these files from time to time, affecting our code. Our code was last updated on March 15th, 2022.
Email me at nedelkoska@csh.ac.at if the code runs into errors that you cannot address.

UPDATE (01.03.2025): OECD changed the location of the PIAAC PUF to: https://webfs.oecd.org/piaac/cy1-puf-data. Please change 
the path to the data in the 0_piaac_PUF.do file accordingly
