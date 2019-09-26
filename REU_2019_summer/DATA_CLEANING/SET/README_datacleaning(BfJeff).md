# GRYD - SET Data Cleaning

Cleaning raw SET data provided by GRYD

Last updated 07/08/2019 by Avery Edson and Charlotte Huang

## Getting Started

All cleaning happens the Python 3.6.6 script `Data Cleaning for SET Data.ipynb`. The script uses packages `pandas` and `numpy`.

Specify the raw data filepath string in `raw_filepath`.


## Cleaning Process

### Filtering out old versions of the SET Questionnaire

According to `VERSION_1`, there are four different SET versions that were used throughout the years.  As we only had access to the 1 FEB 2016 questionnaire, we dropped the data of those who had taken any previous version (before 1 FEB 2016) of the questionnaire.  The data kept were those with the value 3 in `SET_version_num`.  This left us with 2800 entries (out of 3416).


### Creating a unique identifier `UniqueID`

The `UniqueID` assigned to each row comes form the `ETO_ID_FB`.  For our dataset, none of the `ETO_ID_FB` values were missing and so we renamed this column to be the `UniqueID`.  

****At this moment, we have assumed that the `UniqueID` values are indeed unique within the dataset and so we must still validate this by observing the intake/retake validity.

### Only keep completed Processing_status

The `Processing_status` variable denotes the completion status of the SET questionnaire. We only kept those that were either 'Completed' or 'Completed(archive)'.  In other words, we dropped the data that was marked as incomplete. This left us with 2,769 entries (we dropped 31 entries). 

### Family Section (Cleaning for those with no family)

Question F3 inquires how many people are in the participant's family.  There is an option for the participant to respond "NO" if they have no family, and in this case they should skip to page 9 of the questionnaire (the group section).

The responses to questions F4 through F31b are then disregarded for those who responded to having "NO" family in F3.  All of these responses were converted to not available (Nan).  In addition, their responses to questions F1 through F3 were converted to zero to reflect an absence of family.

### Group Section (Cleaning for those with no group)

If the client's response is "NO GROUP" at the beginning of the group portion of the questionnaire, they then skip to page 12.  To account for this, we converted these clients' responses to questions G2 through G37 to not available.

### Other Group Section (Cleaning for those with no other group)

If the client's response is "NO OTHER GROUP" at the beginning of the other group portion of the questionnaire, they then skip to page 15.  To account for this, we converted these clients' responses to questions O4 through O24 to not available.

### Converting Invalid Values to Not Available

Using the SETVariables.htm file in Box, we were able to discern which invalid values were found in the dataset.  We converted each of these values in the data set to not available.

### Questions for Jeff

As we sorted through each SET Variable in the dataset we made the decision to drop many we believed weren't necessary for out purposes.  We also came across several which we lacked an understanding of and so it is these that we want to ask Jeff about.  For the most part, they pertain to process of the GRYD program (cycle/phase) and if there are different services or subprograms that operate within.

### Creating  new dataframe representative of the full questionnaire (including comments)

Our first goal is to create a cleaned dataframe that is representative of the full questionnaire (including non-numeric responses).  We began to do this, however, we did include the SET Variables that we are unsure about as a reminder to ask Jeff about them.  Otherwise, we included SET Variables we thought to be pertinent and all of the questionnaire responses.

