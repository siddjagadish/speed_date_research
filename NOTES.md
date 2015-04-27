##Mon Apr 27 02:20:22 UTC 2015 (Sidd)
### Basic strategy to get myself setup for modeling without cheating!
* Get Positive and Negative Features
    * Group by the rater gender
    * Take the rows corresponding top x and bottom x ratings, provided that at least some configurable proportion is left out (by default, middle 50%)  Otherwise, take out the maximum number of values such that the desired proportion is left out.  Label the top x ratings 1 and the bottom x ratings 0.

* Cross-validation splitting
    * We've normalized ratings by rater, so it doesn't matter if the rater appears in both the training and the test sets.  There's nothing to gain by identifying the rater unless different raters use different criteria by which to rate the other participants, and if this is the case, it's fair game.
    * As such, split by the other gender (so if we're using a male rating as an outcome, ensure each fold of cross-validation has a separate group of female participants) 

##Mon Apr 27 02:03:24 UTC 2015 (Sidd)
* First run featurize to get full feature set (data/featurized.tsv), then run analyze (in progress) to run models and such.
*  Major assumption: In speeddateoutcomes.csv, o_crteos means the rating the speaker gave the other participant for courteousness, not vice-versa.
* I have codified everything as male or female, with one row per conversation. o_crteos_MALE means the courteousness rating the male assigned to the female.
