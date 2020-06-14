# Reference: https://github.com/raedovj/NN19_Project_Football/blob/master/ModelTester.py
# imports
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical

def predict_always_on_one_thing_benefit(labels, betting_odds, predictable_value):
    predictable_indices = np.zeros((labels.shape[0], 3))
    predictable_indices[:, predictable_value] = 1
    
    agency = "B365"
    odds = pd.concat([betting_odds[agency+'H'], betting_odds[agency+'D'], betting_odds[agency+'A']], axis=1)
    # r holds betting results. 0 indicates loss, value otherwise shows the win amount
    r = odds * predictable_indices * to_categorical(labels)
    # Take max value of win, draw, other win. 
    r = r.values.max(axis=1)
    # Let's say we bet 1€. then our profit(or loss) would be = earnings€ - 1€ per bet
    r -= len(predictable_value)
    r

#     print("Agency %s, \twin amount: %.2f" % (agency, r.sum()))
    print("Agency %s, \twin amount: %.2f" % (agency, np.nansum(r)))
    
def always_bet_predicted_winner_profit(predictions, labels, betting_odds):      
    predictions_categorical = to_categorical(predictions)
    
    agency = "B365"
    odds = pd.concat([betting_odds[agency+'H'], betting_odds[agency+'D'], betting_odds[agency+'A']], axis=1)
    # r holds betting results. 0 indicates loss, value otherwise shows the win amount
    r = odds * predictions_categorical * to_categorical(labels)
    # Take max value of win, draw, other win. 
    r = r.values.max(axis=1)
    # Let's say we bet 1€. then our profit(or loss) would be = earnings€ - 1€
    r -= 1
    
#     print("Agency %s, \twin amount: %.2f" % (agency, r.sum()))
    print("Agency %s, \twin amount: %.2f" % (agency, np.nansum(r)))

def bet_predicted_winner_with_threshold_profit(predictions_3x1, predictions, labels, 
                                                          betting_odds, threshold):
    predictions_categorical = to_categorical(predictions)
    
    agency = "B365"
    odds = pd.concat([betting_odds[agency+'H'], betting_odds[agency+'D'], betting_odds[agency+'A']], axis=1)
    # r holds betting results. 0 indicates loss, value otherwise shows the win amount
    bet = odds * predictions_categorical * predictions_3x1
    bet = bet > threshold
    r = odds * predictions_categorical * to_categorical(labels)
    r -= 1
    # Set win/lose amount to 0 on matched it didn't bet
    r[np.invert(bet)] = 0
    # Take max value of win, draw, other win. 
    r = r.values.max(axis=1)

    skip_percentage = (r==0).sum() / r.shape[0] * 100   
    print("Agency %s, \twin amount: %.2f. Didn't bet on %.2f%% of matches" % (agency, r.sum(), skip_percentage))
    
def predict_on_highest_return(predictions_3x1, labels, betting_odds, threshold):
        agency = "B365"
        odds = pd.concat([betting_odds[agency+'H'], betting_odds[agency+'D'], betting_odds[agency+'A']], axis=1)
        # Expected earning value. Basically expects that our NN predicts real match outcomes
        expected = (odds * predictions_3x1).values

        # Threshold matches, when we'd actually would make a bet. If expected yield is too low, it'll pass
        bet = np.max(expected > threshold, axis=1)

        # Take the highest yield of [home win, draw, other win]
        r = np.argmax(expected, axis=1) 

        # Calculate wins/losses according to real match results
        r = to_categorical(r) * to_categorical(labels)
        r -= 1 # subtract our input bet

        # Calculate earnings
        r = r.max(axis=1) # Take max value of win, draw, other win. 
        r[np.invert(bet)] = 0 # Set win/lose amount to 0 on matched it didn't bet

        skip_percentage = (bet==0).sum() / bet.shape[0] * 100   
        print("Agency %s, \twin amount: %.2f. Didn't bet on %.2f%% of matches" % (agency, r.sum(), skip_percentage))  