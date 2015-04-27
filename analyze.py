import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from analyzer import ClassifierModel, Analyzer 

def compare_to_cutoffs(row, split_var, label, lower_cutoffs, upper_cutoffs):
    cur_lb = max(1, lower_cutoffs[row[split_var]] - 1)
    cur_ub = min(10, upper_cutoffs[row[split_var]] + 1)
    if row[label] <= cur_lb:
        return 0
    if row[label] >= cur_ub: 
        return 1
    else:
        return None

def get_pos_neg_by_sorting_for_one_rater(x, label, prop=3.0, max_num_examples=3.0):
    sorted = x[~x[label].isnull()].sort(label, ascending=True)
    num_examples_to_take = min(int(len(sorted) / prop), int(max_num_examples))
    '''
    if num_examples_to_take < max_num_examples:
        print x[['male', 'female']].iloc[0] 
    '''
    ret_df = pd.concat([sorted.iloc[:num_examples_to_take], sorted.iloc[-num_examples_to_take:]], axis=0)
    ret_df[label + '_for_classification'] = 1
    ret_df[label + '_for_classification'].iloc[:num_examples_to_take] = 0 
    return ret_df

def get_pos_neg_examples(df, rater_ids, label, prop=4.0, max_num_examples=3.0):
    grouped = df.groupby(rater_ids)
    return grouped.apply(lambda x: get_pos_neg_by_sorting_for_one_rater(x, label, prop=prop, max_num_examples=max_num_examples)) 

#def main()
df = pd.read_csv('data/featurized.tsv', sep='\t', header=0)

#Courteous Male Rater's ratings for Female speaker's acoustic features only
female_acoustic_df = df[df.has_female_acoustic].copy()
crteos_classification = get_pos_neg_examples(female_acoustic_df, 'male', 'o_crteos_MALE')
analyzer = Analyzer(crteos_classification) 
'''
crteos_classification = get_pos_neg_examples(df, 'selfid', 'o_crteos', [0.25, 0.75])
crteos_classification = crteos_classification[~crteos_classification['o_crteos_for_classification'].isnull()]
analyzer = Analyzer(crteos_classification)
'''
