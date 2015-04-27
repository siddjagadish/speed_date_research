import pandas as pd
import json
import gzip

def get_self_id(row):
    gender = row['gender']
    participants = row['dateid'].split('_')
    return int(max(participants)) if gender == 'FEMALE' else int(min(participants))

def get_other_id(row):
    gender = row['gender']
    participants = row['dateid'].split('_')
    return int(min(participants)) if gender == 'FEMALE' else int(max(participants))

def get_male(dateid):
    return int(min(dateid.split('_')))

def get_female(dateid):
    return int(max(dateid.split('_')))

# Gets lexical predictors, one row per conversation
def get_lexical_predictors(convert_to_proportions = True):
    print 'Getting lexical predictors...'
    json_rows = gzip.open('data/conversations.json.gz', 'r').readlines()
    logs = [json.loads(x) for x in json_rows]
    lex_logs = [log for log in logs if log.get('lexical_features') is not None]
    lex_feats = [log.get('lexical_features') for log in lex_logs] 
    male_lex_feats, female_lex_feats = pd.DataFrame([x['male'] for x in lex_feats]),  pd.DataFrame([x['female'] for x in lex_feats])
    if convert_to_proportions:
        male_prop_df = male_lex_feats.copy()
        female_prop_df = female_lex_feats.copy()
        male_prop_df, female_prop_df = male_prop_df.apply(lambda x: x / male_lex_feats['totalwords'], axis=0), female_prop_df.apply(lambda x: x / female_lex_feats['totalwords'], axis=0)
        male_prop_df['totalwords'], female_prop_df['totalwords'] = male_lex_feats['totalwords'], female_lex_feats['totalwords']    
        male_prop_df.columns, female_prop_df.columns = ['%s_MALE' % x for x in male_lex_feats.columns], ['%s_FEMALE' % x for x in female_lex_feats.columns]
        ret_df = pd.concat([male_prop_df, female_prop_df], axis=1)
    else:
        ret_df = pd.concat([male_lex_feats, female_lex_feats], axis=1)
    ret_df['male'] = [log.get('male_id') for log in lex_logs]
    ret_df['female'] = [log.get('female_id') for log in lex_logs]
    return ret_df

#Gets acoustic predictors, one row per conversation
def get_acoustic_predictors():
    print 'Getting acoustic predictors...'
    acoustic_predictors = pd.read_csv('data/acoustic_final_feats.tsv', sep='\t', header=0)
    acoustic_predictors['male'] = acoustic_predictors['dateid'].map(get_male)
    acoustic_predictors['female'] = acoustic_predictors['dateid'].map(get_female)
    male_acoustic_predictors, female_acoustic_predictors = acoustic_predictors[acoustic_predictors['gender'] == 'MALE'], acoustic_predictors[acoustic_predictors['gender'] == 'FEMALE']
    male_acoustic_predictors.columns, female_acoustic_predictors.columns = ['%s_MALE' % col for col in male_acoustic_predictors.columns],['%s_FEMALE' % col for col in female_acoustic_predictors.columns]
    total = pd.merge(male_acoustic_predictors, female_acoustic_predictors, how='outer', left_on=['male_MALE', 'female_MALE'], right_on=['male_FEMALE', 'female_FEMALE'])
    total['male'] = total['male_FEMALE'].fillna(total['male_MALE'])
    total['female'] = total['female_FEMALE'].fillna(total['female_MALE'])
    total = total.drop(['gender_MALE', 'gender_FEMALE', 'male_MALE', 'male_FEMALE', 'female_MALE', 'female_FEMALE'], axis=1)
    return total

def combine_lex_and_acoustic(lexical_predictors, acoustic_predictors):
    all_predictors = pd.merge(lexical_predictors, acoustic_predictors, how='outer', on = ['male', 'female'])
    all_predictors['has_lex'] = ~all_predictors['totalwords_MALE'].isnull()
    all_predictors['has_acoustic'] = ~(all_predictors['dateid_MALE'].isnull() & all_predictors['dateid_FEMALE'].isnull())
    all_predictors['has_male_acoustic'] = ~all_predictors['dateid_MALE'].isnull()
    all_predictors['has_female_acoustic'] = ~all_predictors['dateid_FEMALE'].isnull()
    return all_predictors

def get_outcomes():
    print 'Getting outcomes...'
    orig_df = pd.read_csv('data/speeddateoutcomes.csv', sep=',', header=0, na_values = ['.', ' .', '. '])
    outcome_get_male = lambda row: min(row['selfid'], row['otherid'])
    outcome_get_female = lambda row: max(row['selfid'], row['otherid'])
    outcome_gender = lambda row: 'MALE' if row['selfid'] < row['otherid'] else 'FEMALE'
    orig_df['male'] = orig_df.apply(outcome_get_male, axis=1)
    orig_df['female'] = orig_df.apply(outcome_get_female, axis=1)
    orig_df['speakergender'] = orig_df.apply(outcome_gender, axis=1)
    male_outcomes, female_outcomes = orig_df[orig_df['speakergender'] == 'MALE'], orig_df[orig_df['speakergender'] == 'FEMALE'] 
    total = pd.merge(male_outcomes, female_outcomes, on=['male', 'female'], suffixes=['_MALE', '_FEMALE'], how='inner')
    total = total.drop(['speakergender_MALE', 'speakergender_FEMALE', 'selfid_MALE', 'otherid_MALE', 'selfid_FEMALE', 'otherid_FEMALE'], axis=1)
    return total

def main():
    lexical_predictors = get_lexical_predictors()
    acoustic_predictors = get_acoustic_predictors()
    all_predictors = combine_lex_and_acoustic(lexical_predictors, acoustic_predictors)
    outcomes = get_outcomes()
    full_df = pd.merge(all_predictors, outcomes, how='left', on=['male', 'female'], suffixes=['_PREDICTORS', '_OUTCOMES'])
    full_df.to_csv('data/featurized.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()
