from pathlib import Path

# Define constants used in majority_vote_sorting.py
MV_BG_LABEL_DISAGREE = 'mv_all_bg_label_disagrees'
LABEL_BG_MV_DISAGREE = 'label_all_bg_pred_disagrees'
LABEL_BG_MV_AGREE = 'label_all_bg_pred_agrees'
STRONG_AGREE = 'strong_agreement'
LANES_SWAPPED = 'lanes_swapped'
DIFFICULT_SITUATIONS = 'difficult_situations'
CROSSING = 'crossing'


# Join sub paths given in argv and return string representation
def path_join(*argv):
    return str(Path('').joinpath(*argv))
