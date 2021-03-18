import itertools
import yaml
import numpy as np
from scipy import stats

def extract_score_list():
    num_ops = 4
    model_encod_list = [
        # 0
        [[1, 1],
         [1, 1, 1, 1],
         [1, 1, 0, 1],
         [1, 0, 1, 1],
         [1, 1, 0, 1]],
        # 10
        [[1, 1],
         [1, 0, 3, 1],
         [1, 3, 0, 1],
         [1, 1, 1, 0],
         [3, 0, 1, 1]],
        # 20
        [[1, 0],
         [1, 2, 3, 1],
         [1, 3, 1, 1],
         [1, 2, 2, 1],
         [3, 1, 3, 1]],
        # 30
        [[1, 0],
         [3, 1, 1, 0],
         [1, 2, 2, 0],
         [1, 0, 0, 0],
         [3, 0, 2, 1]],
        # 40
        [[0, 1],
         [3, 1, 0, 0],
         [3, 2, 0, 1],
         [3, 0, 1, 1],
         [0, 3, 0, 1]],
        # 50
        [[0, 0],
         [3, 0, 2, 1],
         [3, 1, 2, 0],
         [3, 3, 1, 1],
         [0, 0, 3, 1]],
        # 60
        [[0, 0],
         [3, 2, 3, 1],
         [3, 3, 3, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        # 70
        [[3, 1],
         [2, 1, 1, 1],
         [1, 0, 1, 2],
         [2, 1, 0, 1],
         [3, 2, 1, 0]],
        # 80
        [[2, 1],
         [0, 3, 3, 1],
         [1, 0, 3, 3],
         [3, 2, 0, 1],
         [0, 2, 2, 1]],
        # 90
        [[2, 1],
         [2, 1, 3, 1],
         [1, 2, 2, 3],
         [2, 0, 3, 1],
         [1, 2, 0, 0]],
        # 100
        [[3, 0],
         [0, 0, 0, 0],
         [0, 1, 0, 0],
         [2, 3, 2, 1],
         [3, 0, 2, 0]],
        # 110
        [[3, 0],
         [3, 2, 2, 0],
         [3, 0, 1, 3],
         [2, 0, 2, 1],
         [0, 0, 3, 0]],
        # 120
        [[2, 0],
         [0, 3, 2, 0],
         [0, 0, 2, 0],
         [1, 1, 1, 2],
         [0, 0, 0, 0]],
        # 130
        [[1, 3],
         [2, 3, 2, 1],
         [0, 2, 0, 1],
         [1, 1, 2, 3],
         [1, 1, 0, 3]],
        # 140
        [[1, 3],
         [3, 1, 0, 3],
         [0, 2, 0, 0],
         [1, 2, 3, 3],
         [1, 3, 3, 3]],
        # 150
        [[1, 2],
         [0, 2, 2, 0],
         [0, 3, 3, 0],
         [1, 0, 0, 2],
         [1, 3, 1, 2]],
        # 160
        [[0, 3],
         [2, 0, 2, 0],
         [2, 1, 0, 1],
         [1, 0, 2, 2],
         [3, 2, 3, 3]],
        # 170
        [[0, 3],
         [0, 0, 1, 3],
         [2, 0, 3, 1],
         [0, 3, 1, 2],
         [0, 1, 0, 3]],
        # 180
        [[0, 2],
         [3, 2, 0, 3],
         [2, 3, 1, 1],
         [3, 1, 3, 3],
         [0, 2, 1, 3]],
        # 190
        [[0, 2],
         [3, 0, 3, 2],
         [2, 3, 2, 0],
         [3, 2, 1, 3],
         [0, 3, 0, 3]],
        # 200
        [[3, 3],
         [0, 0, 3, 3],
         [0, 1, 2, 3],
         [3, 3, 3, 3],
         [1, 0, 0, 2]],
        # 210 (not retrained)
        # [[3, 2],
        #  [0, 3, 0, 2],
        #  [0, 2, 0, 3],
        #  [3, 0, 3, 2],
        #  [2, 3, 3, 3]],
        # 220
        [[3, 2],
         [2, 1, 3, 3],
         [2, 1, 0, 2],
         [0, 2, 2, 3],
         [0, 0, 2, 3]],
        # 230 (not retrained)
        # [[2, 3],
        #  [3, 2, 2, 2],
        #  [2, 0, 1, 3],
        #  [3, 2, 2, 3],
        #  [0, 1, 2, 2]],
        # 240 (not retrained)
        # [[2, 2],
        #  [2, 3, 0, 2],
        #  [2, 3, 0, 3],
        #  [2, 3, 3, 2],
        #  [2, 1, 2, 2]],
        # 250
        [[2, 2],
         [2, 2, 0, 2],
         [2, 3, 2, 3],
         [2, 0, 0, 2],
         [0, 0, 2, 2]]
    ]
    potential_yaml = [
        ['./path_rank/path_rank_0.yml'],
        ['./path_rank/path_rank_1.yml'],
        ['./path_rank/path_rank_2.yml'],
        ['./path_rank/path_rank_3.yml'],
        ['./path_rank/path_rank_4.yml'],

    ]
    loss_list = []
    for model_encod in model_encod_list:
        total_loss = 0

        for stage, stage_encod in enumerate(model_encod):
            with open(potential_yaml[stage][0], 'r') as f:
                # potential = str_to_dict(f.read())
                potential = dict(yaml.load(f))

            stage_model_pool = list(
                itertools.product(list(range(num_ops)), repeat=len(stage_encod)))
            assert len(stage_model_pool) == len(potential), \
                'length mismatch in stage {}. model pool {},  potential {}'.format(
                    stage, len(stage_model_pool), len(potential))
            stage_encod = ''.join([str(code) for code in stage_encod])
            loss = potential[stage_encod] + 4.
            # loss = 100 - potential[stage_encod]
            total_loss += loss
        loss_list.append(round(total_loss, 4))
    # print('\nloss:\n', loss_list)
    return loss_list


def str_to_dict(a: str):
    return dict([(b.split(', ')[0].strip('\''), float(b.split(', ')[1]))
                 for b in a.strip('[]()').split('), (')])


if __name__ == '__main__':
    loss_list = extract_score_list()
    pre_loss = np.array(loss_list)
    TrueAcc = [75.52, 75.00, 74.66, 74.79, 74.76, 74.46, 74.93, 74.766, 74.36, 74.43, 74.57, 74.52, 74.89, 74.55, 74.34,
               74.62, 74.03, 74.33, 74.14, 73.86, 74.38, 73.85, 73.58]
    TrueAcc = np.array(TrueAcc)
    print("BossNAS :\n{}\n{}\n{}\n".format(stats.kendalltau(TrueAcc, pre_loss),
                                           stats.pearsonr(TrueAcc, pre_loss),
                                           stats.spearmanr(TrueAcc, pre_loss)))
    