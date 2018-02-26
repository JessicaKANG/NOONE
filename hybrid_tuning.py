from utils.DataReader import Dataset
from utils.MapEvaluator import Evaluator
from utils.predictor import predict
from utils.sample_generator import generate_sample
from recommenders.Hybrid import HYBRID
import random
import numpy as np


def generate_gene(e_best):
    e = e_best
    for i in range(0, 4):
        e[random.randint(0, 3)] = random.random()
    return e


def create_chromosome(number, e_best):
    chromosome = []
    evaluation = []
    for i in range(0, number):
        e = generate_gene(e_best.copy())
        chromosome.append(e)
        evaluation.append(0)
    return chromosome, evaluation



ds = Dataset()
ev = Evaluator()
hybrid = HYBRID()

e_best = [1, 1, 1, 1]
while 1 > 0:
    size = 1000
    chromosome, evaluation = create_chromosome(size, e_best.copy())
    best_ev = 0
    for n in range(0, 10):
        idx = random.randint(0, 99)
        name = str(idx).zfill(5)

        urm, target_playlist, target_tracks, answer = generate_sample("a", name)

        coefficient = e_best
        hybrid.fit(urm, target_playlist, target_tracks, ds, coefficient)

        R_hat = hybrid.get_estimate_rating()
        R_hat_ucf = hybrid.get_eR_ucf()
        R_hat_ucbf = hybrid.get_eR_ucbf()
        R_hat_icf = hybrid.get_eR_icf()
        R_hat_icbf = hybrid.get_eR_icbf()

        recs = predict(R_hat, target_playlist, target_tracks)
        map_5_org = ev.evaluate_ent(recs, answer)
        #print("R_hat Map@5: " + str(map_5_org))
        best_ev += map_5_org

        for i in range(0, size):
            e = chromosome[i]
            #coefficient: [ucbf, ucf, icbf, icf]
            R2 = e[0] * R_hat_ucbf + e[1] * R_hat_ucf + e[2] * R_hat_icbf + e[3] * R_hat_icf
            recs = predict(R2, target_playlist, target_tracks)
            map_5_new = ev.evaluate_ent(recs, answer)
            print(" Map@5: " + str(map_5_new))

            evaluation[i] += map_5_new

    best_gene_idx = np.argmax(evaluation)
    if evaluation[best_gene_idx] > best_ev:
        e_best = chromosome[best_gene_idx]
        print("change e_best to ", e_best)

# best for now [0.27557363982735483, 0.08170386431760779, 0.13190978963900768, 0.5666835731111661]
# e_best = [0.27557363982735483, 0.08170386431760779, 0.13190978963900768, 0.5666835731111661]
# target_playlist = ds.get_target_pl()
# target_tracks = ds.get_target_tr()
# urm = ds.build_urm()
# hybrid.fit(urm, target_playlist, target_tracks, ds, e_best)
# hybrid.submit_result()
