from phospho import extract_phospholingo_score

def phospho():
    model_loc = r"H:\PhosphoLingo_ST_new.ckpt"
    sequence_ids = ["test1"]
    sequences = ["AsWWWWWA"]
    sites = [2]
    scores = extract_phospholingo_score(sequences, sites, model_loc, sequence_ids)
    print(scores)


if __name__ == "__main__":
    phospho()