if __name__ == "__main__":
    import pandas as pd

    #DO NOT CHANGE ANYTHING IN THIS CELL. MOVE ON TO THE FOLLOWING ONE TO GET THE PREDICTION.
    import os
    os.environ['PATH'] = "/Users/newuser/ncbi-blast-2.16.0+/bin:" + os.environ['PATH']

    import numpy as np
    from gensim.models import word2vec

    class ProtVec(word2vec.Word2Vec):

        def __init__(self, fasta_fname=None, corpus=None, n=3, size=100, corpus_fname="corpus.txt",  sg=1, window=25, min_count=1, workers=20):
            """
            Either fname or corpus is required.
            fasta_fname: fasta file for corpus
            corpus: corpus object implemented by gensim
            n: n of n-gram
            corpus_fname: corpus file path
            min_count: least appearance count in corpus. if the n-gram appear k times which is below min_count, the model does not remember the n-gram
            """

            self.n = n
            self.size = size
            self.fasta_fname = fasta_fname

            if corpus is None and fasta_fname is None:
                raise Exception("Either fasta_fname or corpus is needed!")

            if fasta_fname is not None:
                print('Generate Corpus file from fasta file...')
                generate_corpusfile(fasta_fname, n, corpus_fname)
                corpus = word2vec.Text8Corpus(corpus_fname)

            word2vec.Word2Vec.__init__(self, corpus, size=size, sg=sg, window=window, min_count=min_count, workers=workers)

        def to_vecs(self, seq):
            """
            convert sequence to three n-length vectors
            e.g. 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ]
            """
            ngram_patterns = split_ngrams(seq, self.n)

            protvecs = []
            for ngrams in ngram_patterns:
                ngram_vecs = []
                for ngram in ngrams:
                    try:
                        ngram_vecs.append(self.wv[ngram])
                    except:
                        raise Exception("Model has never trained this n-gram: " + ngram)
                protvecs.append(sum(ngram_vecs))
            return protvecs
        
        
        def get_vector(self, seq):
            """
            sum and normalize the three n-length vectors returned by self.to_vecs
            """
            #return normalize(sum(self.to_vecs(seq)))
            return sum(self.to_vecs(seq))

        
    def load_protvec(model_fname):
        return word2vec.Word2Vec.load(model_fname)

    pv = load_protvec('__PREDICT/tools/Embeddings/swissprot_size200_window25.model')

    SEED = 42
    np.random.seed(SEED)

    from __PREDICT.deephase_utils import *

    def extract_deephase_score(seq):
        # # Create a DataFrame with the input sequence
        df = pd.DataFrame({'sequence_final': [seq]})
        
        # Call the DeePhase function (assuming it returns a string)
        deephase_result = DeePhase(df)
        
        # Extract the score from the result (assuming it's the last element after splitting)
        score_str = deephase_result.split()[-1]
        
        # # Create a new DataFrame with the sequence and the score
        # new_df = pd.DataFrame({
        #     'sequence_final': [seq],  # Add the sequence
        #     'deephase_score': [score_str]  # Add the score
        # })
        
        # # Print the result and score for debugging
        # print(deephase_result)
        # print(score_str)
        
        # Return the new DataFrame
        return float(score_str)

    def main():

        # Load the CSV file from the current directory
        input_csv_file = "input_seqs.csv"
        dataframe = pd.read_csv(input_csv_file)

        # Ensure the 'seq' column exists
        if 'seq' not in dataframe.columns:
            raise ValueError("CSV file must contain a column named 'seq'")

        # Create a new column 'deephase_score' by applying the scoring function
        dataframe['deephase_score'] = dataframe['seq'].apply(extract_deephase_score)

        # Save the modified dataframe to a new CSV file with the results
        output_csv_file = "output_with_deephase_score.csv"
        dataframe.to_csv(output_csv_file, index=False)

        print(f"Processed results saved to {output_csv_file}")
        
    main()