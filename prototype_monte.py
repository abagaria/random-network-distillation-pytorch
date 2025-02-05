from experiment import Experiment

segmentor_args = {
    "n_segments": 50,
    "compactness": 3
}

experiment = Experiment(batch_size=32,
                        learning_rate=0.1,
                        type="ig",
                        segmentor_type="sam",
                        **segmentor_args)

# experiment.load_train_data("/mnt/nfs/home/ademello/research/acme/resources/trajectories/monte")
experiment.load_train_data("/mnt/nfs/home/ademello/research/acme/resources/trajectories/monte-test")
experiment.load_test_data("/mnt/nfs/home/ademello/research/acme/resources/trajectories/monte-test")

experiment.train(epochs=10)
experiment.get_classifier()